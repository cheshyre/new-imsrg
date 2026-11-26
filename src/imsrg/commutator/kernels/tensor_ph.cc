// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The scalar-tensor particle-hole term comm222_phst (imsrg-devel
// Commutator/TensorCommutators.cc:536-1413), evaluated over ALL ordered
// cross-coupled pairs: Pandya transforms of X (scalar) and Y (rank lambda,
// 9j recoupling), one pair of gemms per cross-coupled channel pair, and the
// inverse transform as a gather kernel. Kernel TU.

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/commutator/detail/tensor_impl.h"
#include "imsrg/commutator/tensor.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/device_blas.h"

namespace imsrg::commutator::tensor_terms {

namespace {

using detail::CrossCoupledChannel;
using detail::Phase;
using detail::TensorTwoBodyView;

IMSRG_INLINE_FUNCTION int Imin(int a, int b) { return a < b ? a : b; }
IMSRG_INLINE_FUNCTION int Imax(int a, int b) { return a > b ? a : b; }
IMSRG_INLINE_FUNCTION int Iabs(int a) { return a < 0 ? -a : a; }

template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> b(v.size(), label);
  if (!v.empty()) b.CopyFromHost(v);
  return b;
}

/// Device arguments of the Pandya kernels.
struct PandyaArgs {
  detail::CrossCoupledView cc;
  const std::int32_t* ph_offset = nullptr;  ///< Per channel + total.
  const std::int32_t* ph_local = nullptr;   ///< Local pair indices.
  const double* ph_w = nullptr;             ///< n_a - n_b per ph pair.
  const std::int32_t* j2 = nullptr;
  const std::int32_t* l = nullptr;
  const std::int32_t* tz2 = nullptr;
  angmom::SixJView sixj;
  angmom::NineJView ninej;
  TensorTwoBodyView x2;
  TensorTwoBodyView y2;
  int L = 0;
};
static_assert(std::is_trivially_copyable_v<PandyaArgs>);

/// Xbar^{J3}_{(il),(ab)} = - sum_J' (2J'+1) {j_i j_l J3; j_a j_b J'}
/// X^{J'}_{ib,al} (unnormalized). O(J range).
IMSRG_INLINE_FUNCTION double XbarElement(const PandyaArgs& a, int i, int l,
                                         int p, int b, int J3) {
  const int ji = a.j2[i], jl = a.j2[l], ja = a.j2[p], jb = a.j2[b];
  const int jmin = Imax(Iabs(ji - jb), Iabs(ja - jl)) / 2;
  const int jmax = Imin(ji + jb, ja + jl) / 2;
  double s = 0.0;
  for (int Jp = jmin; Jp <= jmax; ++Jp) {
    s -= (2 * Jp + 1) *
         a.sixj(TwoJ{ji}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{ja}, TwoJ{jb},
                TwoJ{2 * Jp}) *
         a.x2.get_unnormalized(i, b, p, l, 2 * Jp, 2 * Jp);
  }
  return s;
}

/// Ybar^{J3 J4}_{(ab),(cd)} = - sum_{J1 J2} hat(J1 J2 J3 J4) (-1)^(j_b + j_d
/// + J4 + J2) {j_a j_d J1; j_b j_c J2; J3 J4 L} Y^{J1 J2}_{ad,cb}
/// (imsrg++ DoTensorPandyaTransformation_SingleChannel). O(J range^2).
IMSRG_INLINE_FUNCTION double YbarElement(const PandyaArgs& a, int p, int b,
                                         int c, int d, int J3, int J4) {
  const int ja = a.j2[p], jb = a.j2[b], jc = a.j2[c], jd = a.j2[d];
  double s = 0.0;
  for (int J1 = Iabs(ja - jd) / 2; J1 <= (ja + jd) / 2; ++J1) {
    const int J2min = Imax(Iabs(jc - jb) / 2, Iabs(J1 - a.L));
    const int J2max = Imin((jc + jb) / 2, J1 + a.L);
    for (int J2 = J2min; J2 <= J2max; ++J2) {
      const double nj =
          a.ninej(TwoJ{ja}, TwoJ{jd}, TwoJ{2 * J1}, TwoJ{jb}, TwoJ{jc},
                  TwoJ{2 * J2}, TwoJ{2 * J3}, TwoJ{2 * J4});
      if (nj == 0.0) continue;
      const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                      (2.0 * J3 + 1) * (2.0 * J4 + 1));
      s -= hat * Phase((jb + jd) / 2 + J4 + J2) * nj *
           a.y2.get_unnormalized(p, d, c, b, 2 * J1, 2 * J2);
    }
  }
  return s;
}

/// Arguments of the inverse transform.
struct InverseArgs {
  detail::CrossCoupledView cc;
  const std::int32_t* j2 = nullptr;
  const std::int32_t* l = nullptr;
  const std::int32_t* tz2 = nullptr;
  angmom::NineJView ninej;
  basis::TwoBodyTableView tb;
  const std::int32_t* pair_of = nullptr;  ///< ncc^2 -> pair index or -1.
  const std::int64_t* zbar_off = nullptr;
  const std::int32_t* cc_size = nullptr;  ///< Pairs per cc channel.
  const double* zbar = nullptr;
  /// Fork mode only: the part Xbar3[:, ph3] YA of `zbar` (same layout).
  const double* zbar_xy = nullptr;
  std::int32_t ncc = 0;
  int L = 0;
  int hZ = 1;
  /// Reproduce the 79e9330 fork (odd rank L only), see
  /// `TensorOptions::fork_ph_missing_rank_phase`: `Zbar` reads the matrices
  /// the way the fork does (the stored matrices then already carry the fork's
  /// sign error of the flipped ket orientation).
  bool fork_odd_rank_phase = false;
  // Z blocks
  const std::int32_t* ch_bra = nullptr;
  const std::int32_t* ch_ket = nullptr;
  const std::int32_t* nbra = nullptr;
  const std::int32_t* nket = nullptr;
  const std::int64_t* offset = nullptr;
  const std::int32_t* row_offset = nullptr;
  std::int32_t num_blocks = 0;
  double* zd = nullptr;

  /// Zbar^{J3 J4}_{(il),(kj)}, 0 if the channel pair was not computed.
  IMSRG_INLINE_FUNCTION double Zbar(int i, int ll, int k, int j, int J3,
                                    int J4) const {
    const int c3 = CrossCoupledChannel(l, tz2, i, ll, J3);
    const int c4 = CrossCoupledChannel(l, tz2, k, j, J4);
    if (fork_odd_rank_phase) return ZbarFork(i, ll, k, j, J3, J4, c3, c4);
    const int r = cc.local_index(i, ll, J3);
    const int c = cc.local_index(k, j, J4);
    if (c3 <= c4) {
      const int P = pair_of[c3 * ncc + c4];
      if (P < 0) return 0.0;
      return zbar[zbar_off[P] + r + static_cast<std::int64_t>(c) * cc_size[c3]];
    }
    // Only c3 <= c4 is stored: Zbar^{c3 c4}_{x,y} = hZ (-1)^(J3-J4)
    // Zbar^{c4 c3}_{y,x} (hermiticity of the tensor Pandya transform,
    // imsrg++ AddInverseTensorPandyaTransformation).
    const int P = pair_of[c4 * ncc + c3];
    if (P < 0) return 0.0;
    return hZ * Phase(J3 - J4) *
           zbar[zbar_off[P] + c + static_cast<std::int64_t>(r) * cc_size[c4]];
  }

  /// Stored element (row pair (a, b) at Ja, column pair (c, d) at Jb) of the
  /// matrix of the channel pair (ca, cb), ca <= cb; 0 if not computed.
  IMSRG_INLINE_FUNCTION double Stored(int ca, int cb, int a, int b, int Ja,
                                      int c, int d, int Jb) const {
    const int P = pair_of[ca * ncc + cb];
    if (P < 0) return 0.0;
    return zbar[zbar_off[P] + cc.local_index(a, b, Ja) +
                static_cast<std::int64_t>(cc.local_index(c, d, Jb)) *
                    cc_size[ca]];
  }

  /// The fork's "flipped" copy of a column whose pair is diagonal (c == d):
  /// the fork stores every ket pair twice (k <= j and, flipped, k >= j), so a
  /// diagonal pair (j, j) has a second column built from the flipped Ybar
  /// whose (-1)^L is missing. Its Xbar YA part therefore has the opposite
  /// sign for odd L: Stored - 2 Xbar YA.
  IMSRG_INLINE_FUNCTION double StoredFlippedDiagonal(int ca, int cb, int a,
                                                     int b, int Ja, int c,
                                                     int Jb) const {
    const int P = pair_of[ca * ncc + cb];
    if (P < 0) return 0.0;
    const std::int64_t e =
        zbar_off[P] + cc.local_index(a, b, Ja) +
        static_cast<std::int64_t>(cc.local_index(c, c, Jb)) * cc_size[ca];
    return zbar[e] - 2.0 * zbar_xy[e];
  }

  /// The fork's reading of Zbar^{J3 J4}_{(il),(kj)} (Commutator.cc:12396-
  /// 12405): the stored rows are the pairs (a, b) with a <= b; the others
  /// are reached through the swap of both pairs, with the phases
  /// (-1)^(J3 - J4 + j_i + j_j + j_k + j_l) (c3 <= c4, i > l) and
  /// (-1)^(j_i + j_j + j_k + j_l) (c3 > c4, k > j), which lack the factor
  /// (-1)^L of imsrg-devel. The matrices must have been built with the sign
  /// error of the fork (YA columns of pairs (k, j) with k > j negated). The
  /// fork reads the flipped half of its doubled columns also for a diagonal
  /// pair ((k > j ? 0 : nkets) with k == j, Commutator.cc:12399 and 12404,
  /// 12472 and 12477): `StoredFlippedDiagonal`.
  IMSRG_INLINE_FUNCTION double ZbarFork(int i, int ll, int k, int j, int J3,
                                        int J4, int c3, int c4) const {
    const int S = (j2[i] + j2[j] + j2[k] + j2[ll]) / 2;
    if (c3 <= c4) {
      if (i <= ll) return Stored(c3, c4, i, ll, J3, k, j, J4);
      const double z = (k == j)
                           ? StoredFlippedDiagonal(c3, c4, ll, i, J3, j, J4)
                           : Stored(c3, c4, ll, i, J3, j, k, J4);
      return hZ * Phase(J3 - J4 + S) * z;
    }
    if (k <= j)
      return hZ * Phase(J3 - J4) * Stored(c4, c3, k, j, J4, i, ll, J3);
    const double z = (i == ll) ? StoredFlippedDiagonal(c4, c3, j, k, J4, i, J3)
                               : Stored(c4, c3, j, k, J4, ll, i, J3);
    return Phase(S) * z;
  }
};
static_assert(std::is_trivially_copyable_v<InverseArgs>);

/// One element (row rr of the stacked stored blocks, column iket) of the
/// inverse transform of comm222_phst, plus its mirror in a diagonal block.
/// Every element is written by exactly one (rr, iket), so elements run
/// concurrently without atomics. O((J3 range) x (J4 range)) 9j-weighted
/// Zbar gathers.
IMSRG_FORCEINLINE_FUNCTION void PhstInverseElement(const InverseArgs& v,
                                                   exec::Index rr, int iket) {
  const int k = detail::FindBlock(v.row_offset, v.num_blocks, rr);
  const int ibra = static_cast<int>(rr) - v.row_offset[k];
  const int cb = v.ch_bra[k];
  const int ck = v.ch_ket[k];
  const int J1 = v.tb.channel_two_j[cb] / 2;
  const int J2 = v.tb.channel_two_j[ck] / 2;
  const int nb = v.nbra[k];
  const KetIndex gb = v.tb.global_ket(ChannelIndex{cb}, LocalKetIndex{ibra});
  const int i = v.tb.ket_p[gb.value()];
  const int j = v.tb.ket_q[gb.value()];
  const int ji = v.j2[i];
  const int jj = v.j2[j];
  double* blk = v.zd + v.offset[k];
  const KetIndex gk = v.tb.global_ket(ChannelIndex{ck}, LocalKetIndex{iket});
  const int kk = v.tb.ket_p[gk.value()];
  const int ll = v.tb.ket_q[gk.value()];
  const int jk = v.j2[kk];
  const int jl = v.j2[ll];
  double commij = 0.0;
  for (int J3 = Iabs(ji - jl) / 2; J3 <= (ji + jl) / 2; ++J3) {
    const int J4min = Imax(Iabs(jk - jj) / 2, Iabs(J3 - v.L));
    const int J4max = Imin((jk + jj) / 2, J3 + v.L);
    for (int J4 = J4min; J4 <= J4max; ++J4) {
      const double nj =
          v.ninej(TwoJ{ji}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{jj}, TwoJ{jk},
                  TwoJ{2 * J4}, TwoJ{2 * J1}, TwoJ{2 * J2});
      if (nj == 0.0) continue;
      const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                      (2.0 * J3 + 1) * (2.0 * J4 + 1));
      commij += hat * Phase((jj + jl) / 2 + J2 + J4) * nj *
                v.Zbar(i, ll, kk, j, J3, J4);
    }
  }
  double commji = commij;
  if (i != j) {
    commji = 0.0;
    for (int J3 = Iabs(jj - jl) / 2; J3 <= (jj + jl) / 2; ++J3) {
      const int J4min = Imax(Iabs(jk - ji) / 2, Iabs(J3 - v.L));
      const int J4max = Imin((jk + ji) / 2, J3 + v.L);
      for (int J4 = J4min; J4 <= J4max; ++J4) {
        const double nj =
            v.ninej(TwoJ{jj}, TwoJ{jl}, TwoJ{2 * J3}, TwoJ{ji}, TwoJ{jk},
                    TwoJ{2 * J4}, TwoJ{2 * J1}, TwoJ{2 * J2});
        if (nj == 0.0) continue;
        const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                        (2.0 * J3 + 1) * (2.0 * J4 + 1));
        commji += hat * Phase((ji + jl) / 2 + J2 + J4) * nj *
                  v.Zbar(j, ll, kk, i, J3, J4);
      }
    }
  }
  const bool dij = (i == j);
  const bool dkl = (kk == ll);
  const double norm = (dij == dkl) ? (dij ? 2.0 : 1.0) : constants::kSqrt2;
  const double val = (commij - Phase((ji + jj) / 2 - J1) * commji) / norm;
  blk[ibra + static_cast<std::int64_t>(iket) * nb] += val;
  if (cb == ck && iket > ibra)
    blk[iket + static_cast<std::int64_t>(ibra) * nb] += v.hZ * val;
}

/// Device arguments of the quadruple form of the inverse transform.
struct QuadArgs {
  InverseArgs v;
  const std::int32_t* class_kets = nullptr;  ///< Kets grouped by class.
  const std::int32_t* class_off = nullptr;   ///< 7 offsets.
  const std::int32_t* pair_a = nullptr;      ///< Class of ket a per pair.
  const std::int32_t* pair_b = nullptr;      ///< Class of ket b per pair.
  const std::int64_t* pair_items = nullptr;  ///< Cumulative items.
  std::int32_t num_pairs = 0;
  const std::int64_t* zblock = nullptr;  ///< nch^2 Z block offsets, -1 absent.
  std::int32_t nch = 0;
  std::int32_t nj = 0;  ///< Scratch J extent (2 emax + 2).
};
static_assert(std::is_trivially_copyable_v<QuadArgs>);

/// The inverse transform of comm222_phst for one ket pair (a = (i, j),
/// b = (k, l)) and all (J1, J2) (M14): the team gathers
/// Zbar^{J3 J4}_{(il),(kj)} and Zbar^{J3 J4}_{(jl),(ki)} once into team
/// scratch, then its threads compute the elements of every stored block
/// (J1, J2) that contains the pair, in exactly the order of
/// PhstInverseElement (bitwise equal); the element-wise form gathers the
/// same Zbar values once per (J1, J2).
IMSRG_FORCEINLINE_FUNCTION void PhstQuad(const exec::TeamMember& member,
                                         const QuadArgs& q, double* scratch) {
  const InverseArgs& v = q.v;
  const exec::Index item = member.league_rank();
  const int pr = detail::FindBlock(q.pair_items, q.num_pairs, item);
  const std::int64_t local = item - q.pair_items[pr];
  const int ca = q.pair_a[pr];
  const int cbl = q.pair_b[pr];
  const std::int64_t nbk = q.class_off[cbl + 1] - q.class_off[cbl];
  const int ka = q.class_kets[q.class_off[ca] + local / nbk];
  const int kb = q.class_kets[q.class_off[cbl] + local % nbk];
  const int i = v.tb.ket_p[ka];
  const int j = v.tb.ket_q[ka];
  const int kk = v.tb.ket_p[kb];
  const int ll = v.tb.ket_q[kb];
  const int ji = v.j2[i], jj = v.j2[j], jk = v.j2[kk], jl = v.j2[ll];
  const int nj = q.nj;
  double* z1 = scratch;            // [J3 - J3a][J4 - J4a], term (i, j)
  double* z2 = scratch + nj * nj;  // term (j, i)
  // 1. Gather.
  const int J3a1 = Iabs(ji - jl) / 2, n31 = (ji + jl) / 2 - J3a1 + 1;
  const int J4a1 = Iabs(jk - jj) / 2, n41 = (jk + jj) / 2 - J4a1 + 1;
  const int J3a2 = Iabs(jj - jl) / 2, n32 = (jj + jl) / 2 - J3a2 + 1;
  const int J4a2 = Iabs(jk - ji) / 2, n42 = (jk + ji) / 2 - J4a2 + 1;
  const bool two = i != j;
  const int g1 = n31 * n41;
  const int g2 = two ? n32 * n42 : 0;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, g1 + g2),
                       [&](const int t) {
                         if (t < g1) {
                           const int J3 = J3a1 + t / n41;
                           const int J4 = J4a1 + t % n41;
                           z1[(J3 - J3a1) * nj + (J4 - J4a1)] =
                               J4 >= Iabs(J3 - v.L) && J4 <= J3 + v.L
                                   ? v.Zbar(i, ll, kk, j, J3, J4)
                                   : 0.0;
                         } else {
                           const int u = t - g1;
                           const int J3 = J3a2 + u / n42;
                           const int J4 = J4a2 + u % n42;
                           z2[(J3 - J3a2) * nj + (J4 - J4a2)] =
                               J4 >= Iabs(J3 - v.L) && J4 <= J3 + v.L
                                   ? v.Zbar(j, ll, kk, i, J3, J4)
                                   : 0.0;
                         }
                       });
  member.team_barrier();
  // 2. The elements of every (J1, J2).
  const int J1a = Iabs(ji - jj) / 2, n1 = (ji + jj) / 2 - J1a + 1;
  const int J2a = Iabs(jk - jl) / 2, n2 = (jk + jl) / 2 - J2a + 1;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n1 * n2), [&](const int
                                                                         t) {
    const int J1 = J1a + t / n2;
    const int J2 = J2a + t % n2;
    const std::int32_t ibra = v.tb.local_index(KetIndex{ka}, TwoJ{2 * J1});
    if (ibra < 0) return;
    const std::int32_t iket = v.tb.local_index(KetIndex{kb}, TwoJ{2 * J2});
    if (iket < 0) return;
    const int cb =
        v.tb.channel_of(OrbitIndex{i}, OrbitIndex{j}, TwoJ{2 * J1}).value();
    const int ck =
        v.tb.channel_of(OrbitIndex{kk}, OrbitIndex{ll}, TwoJ{2 * J2}).value();
    const std::int64_t off =
        q.zblock[static_cast<std::int64_t>(cb) * q.nch + ck];
    if (off < 0) return;
    if (cb == ck && iket < ibra) return;  // lower triangle: the mirror
    const int nb = v.tb.channel_num_kets[cb];
    // The 9j boxes of both sums, looked up once per (J1, J2).
    const angmom::NineJView::Slice s1 = v.ninej.MakeSlice(
        TwoJ{ji}, TwoJ{jl}, TwoJ{jj}, TwoJ{jk}, TwoJ{2 * J1}, TwoJ{2 * J2});
    const angmom::NineJView::Slice s2 = v.ninej.MakeSlice(
        TwoJ{jj}, TwoJ{jl}, TwoJ{ji}, TwoJ{jk}, TwoJ{2 * J1}, TwoJ{2 * J2});
    double commij = 0.0;
    for (int J3 = Iabs(ji - jl) / 2; J3 <= (ji + jl) / 2; ++J3) {
      const int J4min = Imax(Iabs(jk - jj) / 2, Iabs(J3 - v.L));
      const int J4max = Imin((jk + jj) / 2, J3 + v.L);
      for (int J4 = J4min; J4 <= J4max; ++J4) {
        const double nj9 = s1(J3, J4);
        if (nj9 == 0.0) continue;
        const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                        (2.0 * J3 + 1) * (2.0 * J4 + 1));
        commij += hat * Phase((jj + jl) / 2 + J2 + J4) * nj9 *
                  z1[(J3 - J3a1) * nj + (J4 - J4a1)];
      }
    }
    double commji = commij;
    if (two) {
      commji = 0.0;
      for (int J3 = Iabs(jj - jl) / 2; J3 <= (jj + jl) / 2; ++J3) {
        const int J4min = Imax(Iabs(jk - ji) / 2, Iabs(J3 - v.L));
        const int J4max = Imin((jk + ji) / 2, J3 + v.L);
        for (int J4 = J4min; J4 <= J4max; ++J4) {
          const double nj9 = s2(J3, J4);
          if (nj9 == 0.0) continue;
          const double hat = Kokkos::sqrt((2.0 * J1 + 1) * (2.0 * J2 + 1) *
                                          (2.0 * J3 + 1) * (2.0 * J4 + 1));
          commji += hat * Phase((ji + jl) / 2 + J2 + J4) * nj9 *
                    z2[(J3 - J3a2) * nj + (J4 - J4a2)];
        }
      }
    }
    const bool dij = (i == j);
    const bool dkl = (kk == ll);
    const double norm = (dij == dkl) ? (dij ? 2.0 : 1.0) : constants::kSqrt2;
    const double val = (commij - Phase((ji + jj) / 2 - J1) * commji) / norm;
    double* blk = v.zd + off;
    blk[ibra + static_cast<std::int64_t>(iket) * nb] += val;
    if (cb == ck && iket > ibra)
      blk[iket + static_cast<std::int64_t>(ibra) * nb] += v.hZ * val;
  });
}

}  // namespace

void Comm222Phst(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 TensorWorkspace& ws, double ninej_threshold,
                 bool fork_ph_missing_rank_phase) {
  IMSRG_CHECK(ws.Compatible(X) && ws.Compatible(Y) && ws.Compatible(Z),
              "comm222_phst: operators not on the workspace's space");
  const op::OperatorSpec zs = TensorCommutatorSpec(X.spec(), Y.spec());
  IMSRG_CHECK(Z.spec().SameSymmetry(zs) && Z.hermiticity() == zs.hermiticity,
              "comm222_phst: Z must have the spec of [X, Y]");
  if (!X.two_body().allocated() || !Y.two_body().allocated() ||
      !Z.two_body().allocated())
    return;
  exec::Fence("commutator::comm222_phst");
  auto& ctx = ws.context();
  const auto& cc = ctx.cross_coupled();
  const auto& ninej = ctx.ninej(Y.spec(), ninej_threshold);
  const auto& ref = ctx.ms->reference();
  const int L = Y.spec().rank_J;
  const int piY = static_cast<int>(Y.spec().parity);
  const int dTz = Y.spec().delta_Tz;
  const int hX = op::HermiticitySign(X.hermiticity());
  const int hZ = op::HermiticitySign(zs.hermiticity);
  const std::int32_t ncc = cc.num_channels;

  // ph pairs (n_a != n_b) per cross-coupled channel.
  std::vector<std::int32_t> ph_off{0};
  std::vector<std::int32_t> ph_local;
  std::vector<double> ph_w;
  std::vector<std::int32_t> size(static_cast<std::size_t>(ncc));
  for (std::int32_t c = 0; c < ncc; ++c) {
    const std::int32_t b = cc.offset[static_cast<std::size_t>(c)];
    const std::int32_t e = cc.offset[static_cast<std::size_t>(c) + 1];
    size[static_cast<std::size_t>(c)] = e - b;
    for (std::int32_t k = b; k < e; ++k) {
      const double na =
          ref.occupation(OrbitIndex{cc.pair_a[static_cast<std::size_t>(k)]});
      const double nb =
          ref.occupation(OrbitIndex{cc.pair_b[static_cast<std::size_t>(k)]});
      if (na != nb) {
        ph_local.push_back(k - b);
        ph_w.push_back(na - nb);
      }
    }
    ph_off.push_back(static_cast<std::int32_t>(ph_local.size()));
  }
  const auto nph = [&](std::int32_t c) {
    return ph_off[static_cast<std::size_t>(c) + 1] -
           ph_off[static_cast<std::size_t>(c)];
  };
  // Channel pairs (c3, c4).
  std::vector<std::int32_t> p3, p4;
  std::vector<std::int32_t> pair_of(
      static_cast<std::size_t>(ncc) * static_cast<std::size_t>(ncc), -1);
  std::vector<std::int64_t> ya_off{0}, yb_off{0}, zb_off{0};
  std::vector<std::int64_t> e_off{0};
  std::vector<char> xc_needed(static_cast<std::size_t>(ncc), 0);
  for (std::int32_t c3 = 0; c3 < ncc; ++c3) {
    const std::int32_t n3 = size[static_cast<std::size_t>(c3)];
    if (n3 == 0) continue;
    // c4 >= c3 only: the other half follows from the hermiticity relation
    // used in `InverseArgs::Zbar`.
    for (std::int32_t c4 = c3; c4 < ncc; ++c4) {
      const std::int32_t n4 = size[static_cast<std::size_t>(c4)];
      if (n4 == 0) continue;
      const int J3 = cc.J[static_cast<std::size_t>(c3)];
      const int J4 = cc.J[static_cast<std::size_t>(c4)];
      if (L < std::abs(J3 - J4) || L > J3 + J4) continue;
      if ((cc.parity[static_cast<std::size_t>(c3)] +
           cc.parity[static_cast<std::size_t>(c4)] + piY) %
              2 !=
          0)
        continue;
      const int t3 = cc.tz[static_cast<std::size_t>(c3)];
      const int t4 = cc.tz[static_cast<std::size_t>(c4)];
      if (!(t3 + t4 == dTz || std::abs(t3 - t4) == dTz)) continue;
      pair_of[static_cast<std::size_t>(c3) * static_cast<std::size_t>(ncc) +
              static_cast<std::size_t>(c4)] =
          static_cast<std::int32_t>(p3.size());
      p3.push_back(c3);
      p4.push_back(c4);
      ya_off.push_back(ya_off.back() + static_cast<std::int64_t>(nph(c3)) * n4);
      yb_off.push_back(yb_off.back() + static_cast<std::int64_t>(n3) * nph(c4));
      zb_off.push_back(zb_off.back() + static_cast<std::int64_t>(n3) * n4);
      e_off.push_back(e_off.back() + static_cast<std::int64_t>(nph(c3)) * n4 +
                      static_cast<std::int64_t>(n3) * nph(c4));
      xc_needed[static_cast<std::size_t>(c3)] = 1;
      xc_needed[static_cast<std::size_t>(c4)] = 1;
    }
  }
  const std::int32_t npairs = static_cast<std::int32_t>(p3.size());
  if (npairs == 0) return;
  // XbarCols per channel: N_c x Nph_c.
  std::vector<std::int64_t> xc_off(static_cast<std::size_t>(ncc) + 1, 0);
  for (std::int32_t c = 0; c < ncc; ++c)
    xc_off[static_cast<std::size_t>(c) + 1] =
        xc_off[static_cast<std::size_t>(c)] +
        (xc_needed[static_cast<std::size_t>(c)]
             ? static_cast<std::int64_t>(size[static_cast<std::size_t>(c)]) *
                   nph(c)
             : 0);

  const auto d_ph_off = ToDevice(ph_off, "imsrg::ph::ph_off");
  const auto d_ph_local = ToDevice(ph_local, "imsrg::ph::ph_local");
  const auto d_ph_w = ToDevice(ph_w, "imsrg::ph::ph_w");
  const auto d_xc_off = ToDevice(xc_off, "imsrg::ph::xc_off");
  const auto d_p3 = ToDevice(p3, "imsrg::ph::p3");
  const auto d_p4 = ToDevice(p4, "imsrg::ph::p4");
  const auto d_ya_off = ToDevice(ya_off, "imsrg::ph::ya_off");
  const auto d_yb_off = ToDevice(yb_off, "imsrg::ph::yb_off");
  const auto d_e_off = ToDevice(e_off, "imsrg::ph::e_off");
  const auto d_size = ToDevice(size, "imsrg::ph::size");
  exec::DeviceBuffer<double> xc(static_cast<std::size_t>(xc_off.back()),
                                "imsrg::ph::xbar");
  exec::DeviceBuffer<double> ya(static_cast<std::size_t>(ya_off.back()),
                                "imsrg::ph::ybarA");
  exec::DeviceBuffer<double> yb(static_cast<std::size_t>(yb_off.back()),
                                "imsrg::ph::ybarB");
  exec::DeviceBuffer<double> zb(static_cast<std::size_t>(zb_off.back()),
                                "imsrg::ph::zbar");
  // Fork mode (odd L): the Xbar YA part of Zbar, see
  // `InverseArgs::StoredFlippedDiagonal`.
  const bool fork_odd = fork_ph_missing_rank_phase && (L % 2 != 0);
  exec::DeviceBuffer<double> zxy(
      fork_odd ? static_cast<std::size_t>(zb_off.back()) : 0,
      "imsrg::ph::zbar_xy");

  const std::int32_t nch = ctx.tables.two_body().num_channels;
  const auto xoff = detail::BlockOffsetTable(X.two_body(), nch);
  const auto yoff = detail::BlockOffsetTable(Y.two_body(), nch);
  PandyaArgs a;
  a.cc = detail::CrossCoupledView{
      cc.d_offset.device_data(),      cc.d_pair_a.device_data(),
      cc.d_pair_b.device_data(),      cc.d_local.device_data(),
      ctx.tables.orbits().num_orbits, cc.max_J};
  a.ph_offset = d_ph_off.device_data();
  a.ph_local = d_ph_local.device_data();
  a.ph_w = d_ph_w.device_data();
  a.j2 = ctx.tables.orbits().j2;
  a.l = ctx.tables.orbits().l;
  a.tz2 = ctx.tables.orbits().tz2;
  a.sixj = ctx.sixj.device_view();
  a.ninej = ninej.device_view();
  a.x2 = detail::MakeTensorView(X.two_body(), ctx.tables.two_body(), xoff);
  a.y2 = detail::MakeTensorView(Y.two_body(), ctx.tables.two_body(), yoff);
  a.L = L;

  // 1. Xbar columns of the ph pairs, all needed channels.
  {
    const std::int64_t* xo = d_xc_off.device_data();
    const std::int32_t* sz = d_size.device_data();
    double* out = xc.device_data();
    const std::int32_t ncc_ = ncc;
    exec::ParallelFor(
        "commutator::comm222_phst::xbar", xc_off.back(),
        IMSRG_LAMBDA(const exec::Index e) {
          const int c = detail::FindBlock(xo, ncc_, e);
          const std::int64_t loc = e - xo[c];
          const int n = sz[c];
          const int r = static_cast<int>(loc % n);
          const int q = static_cast<int>(loc / n);
          const int base = a.cc.offset[c];
          const int i = a.cc.pair_a[base + r];
          const int ll = a.cc.pair_b[base + r];
          const int s = base + a.ph_local[a.ph_offset[c] + q];
          out[e] = XbarElement(a, i, ll, a.cc.pair_a[s], a.cc.pair_b[s], c / 4);
        });
  }
  // 2. Ybar pieces, scaled by n_a - n_b on the ph index: YA = W3 Ybar[ph3, :],
  //    YB = Ybar[:, ph4] W4.
  {
    const std::int64_t* eo = d_e_off.device_data();
    const std::int64_t* yao = d_ya_off.device_data();
    const std::int64_t* ybo = d_yb_off.device_data();
    const std::int32_t* q3 = d_p3.device_data();
    const std::int32_t* q4 = d_p4.device_data();
    const std::int32_t* sz = d_size.device_data();
    double* yad = ya.device_data();
    double* ybd = yb.device_data();
    const std::int32_t np = npairs;
    // The fork's flipped orientation (k > j) of the ket pair of Ybar lacks
    // (-1)^L (Commutator.cc:12696 vs TensorCommutators.cc phase(Jbra + Jket +
    // Lambda)); only odd L is affected.
    const bool fork_flip = fork_odd;
    exec::ParallelFor(
        "commutator::comm222_phst::ybar", e_off.back(),
        IMSRG_LAMBDA(const exec::Index e) {
          const int P = detail::FindBlock(eo, np, e);
          const std::int64_t loc = e - eo[P];
          const int c3 = q3[P];
          const int c4 = q4[P];
          const int n3 = sz[c3];
          const int n4 = sz[c4];
          const int nph3 = a.ph_offset[c3 + 1] - a.ph_offset[c3];
          const int b3 = a.cc.offset[c3];
          const int b4 = a.cc.offset[c4];
          const std::int64_t na = static_cast<std::int64_t>(nph3) * n4;
          if (loc < na) {
            const int q = static_cast<int>(loc % nph3);
            const int col = static_cast<int>(loc / nph3);
            const int s = b3 + a.ph_local[a.ph_offset[c3] + q];
            const int t = b4 + col;
            yad[yao[P] + loc] =
                (fork_flip && a.cc.pair_a[t] > a.cc.pair_b[t] ? -1.0 : 1.0) *
                a.ph_w[a.ph_offset[c3] + q] *
                YbarElement(a, a.cc.pair_a[s], a.cc.pair_b[s], a.cc.pair_a[t],
                            a.cc.pair_b[t], c3 / 4, c4 / 4);
          } else {
            const std::int64_t lb = loc - na;
            const int r = static_cast<int>(lb % n3);
            const int q = static_cast<int>(lb / n3);
            const int s = b3 + r;
            const int t = b4 + a.ph_local[a.ph_offset[c4] + q];
            ybd[ybo[P] + lb] =
                a.ph_w[a.ph_offset[c4] + q] *
                YbarElement(a, a.cc.pair_a[s], a.cc.pair_b[s], a.cc.pair_a[t],
                            a.cc.pair_b[t], c3 / 4, c4 / 4);
          }
        });
  }
  exec::Fence("commutator::comm222_phst::pandya");
  // 3. Zbar^{c3 c4} = Xbar3[:, ph3] YA - hX YB Xbar4[:, ph4]^T (device BLAS:
  //    the first products in one grouped call, then the subtractions).
  {
    const double* xcd = xc.device_data();
    const double* yad = ya.device_data();
    const double* ybd = yb.device_data();
    double* zbd = zb.device_data();
    std::vector<linalg::GemmProblem> first;
    std::vector<linalg::GemmProblem> second;
    for (std::int32_t P = 0; P < npairs; ++P) {
      const std::int32_t c3 = p3[static_cast<std::size_t>(P)];
      const std::int32_t c4 = p4[static_cast<std::size_t>(P)];
      const std::int64_t n3 = size[static_cast<std::size_t>(c3)];
      const std::int64_t n4 = size[static_cast<std::size_t>(c4)];
      const std::int64_t m3 = nph(c3);
      const std::int64_t m4 = nph(c4);
      linalg::MatrixView<double> Zm{zbd + zb_off[static_cast<std::size_t>(P)],
                                    n3, n4, n3};
      linalg::MatrixView<const double> X3{
          xcd + xc_off[static_cast<std::size_t>(c3)], n3, m3, n3};
      linalg::MatrixView<const double> YA{
          yad + ya_off[static_cast<std::size_t>(P)], m3, n4, m3 > 0 ? m3 : 1};
      linalg::MatrixView<const double> YB{
          ybd + yb_off[static_cast<std::size_t>(P)], n3, m4, n3};
      linalg::MatrixView<const double> X4{
          xcd + xc_off[static_cast<std::size_t>(c4)], n4, m4, n4};
      first.push_back(linalg::GemmProblem{1.0, X3, linalg::Op::kNoTrans, YA,
                                          linalg::Op::kNoTrans, 0.0, Zm});
      if (fork_odd) {
        // The Xbar YA part alone (a separate C, same grouped call).
        linalg::MatrixView<double> Zx{
            zxy.device_data() + zb_off[static_cast<std::size_t>(P)], n3, n4,
            n3};
        first.push_back(linalg::GemmProblem{1.0, X3, linalg::Op::kNoTrans, YA,
                                            linalg::Op::kNoTrans, 0.0, Zx});
      }
      second.push_back(linalg::GemmProblem{-1.0 * hX, YB, linalg::Op::kNoTrans,
                                           X4, linalg::Op::kTrans, 1.0, Zm});
    }
    linalg::DeviceGroupedGemm(first);
    linalg::DeviceGroupedGemm(second);
  }
  // 4. Inverse transform into the stored blocks of Z.
  std::vector<std::int32_t> zcb, zck, znb, znk, zro{0};
  std::vector<std::int64_t> zoffv;
  for (const auto& info : Z.two_body().matrix().blocks()) {
    zcb.push_back(info.row_block.value());
    zck.push_back(info.col_block.value());
    znb.push_back(static_cast<std::int32_t>(info.rows));
    znk.push_back(static_cast<std::int32_t>(info.cols));
    zoffv.push_back(info.offset);
    zro.push_back(zro.back() + static_cast<std::int32_t>(info.rows));
  }
  const auto d_zcb = ToDevice(zcb, "imsrg::ph::zcb");
  const auto d_zck = ToDevice(zck, "imsrg::ph::zck");
  const auto d_znb = ToDevice(znb, "imsrg::ph::znb");
  const auto d_znk = ToDevice(znk, "imsrg::ph::znk");
  const auto d_zro = ToDevice(zro, "imsrg::ph::zro");
  const auto d_zoff = ToDevice(zoffv, "imsrg::ph::zoff");
  const auto d_pair_of = ToDevice(pair_of, "imsrg::ph::pair_of");
  const auto d_zb_off = ToDevice(zb_off, "imsrg::ph::zb_off");
  InverseArgs v;
  v.cc = a.cc;
  v.j2 = a.j2;
  v.l = a.l;
  v.tz2 = a.tz2;
  v.ninej = a.ninej;
  v.tb = ctx.tables.two_body();
  v.pair_of = d_pair_of.device_data();
  v.zbar_off = d_zb_off.device_data();
  v.cc_size = d_size.device_data();
  v.zbar = zb.device_data();
  v.ncc = ncc;
  v.L = L;
  v.hZ = hZ;
  v.fork_odd_rank_phase = fork_odd;
  v.zbar_xy = fork_odd ? zxy.device_data() : nullptr;
  v.ch_bra = d_zcb.device_data();
  v.ch_ket = d_zck.device_data();
  v.nbra = d_znb.device_data();
  v.nket = d_znk.device_data();
  v.offset = d_zoff.device_data();
  v.row_offset = d_zro.device_data();
  v.num_blocks = static_cast<std::int32_t>(zcb.size());
  v.zd = Z.two_body().matrix().buffer().device_data();
  // One item per row of the stacked stored blocks, its columns iket >= the
  // first stored one (iket >= ibra in diagonal blocks): looped inside the
  // item on the host, over a team's threads on the device (M14).
  // Quadruple form (M14): one team per ket pair (a, b) of compatible classes;
  // the Zbar values of the pair are gathered once for all (J1, J2) blocks
  // (the element form below gathers them once per block). Bitwise equal to
  // the element form. Host: always (team size 1; emax 8 rank 5 on 56 cores
  // 2348 -> 882 ms, emax 10 rank 5 18.9 -> 4.9 s, rank 2 5.3 -> 2.4 s).
  // Device: only for L >= 3, since for small L few (J1, J2) share a pair and
  // the element form fills the GPU better; emax 10 (ms, element / quad):
  // L = 1 275 / 583, 2 675 / 775, 3 1173 / 944, 4 1642 / 1094, 5 2150 / 1206.
  constexpr int kQuadMinRankDevice = 3;
  if (!exec::kDeviceKernels || L >= kQuadMinRankDevice) {
    const basis::TwoBodyChannels& tbh = ctx.ms->two_body();
    const basis::SingleParticleBasis& sp = ctx.ms->basis();
    std::vector<std::vector<std::int32_t>> by_class(6);
    for (std::int32_t kidx = 0; kidx < tbh.num_kets(); ++kidx) {
      const auto& ket = tbh.ket(KetIndex{kidx});
      const auto& op = sp.orbit(ket.p);
      const auto& oq = sp.orbit(ket.q);
      const int cls = op.tz2.twice + oq.tz2.twice + 2 + (op.l + oq.l) % 2;
      by_class[static_cast<std::size_t>(cls)].push_back(kidx);
    }
    std::vector<std::int32_t> class_kets;
    std::vector<std::int32_t> class_off{0};
    for (const auto& c : by_class) {
      class_kets.insert(class_kets.end(), c.begin(), c.end());
      class_off.push_back(static_cast<std::int32_t>(class_kets.size()));
    }
    // Class pairs that own a stored block of Z (channel = 6 J + class).
    bool used[6][6] = {};
    for (const auto& info : Z.two_body().matrix().blocks()) {
      if (info.rows == 0 || info.cols == 0) continue;
      used[info.row_block.value() % 6][info.col_block.value() % 6] = true;
    }
    std::vector<std::int32_t> pa, pb;
    std::vector<std::int64_t> pitems{0};
    for (int x = 0; x < 6; ++x)
      for (int y = 0; y < 6; ++y) {
        const auto na = static_cast<std::int64_t>(
            by_class[static_cast<std::size_t>(x)].size());
        const auto nbk = static_cast<std::int64_t>(
            by_class[static_cast<std::size_t>(y)].size());
        if (!used[x][y] || na == 0 || nbk == 0) continue;
        pa.push_back(x);
        pb.push_back(y);
        pitems.push_back(pitems.back() + na * nbk);
      }
    const auto zblock = detail::BlockOffsetTable(Z.two_body(), nch);
    const auto d_ck = ToDevice(class_kets, "imsrg::phst::class_kets");
    const auto d_co = ToDevice(class_off, "imsrg::phst::class_off");
    const auto d_pa = ToDevice(pa, "imsrg::phst::pair_a");
    const auto d_pb = ToDevice(pb, "imsrg::phst::pair_b");
    const auto d_pi = ToDevice(pitems, "imsrg::phst::pair_items");
    QuadArgs q;
    q.v = v;
    q.class_kets = d_ck.device_data();
    q.class_off = d_co.device_data();
    q.pair_a = d_pa.device_data();
    q.pair_b = d_pb.device_data();
    q.pair_items = d_pi.device_data();
    q.num_pairs = static_cast<std::int32_t>(pa.size());
    q.zblock = zblock.device_data();
    q.nch = nch;
    q.nj = 2 * ws.model_space().basis().emax() + 2;
    IMSRG_CHECK(pitems.back() <= std::int64_t{2147483647},
                "comm222_phst: too many ket pairs for one launch");
    exec::TeamOptions opts;
    opts.team_size = exec::kDeviceKernels ? 64 : 1;
    opts.schedule = exec::kDeviceKernels ? exec::Schedule::kStatic
                                         : exec::Schedule::kDynamic;
    opts.vector_length = 1;
    opts.scratch_bytes = 2 * static_cast<std::size_t>(q.nj) *
                         static_cast<std::size_t>(q.nj) * sizeof(double);
    const int level = exec::ResolveScratchLevel(opts);
    exec::TeamFor(
        "commutator::comm222_phst::inverse_quad",
        static_cast<int>(pitems.back()), opts,
        IMSRG_LAMBDA(const exec::TeamMember& member) {
          auto* scratch = static_cast<double*>(
              member.team_scratch(level).get_shmem(opts.scratch_bytes));
          PhstQuad(member, q, scratch);
        });
    exec::Fence("commutator::comm222_phst");
    return;
  }
  exec::NestedFor(
      "commutator::comm222_phst::inverse", zro.back(),
      IMSRG_LAMBDA(const exec::Index rr) {
        const int k = detail::FindBlock(v.row_offset, v.num_blocks, rr);
        const int ibra = static_cast<int>(rr) - v.row_offset[k];
        const int first = v.ch_bra[k] == v.ch_ket[k] ? ibra : 0;
        return static_cast<exec::Index>(v.nket[k] - first);
      },
      IMSRG_LAMBDA(const exec::Index rr, const exec::Index t) {
        const int k = detail::FindBlock(v.row_offset, v.num_blocks, rr);
        const int ibra = static_cast<int>(rr) - v.row_offset[k];
        const int first = v.ch_bra[k] == v.ch_ket[k] ? ibra : 0;
        PhstInverseElement(v, rr, first + static_cast<int>(t));
      });
  exec::Fence("commutator::comm222_phst");
}

}  // namespace imsrg::commutator::tensor_terms
