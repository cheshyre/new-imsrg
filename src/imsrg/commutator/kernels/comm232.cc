// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The N7 term comm232ss, [X2, Y3] + [X3, Y2] -> Z2 (imsrg3.h). Kernel TU.
//
// m-scheme (three contractions; o_abc = n_a n_b nbar_c + nbar_a nbar_b n_c):
//   [X2, Y3]_{ij,kl} = (s/2) sum_{abc} o_abc
//       { X_{ic,ab} Y_{abj,ckl} - X_{jc,ab} Y_{abi,ckl}
//         - X_{ab,kc} Y_{cij,abl} + X_{ab,lc} Y_{cij,abk} },
//   Z_{ij,kl} += [X2, Y3]_{ij,kl} - [Y2, X3]_{ij,kl},
// with s = -1 (fixed by the m-scheme brute force; imsrg++ writes the same
// sign, IMSRG3Commutators.cc:828).
//
// J-coupled form (unnormalized elements; bra |(i j) J>, ket |(k l) J>):
//   [X2, Y3]^J_{ijkl} = s { F[X, Y](i, j; k l) - p_ij F[X, Y](j, i; k l)
//                           - F[X^T, Y^T](k, l; i j)
//                           + p_kl F[X^T, Y^T](l, k; i j) },
// p_ij = (-1)^(j_i + j_j - J), with the creator-side half sum
//   F[A, B](p, q; r s) = sum_c sum_{J'} sum_{a <= b} w_ab o_abc
//       sum_{J3} C(p, q, c, J', J, J3) A^{J'}_{(p c),(a b)}
//       B^{J3}((a b) J' q; (r s) J c),
//   C = (2 J3 + 1) Hat(J') / Hat(J) (-1)^(j_q + j_c + J + J')
//       {j_q j_p J; j_c J3 J'},
// w_ab = 1 for a < b and 1/2 for a == b (the m-scheme 1/2 sum over all a,
// b), A^T_{(p c),(a b)} = A_{(a b),(p c)} and B^T(bra; ket) = B(ket; bra).
// The coefficient is (2 J3 + 1) / (2 J + 1) times the recoupling
// <((p q) J, c) J3 | ((p c) J', q) J3> (each m-scheme pair (a b) and (r s)
// keeps its coupling). The stored element is the sum divided by
// sqrt((1 + d_ij)(1 + d_kl)). The annihilator-side terms (with X^T, Y^T) are
// computed explicitly, so no hermiticity of X or Y is assumed (imsrg++
// replaces them by h_X h_Y times the creator side,
// IMSRG3Commutators.cc:2047).
//
// Two evaluations of F, both one work item per stored Z2 element (both
// triangles of every block, no atomics):
// * reference (`detail::Comm232ssReference`): B^{J3} read with
//   `ThreeBodyElement` (recoupled from the canonical kets per read);
// * pair third (`terms::Comm232ss`): B^{J3}((a b) J' q; (r s) J c) is the
//   element (row (a b) J' q, column (r s) J c) of the pair-third block W =
//   E B3 E^T of channel J3 (`op::PairThirdSpace`, layout C), built per call.
//   Within a channel the states are sorted by (third, Jpq, occupation class
//   of the pair, p, q), so the internal pairs (a b) of fixed (J', q) are a
//   contiguous run of rows, read down one column of W, and the column
//   (r s) J c is a binary search in the block (c, J) and class of (r s).
//   The classes (hh-only: n_a n_b != 0 = nbar_a nbar_b; mixed; pp-only;
//   inactive: both products 0) restrict the rows: an inactive pair (a b)
//   never contributes (o_abc = 0), so W keeps only the active rows, and for
//   n_c = 0 (1) only the pairs with n_a n_b != 0 (nbar_a nbar_b != 0) are
//   visited. B^T uses V = W^T, which is h_B W for a three-body part of
//   definite hermiticity (not stored; its active rows are those of W).
//   Memory: one (non-Hermitian part: two) sum over channels of A_ch S_ch
//   doubles per three-body operand (S_ch states, A_ch active ones); channels
//   are processed in batches whose blocks fit `max_block_bytes` (the element
//   kernel runs once per batch, adding the batch's channels).

#include "imsrg/commutator/detail/comm232.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator {

namespace {

/// The overall sign s of the m-scheme expression (file comment), fixed by
/// the m-scheme brute force (commutator_ref::mscheme).
constexpr double kSign = -1.0;

/// Number of occupation classes of a pair (`PairClass`).
constexpr int kNumClasses = 4;

/// Range policy with a dynamic schedule: the cost of a work item varies by
/// an order of magnitude (channel sizes, orbit j), and the cores of the dev
/// machine differ in speed (as in kernels/comm223.cc, which also uses a
/// dynamic team policy directly).
using DynamicPolicy = Kokkos::RangePolicy<exec::DefaultExecSpace,
                                          Kokkos::Schedule<Kokkos::Dynamic>,
                                          Kokkos::IndexType<exec::Index>>;

/// (-1)^n for an integer n.
IMSRG_INLINE_FUNCTION int Phase(int n) noexcept { return (n & 1) ? -1 : 1; }

/// |a - b| for ints.
IMSRG_INLINE_FUNCTION int AbsDiff(int a, int b) noexcept {
  return a > b ? a - b : b - a;
}

/// Occupation class of a pair with occupations np, nq: 0 if only n_p n_q is
/// nonzero (hh-only), 1 if n_p n_q and nbar_p nbar_q are both nonzero
/// (mixed, fractional occupations), 2 if only nbar_p nbar_q is nonzero
/// (pp-only), 3 if both vanish (inactive, e.g. a particle-hole pair).
/// Host and device use this one function, so the classes agree exactly.
IMSRG_INLINE_FUNCTION int PairClass(double np, double nq) noexcept {
  const bool hh = np != 0.0 && nq != 0.0;
  const bool pp = np != 1.0 && nq != 1.0;
  if (hh) return pp ? 1 : 0;
  return pp ? 2 : 3;
}

/// The 2 J3 range [lo, hi] of F: triangle(J', j_q, J3) and triangle(J, j_c,
/// J3) (`two_jp` = 2 J', `two_J` = 2 J); empty if lo > hi.
struct J3Range {
  int lo = 0;
  int hi = -1;
};
static_assert(std::is_trivially_copyable_v<J3Range>);

IMSRG_INLINE_FUNCTION J3Range RangeOfJ3(int two_jp, int jq, int two_J,
                                        int jc) noexcept {
  J3Range r;
  const int d1 = AbsDiff(two_jp, jq);
  const int d2 = AbsDiff(two_J, jc);
  r.lo = d1 > d2 ? d1 : d2;
  r.hi = two_jp + jq < two_J + jc ? two_jp + jq : two_J + jc;
  return r;
}

/// The creator-side half sum F[A, B](p, q; r s) at pair J (file comment),
/// reference version: three-body elements through `ThreeBodyElement`.
/// `trans` != 0 evaluates F[A^T, B^T]. Zero if A or B is missing. Cost
/// O(norb * J'-range * kets * J3-range) element reads.
IMSRG_INLINE_FUNCTION double HalfSumReference(
    const detail::N7Views& v, const detail::TwoBodyView& a2,
    const detail::ThreeBodyPartView& b3, int trans, int p, int q, int r, int s,
    int J) noexcept {
  if (a2.data == nullptr || b3.data == nullptr) return 0.0;
  const basis::TwoBodyTableView& tb = a2.tb;
  const int jp = v.ob.j2[p];
  const int jq = v.ob.j2[q];
  const int two_J = 2 * J;
  double total = 0.0;
  for (int c = 0; c < v.norb; ++c) {
    const int jc = v.ob.j2[c];
    const double nc = v.ob.occupation[c];
    for (int Jp = AbsDiff(jp, jc) / 2; Jp <= (jp + jc) / 2; ++Jp) {
      const int ch2 =
          tb.channel_of(OrbitIndex{p}, OrbitIndex{c}, TwoJ{2 * Jp}).value();
      if (ch2 >= tb.num_channels) continue;
      const J3Range j3 = RangeOfJ3(2 * Jp, jq, two_J, jc);
      if (j3.lo > j3.hi) continue;
      const double pref =
          Phase((jq + jc) / 2 + J + Jp) * v.sj.Hat(Jp) / v.sj.Hat(J);
      const int nk = tb.channel_num_kets[ch2];
      const std::int32_t* kets = tb.channel_kets + tb.channel_ket_offset[ch2];
      for (int ka = 0; ka < nk; ++ka) {
        const int a = tb.ket_p[kets[ka]];
        const int b = tb.ket_q[kets[ka]];
        const double na = v.ob.occupation[a];
        const double nb = v.ob.occupation[b];
        double occ = na * nb * (1.0 - nc) + (1.0 - na) * (1.0 - nb) * nc;
        if (occ == 0.0) continue;
        if (a == b) occ *= 0.5;
        const double x = trans != 0 ? a2.get_unnormalized(a, b, p, c, 2 * Jp)
                                    : a2.get_unnormalized(p, c, a, b, 2 * Jp);
        if (x == 0.0) continue;
        double acc = 0.0;
        for (int two_j3 = j3.lo; two_j3 <= j3.hi; two_j3 += 2) {
          const double six = v.sj(TwoJ{jq}, TwoJ{jp}, TwoJ{two_J}, TwoJ{jc},
                                  TwoJ{two_j3}, TwoJ{2 * Jp});
          if (six == 0.0) continue;
          const double y = trans != 0
                               ? detail::ThreeBodyElement(v, b3, two_j3, r, s,
                                                          c, J, a, b, q, Jp)
                               : detail::ThreeBodyElement(v, b3, two_j3, a, b,
                                                          q, Jp, r, s, c, J);
          acc += (two_j3 + 1) * six * y;
        }
        total += pref * occ * x * acc;
      }
    }
  }
  return total;
}

/// The four bracket terms of [A2, B3]^J_{ijkl} without the sign s (file
/// comment), reference version. O(4 HalfSumReference).
IMSRG_INLINE_FUNCTION double BracketReference(
    const detail::N7Views& v, const detail::TwoBodyView& a2,
    const detail::ThreeBodyPartView& b3, int i, int j, int k, int l,
    int J) noexcept {
  const int p_ij = Phase((v.ob.j2[i] + v.ob.j2[j]) / 2 - J);
  const int p_kl = Phase((v.ob.j2[k] + v.ob.j2[l]) / 2 - J);
  return HalfSumReference(v, a2, b3, 0, i, j, k, l, J) -
         p_ij * HalfSumReference(v, a2, b3, 0, j, i, k, l, J) -
         HalfSumReference(v, a2, b3, 1, k, l, i, j, J) +
         p_kl * HalfSumReference(v, a2, b3, 1, l, k, i, j, J);
}

/// The pair-third states of the context's kept space in the order of the
/// pair-third path (file comment): per channel, sorted by (third r, Jpq,
/// class, p, q). "Sorted index" t: the position in that order, flat over the
/// channels (channel ch owns [ch_begin[ch], ch_begin[ch + 1]), the flat
/// range of its states in `op::PairThirdSpace`). Block b = (ch norb + r) nj
/// + Jpq holds the states of channel ch with third r and pair J Jpq; its
/// class k starts at blk[kNumClasses b + k] and the block ends at
/// blk[kNumClasses (b + 1)]. The active rows (classes 0..2) of a channel
/// have compact indices 0..A_ch - 1 in sorted order; those of block b start
/// at arow0[b]. Device pointers; captured by value.
struct PairBlocksView {
  const std::int32_t* blk = nullptr;        ///< kNumClasses nblocks + 1.
  const std::int32_t* arow0 = nullptr;      ///< Per block.
  const std::int32_t* ch_begin = nullptr;   ///< num_channels + 1 entries.
  const std::int32_t* ch_active = nullptr;  ///< A_ch per channel.
  /// Active rows of all channels in compact order (channel ch from
  /// act_begin[ch]): the sorted index of each.
  const std::int32_t* act_row = nullptr;
  const std::int32_t* act_begin = nullptr;  ///< num_channels entries.
  const std::int32_t* orig = nullptr;       ///< Sorted -> PairThirdSpace flat.
  const std::int32_t* key = nullptr;    ///< p norb + q of the pair (p <= q).
  const std::int32_t* local = nullptr;  ///< Local index of (p, q) at Jpq.
  const double* occ_hh = nullptr;       ///< w n_p n_q, w = 1/sqrt 2 for p == q.
  const double* occ_pp = nullptr;       ///< w nbar_p nbar_q.
  /// Offset of channel ch's dense column-major A_ch x S_ch block of active
  /// rows in a buffer holding all channels (num_channels + 1 entries); a
  /// batch buffer starting at channel ch_lo is offset by w_off[ch_lo].
  const std::int64_t* w_off = nullptr;
  std::int32_t norb = 0;          ///< Orbits of the basis.
  std::int32_t nj = 0;            ///< Two-body J values (max J + 1).
  std::int32_t num_channels = 0;  ///< Three-body channels.
};
static_assert(std::is_trivially_copyable_v<PairBlocksView>);

/// The blocks of one batch of channels [ch_lo, ch_hi) for the element kernel
/// (null pointers: that operand does not contribute).
struct BatchBlocks {
  const double* wx = nullptr;   ///< Active rows of W(X3).
  const double* wxt = nullptr;  ///< Active rows of W(X3^T) (V or W).
  const double* wy = nullptr;   ///< Active rows of W(Y3).
  const double* wyt = nullptr;  ///< Active rows of W(Y3^T).
  double sxt = 1.0;             ///< Sign of wxt (h_X3 when it is W(X3)).
  double syt = 1.0;             ///< Sign of wyt.
  std::int64_t w_base = 0;      ///< w_off[ch_lo].
  std::int32_t ch_lo = 0;       ///< First channel of the batch.
  std::int32_t ch_hi = 0;       ///< One past the last.
};
static_assert(std::is_trivially_copyable_v<BatchBlocks>);

/// The position of `key` in the ascending `keys[lo, hi)`, or -1. Binary
/// search, O(log(hi - lo)).
IMSRG_INLINE_FUNCTION int FindKey(const std::int32_t* keys, int lo, int hi,
                                  int key) noexcept {
  int a = lo;
  int b = hi - 1;
  while (a <= b) {
    const int mid = a + (b - a) / 2;
    const int k = keys[mid];
    if (k == key) return mid;
    if (k < key) {
      a = mid + 1;
    } else {
      b = mid - 1;
    }
  }
  return -1;
}

/// The creator-side half sum F[A, B](p, q; r s) at pair J through the
/// active-row blocks `w` of B (file comment) for the three-body channels of
/// the batch `bb`; `trans` != 0 evaluates F[A^T, B^T] with `w` the blocks of
/// B^T. `w_sign` multiplies the result (h_B when B^T is read from the blocks
/// of B). Zero if A or `w` is missing. Cost O(norb * J'-range * J3-range *
/// (visited rows of the block (q, J') + log S)).
IMSRG_INLINE_FUNCTION double HalfSumPairThird(
    const detail::N7Views& v, const PairBlocksView& pb, const BatchBlocks& bb,
    const detail::TwoBodyView& a2, const double* w, double w_sign, int trans,
    int p, int q, int r, int s, int J) noexcept {
  if (a2.data == nullptr || w == nullptr) return 0.0;
  const basis::TwoBodyTableView& tb = a2.tb;
  const int jp = v.ob.j2[p];
  const int jq = v.ob.j2[q];
  const int two_J = 2 * J;
  const int rs_key = r * pb.norb + s;
  const int rs_class = PairClass(v.ob.occupation[r], v.ob.occupation[s]);
  double total = 0.0;
  for (int c = 0; c < v.norb; ++c) {
    const int jc = v.ob.j2[c];
    const double nc = v.ob.occupation[c];
    const double nbar_c = 1.0 - nc;
    // Visited classes: n_c == 0 needs n_a n_b != 0 (classes 0, 1), n_c == 1
    // needs nbar_a nbar_b != 0 (classes 1, 2).
    const int k_lo = nc == 1.0 ? 1 : 0;
    const int k_hi = nc == 0.0 ? 2 : 3;
    const KetIndex pc =
        tb.ket_index(OrbitIndex{p < c ? p : c}, OrbitIndex{p < c ? c : p});
    for (int Jp = AbsDiff(jp, jc) / 2; Jp <= (jp + jc) / 2; ++Jp) {
      const std::int32_t xloc = tb.local_index(pc, TwoJ{2 * Jp});
      if (xloc < 0) continue;
      const J3Range j3 = RangeOfJ3(2 * Jp, jq, two_J, jc);
      if (j3.lo > j3.hi) continue;
      const int ch2 =
          tb.channel_of(OrbitIndex{p}, OrbitIndex{c}, TwoJ{2 * Jp}).value();
      const std::int64_t n2 = tb.channel_num_kets[ch2];
      const double* xb = a2.data + a2.block_offset[ch2];
      // A^{J'}_{(p c),(a b)} = xf * stored(row (p c), column (a b)); the
      // factor sqrt 2 of a == b is in occ_hh / occ_pp.
      double xf = 1.0;
      if (p == c) {
        xf = constants::kSqrt2;
      } else if (p > c) {
        xf = basis::ExchangePhase(TwoJ{jp}, TwoJ{jc}, TwoJ{2 * Jp});
      }
      const std::int64_t x0 = trans != 0 ? xloc * n2 : xloc;
      const std::int64_t xs = trans != 0 ? 1 : n2;
      const double pref =
          Phase((jq + jc) / 2 + J + Jp) * xf * v.sj.Hat(Jp) / v.sj.Hat(J);
      for (int two_j3 = j3.lo; two_j3 <= j3.hi; two_j3 += 2) {
        const int ch3 = detail::Channel3Of(v, two_j3, r, s, c);
        if (ch3 < bb.ch_lo || ch3 >= bb.ch_hi) continue;
        const int rb = (ch3 * pb.norb + q) * pb.nj + Jp;
        const int r0 = pb.blk[kNumClasses * rb + k_lo];
        const int r1 = pb.blk[kNumClasses * rb + k_hi];
        if (r0 >= r1) continue;
        const int cb = kNumClasses * ((ch3 * pb.norb + c) * pb.nj + J);
        const int col = FindKey(pb.key, pb.blk[cb + rs_class],
                                pb.blk[cb + rs_class + 1], rs_key);
        if (col < 0) continue;
        const double six = v.sj(TwoJ{jq}, TwoJ{jp}, TwoJ{two_J}, TwoJ{jc},
                                TwoJ{two_j3}, TwoJ{2 * Jp});
        if (six == 0.0) continue;
        const std::int64_t na = pb.ch_active[ch3];
        // W[t, col] at w[wc + t] for the active rows t of block rb.
        const std::int64_t wc = pb.w_off[ch3] - bb.w_base +
                                (col - pb.ch_begin[ch3]) * na + pb.arow0[rb] -
                                pb.blk[kNumClasses * rb];
        double acc_hh = 0.0;
        double acc_pp = 0.0;
        for (int t = r0; t < r1; ++t) {
          const double xw = xb[x0 + pb.local[t] * xs] * w[wc + t];
          acc_hh += pb.occ_hh[t] * xw;
          acc_pp += pb.occ_pp[t] * xw;
        }
        total += pref * (two_j3 + 1) * six * (nbar_c * acc_hh + nc * acc_pp);
      }
    }
  }
  return w_sign * total;
}

/// The four bracket terms of [A2, B3]^J_{ijkl} without the sign s through
/// the active-row blocks `w` of B3 and `wt` (sign `wt_sign`) of its
/// transpose, for the channels of the batch. O(4 HalfSumPairThird).
IMSRG_INLINE_FUNCTION double BracketPairThird(
    const detail::N7Views& v, const PairBlocksView& pb, const BatchBlocks& bb,
    const detail::TwoBodyView& a2, const double* w, const double* wt,
    double wt_sign, int i, int j, int k, int l, int J) noexcept {
  const int p_ij = Phase((v.ob.j2[i] + v.ob.j2[j]) / 2 - J);
  const int p_kl = Phase((v.ob.j2[k] + v.ob.j2[l]) / 2 - J);
  return HalfSumPairThird(v, pb, bb, a2, w, 1.0, 0, i, j, k, l, J) -
         p_ij * HalfSumPairThird(v, pb, bb, a2, w, 1.0, 0, j, i, k, l, J) -
         HalfSumPairThird(v, pb, bb, a2, wt, wt_sign, 1, k, l, i, j, J) +
         p_kl * HalfSumPairThird(v, pb, bb, a2, wt, wt_sign, 1, l, k, i, j, J);
}

/// Owner of the device tables behind a `PairBlocksView` (built per call on
/// the host from the context's `op::PairThirdSpace` and model space), with
/// the host copies the batching needs.
struct PairBlocks {
  exec::DeviceBuffer<std::int32_t> blk;
  exec::DeviceBuffer<std::int32_t> arow0;
  exec::DeviceBuffer<std::int32_t> ch_begin;
  exec::DeviceBuffer<std::int32_t> ch_active;
  exec::DeviceBuffer<std::int32_t> act_row;
  exec::DeviceBuffer<std::int32_t> act_begin;
  exec::DeviceBuffer<std::int32_t> orig;
  exec::DeviceBuffer<std::int32_t> key;
  exec::DeviceBuffer<std::int32_t> local;
  exec::DeviceBuffer<double> occ_hh;
  exec::DeviceBuffer<double> occ_pp;
  exec::DeviceBuffer<std::int64_t> w_off;
  std::vector<std::int64_t> w_off_host;  ///< Host copy of w_off.
  PairBlocksView view;  ///< Device view (valid while *this lives).
};

/// Copies a host vector into a new device buffer.
template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& h, const char* label) {
  exec::DeviceBuffer<T> d(h.size(), label);
  if (!h.empty()) d.CopyFromHost(std::span<const T>(h));
  return d;
}

/// Builds the sorted pair-third tables of the context (PairBlocksView).
/// Cost O(states log states) on the host plus the copies.
PairBlocks BuildPairBlocks(const ThreeBodyContext& ctx) {
  const op::PairThirdView pt = ctx.impl().pair_third.host_view();
  const basis::ModelSpace& ms = *ctx.model_space_ptr();
  const basis::TwoBodyChannels& two = ms.two_body();
  const std::span<const double> occ = ms.reference().occupations();
  const int norb = ms.num_orbits();
  const int nj = two.max_J() + 1;
  const int nch = pt.num_channels;
  const auto ns = static_cast<std::size_t>(pt.num_states);
  const auto at = [](auto i) { return static_cast<std::size_t>(i); };
  // Block index of (ch, r, Jpq).
  const auto block_of = [norb, nj](int ch, int r, int jpq) {
    return static_cast<std::size_t>(
        (static_cast<std::int64_t>(ch) * norb + r) * nj + jpq);
  };
  const std::size_t nblocks = block_of(nch, 0, 0);
  const auto class_of = [&](std::int32_t f) {
    return PairClass(occ[at(pt.st_p[f])], occ[at(pt.st_q[f])]);
  };
  std::vector<std::int32_t> count(kNumClasses * nblocks + 1, 0);
  std::vector<std::int32_t> ch_begin(at(nch) + 1, 0);
  std::vector<std::int32_t> orig(ns), key(ns), local(ns);
  std::vector<double> occ_hh(ns), occ_pp(ns);
  std::vector<std::int32_t> perm;
  std::int32_t next = 0;
  for (int ch = 0; ch < nch; ++ch) {
    const std::int32_t off = pt.ch_state_offset[ch];
    const std::int32_t S = pt.ch_num_states[ch];
    IMSRG_CHECK(S == 0 || off == next,
                "comm232ss: pair-third channels are not contiguous");
    ch_begin[at(ch)] = next;
    perm.resize(at(S));
    std::iota(perm.begin(), perm.end(), off);
    std::sort(perm.begin(), perm.end(), [&](std::int32_t x, std::int32_t y) {
      if (pt.st_r[x] != pt.st_r[y]) return pt.st_r[x] < pt.st_r[y];
      if (pt.st_two_jpq[x] != pt.st_two_jpq[y]) {
        return pt.st_two_jpq[x] < pt.st_two_jpq[y];
      }
      const int cx = class_of(x);
      const int cy = class_of(y);
      if (cx != cy) return cx < cy;
      if (pt.st_p[x] != pt.st_p[y]) return pt.st_p[x] < pt.st_p[y];
      return pt.st_q[x] < pt.st_q[y];
    });
    for (std::int32_t pos = 0; pos < S; ++pos) {
      const std::int32_t f = perm[at(pos)];
      const std::size_t t = at(next + pos);
      const int p = pt.st_p[f];
      const int q = pt.st_q[f];
      const int two_jpq = pt.st_two_jpq[f];
      orig[t] = f;
      key[t] = p * norb + q;
      const std::optional<LocalKetIndex> li = two.local_index(
          two.ket_index(OrbitIndex{p}, OrbitIndex{q}), TwoJ{two_jpq});
      IMSRG_CHECK(li.has_value(),
                  "comm232ss: a pair-third state has no two-body ket");
      local[t] = li->value();
      const double w = p == q ? 1.0 / constants::kSqrt2 : 1.0;
      const double np = occ[at(p)];
      const double nq = occ[at(q)];
      occ_hh[t] = w * np * nq;
      occ_pp[t] = w * (1.0 - np) * (1.0 - nq);
      ++count[kNumClasses * block_of(ch, pt.st_r[f], two_jpq / 2) +
              at(class_of(f))];
    }
    next += S;
  }
  ch_begin[at(nch)] = next;
  // blk[k b + c] = first sorted index of class c in block b (exclusive scan;
  // blocks, and classes within a block, follow the sorted order).
  std::vector<std::int32_t> blk(count.size(), 0);
  std::exclusive_scan(count.begin(), count.end(), blk.begin(), 0);
  // Compact active rows per channel.
  std::vector<std::int32_t> arow0(nblocks, 0);
  std::vector<std::int32_t> ch_active(at(nch), 0);
  std::vector<std::int32_t> act_begin(at(nch), 0);
  std::vector<std::int32_t> act_row;
  act_row.reserve(ns);
  std::vector<std::int64_t> w_off(at(nch) + 1, 0);
  const std::size_t blocks_per_channel = block_of(1, 0, 0);
  for (int ch = 0; ch < nch; ++ch) {
    act_begin[at(ch)] = static_cast<std::int32_t>(act_row.size());
    std::int32_t active = 0;
    for (std::size_t b = at(ch) * blocks_per_channel;
         b < at(ch + 1) * blocks_per_channel; ++b) {
      arow0[b] = active;
      for (std::int32_t t = blk[kNumClasses * b]; t < blk[kNumClasses * b + 3];
           ++t) {
        act_row.push_back(t);
        ++active;
      }
    }
    ch_active[at(ch)] = active;
    w_off[at(ch) + 1] =
        w_off[at(ch)] + static_cast<std::int64_t>(active) *
                            (ch_begin[at(ch) + 1] - ch_begin[at(ch)]);
  }
  PairBlocks t;
  t.blk = ToDevice(blk, "commutator::comm232_blk");
  t.arow0 = ToDevice(arow0, "commutator::comm232_arow0");
  t.ch_begin = ToDevice(ch_begin, "commutator::comm232_ch_begin");
  t.ch_active = ToDevice(ch_active, "commutator::comm232_ch_active");
  t.act_row = ToDevice(act_row, "commutator::comm232_act_row");
  t.act_begin = ToDevice(act_begin, "commutator::comm232_act_begin");
  t.orig = ToDevice(orig, "commutator::comm232_orig");
  t.key = ToDevice(key, "commutator::comm232_key");
  t.local = ToDevice(local, "commutator::comm232_local");
  t.occ_hh = ToDevice(occ_hh, "commutator::comm232_occ_hh");
  t.occ_pp = ToDevice(occ_pp, "commutator::comm232_occ_pp");
  t.w_off = ToDevice(w_off, "commutator::comm232_w_off");
  t.w_off_host = std::move(w_off);
  t.view.blk = t.blk.device_data();
  t.view.arow0 = t.arow0.device_data();
  t.view.ch_begin = t.ch_begin.device_data();
  t.view.ch_active = t.ch_active.device_data();
  t.view.act_row = t.act_row.device_data();
  t.view.act_begin = t.act_begin.device_data();
  t.view.orig = t.orig.device_data();
  t.view.key = t.key.device_data();
  t.view.local = t.local.device_data();
  t.view.occ_hh = t.occ_hh.device_data();
  t.view.occ_pp = t.occ_pp.device_data();
  t.view.w_off = t.w_off.device_data();
  t.view.norb = norb;
  t.view.nj = nj;
  t.view.num_channels = nch;
  return t;
}

/// The output pointers of the pair-third build of one batch (null: not
/// needed); each indexed like the batch buffer (w_off[ch] - w_base).
struct PairThirdOut {
  double* wx = nullptr;  ///< Active rows of W(X3).
  double* vx = nullptr;  ///< Active rows of W(X3^T) (non-Hermitian X3).
  double* wy = nullptr;  ///< Active rows of W(Y3).
  double* vy = nullptr;  ///< Active rows of W(Y3^T) (non-Hermitian Y3).
};
static_assert(std::is_trivially_copyable_v<PairThirdOut>);

/// Kernel of the pair-third build: the active rows of W = E B3 E^T (and of
/// V = E B3^T E^T where requested) of X3 and Y3 for the block elements
/// [w_base, w_base + size) of the layout `pb.w_off` (the channels of one
/// batch), written to `o` at index (element - w_base). Element (t, u):
/// W[t, u] = sum_{I in E_t, K in E_u} E_tI E_uK B3[I, K] with E the map `e`
/// (rows: PairThirdSpace flat states, columns: channel-local kept rows).
/// A functor rather than a lambda, so the launch may sit in a function of
/// this TU's unnamed namespace (nvcc restricts the enclosing functions of
/// extended lambdas). Cost per element O(nnz(E row)^2).
struct BuildBlocksFunctor {
  detail::N7Views v;
  PairBlocksView pb;
  op::CsrView e;
  PairThirdOut o;
  std::int64_t w_base = 0;

  IMSRG_FUNCTION void operator()(const exec::Index idx) const {
    const std::int64_t g = idx + w_base;
    const int ch = detail::FindBlock(pb.w_off, pb.num_channels, g);
    const std::int64_t na = pb.ch_active[ch];
    const std::int64_t loc = g - pb.w_off[ch];
    const int fr = pb.orig[pb.act_row[pb.act_begin[ch] + loc % na]];
    const int fc = pb.orig[pb.ch_begin[ch] + loc / na];
    double wx = 0.0;
    double vx = 0.0;
    double wy = 0.0;
    double vy = 0.0;
    for (std::int64_t a = e.ptr[fr]; a < e.ptr[fr + 1]; ++a) {
      const int I = e.col[a];
      for (std::int64_t b = e.ptr[fc]; b < e.ptr[fc + 1]; ++b) {
        const int K = e.col[b];
        const double ee = e.val[a] * e.val[b];
        if (o.wx != nullptr) {
          wx += ee * detail::ThreeBodyLocal(v.x3, v.sp, ch, I, K);
        }
        if (o.vx != nullptr) {
          vx += ee * detail::ThreeBodyLocal(v.x3, v.sp, ch, K, I);
        }
        if (o.wy != nullptr) {
          wy += ee * detail::ThreeBodyLocal(v.y3, v.sp, ch, I, K);
        }
        if (o.vy != nullptr) {
          vy += ee * detail::ThreeBodyLocal(v.y3, v.sp, ch, K, I);
        }
      }
    }
    if (o.wx != nullptr) o.wx[idx] = wx;
    if (o.vx != nullptr) o.vx[idx] = vx;
    if (o.wy != nullptr) o.wy[idx] = wy;
    if (o.vy != nullptr) o.vy[idx] = vy;
  }
};
static_assert(std::is_trivially_copyable_v<BuildBlocksFunctor>);

/// Builds the blocks of one batch (`BuildBlocksFunctor`, one work item per
/// block element).
void BuildPairThirdBlocks(const detail::N7Views& v, const PairBlocksView& pb,
                          const op::CsrView& e, std::int64_t w_base,
                          std::int64_t size, const PairThirdOut& o) {
  if (size == 0) return;
  BuildBlocksFunctor f;
  f.v = v;
  f.pb = pb;
  f.e = e;
  f.o = o;
  f.w_base = w_base;
  Kokkos::parallel_for("commutator::comm232ss_pair_third",
                       DynamicPolicy(0, size), f);
}

/// Kernel of the element pass: adds the contribution of the channels of the
/// batch `bb` to the stored Z2 element `e` (functor for the reason given at
/// `BuildBlocksFunctor`). Cost per element O(8 HalfSumPairThird).
struct AddBatchFunctor {
  detail::N7Views v;
  PairBlocksView pb;
  BatchBlocks bb;
  detail::TwoBodyOutView out;

  IMSRG_FUNCTION void operator()(const exec::Index e) const {
    const basis::TwoBodyTableView& tb = v.x2.tb;
    const int ch = detail::FindBlock(out.block_offset, tb.num_channels, e);
    const std::int64_t n = tb.channel_num_kets[ch];
    const std::int64_t idx = e - out.block_offset[ch];
    const int ib = static_cast<int>(idx % n);
    const int ik = static_cast<int>(idx / n);
    const int J = tb.channel_two_j[ch] / 2;
    const int koff = tb.channel_ket_offset[ch];
    const int bra = tb.channel_kets[koff + ib];
    const int ket = tb.channel_kets[koff + ik];
    const int i = tb.ket_p[bra];
    const int j = tb.ket_q[bra];
    const int k = tb.ket_p[ket];
    const int l = tb.ket_q[ket];
    double z = 0.0;
    if (bb.wy != nullptr) {
      z += BracketPairThird(v, pb, bb, v.x2, bb.wy, bb.wyt, bb.syt, i, j, k, l,
                            J);
    }
    if (bb.wx != nullptr) {
      z -= BracketPairThird(v, pb, bb, v.y2, bb.wx, bb.wxt, bb.sxt, i, j, k, l,
                            J);
    }
    z *= kSign;
    if (i == j) z /= constants::kSqrt2;
    if (k == l) z /= constants::kSqrt2;
    out.data[e] += z;
  }
};
static_assert(std::is_trivially_copyable_v<AddBatchFunctor>);

/// Adds the contribution of the channels of batch `bb` to every stored Z2
/// element (`AddBatchFunctor`, one work item per element; `total` elements).
void AddBatch(const detail::N7Views& v, const PairBlocksView& pb,
              const BatchBlocks& bb, const detail::TwoBodyOutView& out,
              std::int64_t total) {
  AddBatchFunctor f;
  f.v = v;
  f.pb = pb;
  f.bb = bb;
  f.out = out;
  Kokkos::parallel_for("commutator::comm232ss", DynamicPolicy(0, total), f);
}

/// A batch of consecutive three-body channels [lo, hi) and the doubles of
/// one set of its blocks.
struct ChannelBatch {
  int lo = 0;
  int hi = 0;
  std::int64_t size = 0;
};

/// Splits the channels into batches of consecutive channels whose blocks
/// (offsets `w_off`, num_channels + 1 entries) hold at most `budget` doubles,
/// each batch at least one channel; batches without blocks are dropped.
/// O(num_channels).
std::vector<ChannelBatch> MakeBatches(const std::vector<std::int64_t>& w_off,
                                      std::int64_t budget) {
  std::vector<ChannelBatch> out;
  const int nch = static_cast<int>(w_off.size()) - 1;
  const auto at = [](int i) { return static_cast<std::size_t>(i); };
  for (int lo = 0, hi = 0; lo < nch; lo = hi) {
    hi = lo + 1;
    while (hi < nch && w_off[at(hi + 1)] - w_off[at(lo)] <= budget) ++hi;
    const std::int64_t size = w_off[at(hi)] - w_off[at(lo)];
    if (size > 0) out.push_back(ChannelBatch{lo, hi, size});
  }
  return out;
}

/// A device buffer of `n` doubles if `wanted`, else empty.
exec::DeviceBuffer<double> MaybeBuffer(bool wanted, std::int64_t n,
                                       const char* label) {
  return wanted ? exec::DeviceBuffer<double>(static_cast<std::size_t>(n), label)
                : exec::DeviceBuffer<double>();
}

/// Device pointer of a buffer, null if it is empty.
double* DataOrNull(exec::DeviceBuffer<double>& b) {
  return b.empty() ? nullptr : b.device_data();
}

}  // namespace

namespace detail {

void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm232ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm232ss", Z, ctx);
  if (v.x3.data == nullptr && v.y3.data == nullptr) return;
  const std::int64_t total = Z.two_body().matrix().total_size();
  const basis::TwoBodyTableView tb = v.x2.tb;
  exec::ParallelFor(
      "commutator::comm232ss_reference", static_cast<exec::Index>(total),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = FindBlock(out.block_offset, tb.num_channels, e);
        const std::int64_t n = tb.channel_num_kets[ch];
        const std::int64_t idx = e - out.block_offset[ch];
        const int ib = static_cast<int>(idx % n);
        const int ik = static_cast<int>(idx / n);
        const int J = tb.channel_two_j[ch] / 2;
        const int koff = tb.channel_ket_offset[ch];
        const int bra = tb.channel_kets[koff + ib];
        const int ket = tb.channel_kets[koff + ik];
        const int i = tb.ket_p[bra];
        const int j = tb.ket_q[bra];
        const int k = tb.ket_p[ket];
        const int l = tb.ket_q[ket];
        double z = BracketReference(v, v.x2, v.y3, i, j, k, l, J) -
                   BracketReference(v, v.y2, v.x3, i, j, k, l, J);
        z *= kSign;
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        out.data[e] += z;
      });
  exec::Fence("commutator::comm232ss_reference");
}

void Comm232ssPairThird(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx,
                        std::int64_t max_block_bytes) {
  const N7Views v = PrepareN7("Comm232ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm232ss", Z, ctx);
  // [X2, Y3] needs X2 and Y3, [Y2, X3] needs Y2 and X3.
  const bool xy = v.x2.data != nullptr && v.y3.data != nullptr;
  const bool yx = v.y2.data != nullptr && v.x3.data != nullptr;
  if (!xy && !yx) return;
  const PairBlocks blocks = BuildPairBlocks(ctx);
  const PairBlocksView pb = blocks.view;
  const std::vector<std::int64_t>& w_off = blocks.w_off_host;
  // Block sets per channel element: W(X3), W(X3^T), W(Y3), W(Y3^T).
  const int sets = (yx ? 1 : 0) + (yx && v.x3.packed == 0 ? 1 : 0) +
                   (xy ? 1 : 0) + (xy && v.y3.packed == 0 ? 1 : 0);
  const std::int64_t budget = std::max<std::int64_t>(
      1, max_block_bytes / (sets * static_cast<std::int64_t>(sizeof(double))));
  const op::CsrView e = ctx.impl().pair_third.e_map().device_view();
  const std::int64_t total = Z.two_body().matrix().total_size();
  const std::vector<ChannelBatch> batches = MakeBatches(w_off, budget);
  std::int64_t batch_max = 0;
  for (const ChannelBatch& b : batches) batch_max = std::max(batch_max, b.size);
  exec::DeviceBuffer<double> wx =
      MaybeBuffer(yx, batch_max, "commutator::comm232_wx");
  exec::DeviceBuffer<double> vx =
      MaybeBuffer(yx && v.x3.packed == 0, batch_max, "commutator::comm232_vx");
  exec::DeviceBuffer<double> wy =
      MaybeBuffer(xy, batch_max, "commutator::comm232_wy");
  exec::DeviceBuffer<double> vy =
      MaybeBuffer(xy && v.y3.packed == 0, batch_max, "commutator::comm232_vy");
  PairThirdOut o;
  o.wx = DataOrNull(wx);
  o.vx = DataOrNull(vx);
  o.wy = DataOrNull(wy);
  o.vy = DataOrNull(vy);
  BatchBlocks bb;
  bb.wx = o.wx;
  bb.wy = o.wy;
  // B^T: V if stored, else h_B W.
  bb.wxt = o.vx != nullptr ? o.vx : o.wx;
  bb.wyt = o.vy != nullptr ? o.vy : o.wy;
  bb.sxt = o.vx != nullptr ? 1.0 : v.x3.sign;
  bb.syt = o.vy != nullptr ? 1.0 : v.y3.sign;
  // Batches run in order on the execution space, so one set of buffers
  // serves them all.
  for (const ChannelBatch& b : batches) {
    bb.w_base = w_off[static_cast<std::size_t>(b.lo)];
    bb.ch_lo = b.lo;
    bb.ch_hi = b.hi;
    BuildPairThirdBlocks(v, pb, e, bb.w_base, b.size, o);
    AddBatch(v, pb, bb, out, total);
  }
  exec::Fence("commutator::comm232ss");
}

}  // namespace detail

namespace terms {

void Comm232ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm232ssPairThird(X, Y, Z, ctx, detail::kComm232MaxBlockBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
