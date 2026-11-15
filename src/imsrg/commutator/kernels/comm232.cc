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
//   E B3 E^T of channel J3 (`op::PairThirdSpace`, layout C), built once per
//   call for every channel. The states of a channel are sorted by (third,
//   Jpq, p, q), so the internal pairs (a b) of fixed (J', q) are a
//   contiguous block of rows of W, read down one column; the column (r s) J c
//   is found by a binary search in the block (c, J). B^T uses V = W^T, which
//   is h_B W for a three-body part of definite hermiticity (not stored).
//   Memory: one (or two, for a non-Hermitian part) sum over channels of
//   S_ch^2 doubles per three-body operand (S_ch pair-third states).

#include "imsrg/commutator/detail/comm232.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>
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

/// (-1)^n for an integer n.
IMSRG_INLINE_FUNCTION int Phase(int n) noexcept { return (n & 1) ? -1 : 1; }

/// |a - b| for ints.
IMSRG_INLINE_FUNCTION int AbsDiff(int a, int b) noexcept {
  return a > b ? a - b : b - a;
}

/// The 2 J3 range [lo, hi] of F: triangle(J', j_q, J3) and triangle(J, j_c,
/// J3) (`two_jp` = 2 J', `two_J` = 2 J); empty if lo > hi.
struct J3Range {
  int lo = 0;
  int hi = -1;
};

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
/// pair-third path (file comment): per channel, sorted by (third r, Jpq, p,
/// q). "Sorted index" t: the position in that order, flat over the channels
/// (channel ch owns [ch_begin[ch], ch_begin[ch + 1]), the flat range of its
/// states in `op::PairThirdSpace`). Device pointers; captured by value.
struct PairBlocksView {
  /// Block (ch, r, Jpq) = sorted indices [blk[b], blk[b + 1]) with b = (ch
  /// norb + r) nj + Jpq; num_channels norb nj + 1 entries.
  const std::int32_t* blk = nullptr;
  const std::int32_t* ch_begin = nullptr;  ///< num_channels + 1 entries.
  const std::int32_t* orig = nullptr;      ///< Sorted -> PairThirdSpace flat.
  const std::int32_t* key = nullptr;       ///< p norb + q of the pair (p <= q).
  const std::int32_t* local = nullptr;     ///< Local index of (p, q) at Jpq.
  const double* occ_hh = nullptr;  ///< w n_p n_q, w = 1/sqrt 2 for p == q.
  const double* occ_pp = nullptr;  ///< w nbar_p nbar_q.
  /// Offset of the dense column-major S_ch x S_ch block of channel ch in the
  /// W buffers (num_channels + 1 entries); element (t, u) of the channel at
  /// w_off[ch] + (t - ch_begin[ch]) + (u - ch_begin[ch]) S_ch.
  const std::int64_t* w_off = nullptr;
  std::int32_t norb = 0;          ///< Orbits of the basis.
  std::int32_t nj = 0;            ///< Two-body J values (max J + 1).
  std::int32_t num_channels = 0;  ///< Three-body channels.
};
static_assert(std::is_trivially_copyable_v<PairBlocksView>);

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
/// pair-third blocks `w` of B (file comment); `trans` != 0 evaluates F[A^T,
/// B^T] with `w` the blocks of B^T. `w_sign` multiplies the result (h_B when
/// B^T is read from the blocks of B). Zero if A or `w` is missing. Cost
/// O(norb * J'-range * J3-range * (rows of the block (q, J') + log S)).
IMSRG_INLINE_FUNCTION double HalfSumPairThird(const detail::N7Views& v,
                                              const PairBlocksView& pb,
                                              const detail::TwoBodyView& a2,
                                              const double* w, double w_sign,
                                              int trans, int p, int q, int r,
                                              int s, int J) noexcept {
  if (a2.data == nullptr || w == nullptr) return 0.0;
  const basis::TwoBodyTableView& tb = a2.tb;
  const int jp = v.ob.j2[p];
  const int jq = v.ob.j2[q];
  const int two_J = 2 * J;
  const int rs_key = r * pb.norb + s;
  double total = 0.0;
  for (int c = 0; c < v.norb; ++c) {
    const int jc = v.ob.j2[c];
    const double nc = v.ob.occupation[c];
    const double nbar_c = 1.0 - nc;
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
        if (ch3 < 0) continue;
        const int cb = (ch3 * pb.norb + c) * pb.nj + J;
        const int col = FindKey(pb.key, pb.blk[cb], pb.blk[cb + 1], rs_key);
        if (col < 0) continue;
        const int rb = (ch3 * pb.norb + q) * pb.nj + Jp;
        const int r0 = pb.blk[rb];
        const int r1 = pb.blk[rb + 1];
        if (r0 == r1) continue;
        const double six = v.sj(TwoJ{jq}, TwoJ{jp}, TwoJ{two_J}, TwoJ{jc},
                                TwoJ{two_j3}, TwoJ{2 * Jp});
        if (six == 0.0) continue;
        const std::int64_t base = pb.ch_begin[ch3];
        const std::int64_t S = pb.ch_begin[ch3 + 1] - base;
        // W[t, col] of the channel block at w[wc + t] for r0 <= t < r1.
        const std::int64_t wc = pb.w_off[ch3] + (col - base) * S - base;
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
/// the pair-third blocks `w` of B3 and `wt` (sign `wt_sign`) of its
/// transpose. O(4 HalfSumPairThird).
IMSRG_INLINE_FUNCTION double BracketPairThird(const detail::N7Views& v,
                                              const PairBlocksView& pb,
                                              const detail::TwoBodyView& a2,
                                              const double* w, const double* wt,
                                              double wt_sign, int i, int j,
                                              int k, int l, int J) noexcept {
  const int p_ij = Phase((v.ob.j2[i] + v.ob.j2[j]) / 2 - J);
  const int p_kl = Phase((v.ob.j2[k] + v.ob.j2[l]) / 2 - J);
  return HalfSumPairThird(v, pb, a2, w, 1.0, 0, i, j, k, l, J) -
         p_ij * HalfSumPairThird(v, pb, a2, w, 1.0, 0, j, i, k, l, J) -
         HalfSumPairThird(v, pb, a2, wt, wt_sign, 1, k, l, i, j, J) +
         p_kl * HalfSumPairThird(v, pb, a2, wt, wt_sign, 1, l, k, i, j, J);
}

/// Owner of the device tables behind a `PairBlocksView` (built per call on
/// the host from the context's `op::PairThirdSpace` and model space).
struct PairBlocks {
  exec::DeviceBuffer<std::int32_t> blk;
  exec::DeviceBuffer<std::int32_t> ch_begin;
  exec::DeviceBuffer<std::int32_t> orig;
  exec::DeviceBuffer<std::int32_t> key;
  exec::DeviceBuffer<std::int32_t> local;
  exec::DeviceBuffer<double> occ_hh;
  exec::DeviceBuffer<double> occ_pp;
  exec::DeviceBuffer<std::int64_t> w_off;
  std::int64_t w_size = 0;  ///< Doubles of one set of blocks.
  PairBlocksView view;      ///< Device view (valid while *this lives).
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
  // Index of block (ch, r, Jpq) (PairBlocksView::blk).
  const auto block_of = [norb, nj](int ch, int r, int jpq) {
    return static_cast<std::size_t>(
        (static_cast<std::int64_t>(ch) * norb + r) * nj + jpq);
  };
  std::vector<std::int32_t> count(block_of(nch, 0, 0) + 1, 0);
  std::vector<std::int32_t> ch_begin(static_cast<std::size_t>(nch) + 1, 0);
  std::vector<std::int32_t> orig(ns), key(ns), local(ns);
  std::vector<double> occ_hh(ns), occ_pp(ns);
  std::vector<std::int64_t> w_off(static_cast<std::size_t>(nch) + 1, 0);
  std::vector<std::int32_t> perm;
  std::int32_t next = 0;
  for (int ch = 0; ch < nch; ++ch) {
    const std::int32_t off = pt.ch_state_offset[ch];
    const std::int32_t S = pt.ch_num_states[ch];
    IMSRG_CHECK(S == 0 || off == next,
                "comm232ss: pair-third channels are not contiguous");
    ch_begin[static_cast<std::size_t>(ch)] = next;
    perm.resize(static_cast<std::size_t>(S));
    std::iota(perm.begin(), perm.end(), off);
    std::sort(perm.begin(), perm.end(), [&](std::int32_t x, std::int32_t y) {
      if (pt.st_r[x] != pt.st_r[y]) return pt.st_r[x] < pt.st_r[y];
      if (pt.st_two_jpq[x] != pt.st_two_jpq[y]) {
        return pt.st_two_jpq[x] < pt.st_two_jpq[y];
      }
      if (pt.st_p[x] != pt.st_p[y]) return pt.st_p[x] < pt.st_p[y];
      return pt.st_q[x] < pt.st_q[y];
    });
    for (std::int32_t pos = 0; pos < S; ++pos) {
      const std::int32_t f = perm[static_cast<std::size_t>(pos)];
      const auto t = static_cast<std::size_t>(next + pos);
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
      const double np = occ[static_cast<std::size_t>(p)];
      const double nq = occ[static_cast<std::size_t>(q)];
      occ_hh[t] = w * np * nq;
      occ_pp[t] = w * (1.0 - np) * (1.0 - nq);
      ++count[block_of(ch, pt.st_r[f], two_jpq / 2)];
    }
    next += S;
    w_off[static_cast<std::size_t>(ch) + 1] =
        w_off[static_cast<std::size_t>(ch)] + static_cast<std::int64_t>(S) * S;
  }
  ch_begin[static_cast<std::size_t>(nch)] = next;
  // Exclusive prefix sum: blk[b] = first sorted index of block b.
  std::vector<std::int32_t> blk(count.size(), 0);
  std::exclusive_scan(count.begin(), count.end(), blk.begin(), 0);
  PairBlocks t;
  t.blk = ToDevice(blk, "commutator::comm232_blk");
  t.ch_begin = ToDevice(ch_begin, "commutator::comm232_ch_begin");
  t.orig = ToDevice(orig, "commutator::comm232_orig");
  t.key = ToDevice(key, "commutator::comm232_key");
  t.local = ToDevice(local, "commutator::comm232_local");
  t.occ_hh = ToDevice(occ_hh, "commutator::comm232_occ_hh");
  t.occ_pp = ToDevice(occ_pp, "commutator::comm232_occ_pp");
  t.w_off = ToDevice(w_off, "commutator::comm232_w_off");
  t.w_size = w_off.back();
  t.view.blk = t.blk.device_data();
  t.view.ch_begin = t.ch_begin.device_data();
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

/// The output pointers of the pair-third build (null: not needed).
struct PairThirdOut {
  double* wx = nullptr;  ///< Blocks of X3.
  double* vx = nullptr;  ///< Blocks of X3^T (non-Hermitian X3 only).
  double* wy = nullptr;  ///< Blocks of Y3.
  double* vy = nullptr;  ///< Blocks of Y3^T (non-Hermitian Y3 only).
};
static_assert(std::is_trivially_copyable_v<PairThirdOut>);

/// W = E B3 E^T (and V = W^T where requested) of X3 and Y3 for every
/// channel: one work item per block element, W[t, u] = sum_{I in E_t, K in
/// E_u} E_tI E_uK B3[I, K]. Cost O(sum S_ch^2 nnz(E row)^2).
void BuildPairThirdBlocks(const detail::N7Views& v, const PairBlocksView& pb,
                          const op::CsrView& e, std::int64_t w_size,
                          const PairThirdOut& o) {
  if (w_size == 0) return;
  exec::ParallelFor(
      "commutator::comm232ss_pair_third", static_cast<exec::Index>(w_size),
      IMSRG_LAMBDA(const exec::Index idx) {
        const int ch = detail::FindBlock(pb.w_off, pb.num_channels, idx);
        const std::int64_t base = pb.ch_begin[ch];
        const std::int64_t S = pb.ch_begin[ch + 1] - base;
        const std::int64_t loc = idx - pb.w_off[ch];
        const int fr = pb.orig[base + loc % S];
        const int fc = pb.orig[base + loc / S];
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
      });
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

}  // namespace detail

namespace terms {

void Comm232ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  const detail::N7Views v = detail::PrepareN7("Comm232ss", X, Y, Z, ctx);
  const detail::TwoBodyOutView out = detail::TwoBodyOut("Comm232ss", Z, ctx);
  // [X2, Y3] needs X2 and Y3, [Y2, X3] needs Y2 and X3.
  const bool xy = v.x2.data != nullptr && v.y3.data != nullptr;
  const bool yx = v.y2.data != nullptr && v.x3.data != nullptr;
  if (!xy && !yx) return;
  const PairBlocks blocks = BuildPairBlocks(ctx);
  const PairBlocksView pb = blocks.view;
  const std::int64_t w_size = blocks.w_size;
  exec::DeviceBuffer<double> wx =
      MaybeBuffer(yx, w_size, "commutator::comm232_wx");
  exec::DeviceBuffer<double> vx =
      MaybeBuffer(yx && v.x3.packed == 0, w_size, "commutator::comm232_vx");
  exec::DeviceBuffer<double> wy =
      MaybeBuffer(xy, w_size, "commutator::comm232_wy");
  exec::DeviceBuffer<double> vy =
      MaybeBuffer(xy && v.y3.packed == 0, w_size, "commutator::comm232_vy");
  PairThirdOut o;
  o.wx = DataOrNull(wx);
  o.vx = DataOrNull(vx);
  o.wy = DataOrNull(wy);
  o.vy = DataOrNull(vy);
  BuildPairThirdBlocks(v, pb, ctx.impl().pair_third.e_map().device_view(),
                       w_size, o);
  // B^T: V if stored, else h_B W.
  const double* wxt = o.vx != nullptr ? o.vx : o.wx;
  const double* wyt = o.vy != nullptr ? o.vy : o.wy;
  const double sxt = o.vx != nullptr ? 1.0 : v.x3.sign;
  const double syt = o.vy != nullptr ? 1.0 : v.y3.sign;
  const double* wxp = o.wx;
  const double* wyp = o.wy;
  const std::int64_t total = Z.two_body().matrix().total_size();
  const basis::TwoBodyTableView tb = v.x2.tb;
  exec::ParallelFor(
      "commutator::comm232ss", static_cast<exec::Index>(total),
      IMSRG_LAMBDA(const exec::Index e) {
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
        if (wyp != nullptr) {
          z += BracketPairThird(v, pb, v.x2, wyp, wyt, syt, i, j, k, l, J);
        }
        if (wxp != nullptr) {
          z -= BracketPairThird(v, pb, v.y2, wxp, wxt, sxt, i, j, k, l, J);
        }
        z *= kSign;
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        out.data[e] += z;
      });
  exec::Fence("commutator::comm232ss");
}

}  // namespace terms

}  // namespace imsrg::commutator
