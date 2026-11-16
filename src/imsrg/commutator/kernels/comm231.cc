// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm231ss, [X2, Y3] + [X3, Y2] -> Z1 (imsrg3.h). Kernel TU.
//
// J-scheme form (the sum over the J3 channels carries (2 J3 + 1)):
//   Z_ij += 1 / (2 j_i + 1) sum_{J2} sum_{kets C=(a b), R=(c d) of J2}
//      w_C w_R (n_C nbar_R - n_R nbar_C) sum_{J3} (2 J3 + 1)
//      (A2^{J2}_{C,R} B3^{J3}((R i); (C j)))  [A = X, B = Y]  - [X <-> Y],
// w = 1/2 for a == b else 1 (unnormalized elements; the form follows from the
// two terms of `Weighted3` of the reference by exchanging the names of the
// two kets in the second), n_C = n_a n_b, nbar_C = nbar_a nbar_b.
//
// Element-wise reference (`detail::Comm231ssReference`, the kernel before
// the row tiling): one work item per (output pair, two-body channel), the
// three-body elements read through `ThreeBodyElement`.
//
// Batch form (`detail::Comm231Batch`, one batch = the W columns of its
// tiles; needs W). B3^{J3}((R i); (C j)) is the element (row (R i), column
// (C j)) of W(B) of the channel J3 (`op::PairThirdSpace`; rows in the sorted
// order of `Comm232Tables`, whose blocks (channel, third i, Jpq) hold the
// states (R, i) of all pairs R contiguously, classes 0..2 ascending), W
// itself is the unnormalized element, so with the stored normalized element
// A_st the product w_C w_R A_un = cw_C cw_R A_st, cw = 1/sqrt 2 for equal
// orbits else 1; cw_R is in `occ_hh` = cw n_R and `occ_pp` = cw nbar_R. A
// column (C j) adds, for every i of the one-body channel of j,
//   (2 J3 + 1) cw_C [n_C sum_R occ_pp(R) A_st(C, R) W[(R i), (C j)]
//                    - nbar_C sum_R occ_hh(R) A_st(C, R) W[(R i), (C j)]]
// (A = X, B = Y, minus A = Y, B = X). R and C lie in the same two-body
// channel (same parity and Tz, as i and j have the same one-body channel).
// Only the rows of the classes the occupations of C need are visited (n_C
// != 0 needs nbar_R != 0 etc.). No transposed W is needed.
//
// Work items: one per output pair (i, j) of a one-body channel; the host
// groups the batch's columns by third j (counting sort, plan order), and the
// item sums over the columns of its j in that order, so every Z1 element is
// written by one item in a fixed order (no atomics; the result does not
// depend on the thread count). Every column state belongs to exactly one
// batch of a partition of the lines, so the batches add up to the whole term
// (up to the order of the sums). Cost per batch: the sum over its columns
// (C, j) of n_1B(j) (the i of the channel) times the rows of the block
// (channel, i, Jpq) visited.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator {

namespace {

/// True if orbits i and j are in the same one-body channel (l, 2j, 2tz).
IMSRG_INLINE_FUNCTION bool SameChannel(const detail::N7Views& v, int i,
                                       int j) noexcept {
  return v.ob.l[i] == v.ob.l[j] && v.ob.j2[i] == v.ob.j2[j] &&
         v.ob.tz2[i] == v.ob.tz2[j];
}

/// sum_J3 (2 J3 + 1) [A2^{J2}_{ab,cd} B3^{J3}((cd)J2 i; (ab)J2 j) -
/// B3^{J3}((ab)J2 i; (cd)J2 j) A2^{J2}_{cd,ab}] for fixed orbits. O(J3
/// range * element read).
IMSRG_INLINE_FUNCTION double Weighted3(const detail::N7Views& v,
                                       const detail::TwoBodyView& a2,
                                       const detail::ThreeBodyPartView& b3,
                                       int two_j2, int a, int b, int c, int d,
                                       int i, int j) noexcept {
  const double ab_cd = a2.get_unnormalized(a, b, c, d, two_j2);
  const double cd_ab = a2.get_unnormalized(c, d, a, b, two_j2);
  if (ab_cd == 0.0 && cd_ab == 0.0) return 0.0;
  const int ji = v.ob.j2[i];
  const int lo = two_j2 > ji ? two_j2 - ji : ji - two_j2;
  const int hi = two_j2 + ji;
  const int j2 = two_j2 / 2;
  double sum = 0.0;
  for (int two_j3 = lo; two_j3 <= hi; two_j3 += 2) {
    const double t1 =
        ab_cd == 0.0 ? 0.0
                     : ab_cd * detail::ThreeBodyElement(v, b3, two_j3, c, d, i,
                                                        j2, a, b, j, j2);
    const double t2 =
        cd_ab == 0.0 ? 0.0
                     : cd_ab * detail::ThreeBodyElement(v, b3, two_j3, a, b, i,
                                                        j2, c, d, j, j2);
    sum += (two_j3 + 1) * (t1 - t2);
  }
  return sum;
}

/// The part of Z_ij (without the factor 1 / (2 j_i + 1)) from the two-body
/// channel `ch2`: sum over kets (ab), (cd) of the channel, w_ab w_cd n_a n_b
/// nbar_c nbar_d (Weighted3(X2, Y3) - Weighted3(Y2, X3)). Cost O(kets^2 * J3
/// range * element read).
IMSRG_INLINE_FUNCTION double Comm231Channel(const detail::N7Views& v, int ch2,
                                            int i, int j) noexcept {
  const basis::TwoBodyTableView& tb = v.x2.tb;
  const int two_j2 = tb.channel_two_j[ch2];
  const int nk = tb.channel_num_kets[ch2];
  const std::int32_t* kets = tb.channel_kets + tb.channel_ket_offset[ch2];
  double total = 0.0;
  for (int ka = 0; ka < nk; ++ka) {
    const int a = tb.ket_p[kets[ka]];
    const int b = tb.ket_q[kets[ka]];
    const double nab = v.ob.occupation[a] * v.ob.occupation[b];
    if (nab == 0.0) continue;
    const double wab = a == b ? 0.5 : 1.0;
    for (int kc = 0; kc < nk; ++kc) {
      const int c = tb.ket_p[kets[kc]];
      const int d = tb.ket_q[kets[kc]];
      const double ncd =
          (1.0 - v.ob.occupation[c]) * (1.0 - v.ob.occupation[d]);
      if (ncd == 0.0) continue;
      const double w = wab * (c == d ? 0.5 : 1.0) * nab * ncd;
      total += w * (Weighted3(v, v.x2, v.y3, two_j2, a, b, c, d, i, j) -
                    Weighted3(v, v.y2, v.x3, two_j2, a, b, c, d, i, j));
    }
  }
  return total;
}

}  // namespace

namespace detail {

void Comm231ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const detail::N7Views v = PrepareN7("Comm231ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  const exec::Index norb = v.norb;
  const exec::Index nch = v.x2.tb.num_channels;
  if (norb == 0 || nch == 0) return;
  // Pass 1: one partial sum per (output pair, two-body channel).
  exec::DeviceBuffer<double> partial(
      static_cast<std::size_t>(norb * norb * nch), "commutator::comm231_part");
  const exec::DeviceView<double> part = exec::View(partial);
  exec::ParallelFor(
      "commutator::comm231ss_reference_channels", norb * norb * nch,
      IMSRG_LAMBDA(const exec::Index w) {
        const int ch2 = static_cast<int>(w % nch);
        const exec::Index pair = w / nch;
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        double value = 0.0;
        if (SameChannel(v, i, j)) value = Comm231Channel(v, ch2, i, j);
        part(w) = value;
      });
  // Pass 2: ordered reduction over the channels, added to Z1.
  double* z1 = OneBodyOut(Z);
  exec::ParallelFor(
      "commutator::comm231ss_reference_reduce", norb * norb,
      IMSRG_LAMBDA(const exec::Index pair) {
        const int i = static_cast<int>(pair % norb);
        const int j = static_cast<int>(pair / norb);
        if (!SameChannel(v, i, j)) return;
        double sum = 0.0;
        for (exec::Index ch = 0; ch < nch; ++ch) sum += part(pair * nch + ch);
        z1[pair] += sum / (v.ob.j2[i] + 1);
      });
  exec::Fence("commutator::comm231ss_reference");
}

namespace {

/// The W columns of a batch grouped by their third orbit j (host lists,
/// uploaded per batch): the columns of third j are the entries
/// [col_begin[j], col_begin[j + 1]) in plan order (tiles, then states
/// ascending), each with its channel, pair (p, q), pair J and the offset of
/// its W column (`BatchColumn` convention, doubles into a W block).
struct ColumnLists {
  std::vector<std::int32_t> col_begin;  ///< norb + 1 entries.
  std::vector<std::int32_t> ch;         ///< Per entry: three-body channel.
  std::vector<std::int32_t> jpq;        ///< Per entry: pair J (not 2J).
  std::vector<std::int32_t> p;          ///< Per entry: pair p.
  std::vector<std::int32_t> q;          ///< Per entry: pair q >= p.
  std::vector<std::int64_t> offset;     ///< Per entry: W column offset.
};

/// The `ColumnLists` of `plan` (`pt`: host view of the pair-third space of
/// the context). Counting sort by third, stable in plan order, so the lists
/// are deterministic. O(states of the batch + norb).
ColumnLists ListColumnsByThird(const op::PairThirdView& pt,
                               const LineBatch& plan, int norb) {
  ColumnLists out;
  out.col_begin.assign(static_cast<std::size_t>(norb) + 1, 0);
  for (const LineTile& tile : plan.tiles) {
    const std::int32_t base = pt.ch_state_offset[tile.channel];
    for (std::int32_t t = tile.state_begin; t < tile.state_end; ++t) {
      ++out.col_begin[static_cast<std::size_t>(pt.st_r[base + t]) + 1];
    }
  }
  for (std::size_t r = 0; r + 1 < out.col_begin.size(); ++r) {
    out.col_begin[r + 1] += out.col_begin[r];
  }
  const auto n = static_cast<std::size_t>(out.col_begin.back());
  out.ch.resize(n);
  out.jpq.resize(n);
  out.p.resize(n);
  out.q.resize(n);
  out.offset.resize(n);
  std::vector<std::int32_t> next(out.col_begin.begin(),
                                 out.col_begin.end() - 1);
  for (const LineTile& tile : plan.tiles) {
    const std::int32_t base = pt.ch_state_offset[tile.channel];
    const std::int64_t stride = pt.ch_num_states[tile.channel];
    for (std::int32_t t = tile.state_begin; t < tile.state_end; ++t) {
      const std::int32_t f = base + t;
      const auto e = static_cast<std::size_t>(
          next[static_cast<std::size_t>(pt.st_r[f])]++);
      out.ch[e] = tile.channel;
      out.jpq[e] = pt.st_two_jpq[f] / 2;
      out.p[e] = pt.st_p[f];
      out.q[e] = pt.st_q[f];
      out.offset[e] = tile.w_offset + (t - tile.state_begin) * stride;
    }
  }
  return out;
}

/// Everything the kernel of one comm231 batch reads (device pointers by
/// value; trivially copyable).
struct Comm231Args {
  N7Views v;
  Comm232TablesView tab;
  const double* wx = nullptr;  ///< W(X3), or null.
  const double* wy = nullptr;  ///< W(Y3), or null.
  const std::int32_t* col_begin = nullptr;
  const std::int32_t* col_ch = nullptr;
  const std::int32_t* col_jpq = nullptr;
  const std::int32_t* col_p = nullptr;
  const std::int32_t* col_q = nullptr;
  const std::int64_t* col_offset = nullptr;
  double* z1 = nullptr;
};
static_assert(std::is_trivially_copyable_v<Comm231Args>);

/// The batch's contribution to Z_ij (without 1 / (2 j_i + 1)) for one output
/// pair of a one-body channel (file comment): the sum over the batch's
/// columns (C, j) of third j, in list order, and over the rows (R, i) of the
/// column's channel of the visited occupation classes. O(columns of j *
/// rows of (channel, i, Jpq)).
IMSRG_INLINE_FUNCTION double Comm231Pair(const Comm231Args& a, int i,
                                         int j) noexcept {
  const N7Views& v = a.v;
  const Comm232TablesView& tab = a.tab;
  const basis::TwoBodyTableView& tb = v.x2.tb;
  double total = 0.0;
  for (int e = a.col_begin[j]; e < a.col_begin[j + 1]; ++e) {
    const int ch = a.col_ch[e];
    const int jpq = a.col_jpq[e];
    const int p = a.col_p[e];
    const int q = a.col_q[e];
    const double np = v.ob.occupation[p];
    const double nq = v.ob.occupation[q];
    // Class of C = (p q): n_C != 0 needs the pp rows R (nbar_R != 0), nbar_C
    // != 0 the hh rows.
    const bool c_hh = np != 0.0 && nq != 0.0;
    const bool c_pp = np != 1.0 && nq != 1.0;
    if (!c_hh && !c_pp) continue;
    const std::int64_t b =
        (static_cast<std::int64_t>(ch) * tab.norb + i) * tab.nj + jpq;
    const int r_lo = tab.blk[kComm232Classes * b + (c_pp ? 0 : 1)];
    const int r_hi = tab.blk[kComm232Classes * b + (c_hh ? 3 : 2)];
    if (r_lo >= r_hi) continue;
    const int ch2 =
        tb.channel_of(OrbitIndex{p}, OrbitIndex{q}, TwoJ{2 * jpq}).value();
    if (ch2 >= tb.num_channels) continue;
    const std::int64_t n2 = tb.channel_num_kets[ch2];
    const std::int32_t loc_c = tb.local_index(
        tb.ket_index(OrbitIndex{p}, OrbitIndex{q}), TwoJ{2 * jpq});
    if (loc_c < 0) continue;
    const double n_c = np * nq;
    const double nbar_c = (1.0 - np) * (1.0 - nq);
    const double cw_c = p == q ? constants::kInvSqrt2 : 1.0;
    // W column of C: rows R at wcol[t] for the flat sorted row t.
    const std::int64_t wcol = a.col_offset[e] - tab.ch_begin[ch];
    double col_sum = 0.0;
    for (int term = 0; term < 2; ++term) {
      const TwoBodyView& a2 = term == 0 ? v.x2 : v.y2;
      const double* w = term == 0 ? a.wy : a.wx;
      if (a2.data == nullptr || w == nullptr) continue;
      const double* xc = a2.data + a2.block_offset[ch2] + loc_c;
      double acc_hh = 0.0;
      double acc_pp = 0.0;
      for (int t = r_lo; t < r_hi; ++t) {
        const double xw = xc[tab.local[t] * n2] * w[wcol + t];
        acc_hh += tab.occ_hh[t] * xw;
        acc_pp += tab.occ_pp[t] * xw;
      }
      const double s = n_c * acc_pp - nbar_c * acc_hh;
      col_sum += term == 0 ? s : -s;
    }
    total += (v.sp.ch_two_j[ch] + 1) * cw_c * col_sum;
  }
  return total;
}

/// Copies a host vector into a new device buffer.
template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& h, const char* label) {
  exec::DeviceBuffer<T> d(h.size(), label);
  if (!h.empty()) d.CopyFromHost(h);
  return d;
}

}  // namespace

void Comm231Batch(const N7BatchInputs& in, double* z1) {
  IMSRG_CHECK(in.plan != nullptr && in.ctx != nullptr,
              "Comm231Batch: the batch inputs need the host plan and context");
  const LineBatch& plan = *in.plan;
  if (plan.tiles.empty()) return;
  // A term needs the two-body part of one operand and the W block of the
  // other's three-body part.
  const bool t1 = in.v.x2.data != nullptr && in.wy != nullptr;
  const bool t2 = in.v.y2.data != nullptr && in.wx != nullptr;
  if (!t1 && !t2) return;
  const exec::Index norb = in.v.norb;
  if (norb == 0) return;
  const ColumnLists cols = ListColumnsByThird(
      in.ctx->impl().pair_third.host_view(), plan, static_cast<int>(norb));
  if (cols.ch.empty()) return;
  const exec::DeviceBuffer<std::int32_t> col_begin =
      Upload(cols.col_begin, "commutator::comm231_col_begin");
  const exec::DeviceBuffer<std::int32_t> col_ch =
      Upload(cols.ch, "commutator::comm231_col_ch");
  const exec::DeviceBuffer<std::int32_t> col_jpq =
      Upload(cols.jpq, "commutator::comm231_col_jpq");
  const exec::DeviceBuffer<std::int32_t> col_p =
      Upload(cols.p, "commutator::comm231_col_p");
  const exec::DeviceBuffer<std::int32_t> col_q =
      Upload(cols.q, "commutator::comm231_col_q");
  const exec::DeviceBuffer<std::int64_t> col_offset =
      Upload(cols.offset, "commutator::comm231_col_offset");
  Comm231Args a;
  a.v = in.v;
  a.tab = in.tab;
  a.wx = t2 ? in.wx : nullptr;
  a.wy = t1 ? in.wy : nullptr;
  a.col_begin = col_begin.device_data();
  a.col_ch = col_ch.device_data();
  a.col_jpq = col_jpq.device_data();
  a.col_p = col_p.device_data();
  a.col_q = col_q.device_data();
  a.col_offset = col_offset.device_data();
  a.z1 = z1;
  exec::ParallelForDynamic(
      "commutator::comm231_batch", norb * norb,
      IMSRG_LAMBDA(const exec::Index w) {
        const int i = static_cast<int>(w % norb);
        const int j = static_cast<int>(w / norb);
        if (!SameChannel(a.v, i, j)) return;
        a.z1[w] += Comm231Pair(a, i, j) / (a.v.ob.j2[i] + 1);
      });
  exec::Fence("commutator::comm231_batch");
}

void Comm231Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  const N7Views v = PrepareN7("Comm231ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  if (v.norb == 0 || v.x2.tb.num_channels == 0) return;
  double* z1 = OneBodyOut(Z);
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{false, true},
                 [&](const N7BatchInputs& in) { Comm231Batch(in, z1); });
}

}  // namespace detail

namespace terms {

void Comm231ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm231Rows(
      X, Y, Z, ctx,
      basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
      detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
