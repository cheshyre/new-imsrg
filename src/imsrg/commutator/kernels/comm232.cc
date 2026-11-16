// SPDX-License-Identifier: GPL-3.0-only
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
// Two evaluations of F:
// * reference (`detail::Comm232ssReference`): one work item per stored Z2
//   element (both triangles of every block), B^{J3} read with
//   `ThreeBodyElement` (recoupled from the canonical kets per read);
// * batched (`detail::Comm232Batch`, run by `detail::Comm232Rows` over the
//   batches of the lines of a `basis::RowSet`; `terms::Comm232ss` is the
//   whole space): B^{J3}((a b) J' q; (r s) J c) is the element (row
//   (a b) J' q, column (r s) J c) of the pair-third block W = E B3 E^T of
//   channel J3 (`op::PairThirdSpace`), which the N7 batch loop builds for
//   the columns of the batch's tiles, every row in the sorted order of
//   `Comm232Tables` (third, Jpq, occupation class of the pair, p, q;
//   detail/n7_batch.h). So the internal pairs (a b) of fixed (J', q) are a
//   contiguous run of rows, read down one column of W. The classes (hh-only:
//   n_a n_b != 0 = nbar_a nbar_b; mixed; pp-only; inactive: both products
//   0) restrict the rows: an inactive pair (a b) never contributes (o_abc =
//   0), and for n_c = 0 (1) only the pairs with n_a n_b != 0 (nbar_a nbar_b
//   != 0) are visited. B^T is read from W(B^T) = sign * wt
//   (`N7BatchInputs`).
//
// Work items of a batch: a column state ((r s) J, c) of channel J3 enters
// F only through its pair, in the direct half sums F(i, j; k l), F(j, i;
// k l) of the Z2 elements (ij, kl; J) with (k l) = (r s), and in the
// transposed half sums F[A^T, B^T](k, l; i j), F[A^T, B^T](l, k; i j) of
// the elements with (i j) = (r s) (columns of W(B^T)). In both, the
// creator-side pair is the OTHER pair of the element. The host groups the
// batch's column states by key (r, s, J), i.e. by the Z2 channel of (r s)
// at J and the local index of (r s) there; the direct pass runs one work
// item per (key, bra of that channel) adding into element (bra, key), the
// transposed pass one per (key, ket) adding into element (key, ket). Within
// a pass the items write distinct elements and the passes run one after the
// other, so there are no atomics; every column state belongs to exactly one
// batch of a partition of the lines, so the batches add up to the whole
// term. The cost of a batch is proportional to its columns: the sum over
// its keys of n_ch2 * (entries of the key) * (J' range) * (visited rows).
// The arithmetic per (element, column, J') is that of the element-wise
// pair-third kernel this replaces; only the order of the sums over the
// columns (and the batches) differs, which changes results at the rounding
// level (M10-Q79).

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
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/device_views.h"
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

/// The overall sign s of the m-scheme expression (file comment), fixed by
/// the m-scheme brute force (commutator_ref::mscheme).
constexpr double kSign = -1.0;

/// (-1)^n for an integer n.
IMSRG_INLINE_FUNCTION int Phase(int n) noexcept { return (n & 1) ? -1 : 1; }

/// |a - b| for ints.
IMSRG_INLINE_FUNCTION int AbsDiff(int a, int b) noexcept {
  return a > b ? a - b : b - a;
}

/// Larger of two ints.
IMSRG_INLINE_FUNCTION int Max(int a, int b) noexcept { return a > b ? a : b; }
/// Smaller of two ints.
IMSRG_INLINE_FUNCTION int Min(int a, int b) noexcept { return a < b ? a : b; }

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
  if (a2.data == nullptr || b3.local == nullptr) return 0.0;
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

/// One W column of a batch: the pair-third state ((r s) J, c) of channel
/// `ch3` (2 J3 = `two_j3`), whose rows (sorted flat index t of `tab`) are
/// `w[offset + t - tab.ch_begin[ch3]]` of a W block of the batch.
struct Column {
  int ch3 = 0;
  int two_j3 = 0;
  int c = 0;
  std::int64_t offset = 0;
};
static_assert(std::is_trivially_copyable_v<Column>);

/// The creator-side half sum F[A, B](p, q; r s) at pair J restricted to one
/// column `col` = ((r s) J, c) of W(B) (`w`, a W block of the batch;
/// `trans` != 0 evaluates F[A^T, B^T] with `w` the block of B^T, the sign
/// of which the caller applies). J3 and c are fixed by the column, so the
/// sum runs over the J' allowed by triangle(j_p, j_c, J') and
/// triangle(J', j_q, J3), and per J' over the rows ((a b) J' q) of the
/// visited classes (by n_c). The arithmetic per J' is that of the
/// element-wise pair-third kernel this replaces. Zero if A or `w` is
/// missing. Cost O(J'-range * visited rows of the block (q, J')).
IMSRG_INLINE_FUNCTION double HalfSumColumn(const detail::N7Views& v,
                                           const detail::Comm232TablesView& tab,
                                           const detail::TwoBodyView& a2,
                                           const double* w, int trans, int p,
                                           int q, int J,
                                           const Column& col) noexcept {
  if (a2.data == nullptr || w == nullptr) return 0.0;
  const basis::TwoBodyTableView& tb = a2.tb;
  const int jp = v.ob.j2[p];
  const int jq = v.ob.j2[q];
  const int c = col.c;
  const int jc = v.ob.j2[c];
  const int two_J = 2 * J;
  const int two_j3 = col.two_j3;
  const double nc = v.ob.occupation[c];
  const double nbar_c = 1.0 - nc;
  // Visited classes: n_c == 0 needs n_a n_b != 0 (classes 0, 1), n_c == 1
  // needs nbar_a nbar_b != 0 (classes 1, 2).
  const int k_lo = nc == 1.0 ? 1 : 0;
  const int k_hi = nc == 0.0 ? 2 : 3;
  const KetIndex pc =
      tb.ket_index(OrbitIndex{p < c ? p : c}, OrbitIndex{p < c ? c : p});
  // J' couples p and c, and (J', j_q) must reach 2 J3 (2 J3 and 2 j_q are
  // odd, so the differences are even).
  const int jp_lo = Max(AbsDiff(jp, jc) / 2, AbsDiff(two_j3, jq) / 2);
  const int jp_hi = Min((jp + jc) / 2, (two_j3 + jq) / 2);
  // W[t, col] at w[wc + t] for the sorted rows t of channel ch3.
  const std::int64_t wc = col.offset - tab.ch_begin[col.ch3];
  double total = 0.0;
  for (int Jp = jp_lo; Jp <= jp_hi; ++Jp) {
    const std::int32_t xloc = tb.local_index(pc, TwoJ{2 * Jp});
    if (xloc < 0) continue;
    const std::int64_t rb =
        (static_cast<std::int64_t>(col.ch3) * tab.norb + q) * tab.nj + Jp;
    const int r0 = tab.blk[detail::kComm232Classes * rb + k_lo];
    const int r1 = tab.blk[detail::kComm232Classes * rb + k_hi];
    if (r0 >= r1) continue;
    const double six = v.sj(TwoJ{jq}, TwoJ{jp}, TwoJ{two_J}, TwoJ{jc},
                            TwoJ{two_j3}, TwoJ{2 * Jp});
    if (six == 0.0) continue;
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
    double acc_hh = 0.0;
    double acc_pp = 0.0;
    for (int t = r0; t < r1; ++t) {
      const double xw = xb[x0 + tab.local[t] * xs] * w[wc + t];
      acc_hh += tab.occ_hh[t] * xw;
      acc_pp += tab.occ_pp[t] * xw;
    }
    total += pref * (two_j3 + 1) * six * (nbar_c * acc_hh + nc * acc_pp);
  }
  return total;
}

/// Fields per key of a work list: the pair p <= q, J, the Z2 channel of
/// (p, q) at J and the local index of (p, q) in it.
constexpr int kKeyFields = 5;
/// Fields per entry of a work list: the channel and the third orbit c of
/// the column state.
constexpr int kEntryFields = 2;

/// The work list of one batch (host): the W columns of the batch (its
/// tiles' pair-third states ((p q) J, c)) grouped by key (p, q, J), keys
/// ascending in (p, q, J), the entries of a key in plan order (tiles, then
/// states ascending). Key k owns the entries [key_entry[k], key_entry[k +
/// 1]) and the work items [key_item[k], key_item[k + 1]), one per local ket
/// of its Z2 channel.
struct WorkList {
  std::vector<std::int32_t> key;        ///< kKeyFields per key.
  std::vector<std::int32_t> key_entry;  ///< num_keys + 1 entry offsets.
  std::vector<std::int64_t> key_item;   ///< num_keys + 1 item offsets.
  std::vector<std::int32_t> entry;      ///< kEntryFields per entry.
  std::vector<std::int64_t> column;     ///< Per entry: W column offset.
};

/// Device view of a `WorkList` (raw device pointers, captured by value).
struct WorkView {
  const std::int32_t* key = nullptr;
  const std::int32_t* key_entry = nullptr;
  const std::int64_t* key_item = nullptr;
  const std::int32_t* entry = nullptr;
  const std::int64_t* column = nullptr;
  std::int32_t num_keys = 0;
  std::int64_t num_items = 0;
};
static_assert(std::is_trivially_copyable_v<WorkView>);

/// A `WorkList` in device buffers.
struct WorkBuffers {
  exec::DeviceBuffer<std::int32_t> key;
  exec::DeviceBuffer<std::int32_t> key_entry;
  exec::DeviceBuffer<std::int64_t> key_item;
  exec::DeviceBuffer<std::int32_t> entry;
  exec::DeviceBuffer<std::int64_t> column;
};

/// Copies a host vector into a new device buffer.
template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& h, const char* label) {
  exec::DeviceBuffer<T> d(h.size(), label);
  if (!h.empty()) d.CopyFromHost(std::span<const T>(h));
  return d;
}

/// Bytes of a device buffer.
template <class T>
std::int64_t BufferBytes(const exec::DeviceBuffer<T>& b) {
  return static_cast<std::int64_t>(b.size() * sizeof(T));
}

/// The column states of a batch in plan order: per state its key slot
/// (p norb + q) nj + J, channel, third orbit and W column offset.
struct BatchColumns {
  std::vector<std::int64_t> slot;
  std::vector<std::int32_t> ch3;
  std::vector<std::int32_t> third;
  std::vector<std::int64_t> offset;
};

/// Lists the pair-third states of the tiles of `plan` (`pt`: host view;
/// `norb`, `nj` of the comm232 tables). O(states of the batch).
BatchColumns ListColumns(const op::PairThirdView& pt,
                         const detail::LineBatch& plan, std::int64_t norb,
                         std::int64_t nj) {
  BatchColumns cols;
  for (const detail::LineTile& tile : plan.tiles) {
    const std::int32_t ch = tile.channel;
    const std::int64_t S = pt.ch_num_states[ch];
    for (std::int32_t t = tile.state_begin; t < tile.state_end; ++t) {
      const std::int32_t f = pt.ch_state_offset[ch] + t;
      cols.slot.push_back((pt.st_p[f] * norb + pt.st_q[f]) * nj +
                          pt.st_two_jpq[f] / 2);
      cols.ch3.push_back(ch);
      cols.third.push_back(pt.st_r[f]);
      cols.offset.push_back(tile.w_offset + (t - tile.state_begin) * S);
    }
  }
  return cols;
}

/// Appends the key of `slot` (decoded with `norb`, `nj`) to `w`: the pair,
/// J, its Z2 channel in `two`, its local index there, and the items (one
/// per local ket of the channel).
/// @throws InvariantError if the pair has no ket at J (not a pair-third
///         state of the context).
void AppendKey(const basis::TwoBodyChannels& two, std::int64_t slot,
               std::int64_t norb, std::int64_t nj, WorkList& w) {
  const auto J = static_cast<std::int32_t>(slot % nj);
  const auto p = static_cast<std::int32_t>(slot / nj / norb);
  const auto q = static_cast<std::int32_t>(slot / nj % norb);
  const KetIndex ket = two.ket_index(OrbitIndex{p}, OrbitIndex{q});
  const std::optional<LocalKetIndex> loc = two.local_index(ket, TwoJ{2 * J});
  IMSRG_CHECK(loc.has_value(), "comm232ss: a column pair has no two-body ket");
  const ChannelIndex ch2 =
      two.channel_of(OrbitIndex{p}, OrbitIndex{q}, TwoJ{2 * J});
  w.key.insert(w.key.end(), {p, q, J, ch2.value(), loc->value()});
  w.key_item.push_back(w.key_item.back() + two.channel(ch2).num_kets);
}

/// The work list of the batch `plan` of the context `ctx` (file comment):
/// its column states sorted by (key slot, plan position), so the grouping
/// is deterministic. Host; O(m log m) for the m states of the batch.
WorkList BuildWorkList(const ThreeBodyContext& ctx,
                       const detail::LineBatch& plan) {
  const ThreeBodyContext::Impl& im = ctx.impl();
  const std::int64_t norb = im.comm232.norb;
  const std::int64_t nj = im.comm232.nj;
  const BatchColumns cols =
      ListColumns(im.pair_third.host_view(), plan, norb, nj);
  // Sort (slot, position) pairs packed into one integer: slots are below
  // norb^2 nj < 2^31 and positions below 2^31.
  constexpr std::int64_t kShift = std::int64_t{1} << 32;
  std::vector<std::int64_t> order(cols.slot.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = cols.slot[i] * kShift + static_cast<std::int64_t>(i);
  }
  std::sort(order.begin(), order.end());
  const basis::TwoBodyChannels& two = ctx.model_space_ptr()->two_body();
  WorkList w;
  w.key_item.push_back(0);
  std::int64_t prev = -1;
  for (const std::int64_t o : order) {
    const std::int64_t slot = o / kShift;
    const auto i = static_cast<std::size_t>(o % kShift);
    if (slot != prev) {
      w.key_entry.push_back(static_cast<std::int32_t>(w.column.size()));
      AppendKey(two, slot, norb, nj, w);
      prev = slot;
    }
    w.entry.push_back(cols.ch3[i]);
    w.entry.push_back(cols.third[i]);
    w.column.push_back(cols.offset[i]);
  }
  w.key_entry.push_back(static_cast<std::int32_t>(w.column.size()));
  return w;
}

/// Uploads `w`. O(size of w).
WorkBuffers Upload(const WorkList& w) {
  WorkBuffers b;
  b.key = ToDevice(w.key, "commutator::comm232_work_key");
  b.key_entry = ToDevice(w.key_entry, "commutator::comm232_work_key_entry");
  b.key_item = ToDevice(w.key_item, "commutator::comm232_work_key_item");
  b.entry = ToDevice(w.entry, "commutator::comm232_work_entry");
  b.column = ToDevice(w.column, "commutator::comm232_work_column");
  return b;
}

/// The device view of the buffers `b` of the work list `w`.
WorkView ViewOf(const WorkBuffers& b, const WorkList& w) {
  WorkView v;
  v.key = b.key.device_data();
  v.key_entry = b.key_entry.device_data();
  v.key_item = b.key_item.device_data();
  v.entry = b.entry.device_data();
  v.column = b.column.device_data();
  v.num_keys = static_cast<std::int32_t>(w.key_item.size()) - 1;
  v.num_items = w.key_item.back();
  return v;
}

/// Kernel of one pass over the work list of a batch (file comment; a
/// functor rather than a lambda, so the launch may sit in a function of
/// this TU's unnamed namespace: nvcc restricts the enclosing functions of
/// extended lambdas). Item (key, o), o a local ket of the key's Z2 channel
/// with pair (p, q) (the creator-side pair of the half sums): for
/// `transposed` == 0 the direct terms F(p, q; key) - p_pq F(q, p; key) of
/// element (o, key), with W(Y3) = `wy` and W(X3) = `wx`; otherwise the
/// transposed terms -F[A^T, B^T](p, q; key) + p_pq F[A^T, B^T](q, p; key)
/// of element (key, o), with W(Y3^T) = `sy` * `wy`, W(X3^T) = `sx` * `wx`.
/// [X2, Y3] minus [Y2, X3], times s, normalized, added to Z2. Cost per item
/// O(4 entries HalfSumColumn).
struct PassFunctor {
  detail::N7Views v;
  detail::Comm232TablesView tab;
  WorkView work;
  detail::TwoBodyOutView out;
  const double* wx = nullptr;  ///< W(X3), or W(X3^T) up to sx.
  const double* wy = nullptr;  ///< W(Y3), or W(Y3^T) up to sy.
  double sx = 1.0;             ///< Sign of wx (transposed pass).
  double sy = 1.0;             ///< Sign of wy (transposed pass).
  int transposed = 0;          ///< 0: direct pass, 1: transposed pass.

  IMSRG_FUNCTION void operator()(const exec::Index idx) const {
    const int k = detail::FindBlock(work.key_item, work.num_keys, idx);
    const std::int32_t* key = work.key + kKeyFields * std::int64_t{k};
    const int J = key[2];
    const int ch2 = key[3];
    const basis::TwoBodyTableView& tb = v.x2.tb;
    const int o = static_cast<int>(idx - work.key_item[k]);
    const int ket = tb.channel_kets[tb.channel_ket_offset[ch2] + o];
    const int p = tb.ket_p[ket];
    const int q = tb.ket_q[ket];
    double f1y = 0.0;  // F[X2, Y3](p, q; key) over the key's columns
    double f2y = 0.0;  // F[X2, Y3](q, p; key)
    double f1x = 0.0;  // F[Y2, X3](p, q; key)
    double f2x = 0.0;  // F[Y2, X3](q, p; key)
    for (int e = work.key_entry[k]; e < work.key_entry[k + 1]; ++e) {
      Column col;
      col.ch3 = work.entry[kEntryFields * std::int64_t{e}];
      col.c = work.entry[kEntryFields * std::int64_t{e} + 1];
      col.two_j3 = v.sp.ch_two_j[col.ch3];
      col.offset = work.column[e];
      f1y += HalfSumColumn(v, tab, v.x2, wy, transposed, p, q, J, col);
      f2y += HalfSumColumn(v, tab, v.x2, wy, transposed, q, p, J, col);
      f1x += HalfSumColumn(v, tab, v.y2, wx, transposed, p, q, J, col);
      f2x += HalfSumColumn(v, tab, v.y2, wx, transposed, q, p, J, col);
    }
    const int p_pq = Phase((v.ob.j2[p] + v.ob.j2[q]) / 2 - J);
    const double gy = transposed != 0 ? -sy : 1.0;
    const double gx = transposed != 0 ? -sx : 1.0;
    double z = 0.0;
    if (wy != nullptr) z += gy * (f1y - p_pq * f2y);
    if (wx != nullptr) z -= gx * (f1x - p_pq * f2x);
    z *= kSign;
    if (p == q) z /= constants::kSqrt2;
    if (key[0] == key[1]) z /= constants::kSqrt2;
    const std::int64_t n = tb.channel_num_kets[ch2];
    const std::int64_t loc = key[4];
    const std::int64_t e2 = transposed != 0 ? loc + o * n : o + loc * n;
    out.data[out.block_offset[ch2] + e2] += z;
  }
};
static_assert(std::is_trivially_copyable_v<PassFunctor>);

/// Runs the direct and then the transposed pass of the batch `in` over the
/// work list `work` into `z2` (functor `PassFunctor`, one dynamic work item
/// per (key, local ket)), then fences.
void RunPasses(const detail::N7BatchInputs& in,
               const detail::TwoBodyOutView& z2, const WorkView& work) {
  PassFunctor f;
  f.v = in.v;
  f.tab = in.tab;
  f.work = work;
  f.out = z2;
  f.wx = in.wx;
  f.wy = in.wy;
  exec::ParallelForDynamic("commutator::comm232ss_direct", work.num_items, f);
  // The transposed items write other elements than the direct ones of the
  // same key, so they run after the direct pass (same execution space, in
  // order).
  f.wx = in.wxt;
  f.wy = in.wyt;
  f.sx = in.sxt;
  f.sy = in.syt;
  f.transposed = 1;
  exec::ParallelForDynamic("commutator::comm232ss_transposed", work.num_items,
                           f);
  exec::Fence("commutator::comm232ss");
}

}  // namespace

namespace detail {

Comm232TablesView Comm232Tables::View() const noexcept {
  Comm232TablesView t;
  t.blk = blk.device_data();
  t.ch_begin = ch_begin.device_data();
  t.orig = orig.device_data();
  t.key = key.device_data();
  t.local = local.device_data();
  t.occ_hh = occ_hh.device_data();
  t.occ_pp = occ_pp.device_data();
  t.norb = norb;
  t.nj = nj;
  t.num_channels = static_cast<std::int32_t>(ch_active_host.size());
  return t;
}

std::int64_t Comm232Tables::Bytes() const noexcept {
  return BufferBytes(blk) + BufferBytes(ch_begin) + BufferBytes(orig) +
         BufferBytes(key) + BufferBytes(local) + BufferBytes(occ_hh) +
         BufferBytes(occ_pp);
}

Comm232Tables BuildComm232Tables(const op::PairThirdSpace& pair_third,
                                 const basis::ModelSpace& ms3) {
  const op::PairThirdView pt = pair_third.host_view();
  const basis::TwoBodyChannels& two = ms3.two_body();
  const std::span<const double> occ = ms3.reference().occupations();
  const int norb = ms3.num_orbits();
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
  std::vector<std::int32_t> count(kComm232Classes * nblocks + 1, 0);
  std::vector<std::int32_t> ch_begin(at(nch) + 1, 0);
  std::vector<std::int32_t> orig(ns), third(ns), key(ns), local(ns);
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
      third[t] = pt.st_r[f];
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
      ++count[kComm232Classes * block_of(ch, pt.st_r[f], two_jpq / 2) +
              at(class_of(f))];
    }
    next += S;
  }
  ch_begin[at(nch)] = next;
  // blk[k b + c] = first sorted index of class c in block b (exclusive scan;
  // blocks, and classes within a block, follow the sorted order).
  std::vector<std::int32_t> blk(count.size(), 0);
  std::exclusive_scan(count.begin(), count.end(), blk.begin(), 0);
  // Active states (classes 0..2) per channel.
  std::vector<std::int32_t> ch_active(at(nch), 0);
  const std::size_t blocks_per_channel = block_of(1, 0, 0);
  for (int ch = 0; ch < nch; ++ch) {
    for (std::size_t b = at(ch) * blocks_per_channel;
         b < at(ch + 1) * blocks_per_channel; ++b) {
      ch_active[at(ch)] +=
          blk[kComm232Classes * b + 3] - blk[kComm232Classes * b];
    }
  }
  Comm232Tables t;
  t.blk = ToDevice(blk, "commutator::comm232_blk");
  t.ch_begin = ToDevice(ch_begin, "commutator::comm232_ch_begin");
  t.orig = ToDevice(orig, "commutator::comm232_orig");
  t.key = ToDevice(key, "commutator::comm232_key");
  t.local = ToDevice(local, "commutator::comm232_local");
  t.occ_hh = ToDevice(occ_hh, "commutator::comm232_occ_hh");
  t.occ_pp = ToDevice(occ_pp, "commutator::comm232_occ_pp");
  t.ch_begin_host = std::move(ch_begin);
  t.ch_active_host = std::move(ch_active);
  t.third_host = std::move(third);
  t.norb = norb;
  t.nj = nj;
  return t;
}

void Comm232ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm232ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm232ss", Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
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

void Comm232Batch(const N7BatchInputs& in, const TwoBodyOutView& z2) {
  const bool xy = in.v.x2.data != nullptr && in.wy != nullptr;
  const bool yx = in.v.y2.data != nullptr && in.wx != nullptr;
  if (!xy && !yx) return;
  IMSRG_CHECK(in.ctx != nullptr && in.plan != nullptr,
              "Comm232Batch: the batch inputs lack the context or the plan");
  const WorkList w = BuildWorkList(*in.ctx, *in.plan);
  if (w.key_item.back() == 0) return;
  const WorkBuffers b = Upload(w);
  RunPasses(in, z2, ViewOf(b, w));
}

void Comm232Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  N7Views v = PrepareN7("Comm232ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm232ss", Z, ctx);
  // [X2, Y3] needs X2 and Y3, [Y2, X3] needs Y2 and X3: a three-body part
  // without its two-body partner gets no intermediates.
  if (v.x2.data == nullptr) v.y3 = ThreeBodyPartView{};
  if (v.y2.data == nullptr) v.x3 = ThreeBodyPartView{};
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{false, true},
                 [&out](const N7BatchInputs& in) { Comm232Batch(in, out); });
}

void Comm232ssPairThird(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx,
                        std::int64_t max_block_bytes) {
  Comm232Rows(X, Y, Z, ctx,
              basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
              max_block_bytes);
}

}  // namespace detail

namespace terms {

void Comm232ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm232ssPairThird(X, Y, Z, ctx, detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
