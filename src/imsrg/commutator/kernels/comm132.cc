// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm132ss, [X1, Y3] + [X3, Y1] -> Z2 (imsrg3.h). Kernel TU.
//
// J-scheme form (bra |(i j) J>, ket |(k l) J>, a and b in the same one-body
// channel, 2J3 = |2J - 2 j_b| .. 2J + 2 j_b):
//   Z^J_ijkl (unnormalized) = sum_{ab} (n_a - n_b) sum_{J3} (2 J3 + 1)/(2 J +
//   1)
//       [ X_ab Y3^{J3}((ij)J b; (kl)J a) - Y_ab X3^{J3}((ij)J b; (kl)J a) ],
// added to the stored normalized element divided by sqrt((1 + d_ij)(1 +
// d_kl)). One parallel task per stored element of Z's two-body part (both
// triangles of every block), so there are no atomics.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator {

namespace {

/// Fields per key of a 132 work list: the ket pair (k, l) with k <= l, the
/// pair J, the Z2 channel of (k, l) at J and the local index of (k, l) in it.
constexpr int kKeyFields = 5;
/// Fields per entry: the three-body channel and the third orbit a of the W
/// column state.
constexpr int kEntryFields = 2;

/// The work list of one batch (host): the W columns of the batch (its
/// tiles' pair-third states ((k l) J, a)) grouped by key (k, l, J), keys
/// ascending, the entries of a key in plan order (tiles, then states
/// ascending). Key k owns the entries [key_entry[k], key_entry[k + 1]) and
/// the work items [key_item[k], key_item[k + 1]), one per local bra ket of
/// its Z2 channel.
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

/// Copies a host vector into a new device buffer.
template <class T>
exec::DeviceBuffer<T> ToDevice(const std::vector<T>& h, const char* label) {
  exec::DeviceBuffer<T> d(h.size(), label);
  if (!h.empty()) d.CopyFromHost(std::span<const T>(h));
  return d;
}

/// One W column state of a batch (host): its key slot (k norb + l) nj + J,
/// channel, third orbit and W column offset.
struct ColumnState {
  std::int64_t slot = 0;
  std::int32_t channel = 0;
  std::int32_t third = 0;
  std::int64_t offset = 0;
};

/// The work list of the batch `plan` of the context `ctx`: its column states
/// sorted by (key slot, plan position), so the grouping is deterministic.
/// Column states whose pair has no two-body ket at J (nothing to write) are
/// dropped. Host; O(m log m) for the m states of the batch.
WorkList BuildWorkList(const ThreeBodyContext& ctx,
                       const detail::LineBatch& plan) {
  const ThreeBodyContext::Impl& im = ctx.impl();
  const std::int64_t norb = im.comm232.norb;
  const std::int64_t nj = im.comm232.nj;
  const op::PairThirdView pt = im.pair_third.host_view();
  std::vector<ColumnState> cols;
  for (const detail::LineTile& tile : plan.tiles) {
    const std::int32_t ch = tile.channel;
    const std::int64_t S = pt.ch_num_states[ch];
    for (std::int32_t t = tile.state_begin; t < tile.state_end; ++t) {
      const std::int32_t f = pt.ch_state_offset[ch] + t;
      ColumnState c;
      c.slot = (pt.st_p[f] * norb + pt.st_q[f]) * nj + pt.st_two_jpq[f] / 2;
      c.channel = ch;
      c.third = pt.st_r[f];
      c.offset = tile.w_offset + (t - tile.state_begin) * S;
      cols.push_back(c);
    }
  }
  // Stable sort by slot keeps the plan order within a key.
  std::stable_sort(cols.begin(), cols.end(),
                   [](const ColumnState& a, const ColumnState& b) {
                     return a.slot < b.slot;
                   });
  const basis::TwoBodyChannels& two = ctx.model_space_ptr()->two_body();
  WorkList w;
  w.key_item.push_back(0);
  std::int64_t prev = -1;
  bool open = false;
  for (const ColumnState& c : cols) {
    if (c.slot != prev) {
      prev = c.slot;
      open = false;
      const auto J = static_cast<std::int32_t>(c.slot % nj);
      const auto p = static_cast<std::int32_t>(c.slot / nj / norb);
      const auto q = static_cast<std::int32_t>(c.slot / nj % norb);
      const KetIndex ket = two.ket_index(OrbitIndex{p}, OrbitIndex{q});
      const std::optional<LocalKetIndex> loc =
          two.local_index(ket, TwoJ{2 * J});
      if (!loc.has_value()) continue;
      const ChannelIndex ch2 =
          two.channel_of(OrbitIndex{p}, OrbitIndex{q}, TwoJ{2 * J});
      w.key_entry.push_back(static_cast<std::int32_t>(w.column.size()));
      w.key.insert(w.key.end(), {p, q, J, ch2.value(), loc->value()});
      w.key_item.push_back(w.key_item.back() + two.channel(ch2).num_kets);
      open = true;
    } else if (!open) {
      continue;
    }
    w.entry.push_back(c.channel);
    w.entry.push_back(c.third);
    w.column.push_back(c.offset);
  }
  w.key_entry.push_back(static_cast<std::int32_t>(w.column.size()));
  return w;
}

/// Kernel of 132 over the work list of a batch (a functor rather than a
/// lambda: see the 232 pass functor). Item (key, o), o a local bra ket (i,
/// j) of the key's Z2 channel: element z(o, key) of block J gets
///   (1/(2J+1)) sum_{col (ch3, a)} (2 J3 + 1) sum_{b in ch(a)} (n_a - n_b)
///       [X_ab W_Y[((ij) J b), col] - Y_ab W_X[((ij) J b), col]]
/// with 2 J3 = the channel's, divided by sqrt 2 for i == j and for k == l.
/// A row ((ij) J b) missing from the channel reads 0. Every element is
/// written by one item, the sums run in a fixed order (entries in plan
/// order, b in channel order). Cost per item O(entries one-body channel
/// size log S).
struct Pass132 {
  detail::N7Views v;
  detail::Comm232TablesView tab;
  WorkView work;
  detail::TwoBodyOutView out;
  const double* wx = nullptr;  ///< W(X3) (pairs with Y1), or null.
  const double* wy = nullptr;  ///< W(Y3) (pairs with X1), or null.

  IMSRG_FUNCTION void operator()(const exec::Index idx) const {
    const int k = detail::FindBlock(work.key_item, work.num_keys, idx);
    const std::int32_t* key = work.key + kKeyFields * std::int64_t{k};
    const int J = key[2];
    const int ch2 = key[3];
    const basis::TwoBodyTableView& tb = v.x2.tb;
    const int o = static_cast<int>(idx - work.key_item[k]);
    const int ket = tb.channel_kets[tb.channel_ket_offset[ch2] + o];
    const int i = tb.ket_p[ket];
    const int j = tb.ket_q[ket];
    double z = 0.0;
    for (int e = work.key_entry[k]; e < work.key_entry[k + 1]; ++e) {
      const int ch3 = work.entry[kEntryFields * std::int64_t{e}];
      const int a = work.entry[kEntryFields * std::int64_t{e} + 1];
      const int two_j3 = v.sp.ch_two_j[ch3];
      const std::int64_t col = work.column[e];
      const double na = v.ob.occupation[a];
      double acc = 0.0;
      for (int m = v.ob_ch.offsets[a]; m < v.ob_ch.offsets[a + 1]; ++m) {
        const int b = v.ob_ch.orbits[m];
        const double dn = na - v.ob.occupation[b];
        if (dn == 0.0) continue;
        const double xab = detail::OneBodyElement(v.x1, v.norb, a, b);
        const double yab = detail::OneBodyElement(v.y1, v.norb, a, b);
        if (xab == 0.0 && yab == 0.0) continue;
        const int s =
            detail::FindSortedRow(tab, v.ob.occupation, ch3, i, j, J, b);
        if (s < 0) continue;
        double t = 0.0;
        if (wy != nullptr && xab != 0.0) t += xab * wy[col + s];
        if (wx != nullptr && yab != 0.0) t -= yab * wx[col + s];
        acc += dn * t;
      }
      z += (two_j3 + 1) * acc;
    }
    z /= (2 * J + 1);
    if (i == j) z /= constants::kSqrt2;
    if (key[0] == key[1]) z /= constants::kSqrt2;
    const std::int64_t n = tb.channel_num_kets[ch2];
    out.data[out.block_offset[ch2] + o + key[4] * n] += z;
  }
};
static_assert(std::is_trivially_copyable_v<Pass132>);

}  // namespace

namespace detail {

void Comm132ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm132ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm132ss", Z, ctx);
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  const std::int64_t total = Z.two_body().matrix().total_size();
  const basis::TwoBodyTableView tb = v.x2.tb;
  exec::ParallelFor(
      "commutator::comm132ss_reference", static_cast<exec::Index>(total),
      IMSRG_LAMBDA(const exec::Index e) {
        const int ch = FindBlock(out.block_offset, tb.num_channels, e);
        const std::int64_t n = tb.channel_num_kets[ch];
        const std::int64_t idx = e - out.block_offset[ch];
        const int ib = static_cast<int>(idx % n);
        const int ik = static_cast<int>(idx / n);
        const int two_J = tb.channel_two_j[ch];
        const int J = two_J / 2;
        const int koff = tb.channel_ket_offset[ch];
        const int bra = tb.channel_kets[koff + ib];
        const int ket = tb.channel_kets[koff + ik];
        const int i = tb.ket_p[bra];
        const int j = tb.ket_q[bra];
        const int k = tb.ket_p[ket];
        const int l = tb.ket_q[ket];
        double z = 0.0;
        for (int a = 0; a < v.norb; ++a) {
          const double na = v.ob.occupation[a];
          for (int m = v.ob_ch.offsets[a]; m < v.ob_ch.offsets[a + 1]; ++m) {
            const int b = v.ob_ch.orbits[m];
            const double dn = na - v.ob.occupation[b];
            if (dn == 0.0) continue;
            const double xab = OneBodyElement(v.x1, v.norb, a, b);
            const double yab = OneBodyElement(v.y1, v.norb, a, b);
            if (xab == 0.0 && yab == 0.0) continue;
            const int j2b = v.ob.j2[b];
            const int lo = two_J > j2b ? two_J - j2b : j2b - two_J;
            double acc = 0.0;
            for (int two_j3 = lo; two_j3 <= two_J + j2b; two_j3 += 2) {
              double t = 0.0;
              if (xab != 0.0) {
                t += xab * ThreeBodyElement(v, v.y3, two_j3, i, j, b, J,
                                                    k, l, a, J);
              }
              if (yab != 0.0) {
                t -= yab * ThreeBodyElement(v, v.x3, two_j3, i, j, b, J,
                                                    k, l, a, J);
              }
              acc += (two_j3 + 1) * t;
            }
            z += dn * acc;
          }
        }
        z /= (two_J + 1);
        if (i == j) z /= constants::kSqrt2;
        if (k == l) z /= constants::kSqrt2;
        out.data[e] += z;
      });
  exec::Fence("commutator::comm132ss_reference");
}

void Comm132Batch(const N7BatchInputs& in, const TwoBodyOutView& z2) {
  // [X1, Y3] needs X1 and W(Y3), [X3, Y1] needs Y1 and W(X3).
  Pass132 f;
  f.wy = in.v.x1 != nullptr ? in.wy : nullptr;
  f.wx = in.v.y1 != nullptr ? in.wx : nullptr;
  if (f.wx == nullptr && f.wy == nullptr) return;
  IMSRG_CHECK(in.ctx != nullptr && in.plan != nullptr,
              "Comm132Batch: the batch inputs lack the context or the plan");
  const WorkList w = BuildWorkList(*in.ctx, *in.plan);
  if (w.key_item.back() == 0) return;
  const exec::DeviceBuffer<std::int32_t> key =
      ToDevice(w.key, "commutator::comm132_work_key");
  const exec::DeviceBuffer<std::int32_t> key_entry =
      ToDevice(w.key_entry, "commutator::comm132_work_key_entry");
  const exec::DeviceBuffer<std::int64_t> key_item =
      ToDevice(w.key_item, "commutator::comm132_work_key_item");
  const exec::DeviceBuffer<std::int32_t> entry =
      ToDevice(w.entry, "commutator::comm132_work_entry");
  const exec::DeviceBuffer<std::int64_t> column =
      ToDevice(w.column, "commutator::comm132_work_column");
  f.v = in.v;
  f.tab = in.tab;
  f.work.key = key.device_data();
  f.work.key_entry = key_entry.device_data();
  f.work.key_item = key_item.device_data();
  f.work.entry = entry.device_data();
  f.work.column = column.device_data();
  f.work.num_keys = static_cast<std::int32_t>(w.key_item.size()) - 1;
  f.work.num_items = w.key_item.back();
  f.out = z2;
  exec::ParallelForDynamic("commutator::comm132ss_batch", f.work.num_items, f);
  exec::Fence("commutator::comm132ss_batch");
}

void Comm132Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  N7Views v = PrepareN7("Comm132ss", X, Y, Z, ctx);
  const TwoBodyOutView out = TwoBodyOut("Comm132ss", Z, ctx);
  // [X1, Y3] needs X1 and Y3, [X3, Y1] needs Y1 and X3: a three-body part
  // without its one-body partner gets no intermediates.
  if (v.x1 == nullptr) v.y3 = ThreeBodyPartView{};
  if (v.y1 == nullptr) v.x3 = ThreeBodyPartView{};
  if (v.x3.local == nullptr && v.y3.local == nullptr) return;
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{false, true},
                 [&out](const N7BatchInputs& in) { Comm132Batch(in, out); });
}

}  // namespace detail

namespace terms {

void Comm132ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm132Rows(
      X, Y, Z, ctx,
      basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
      detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
