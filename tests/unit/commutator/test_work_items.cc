// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Work items and the per-part work assignment of the expensive IMSRG(2) terms
// (M12b phase 1): sizes against the Workspace's own data, ladder row-block
// items, disjoint and complete assignments, and the Workspace hook-up.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/workspace.h"
#include "tests/unit/commutator/commutator_test_cases.h"

namespace imsrg::commutator::detail {
namespace {

using commutator_test::MakeSpace;
using commutator_test::SpaceCase;
using commutator_test::SpaceCases;

/// The model spaces of the tests: the shared cases (incl. a fractional one)
/// plus He4 and O16 at emax 4.
std::vector<SpaceCase> Cases() {
  std::vector<SpaceCase> cases = SpaceCases();
  cases.push_back({"He4_emax4", 4, "He4", false});
  cases.push_back({"O16_emax4", 4, "O16", false});
  return cases;
}

/// Lifted entry counts per channel from the Workspace's LiftedData.
std::vector<std::int32_t> LiftedCounts(const Workspace::Impl& impl,
                                       std::size_t nch) {
  const auto row_base = impl.lifted.row_base.ToVector();
  const auto entry_offset = impl.lifted.entry_offset.ToVector();
  std::vector<std::int32_t> out(nch, 0);
  for (std::size_t c = 0; c < nch; ++c)
    out[c] = entry_offset[static_cast<std::size_t>(row_base[c + 1])] -
             entry_offset[static_cast<std::size_t>(row_base[c])];
  return out;
}

TEST(commutator_work_items, sizes_equal_workspace_data) {
  for (const SpaceCase& c : Cases()) {
    SCOPED_TRACE(c.label);
    const auto ms = MakeSpace(c);
    const Workspace ws(ms);
    const auto& impl = ws.impl();
    const WorkSizes s = ComputeWorkSizes(*ms);
    const std::size_t nch = s.num_kets.size();
    ASSERT_EQ(nch, impl.ladder.pp.count_host.size());
    EXPECT_EQ(s.k_pp, impl.ladder.pp.count_host);
    EXPECT_EQ(s.k_hh, impl.ladder.hh.count_host);
    EXPECT_EQ(s.ncc, impl.pandya.ncc_host);
    EXPECT_EQ(s.nk, impl.pandya.nk_host);
    EXPECT_EQ(s.lifted, LiftedCounts(impl, nch));
    for (std::size_t ch = 0; ch < nch; ++ch) {
      const auto n = static_cast<std::int64_t>(s.num_kets[ch]);
      EXPECT_EQ(n * n,
                impl.block_offsets_host[ch + 1] - impl.block_offsets_host[ch]);
    }
  }
}

TEST(commutator_work_items, ladder_items_single_part_are_whole_channels) {
  const auto ms = MakeSpace({"O16_emax4", 4, "O16", false});
  const WorkSizes s = ComputeWorkSizes(*ms);
  const LadderItems items = LadderWorkItems(s, 1);
  std::size_t item = 0;
  for (std::size_t c = 0; c < s.num_kets.size(); ++c) {
    if (s.num_kets[c] == 0 || s.k_pp[c] + s.k_hh[c] == 0) continue;
    ASSERT_LT(item, items.blocks.size());
    const RowBlock expected{static_cast<std::int32_t>(c), 0, s.num_kets[c]};
    EXPECT_TRUE(items.blocks[item] == expected) << "channel " << c;
    const double n = s.num_kets[c];
    EXPECT_EQ(items.cost[item], 2.0 * n * n * (s.k_pp[c] + s.k_hh[c]));
    ++item;
  }
  EXPECT_EQ(item, items.blocks.size());
}

TEST(commutator_work_items, ladder_items_split_covers_rows_once) {
  const auto ms = MakeSpace({"O16_emax4", 4, "O16", false});
  const WorkSizes s = ComputeWorkSizes(*ms);
  const LadderItems whole = LadderWorkItems(s, 1);
  double total = 0.0;
  for (const double c : whole.cost) total += c;
  for (const std::int32_t parts : {2, 4, 22}) {
    SCOPED_TRACE(parts);
    const LadderItems items = LadderWorkItems(s, parts);
    const double threshold = total / (4.0 * parts);
    std::size_t item = 0;
    int num_split = 0;
    for (const RowBlock& w : whole.blocks) {
      std::int32_t next = 0, min_rows = w.row_end, max_rows = 0;
      double cost_sum = 0.0;
      std::size_t first = item;
      while (item < items.blocks.size() &&
             items.blocks[item].channel == w.channel) {
        const RowBlock& b = items.blocks[item];
        EXPECT_EQ(b.row_begin, next);  // contiguous, no overlap, no gap
        EXPECT_GT(b.row_end, b.row_begin);
        const std::int32_t rows = b.row_end - b.row_begin;
        min_rows = std::min(min_rows, rows);
        max_rows = std::max(max_rows, rows);
        next = b.row_end;
        cost_sum += items.cost[item];
        if (rows > 1) EXPECT_LE(items.cost[item], threshold);
        ++item;
      }
      EXPECT_EQ(next, w.row_end) << "channel " << w.channel;
      EXPECT_LE(max_rows - min_rows, 1) << "channel " << w.channel;
      EXPECT_NEAR(cost_sum,
                  whole.cost[static_cast<std::size_t>(&w - &whole.blocks[0])],
                  1.0e-9 * cost_sum);
      if (item - first > 1) ++num_split;
    }
    EXPECT_EQ(item, items.blocks.size());
    std::printf("[          ] P = %d: %zu items, %d channels split\n", parts,
                items.blocks.size(), num_split);
  }
}

/// All three terms' item lists of the parts must be disjoint and complete.
void ExpectPartition(const WorkSizes& s, std::int32_t num_parts) {
  SCOPED_TRACE(num_parts);
  const LadderItems lad = LadderWorkItems(s, num_parts);
  const ChannelItems c122 = Comm122WorkItems(s);
  const ChannelItems ph = PhWorkItems(s);
  std::vector<int> lad_count(lad.blocks.size(), 0);
  std::size_t n122 = 0, nph = 0;
  std::vector<int> seen122(s.num_kets.size(), 0), seen_ph(s.ncc.size(), 0);
  for (std::int32_t p = 0; p < num_parts; ++p) {
    const WorkAssignment a = WorkAssignment::Build(s, p, num_parts);
    EXPECT_EQ(a.part(), p);
    EXPECT_EQ(a.num_parts(), num_parts);
    EXPECT_EQ(a.distributed(), num_parts > 1);
    for (const RowBlock& b : a.ladder_blocks()) {
      const auto it = std::find(lad.blocks.begin(), lad.blocks.end(), b);
      ASSERT_NE(it, lad.blocks.end());
      ++lad_count[static_cast<std::size_t>(it - lad.blocks.begin())];
    }
    std::vector<std::int32_t> chans;
    for (const RowBlock& b : a.ladder_blocks())
      if (chans.empty() || chans.back() != b.channel)
        chans.push_back(b.channel);
    EXPECT_EQ(chans, a.ladder_channels());
    EXPECT_TRUE(std::is_sorted(
        a.ladder_blocks().begin(), a.ladder_blocks().end(),
        [](const RowBlock& x, const RowBlock& y) {
          return x.channel != y.channel ? x.channel < y.channel
                                        : x.row_begin < y.row_begin;
        }));
    for (const std::int32_t c : a.comm122_channels())
      ++seen122[static_cast<std::size_t>(c)];
    for (const std::int32_t c : a.ph_channels())
      ++seen_ph[static_cast<std::size_t>(c)];
    n122 += a.comm122_channels().size();
    nph += a.ph_channels().size();
    EXPECT_TRUE(std::is_sorted(a.comm122_channels().begin(),
                               a.comm122_channels().end()));
    EXPECT_TRUE(std::is_sorted(a.ph_channels().begin(), a.ph_channels().end()));
    const WorkAssignment again = WorkAssignment::Build(s, p, num_parts);
    EXPECT_TRUE(again.ladder_blocks() == a.ladder_blocks());
    EXPECT_EQ(again.comm122_channels(), a.comm122_channels());
    EXPECT_EQ(again.ph_channels(), a.ph_channels());
    if (num_parts == 1) {
      EXPECT_EQ(a.ladder_blocks().size(), lad.blocks.size());
      EXPECT_EQ(a.comm122_channels(), c122.channels);
      EXPECT_EQ(a.ph_channels(), ph.channels);
    }
  }
  for (const int n : lad_count) EXPECT_EQ(n, 1);
  EXPECT_EQ(n122, c122.channels.size());
  EXPECT_EQ(nph, ph.channels.size());
  for (const std::int32_t c : c122.channels)
    EXPECT_EQ(seen122[static_cast<std::size_t>(c)], 1);
  for (const std::int32_t c : ph.channels)
    EXPECT_EQ(seen_ph[static_cast<std::size_t>(c)], 1);
}

TEST(commutator_work_items, assignments_partition_all_items) {
  for (const SpaceCase& c : Cases()) {
    SCOPED_TRACE(c.label);
    const WorkSizes s = ComputeWorkSizes(*MakeSpace(c));
    for (const std::int32_t parts : {1, 2, 3, 4}) ExpectPartition(s, parts);
  }
}

TEST(commutator_work_items, invalid_part_throws) {
  const WorkSizes s = ComputeWorkSizes(*MakeSpace({"He4", 1, "He4", false}));
  EXPECT_THROW(WorkAssignment::Build(s, 2, 2), InvariantError);
  EXPECT_THROW(WorkAssignment::Build(s, -1, 2), InvariantError);
  EXPECT_THROW(WorkAssignment::Build(s, 0, 0), InvariantError);
}

/// Device buffer contents as a host vector (empty buffer: empty vector).
std::vector<std::int32_t> Host(const exec::DeviceBuffer<std::int32_t>& b) {
  return b.ToVector();
}

/// The columns (entry indices) of `channels` for the channel sizes `n`.
std::vector<std::int32_t> ExpectedColumns(
    const std::vector<std::int32_t>& channels,
    const std::vector<std::int32_t>& n) {
  std::vector<std::int32_t> base(n.size() + 1, 0);
  for (std::size_t c = 0; c < n.size(); ++c) base[c + 1] = base[c] + n[c];
  std::vector<std::int32_t> cols;
  for (const std::int32_t c : channels)
    for (std::int32_t e = base[static_cast<std::size_t>(c)];
         e < base[static_cast<std::size_t>(c) + 1]; ++e)
      cols.push_back(e);
  return cols;
}

TEST(commutator_work_items, workspace_hookup) {
  const auto ms = MakeSpace({"O16_emax4", 4, "O16", false});
  Workspace ws(ms);
  EXPECT_EQ(ws.impl().work.part(), 0);
  EXPECT_EQ(ws.impl().work.num_parts(), 1);
  EXPECT_FALSE(ws.impl().work.distributed());
  const WorkSizes s = ComputeWorkSizes(*ms);
  for (const std::int32_t parts : {1, 2, 3}) {
    for (std::int32_t part = 0; part < parts; ++part) {
      SCOPED_TRACE(parts * 10 + part);
      AssignWork(ws, part, parts);
      const auto& w = ws.impl().work;
      const auto& d = ws.impl().work_device;
      EXPECT_EQ(w.part(), part);
      EXPECT_EQ(w.num_parts(), parts);
      EXPECT_EQ(w.distributed(), parts > 1);
      const WorkAssignment ref = WorkAssignment::Build(s, part, parts);
      EXPECT_TRUE(w.ladder_blocks() == ref.ladder_blocks());
      std::vector<std::int32_t> active(s.num_kets.size(), 0);
      for (const std::int32_t c : w.ladder_channels())
        active[static_cast<std::size_t>(c)] = 1;
      EXPECT_EQ(Host(d.ladder_active), active);
      EXPECT_EQ(Host(d.ladder_columns),
                ExpectedColumns(w.ladder_channels(), s.num_kets));
      EXPECT_EQ(Host(d.comm122_columns),
                ExpectedColumns(w.comm122_channels(), s.num_kets));
      std::vector<std::int32_t> owned(s.ncc.size(), 0);
      for (const std::int32_t c : w.ph_channels())
        owned[static_cast<std::size_t>(c)] = 1;
      EXPECT_EQ(Host(d.ph_owned), owned);
    }
  }
  AssignWork(ws, 0, 1);
  EXPECT_FALSE(ws.impl().work.distributed());
}

}  // namespace
}  // namespace imsrg::commutator::detail
