// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Work items and the per-part work assignment of the expensive IMSRG(2) terms
// (M12b phase 1): sizes against the Workspace's own data, ladder row-block
// items, disjoint and complete assignments, and the Workspace hook-up.
#include <algorithm>
#include <cmath>
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

/// Row cost of channel c in `s`.
double RowCost(const WorkSizes& s, std::size_t c) {
  return 2.0 * s.num_kets[c] * (s.k_pp[c] + s.k_hh[c]);
}

/// The LPT map of the comm122 items on `parts` ranks.
comm::OwnershipMap Comm122Map(const WorkSizes& s, std::int32_t parts) {
  return comm::OwnershipMap::Greedy(Comm122WorkItems(s).cost, parts);
}

/// The LPT map of the particle-hole items on `parts` ranks.
comm::OwnershipMap PhMap(const WorkSizes& s, std::int32_t parts) {
  return comm::OwnershipMap::Greedy(PhWorkItems(s).cost, parts);
}

/// Checks the contiguous cost-balanced cut of `LadderWorkItems(s, parts)`.
void ExpectContiguousCut(const WorkSizes& s, std::int32_t parts) {
  SCOPED_TRACE(parts);
  const LadderItems items = LadderWorkItems(s, parts);
  const std::size_t nb = items.blocks.size();
  ASSERT_EQ(items.cost.size(), nb);
  ASSERT_EQ(items.owner.size(), nb);
  // Global row sequence: eligible channels ascending.
  std::vector<std::int32_t> row_channel;
  std::vector<std::int32_t> row_local;
  double total = 0.0, max_row = 0.0;
  for (std::size_t c = 0; c < s.num_kets.size(); ++c) {
    if (s.num_kets[c] == 0 || s.k_pp[c] + s.k_hh[c] == 0) continue;
    for (std::int32_t j = 0; j < s.num_kets[c]; ++j) {
      row_channel.push_back(static_cast<std::int32_t>(c));
      row_local.push_back(j);
      total += RowCost(s, c);
    }
    max_row = std::max(max_row, RowCost(s, c));
  }
  // The blocks tile the sequence in order (no gap, no overlap) and the owners
  // ascend, so each rank's blocks are one contiguous run.
  std::size_t pos = 0;
  std::vector<double> load(static_cast<std::size_t>(parts), 0.0);
  for (std::size_t i = 0; i < nb; ++i) {
    const RowBlock& b = items.blocks[i];
    ASSERT_GT(b.row_end, b.row_begin);
    ASSERT_LT(pos, row_channel.size());
    EXPECT_EQ(row_channel[pos], b.channel);
    EXPECT_EQ(row_local[pos], b.row_begin);
    const auto rows = static_cast<std::size_t>(b.row_end - b.row_begin);
    pos += rows;
    ASSERT_LE(pos, row_channel.size());
    EXPECT_EQ(row_channel[pos - 1], b.channel);
    EXPECT_EQ(row_local[pos - 1], b.row_end - 1);
    const auto r = static_cast<std::size_t>(items.owner[i]);
    ASSERT_LT(r, load.size());
    if (i > 0) {
      EXPECT_LE(items.owner[i - 1], items.owner[i]);
    }
    EXPECT_EQ(items.cost[i], RowCost(s, static_cast<std::size_t>(b.channel)) *
                                 static_cast<double>(rows));
    load[r] += items.cost[i];
  }
  EXPECT_EQ(pos, row_channel.size());
  // Loads within one (largest) row of T / P.
  for (std::int32_t r = 0; r < parts; ++r)
    EXPECT_LE(std::abs(load[static_cast<std::size_t>(r)] - total / parts),
              max_row)
        << "rank " << r;
  // Deterministic.
  const LadderItems again = LadderWorkItems(s, parts);
  EXPECT_TRUE(again.blocks == items.blocks);
  EXPECT_EQ(again.owner, items.owner);
  EXPECT_EQ(again.cost, items.cost);
  if (parts == 1) {  // the old items: one whole block per eligible channel
    for (std::size_t i = 0; i < nb; ++i) {
      EXPECT_EQ(items.owner[i], 0);
      EXPECT_EQ(items.blocks[i].row_begin, 0);
      EXPECT_EQ(items.blocks[i].row_end,
                s.num_kets[static_cast<std::size_t>(items.blocks[i].channel)]);
    }
  }
  // Equal capacities (any common scale that is a power of two) are the equal
  // cut, bitwise.
  for (const double cap : {1.0, 8.0}) {
    const LadderItems eq = LadderWorkItems(
        s, std::vector<double>(static_cast<std::size_t>(parts), cap));
    EXPECT_TRUE(eq.blocks == items.blocks);
    EXPECT_EQ(eq.owner, items.owner);
    EXPECT_EQ(eq.cost, items.cost);
  }
  // The assignment's map agrees with the joint cut's owners (which are the
  // equal cut's owners for one part).
  const WorkAssignment a = WorkAssignment::Build(s, 0, parts);
  const LadderItems joint =
      JointLadderItems(s, Comm122Map(s, parts), PhMap(s, parts));
  ASSERT_EQ(a.ladder_map().num_items(),
            static_cast<std::int32_t>(joint.cost.size()));
  for (std::size_t i = 0; i < joint.owner.size(); ++i)
    EXPECT_EQ(a.ladder_map().owner(static_cast<std::int32_t>(i)),
              joint.owner[i]);
  if (parts == 1) {
    EXPECT_TRUE(joint.blocks == items.blocks);
    EXPECT_EQ(joint.owner, items.owner);
  }
}

TEST(commutator_work_items, ladder_cut_synthetic) {
  WorkSizes s;
  s.num_kets = {5, 0, 3, 7, 2, 1};
  s.k_pp = {4, 9, 0, 1, 0, 6};
  s.k_hh = {1, 9, 0, 2, 0, 0};  // channels 1, 2 and 4 are ineligible
  s.lifted.assign(6, 0);
  for (const std::int32_t parts : {1, 2, 3, 4, 7, 22, 64})
    ExpectContiguousCut(s, parts);
  // Hand check, P = 2: row costs 50 (x5), 42 (x7), 12 (x1); T = 556, target
  // 278: S_5 = 250, S_6 = 292, so the nearest boundary is 6 (one row of the
  // second channel goes to rank 0).
  const LadderItems two = LadderWorkItems(s, 2);
  ASSERT_EQ(two.blocks.size(), 4U);
  EXPECT_EQ(two.owner, (std::vector<std::int32_t>{0, 0, 1, 1}));
  EXPECT_TRUE((two.blocks[1] == RowBlock{3, 0, 1}));
  EXPECT_TRUE((two.blocks[2] == RowBlock{3, 1, 7}));
  EXPECT_EQ(two.cost[1], 42.0);
  // No eligible channel: no items for any P.
  WorkSizes none;
  none.num_kets = {0, 3};
  none.k_pp = {0, 0};
  none.k_hh = {0, 0};
  EXPECT_TRUE(LadderWorkItems(none, 4).blocks.empty());
}

TEST(commutator_work_items, ladder_cut_model_spaces) {
  const SpaceCase cases[] = {{"He4_emax2", 2, "He4", false},
                             {"O16_emax4", 4, "O16", false},
                             {"O16_emax6", 6, "O16", false}};
  for (const SpaceCase& c : cases) {
    SCOPED_TRACE(c.label);
    const WorkSizes s = ComputeWorkSizes(*MakeSpace(c));
    for (const std::int32_t parts : {1, 2, 3, 4, 7, 22, 64, 256})
      ExpectContiguousCut(s, parts);
  }
}

/// All three terms' item lists of the parts must be disjoint and complete.
void ExpectPartition(const WorkSizes& s, std::int32_t num_parts) {
  SCOPED_TRACE(num_parts);
  const LadderItems lad =
      JointLadderItems(s, Comm122Map(s, num_parts), PhMap(s, num_parts));
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

/// Checks the joint cut for `parts` ranks: the blocks tile every eligible
/// channel's rows exactly in ascending order, owners ascend (one contiguous
/// segment per rank), and the ladder load of each rank is within one largest
/// row cost of its capacity; where no rank is above the target (no clamping)
/// the calibrated total of every rank is within that of the target.
void ExpectJointCut(const WorkSizes& s, std::int32_t parts) {
  SCOPED_TRACE(parts);
  const comm::OwnershipMap m122 = Comm122Map(s, parts);
  const comm::OwnershipMap mph = PhMap(s, parts);
  const LadderItems items = JointLadderItems(s, m122, mph);
  std::vector<std::int32_t> next_row(s.num_kets.size(), 0);
  std::int32_t last_channel = -1;
  double total = 0.0, max_row = 0.0;
  std::vector<double> load(static_cast<std::size_t>(parts), 0.0);
  for (std::size_t i = 0; i < items.blocks.size(); ++i) {
    const RowBlock& b = items.blocks[i];
    const auto c = static_cast<std::size_t>(b.channel);
    ASSERT_GE(b.channel, last_channel);
    last_channel = b.channel;
    EXPECT_EQ(b.row_begin, next_row[c]);
    ASSERT_GT(b.row_end, b.row_begin);
    next_row[c] = b.row_end;
    if (i > 0) {
      EXPECT_LE(items.owner[i - 1], items.owner[i]);
    }
    load[static_cast<std::size_t>(items.owner[i])] += items.cost[i];
    total += items.cost[i];
  }
  for (std::size_t c = 0; c < s.num_kets.size(); ++c) {
    const bool eligible = s.num_kets[c] > 0 && s.k_pp[c] + s.k_hh[c] > 0;
    EXPECT_EQ(next_row[c], eligible ? s.num_kets[c] : 0);
    if (eligible) max_row = std::max(max_row, RowCost(s, c));
  }
  if (parts == 1) return;
  std::vector<double> other(static_cast<std::size_t>(parts), 0.0);
  double other_sum = 0.0;
  for (std::int32_t r = 0; r < parts; ++r) {
    const auto ri = static_cast<std::size_t>(r);
    other[ri] = kComm122WeightInLadderUnits * m122.load(r) +
                kPhWeightInLadderUnits * mph.load(r);
    other_sum += other[ri];
  }
  const double target = (total + other_sum) / parts;
  double cap_sum = 0.0;
  bool clamped = false;
  for (const double o : other) {
    cap_sum += std::max(0.0, target - o);
    clamped = clamped || o > target;
  }
  for (std::int32_t r = 0; r < parts; ++r) {
    const auto ri = static_cast<std::size_t>(r);
    const double cap = std::max(0.0, target - other[ri]) * total / cap_sum;
    EXPECT_LE(std::abs(load[ri] - cap), max_row) << "rank " << r;
    if (!clamped) {
      EXPECT_LE(std::abs(other[ri] + load[ri] - target), max_row)
          << "rank " << r;
    }
  }
}

TEST(commutator_work_items, joint_cut_synthetic_and_model_spaces) {
  WorkSizes syn;
  syn.num_kets = {5, 0, 3, 7, 2, 1};
  syn.k_pp = {4, 9, 0, 1, 0, 6};
  syn.k_hh = {1, 9, 0, 2, 0, 0};
  syn.lifted = {3, 0, 2, 4, 1, 1};
  syn.ncc = {2, 3, 0, 1};
  syn.nk = {4, 2, 5, 0};
  for (const std::int32_t parts : {1, 2, 3, 4, 7, 22, 64}) {
    ExpectJointCut(syn, parts);
    ExpectPartition(syn, parts);
  }
  const SpaceCase cases[] = {{"He4_emax2", 2, "He4", false},
                             {"O16_emax4", 4, "O16", false},
                             {"O16_emax6", 6, "O16", false}};
  for (const SpaceCase& c : cases) {
    SCOPED_TRACE(c.label);
    const WorkSizes s = ComputeWorkSizes(*MakeSpace(c));
    for (const std::int32_t parts : {1, 2, 3, 4, 7, 22, 64}) {
      ExpectJointCut(s, parts);
      ExpectPartition(s, parts);
    }
  }
}

TEST(commutator_work_items, unequal_capacities_cut) {
  WorkSizes s;
  s.num_kets = {5, 3, 7};
  s.k_pp = {4, 1, 1};
  s.k_hh = {1, 0, 2};
  s.lifted.assign(3, 0);
  // Row costs 50 (x5), 6 (x3), 42 (x7); T = 562. Capacities 1 : 0 : 3 give
  // rank 0 about T / 4, rank 1 nothing, rank 2 the rest.
  const LadderItems li = LadderWorkItems(s, std::vector<double>{1.0, 0.0, 3.0});
  double load[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < li.cost.size(); ++i)
    load[li.owner[i]] += li.cost[i];
  EXPECT_EQ(load[1], 0.0);
  EXPECT_LE(std::abs(load[0] - 562.0 / 4.0), 50.0);
  EXPECT_EQ(load[0] + load[2], 562.0);
  EXPECT_THROW(LadderWorkItems(s, std::vector<double>{}), InvariantError);
  EXPECT_THROW(LadderWorkItems(s, std::vector<double>{1.0, -1.0}),
               InvariantError);
  // All-zero capacities fall back to the equal cut.
  const LadderItems zero = LadderWorkItems(s, std::vector<double>{0.0, 0.0});
  const LadderItems equal = LadderWorkItems(s, 2);
  EXPECT_EQ(zero.owner, equal.owner);
  EXPECT_TRUE(zero.blocks == equal.blocks);
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
