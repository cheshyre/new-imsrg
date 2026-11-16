// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Unit tests of RowRange / RowSet / RowPartition (row ranges of the kept
// three-body space).
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_rows.h"

namespace imsrg::basis {
namespace {

std::shared_ptr<const ModelSpace> O16(std::int32_t emax, std::int32_t e3max,
                                      double de3max = 1.0e30) {
  ThreeBodyCuts c;
  c.e3max = e3max;
  c.de3max = de3max;
  return ModelSpace::FromNames(BasisTruncation{emax}, 16.0, "O16", "O16")
      ->WithThreeBody(c);
}

struct Case {
  std::int32_t emax;
  std::int32_t e3max;
  double de3max;
};

constexpr Case kCases[] = {{2, 6, 1.0e30}, {3, 9, 1.0e30}, {3, 9, 3.0}};

std::int64_t Packed(std::int64_t n, std::int64_t k) {
  return k + k * (2 * n - k - 1) / 2;  // PackedIndex(n, k, k)
}

/// The synthetic weights of the brief: memory = sum of packed row lengths,
/// compute = rows * n.
LineWeights SyntheticWeights(const ThreeBodySpace& s) {
  LineWeights w;
  for (const OrbitTriple& t : s.triples()) {
    const std::int64_t n = s.dim(t.channel);
    double mem = 0.0;
    for (std::int32_t k = t.row_begin.value(); k < t.row_end.value(); ++k) {
      mem += static_cast<double>(n - k);
    }
    w.memory.push_back(mem);
    w.compute.push_back(
        static_cast<double>(t.row_end.value() - t.row_begin.value()) *
        static_cast<double>(n));
  }
  return w;
}

TEST(RowSetTest, WholeCoversEveryRowOnceWithTodaysOffsets) {
  for (const Case& c : kCases) {
    const auto ms = O16(c.emax, c.e3max, c.de3max);
    const ThreeBodySpace& s = *ms->three_body_space();
    const RowSet whole = RowSet::Whole(s);
    EXPECT_EQ(whole.NumRows(), s.NumRows());
    std::int64_t packed = 0;
    std::int64_t dense = 0;
    for (std::int32_t c3 = 0; c3 < s.num_channels(); ++c3) {
      const Channel3Index ch{c3};
      const std::int64_t n = s.dim(ch);
      for (std::int32_t k = 0; k < n; ++k) {
        const KeptRow3Index row{k};
        ASSERT_TRUE(whole.Contains(ch, row));
        ASSERT_EQ(whole.LocalRowOffset(ch, row, true), packed + Packed(n, k));
        ASSERT_EQ(whole.LocalRowOffset(ch, row, false), dense + k * n);
      }
      EXPECT_FALSE(whole.Contains(ch, KeptRow3Index{static_cast<int>(n)}));
      EXPECT_EQ(
          whole.LocalRowOffset(ch, KeptRow3Index{static_cast<int>(n)}, true),
          -1);
      packed += n * (n + 1) / 2;
      dense += n * n;
    }
    EXPECT_EQ(whole.StoredSize(true), packed);
    EXPECT_EQ(whole.StoredSize(false), dense);
  }
}

TEST(RowSetTest, FromRangesSortsAndValidates) {
  const auto ms = O16(3, 9);
  const ThreeBodySpace& s = *ms->three_body_space();
  // The two biggest channels, each split into three pieces, given unsorted.
  std::vector<RowRange> in;
  std::int32_t big = 0;
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    if (s.dim(Channel3Index{c}) > s.dim(Channel3Index{big})) big = c;
  }
  const KeptChannel3& kc = s.channel(Channel3Index{big});
  const std::int32_t t0 = kc.triple_begin.value();
  const std::int32_t t3 = kc.triple_end.value();
  ASSERT_GE(t3 - t0, 3);
  auto range = [&](std::int32_t a, std::int32_t b) {
    return RowRange{Channel3Index{big}, Triple3Index{a}, Triple3Index{b},
                    s.triple(Triple3Index{a}).row_begin,
                    s.triple(Triple3Index{b - 1}).row_end};
  };
  const std::int32_t t1 = t0 + 1;
  const std::int32_t t2 = t0 + 2;
  const RowSet set =
      RowSet::FromRanges(s, {range(t2, t3), range(t0, t1), range(t1, t2)});
  ASSERT_EQ(set.ranges().size(), 3U);
  EXPECT_EQ(set.ranges()[0].triple_begin.value(), t0);
  EXPECT_EQ(set.ranges()[2].triple_begin.value(), t2);
  EXPECT_EQ(set.NumRows(), s.dim(Channel3Index{big}));
  // Local offsets of a split channel equal those of the whole channel.
  const RowSet whole = RowSet::FromRanges(s, {range(t0, t3)});
  for (std::int32_t k = 0; k < s.dim(Channel3Index{big}); ++k) {
    EXPECT_EQ(set.LocalRowOffset(Channel3Index{big}, KeptRow3Index{k}, true),
              whole.LocalRowOffset(Channel3Index{big}, KeptRow3Index{k}, true));
    EXPECT_EQ(
        set.LocalRowOffset(Channel3Index{big}, KeptRow3Index{k}, false),
        whole.LocalRowOffset(Channel3Index{big}, KeptRow3Index{k}, false));
  }
  EXPECT_EQ(set.StoredSize(true), whole.StoredSize(true));
  EXPECT_EQ(set.StoredSize(false), whole.StoredSize(false));

  // Overlap, empty, misaligned rows, channel mismatch.
  EXPECT_THROW(RowSet::FromRanges(s, {range(t0, t2), range(t1, t3)}),
               InputError);
  EXPECT_THROW(RowSet::FromRanges(s, {range(t0, t1), range(t0, t1)}),
               InputError);
  RowRange empty = range(t0, t1);
  empty.triple_end = empty.triple_begin;
  EXPECT_THROW(RowSet::FromRanges(s, {empty}), InputError);
  RowRange bad_rows = range(t0, t2);
  bad_rows.row_end = KeptRow3Index{bad_rows.row_end.value() - 1};
  EXPECT_THROW(RowSet::FromRanges(s, {bad_rows}), InputError);
  RowRange bad_begin = range(t1, t3);
  bad_begin.row_begin = KeptRow3Index{bad_begin.row_begin.value() + 1};
  EXPECT_THROW(RowSet::FromRanges(s, {bad_begin}), InputError);
  RowRange cross = range(t0, t3);  // triples run into the next channel
  cross.triple_end = Triple3Index{t3 + 1};
  EXPECT_THROW(RowSet::FromRanges(s, {cross}), InputError);
  RowRange other_channel = range(t0, t1);
  other_channel.channel = Channel3Index{big + 1};
  EXPECT_THROW(RowSet::FromRanges(s, {other_channel}), InputError);
  RowRange invalid = range(t0, t1);
  invalid.channel = Channel3Index{s.num_channels()};
  EXPECT_THROW(RowSet::FromRanges(s, {invalid}), InputError);
}

/// Checks cover, disjointness, alignment and the two-ranges-per-channel rule.
void ExpectValidPartition(const ThreeBodySpace& s, const RowPartition& p,
                          bool check_two_ranges) {
  std::int64_t rows = 0;
  for (std::int32_t q = 0; q < p.num_parts(); ++q) {
    rows += p.part(q).NumRows();
    std::int32_t per_channel = 0;
    Channel3Index prev{-1};
    for (const RowRange& r : p.part(q).ranges()) {
      per_channel = r.channel == prev ? per_channel + 1 : 1;
      prev = r.channel;
      if (check_two_ranges) {
        ASSERT_LE(per_channel, 2) << "part " << q;
      }
    }
  }
  EXPECT_EQ(rows, s.NumRows());
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const Channel3Index ch{c};
    for (std::int32_t k = 0; k < s.dim(ch); ++k) {
      int owners = 0;
      for (std::int32_t q = 0; q < p.num_parts(); ++q) {
        owners += p.part(q).Contains(ch, KeptRow3Index{k}) ? 1 : 0;
      }
      ASSERT_EQ(owners, 1);
      const std::int32_t o = p.Owner(ch, KeptRow3Index{k});
      ASSERT_GE(o, 0);
      ASSERT_TRUE(p.part(o).Contains(ch, KeptRow3Index{k}));
    }
    EXPECT_EQ(p.Owner(ch, KeptRow3Index{s.dim(ch)}), -1);
  }
}

bool SamePartition(const RowPartition& a, const RowPartition& b) {
  if (a.num_parts() != b.num_parts()) return false;
  for (std::int32_t q = 0; q < a.num_parts(); ++q) {
    const auto ra = a.part(q).ranges();
    const auto rb = b.part(q).ranges();
    if (ra.size() != rb.size()) return false;
    for (std::size_t i = 0; i < ra.size(); ++i) {
      if (ra[i].channel != rb[i].channel ||
          ra[i].triple_begin != rb[i].triple_begin ||
          ra[i].triple_end != rb[i].triple_end) {
        return false;
      }
    }
  }
  return true;
}

TEST(RowPartitionTest, WholeIsOnePart) {
  const auto ms = O16(2, 6);
  const ThreeBodySpace& s = *ms->three_body_space();
  const RowPartition p = RowPartition::Whole(s);
  EXPECT_EQ(p.num_parts(), 1);
  ExpectValidPartition(s, p, true);
}

TEST(RowPartitionTest, BalancedIsValidAndDeterministic) {
  for (const Case& c : kCases) {
    const auto ms = O16(c.emax, c.e3max, c.de3max);
    const ThreeBodySpace& s = *ms->three_body_space();
    const LineWeights w = SyntheticWeights(s);
    const std::int32_t many = s.num_triples() + 5;
    for (const std::int32_t parts : {1, 2, 3, 7, 8, 16, many}) {
      const RowPartition p = RowPartition::Balanced(s, w, parts);
      ASSERT_EQ(p.num_parts(), parts);
      ExpectValidPartition(s, p, true);
      EXPECT_TRUE(SamePartition(p, RowPartition::Balanced(s, w, parts)));
    }
  }
}

TEST(RowPartitionTest, BalancedImbalanceBelowFivePercent) {
  const auto ms = O16(3, 9);
  const ThreeBodySpace& s = *ms->three_body_space();
  const LineWeights w = SyntheticWeights(s);
  for (std::int32_t parts = 1; parts <= 8; ++parts) {
    const RowPartition p = RowPartition::Balanced(s, w, parts);
    const PartitionImbalanceResult r = PartitionImbalance(p, w);
    const PartitionImbalanceResult lpt =
        PartitionImbalance(RowPartition::WholeChannels(s, w, parts), w);
    std::cout << "P=" << parts << " balanced: compute " << r.compute
              << " memory " << r.memory << "; whole channels: compute "
              << lpt.compute << " memory " << lpt.memory << "\n";
    EXPECT_LE(r.compute, 0.05);
    EXPECT_LE(r.memory, 0.05);
  }
}

TEST(RowPartitionTest, WholeChannelsReproducesIndependentLpt) {
  const auto ms = O16(3, 9);
  const ThreeBodySpace& s = *ms->three_body_space();
  const LineWeights w = SyntheticWeights(s);
  for (const std::int32_t parts : {1, 2, 5, 8}) {
    // Independent LPT: repeatedly take the heaviest unassigned channel.
    std::vector<double> total(static_cast<std::size_t>(s.num_channels()), 0.0);
    std::vector<bool> done(total.size(), false);
    for (std::int32_t c = 0; c < s.num_channels(); ++c) {
      const KeptChannel3& kc = s.channel(Channel3Index{c});
      done[static_cast<std::size_t>(c)] = kc.dim == 0;
      for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
           ++t) {
        total[static_cast<std::size_t>(c)] +=
            w.compute[static_cast<std::size_t>(t)];
      }
    }
    std::vector<double> load(static_cast<std::size_t>(parts), 0.0);
    std::vector<std::int32_t> owner(total.size(), -1);
    for (;;) {
      std::size_t best = total.size();
      for (std::size_t c = 0; c < total.size(); ++c) {
        if (!done[c] && (best == total.size() || total[c] > total[best])) {
          best = c;
        }
      }
      if (best == total.size()) break;
      std::size_t p = 0;
      for (std::size_t q = 1; q < load.size(); ++q) {
        if (load[q] < load[p]) p = q;
      }
      load[p] += total[best];
      owner[best] = static_cast<std::int32_t>(p);
      done[best] = true;
    }
    const RowPartition lpt = RowPartition::WholeChannels(s, w, parts);
    ExpectValidPartition(s, lpt, true);
    for (std::int32_t c = 0; c < s.num_channels(); ++c) {
      for (std::int32_t k = 0; k < s.dim(Channel3Index{c}); ++k) {
        ASSERT_EQ(lpt.Owner(Channel3Index{c}, KeptRow3Index{k}),
                  owner[static_cast<std::size_t>(c)]);
      }
    }
  }
}

TEST(RowPartitionTest, RejectsBadArguments) {
  const auto ms = O16(2, 6);
  const ThreeBodySpace& s = *ms->three_body_space();
  LineWeights w = SyntheticWeights(s);
  EXPECT_THROW(RowPartition::Balanced(s, w, 0), InputError);
  EXPECT_THROW(RowPartition::WholeChannels(s, w, -1), InputError);
  LineWeights shortw = w;
  shortw.memory.pop_back();
  EXPECT_THROW(RowPartition::Balanced(s, shortw, 2), InputError);
  w.compute[0] = -1.0;
  EXPECT_THROW(RowPartition::Balanced(s, w, 2), InputError);
}

TEST(RowPartitionTest, ZeroWeightsStillCover) {
  const auto ms = O16(2, 6);
  const ThreeBodySpace& s = *ms->three_body_space();
  LineWeights w;
  w.compute.assign(static_cast<std::size_t>(s.num_triples()), 0.0);
  w.memory = w.compute;
  const RowPartition p = RowPartition::Balanced(s, w, 3);
  ExpectValidPartition(s, p, true);
}

}  // namespace
}  // namespace imsrg::basis
