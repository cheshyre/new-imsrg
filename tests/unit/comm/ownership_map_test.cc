// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Tests of imsrg::comm::OwnershipMap and the message chunking helper.

#include "imsrg/comm/ownership_map.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/comm/detail/chunks.h"

namespace imsrg::comm {
namespace {

std::vector<std::int32_t> Owners(const OwnershipMap& m) {
  std::vector<std::int32_t> o;
  for (std::int32_t i = 0; i < m.num_items(); ++i) o.push_back(m.owner(i));
  return o;
}

std::vector<std::int32_t> ItemsOf(const OwnershipMap& m, std::int32_t r) {
  const auto s = m.items_of(r);
  return {s.begin(), s.end()};
}

TEST(OwnershipMap, HandCalculatedLpt) {
  const std::vector<double> cost{5, 3, 3, 2, 2, 1};
  const auto m = OwnershipMap::Greedy(cost, 2);
  EXPECT_EQ(Owners(m), (std::vector<std::int32_t>{0, 1, 1, 0, 1, 0}));
  EXPECT_EQ(ItemsOf(m, 0), (std::vector<std::int32_t>{0, 3, 5}));
  EXPECT_EQ(ItemsOf(m, 1), (std::vector<std::int32_t>{1, 2, 4}));
  EXPECT_EQ(m.load(0), 8.0);
  EXPECT_EQ(m.load(1), 8.0);
  EXPECT_EQ(m.Imbalance(), 0.0);
}

TEST(OwnershipMap, TiesGoToLowerIndexAndLowerRank) {
  const std::vector<double> cost{1, 1, 1, 1};
  const auto m = OwnershipMap::Greedy(cost, 3);
  EXPECT_EQ(Owners(m), (std::vector<std::int32_t>{0, 1, 2, 0}));
}

TEST(OwnershipMap, RandomCostsPartitionExactly) {
  std::mt19937_64 gen(12345);
  std::uniform_real_distribution<double> dist(0.0, 10.0);
  std::vector<double> cost(500);
  for (double& c : cost) c = dist(gen);
  const double total = std::accumulate(cost.begin(), cost.end(), 0.0);
  for (std::int32_t p = 1; p <= 300; ++p) {
    const auto m = OwnershipMap::Greedy(cost, p);
    ASSERT_EQ(m.num_ranks(), p);
    ASSERT_EQ(m.num_items(), 500);
    std::vector<int> seen(cost.size(), 0);
    double load_sum = 0.0;
    std::size_t count = 0;
    for (std::int32_t r = 0; r < p; ++r) {
      const auto items = m.items_of(r);
      count += items.size();
      double l = 0.0;
      for (std::size_t k = 0; k < items.size(); ++k) {
        if (k > 0) {
          ASSERT_LT(items[k - 1], items[k]);
        }
        ++seen[static_cast<std::size_t>(items[k])];
        ASSERT_EQ(m.owner(items[k]), r);
        l += cost[static_cast<std::size_t>(items[k])];
      }
      EXPECT_NEAR(m.load(r), l, 1.0e-12 * (1.0 + l));
      load_sum += m.load(r);
    }
    ASSERT_EQ(count, cost.size());
    for (const int s : seen) ASSERT_EQ(s, 1);
    EXPECT_NEAR(load_sum, total, 1.0e-10 * total);
  }
}

TEST(OwnershipMap, ImbalanceFormula) {
  const std::vector<double> cost{3, 1};
  const auto m = OwnershipMap::Greedy(cost, 2);
  EXPECT_EQ(m.load(0), 3.0);
  EXPECT_EQ(m.load(1), 1.0);
  EXPECT_DOUBLE_EQ(m.Imbalance(), 0.5);  // 3 / mean 2 - 1
}

TEST(OwnershipMap, EdgeCases) {
  const OwnershipMap def;
  EXPECT_EQ(def.num_items(), 0);
  EXPECT_EQ(def.num_ranks(), 1);
  EXPECT_TRUE(def.items_of(0).empty());
  EXPECT_EQ(def.Imbalance(), 0.0);

  const auto none = OwnershipMap::Greedy({}, 4);
  EXPECT_EQ(none.num_items(), 0);
  EXPECT_EQ(none.num_ranks(), 4);
  EXPECT_EQ(none.Imbalance(), 0.0);

  const std::vector<double> two{2.0, 1.0};
  const auto more_ranks = OwnershipMap::Greedy(two, 5);
  EXPECT_EQ(more_ranks.owner(0), 0);
  EXPECT_EQ(more_ranks.owner(1), 1);
  EXPECT_TRUE(more_ranks.items_of(4).empty());
  EXPECT_EQ(more_ranks.load(4), 0.0);
  EXPECT_DOUBLE_EQ(more_ranks.Imbalance(), 2.0 / (3.0 / 5.0) - 1.0);

  const std::vector<double> zeros(7, 0.0);
  EXPECT_EQ(OwnershipMap::Greedy(zeros, 3).Imbalance(), 0.0);

  const std::vector<double> five{1, 2, 3, 4, 5};
  const auto one = OwnershipMap::Greedy(five, 1);
  EXPECT_EQ(ItemsOf(one, 0), (std::vector<std::int32_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(one.load(0), 15.0);
  EXPECT_EQ(one.Imbalance(), 0.0);
}

TEST(OwnershipMap, InvalidInputThrows) {
  const std::vector<double> ok{1, 2};
  EXPECT_THROW(OwnershipMap::Greedy(ok, 0), InputError);
  EXPECT_THROW(OwnershipMap::Greedy(ok, -1), InputError);
  const std::vector<double> neg{1, -1};
  EXPECT_THROW(OwnershipMap::Greedy(neg, 2), InputError);
  const std::vector<double> nan{1, std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(OwnershipMap::Greedy(nan, 2), InputError);
  const std::vector<double> inf{1, std::numeric_limits<double>::infinity()};
  EXPECT_THROW(OwnershipMap::Greedy(inf, 2), InputError);
}

TEST(OwnershipMap, ManySmallItemsAreBalanced) {
  std::mt19937_64 gen(2024);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<double> cost(1000);
  for (double& c : cost) c = dist(gen);
  const auto m = OwnershipMap::Greedy(cost, 22);
  std::printf("imbalance of 1000 items on 22 ranks: %.3e\n", m.Imbalance());
  EXPECT_LT(m.Imbalance(), 0.05);
}

TEST(OwnershipMap, FromOwnersCsrLoadsAndImbalance) {
  const std::vector<std::int32_t> owner{0, 0, 2, 2, 2, 3};
  const std::vector<double> cost{1, 2, 3, 4, 5, 6};
  const auto m = OwnershipMap::FromOwners(owner, cost, 4);
  EXPECT_EQ(m.num_items(), 6);
  EXPECT_EQ(m.num_ranks(), 4);
  EXPECT_EQ(Owners(m), owner);
  EXPECT_EQ(ItemsOf(m, 0), (std::vector<std::int32_t>{0, 1}));
  EXPECT_TRUE(ItemsOf(m, 1).empty());  // an empty rank
  EXPECT_EQ(ItemsOf(m, 2), (std::vector<std::int32_t>{2, 3, 4}));
  EXPECT_EQ(ItemsOf(m, 3), (std::vector<std::int32_t>{5}));
  EXPECT_EQ(m.load(0), 3.0);
  EXPECT_EQ(m.load(1), 0.0);
  EXPECT_EQ(m.load(2), 12.0);
  EXPECT_EQ(m.load(3), 6.0);
  EXPECT_DOUBLE_EQ(m.Imbalance(), 12.0 / (21.0 / 4.0) - 1.0);
}

TEST(OwnershipMap, FromOwnersUnsortedOwnersGiveAscendingItems) {
  const std::vector<std::int32_t> owner{1, 0, 1, 0};
  const std::vector<double> cost{1, 1, 1, 1};
  const auto m = OwnershipMap::FromOwners(owner, cost, 2);
  EXPECT_EQ(ItemsOf(m, 0), (std::vector<std::int32_t>{1, 3}));
  EXPECT_EQ(ItemsOf(m, 1), (std::vector<std::int32_t>{0, 2}));
  EXPECT_EQ(m.Imbalance(), 0.0);
}

TEST(OwnershipMap, FromOwnersMatchesGreedyOwners) {
  const std::vector<double> cost{5, 3, 3, 2, 2, 1};
  const auto g = OwnershipMap::Greedy(cost, 3);
  const auto f = OwnershipMap::FromOwners(Owners(g), cost, 3);
  for (std::int32_t r = 0; r < 3; ++r) {
    EXPECT_EQ(ItemsOf(f, r), ItemsOf(g, r));
    EXPECT_DOUBLE_EQ(f.load(r), g.load(r));
  }
  EXPECT_DOUBLE_EQ(f.Imbalance(), g.Imbalance());
}

TEST(OwnershipMap, FromOwnersEmptyAndInvalid) {
  const auto empty = OwnershipMap::FromOwners({}, {}, 3);
  EXPECT_EQ(empty.num_items(), 0);
  EXPECT_EQ(empty.num_ranks(), 3);
  EXPECT_EQ(empty.Imbalance(), 0.0);
  const std::vector<std::int32_t> owner{0, 1};
  const std::vector<double> cost{1.0, 2.0};
  EXPECT_THROW(OwnershipMap::FromOwners(owner, cost, 0), InputError);
  EXPECT_THROW(OwnershipMap::FromOwners(owner, cost, -1), InputError);
  const std::vector<double> short_cost{1.0};
  EXPECT_THROW(OwnershipMap::FromOwners(owner, short_cost, 2), InputError);
  EXPECT_THROW(OwnershipMap::FromOwners(owner, cost, 1), InputError);
  const std::vector<std::int32_t> negative{0, -1};
  EXPECT_THROW(OwnershipMap::FromOwners(negative, cost, 2), InputError);
  const std::vector<double> bad_neg{1.0, -2.0};
  EXPECT_THROW(OwnershipMap::FromOwners(owner, bad_neg, 2), InputError);
  const std::vector<double> bad_nan{1.0, std::nan("")};
  EXPECT_THROW(OwnershipMap::FromOwners(owner, bad_nan, 2), InputError);
  const std::vector<double> bad_inf{1.0,
                                    std::numeric_limits<double>::infinity()};
  EXPECT_THROW(OwnershipMap::FromOwners(owner, bad_inf, 2), InputError);
}

TEST(CommChunks, Ranges) {
  using detail::ChunkRanges;
  EXPECT_TRUE(ChunkRanges(0, 4).empty());
  const auto one = ChunkRanges(3, 4);
  ASSERT_EQ(one.size(), 1U);
  EXPECT_EQ(one[0].offset, 0);
  EXPECT_EQ(one[0].count, 3);
  EXPECT_EQ(ChunkRanges(8, 4).size(), 2U);
  const auto three = ChunkRanges(9, 4);
  ASSERT_EQ(three.size(), 3U);
  EXPECT_EQ(three[2].offset, 8);
  EXPECT_EQ(three[2].count, 1);
  EXPECT_THROW(ChunkRanges(-1, 4), InvariantError);
  EXPECT_THROW(ChunkRanges(4, 0), InvariantError);
}

TEST(CommChunks, HugeMessageIsContiguousAndFitsInt) {
  const std::int64_t n = 3'000'000'000;
  const auto chunks = detail::ChunkRanges(n, detail::kMaxMessageElements);
  ASSERT_EQ(chunks.size(), 3U);
  std::int64_t next = 0;
  for (const auto& c : chunks) {
    EXPECT_EQ(c.offset, next);
    EXPECT_GT(c.count, 0);
    EXPECT_LE(c.count, std::numeric_limits<std::int32_t>::max());
    next += c.count;
  }
  EXPECT_EQ(next, n);
}

}  // namespace
}  // namespace imsrg::comm
