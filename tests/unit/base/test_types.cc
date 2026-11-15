// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <type_traits>
#include <unordered_map>

#include <gtest/gtest.h>

#include "imsrg/base/types.h"

namespace {

using namespace imsrg;

static_assert(!std::is_convertible_v<int, OrbitIndex>);
static_assert(!std::is_convertible_v<OrbitIndex, int>);
static_assert(!std::is_same_v<OrbitIndex, KetIndex>);
static_assert(std::is_trivially_copyable_v<ChannelIndex>);
static_assert(sizeof(Ket3Index) == 4);
static_assert(OrbitIndex{5}.value() == 5);
static_assert(TwoJ{3} < TwoJ{5} && TwoJ{3} != TwoJ{5} && TwoJ{3} == TwoJ{3});
static_assert(Degeneracy(TwoJ{3}) == 4);
static_assert(AsDouble(TwoJ{3}) == 1.5 && AsDouble(TwoM{-1}) == -0.5);
static_assert(IsHalfInteger(TwoJ{3}) && !IsHalfInteger(TwoJ{4}));
static_assert(TwoJ::FromInteger(2) == TwoJ{4});
static_assert(TwoTz::Proton().twice == -1 && TwoTz::Neutron().twice == 1);
static_assert(ParityOfL(0) == Parity::kEven && ParityOfL(3) == Parity::kOdd);
static_assert(Parity::kOdd * Parity::kOdd == Parity::kEven);
static_assert(Parity::kOdd * Parity::kEven == Parity::kOdd);
static_assert(Sign(Parity::kEven) == 1 && Sign(Parity::kOdd) == -1);

TEST(Types, DefaultIsZeroAndExplicitConstruction) {
  EXPECT_EQ(OrbitIndex{}.value(), 0);
  EXPECT_EQ(OrbitIndex{7}.value(), 7);
  EXPECT_EQ(OrbitIndex{7}.ToSize(), std::size_t{7});
}

TEST(Types, IncrementDecrement) {
  OrbitIndex i{3};
  EXPECT_EQ((++i).value(), 4);
  EXPECT_EQ((i++).value(), 4);
  EXPECT_EQ(i.value(), 5);
  EXPECT_EQ((--i).value(), 4);
  EXPECT_EQ((i--).value(), 4);
  EXPECT_EQ(i.value(), 3);
}

TEST(Types, Comparisons) {
  const KetIndex a{1}, b{2};
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a == KetIndex{1});
  EXPECT_FALSE(a == b);
}

TEST(Types, HashingInUnorderedMap) {
  std::unordered_map<OrbitIndex, int> m;
  for (int i = 0; i < 100; ++i) m[OrbitIndex{i}] = i * i;
  EXPECT_EQ(m.size(), std::size_t{100});
  EXPECT_EQ(m.at(OrbitIndex{9}), 81);
  EXPECT_EQ(m.count(OrbitIndex{100}), std::size_t{0});
}

TEST(Types, TwoJHelpers) {
  EXPECT_EQ(Degeneracy(TwoJ{7}), 8);
  EXPECT_DOUBLE_EQ(AsDouble(TwoJ{7}), 3.5);
  EXPECT_TRUE(IsHalfInteger(TwoJ{1}));
  EXPECT_FALSE(IsHalfInteger(TwoJ{0}));
  EXPECT_EQ(TwoJ::FromInteger(3).twice, 6);
  EXPECT_TRUE(TwoM{-3} < TwoM{1});
  EXPECT_TRUE(TwoTz::Proton() < TwoTz::Neutron());
}

TEST(Types, ParityProduct) {
  for (int l1 = 0; l1 < 6; ++l1) {
    for (int l2 = 0; l2 < 6; ++l2) {
      EXPECT_EQ(ParityOfL(l1) * ParityOfL(l2), ParityOfL(l1 + l2));
      EXPECT_EQ(Sign(ParityOfL(l1)) * Sign(ParityOfL(l2)),
                Sign(ParityOfL(l1 + l2)));
    }
  }
}

}  // namespace
