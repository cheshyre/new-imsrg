#include <type_traits>
#include <unordered_map>

#include <gtest/gtest.h>

#include "imsrg/base/types.hpp"

namespace {

using namespace imsrg;

static_assert(!std::is_convertible_v<int, OrbitIndex>);
static_assert(!std::is_convertible_v<OrbitIndex, int>);
static_assert(!std::is_same_v<OrbitIndex, KetIndex>);
static_assert(std::is_trivially_copyable_v<ChannelIndex>);
static_assert(sizeof(Ket3Index) == 4);
static_assert(OrbitIndex{5}.value() == 5);
static_assert(TwoJ{3} < TwoJ{5} && TwoJ{3} != TwoJ{5} && TwoJ{3} == TwoJ{3});
static_assert(degeneracy(TwoJ{3}) == 4);
static_assert(as_double(TwoJ{3}) == 1.5 && as_double(TwoM{-1}) == -0.5);
static_assert(is_half_integer(TwoJ{3}) && !is_half_integer(TwoJ{4}));
static_assert(TwoJ::from_integer(2) == TwoJ{4});
static_assert(TwoTz::proton().twice == -1 && TwoTz::neutron().twice == 1);
static_assert(parity_of_l(0) == Parity::Even && parity_of_l(3) == Parity::Odd);
static_assert(Parity::Odd * Parity::Odd == Parity::Even);
static_assert(Parity::Odd * Parity::Even == Parity::Odd);
static_assert(sign(Parity::Even) == 1 && sign(Parity::Odd) == -1);

TEST(Types, DefaultIsZeroAndExplicitConstruction) {
  EXPECT_EQ(OrbitIndex{}.value(), 0);
  EXPECT_EQ(OrbitIndex{7}.value(), 7);
  EXPECT_EQ(OrbitIndex{7}.to_size(), std::size_t{7});
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
  EXPECT_EQ(degeneracy(TwoJ{7}), 8);
  EXPECT_DOUBLE_EQ(as_double(TwoJ{7}), 3.5);
  EXPECT_TRUE(is_half_integer(TwoJ{1}));
  EXPECT_FALSE(is_half_integer(TwoJ{0}));
  EXPECT_EQ(TwoJ::from_integer(3).twice, 6);
  EXPECT_TRUE(TwoM{-3} < TwoM{1});
  EXPECT_TRUE(TwoTz::proton() < TwoTz::neutron());
}

TEST(Types, ParityProduct) {
  for (int l1 = 0; l1 < 6; ++l1) {
    for (int l2 = 0; l2 < 6; ++l2) {
      EXPECT_EQ(parity_of_l(l1) * parity_of_l(l2), parity_of_l(l1 + l2));
      EXPECT_EQ(sign(parity_of_l(l1)) * sign(parity_of_l(l2)), sign(parity_of_l(l1 + l2)));
    }
  }
}

}  // namespace
