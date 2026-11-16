// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

/// @file
/// Semantics of the serial communicator (valid in builds with and without MPI).

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/comm/environment.h"

namespace {

using imsrg::comm::Communicator;

TEST(CommSerial, Basics) {
  const auto c = Communicator::Serial();
  EXPECT_EQ(c.rank(), 0);
  EXPECT_EQ(c.size(), 1);
  EXPECT_TRUE(c.is_root());
  EXPECT_TRUE(c.is_serial());
  c.Barrier();
  const Communicator
      by_default;  // default construction gives the serial communicator
  EXPECT_TRUE(by_default.is_serial());
  EXPECT_EQ(by_default.size(), 1);
}

TEST(CommSerial, ReductionsAreIdentity) {
  const auto c = Communicator::Serial();
  std::array<double, 3> d{1.5, -2.0, 3.0};
  c.AllreduceSum(d);
  c.AllreduceMax(d);
  c.AllreduceMin(d);
  EXPECT_EQ(d, (std::array<double, 3>{1.5, -2.0, 3.0}));
  std::array<float, 2> f{1.0F, 2.0F};
  c.AllreduceSum(f);
  EXPECT_EQ(f[1], 2.0F);
  std::array<std::int32_t, 2> i32{7, -8};
  c.AllreduceMin(i32);
  EXPECT_EQ(i32[1], -8);
  std::array<std::int64_t, 2> i64{1LL << 40, 5};
  c.AllreduceMax(i64);
  EXPECT_EQ(i64[0], 1LL << 40);
  EXPECT_EQ(c.AllreduceSum(2.5), 2.5);
  EXPECT_EQ(c.AllreduceMax(std::int64_t{9}), 9);
  EXPECT_EQ(c.AllreduceMin(std::int64_t{-9}), -9);
  EXPECT_EQ(c.AllreduceSum(std::int64_t{4}), 4);
  EXPECT_EQ(c.AllreduceMax(1.0), 1.0);
  EXPECT_EQ(c.AllreduceMin(1.0), 1.0);
}

TEST(CommSerial, BroadcastKeepsData) {
  const auto c = Communicator::Serial();
  std::array<double, 2> d{1.0, 2.0};
  c.Broadcast(d, 0);
  EXPECT_EQ(d[1], 2.0);
  std::array<std::byte, 2> b{std::byte{1}, std::byte{2}};
  c.Broadcast(b, 0);
  EXPECT_EQ(b[1], std::byte{2});
  std::string s = "hello";
  c.Broadcast(s, 0);
  EXPECT_EQ(s, "hello");
  EXPECT_THROW(c.Broadcast(d, 1), imsrg::InvariantError);
}

TEST(CommSerial, Allgather) {
  const auto c = Communicator::Serial();
  EXPECT_EQ(c.Allgather(3.5), std::vector<double>{3.5});
  EXPECT_EQ(c.Allgather(std::int64_t{7}), std::vector<std::int64_t>{7});
}

TEST(CommSerial, AlltoallvCopies) {
  const auto c = Communicator::Serial();
  const std::vector<double> send{1, 2, 3, 4};
  std::vector<double> recv(5, 0.0);
  const std::array<int, 1> counts{3};
  const std::array<int, 1> sd{1};
  const std::array<int, 1> rd{2};
  c.Alltoallv(send, counts, sd, recv, counts, rd);
  EXPECT_EQ(recv, (std::vector<double>{0, 0, 2, 3, 4}));
  const std::array<int, 1> too_many{4};
  EXPECT_THROW(c.Alltoallv(send, too_many, sd, recv, too_many, rd),
               imsrg::InvariantError);
}

TEST(CommSerial, SplitAndDuplicate) {
  const auto c = Communicator::Serial();
  const auto s = c.Split(3, 0);
  EXPECT_EQ(s.size(), 1);
  EXPECT_EQ(s.rank(), 0);
  EXPECT_EQ(s.AllreduceSum(1.0), 1.0);
  const auto d = c.Duplicate();
  EXPECT_EQ(d.size(), 1);
  EXPECT_TRUE(d.is_serial());
}

TEST(CommSerial, EnvironmentQueries) {
  // No Environment is created here: only the static queries and the MPI-free
  // serial path.
  EXPECT_FALSE(imsrg::comm::Environment::IsInitialized());
  (void)imsrg::comm::Environment::MpiEnabled();
}

}  // namespace
