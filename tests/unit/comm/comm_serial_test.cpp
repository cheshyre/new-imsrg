/// @file
/// Semantics of the serial communicator (valid in builds with and without MPI).

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/comm/communicator.hpp"
#include "imsrg/comm/environment.hpp"

namespace {

using imsrg::comm::Communicator;

TEST(CommSerial, Basics) {
  const auto c = Communicator::serial();
  EXPECT_EQ(c.rank(), 0);
  EXPECT_EQ(c.size(), 1);
  EXPECT_TRUE(c.is_root());
  EXPECT_TRUE(c.is_serial());
  c.barrier();
  const Communicator by_default;  // default construction gives the serial communicator
  EXPECT_TRUE(by_default.is_serial());
  EXPECT_EQ(by_default.size(), 1);
}

TEST(CommSerial, ReductionsAreIdentity) {
  const auto c = Communicator::serial();
  std::array<double, 3> d{1.5, -2.0, 3.0};
  c.allreduce_sum(d);
  c.allreduce_max(d);
  c.allreduce_min(d);
  EXPECT_EQ(d, (std::array<double, 3>{1.5, -2.0, 3.0}));
  std::array<float, 2> f{1.0F, 2.0F};
  c.allreduce_sum(f);
  EXPECT_EQ(f[1], 2.0F);
  std::array<std::int32_t, 2> i32{7, -8};
  c.allreduce_min(i32);
  EXPECT_EQ(i32[1], -8);
  std::array<std::int64_t, 2> i64{1LL << 40, 5};
  c.allreduce_max(i64);
  EXPECT_EQ(i64[0], 1LL << 40);
  EXPECT_EQ(c.allreduce_sum(2.5), 2.5);
  EXPECT_EQ(c.allreduce_max(std::int64_t{9}), 9);
  EXPECT_EQ(c.allreduce_min(std::int64_t{-9}), -9);
  EXPECT_EQ(c.allreduce_sum(std::int64_t{4}), 4);
  EXPECT_EQ(c.allreduce_max(1.0), 1.0);
  EXPECT_EQ(c.allreduce_min(1.0), 1.0);
}

TEST(CommSerial, BroadcastKeepsData) {
  const auto c = Communicator::serial();
  std::array<double, 2> d{1.0, 2.0};
  c.broadcast(d, 0);
  EXPECT_EQ(d[1], 2.0);
  std::array<std::byte, 2> b{std::byte{1}, std::byte{2}};
  c.broadcast(b, 0);
  EXPECT_EQ(b[1], std::byte{2});
  std::string s = "hello";
  c.broadcast(s, 0);
  EXPECT_EQ(s, "hello");
  EXPECT_THROW(c.broadcast(d, 1), imsrg::InvariantError);
}

TEST(CommSerial, Allgather) {
  const auto c = Communicator::serial();
  EXPECT_EQ(c.allgather(3.5), std::vector<double>{3.5});
  EXPECT_EQ(c.allgather(std::int64_t{7}), std::vector<std::int64_t>{7});
}

TEST(CommSerial, AlltoallvCopies) {
  const auto c = Communicator::serial();
  const std::vector<double> send{1, 2, 3, 4};
  std::vector<double> recv(5, 0.0);
  const std::array<int, 1> counts{3};
  const std::array<int, 1> sd{1};
  const std::array<int, 1> rd{2};
  c.alltoallv(send, counts, sd, recv, counts, rd);
  EXPECT_EQ(recv, (std::vector<double>{0, 0, 2, 3, 4}));
  const std::array<int, 1> too_many{4};
  EXPECT_THROW(c.alltoallv(send, too_many, sd, recv, too_many, rd), imsrg::InvariantError);
}

TEST(CommSerial, SplitAndDuplicate) {
  const auto c = Communicator::serial();
  const auto s = c.split(3, 0);
  EXPECT_EQ(s.size(), 1);
  EXPECT_EQ(s.rank(), 0);
  EXPECT_EQ(s.allreduce_sum(1.0), 1.0);
  const auto d = c.duplicate();
  EXPECT_EQ(d.size(), 1);
  EXPECT_TRUE(d.is_serial());
}

TEST(CommSerial, EnvironmentQueries) {
  // No Environment is created here: only the static queries and the MPI-free serial path.
  EXPECT_FALSE(imsrg::comm::Environment::is_initialized());
  (void)imsrg::comm::Environment::mpi_enabled();
}

}  // namespace
