// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// M12b review fix: `CheckReplicatedSetup` agrees across the ranks. Equal
// setups pass; rank-dependent occupations or thread counts throw
// `RuntimeError` on every rank (the decision uses reduced values, so no rank
// is left in a collective). Run as `driver_mpi_check_2ranks` and
// `driver_mpi_check_4ranks`.

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/driver/pipeline/mpi_sync.h"
#include "tests/support/mpi_test_main.h"

namespace imsrg::driver::pipeline {
namespace {

using comm::Communicator;
using test_support::TestWorld;

std::shared_ptr<const basis::ModelSpace> Space(const char* nucleus) {
  return basis::ModelSpace::FromNames({.emax = 2}, 16.0, nucleus, nucleus);
}

/// Number of ranks on which the check threw RuntimeError; `message` receives
/// the message of the calling rank (empty if it did not throw).
std::int64_t CountThrowing(const basis::ModelSpace& ms, int concurrency,
                           std::string& message) {
  const Communicator& world = TestWorld();
  std::int64_t threw = 0;
  try {
    CheckReplicatedSetup(ms, world, concurrency);
  } catch (const RuntimeError& e) {
    threw = 1;
    message = e.what();
  }
  return world.AllreduceSum(threw);
}

TEST(ReplicatedCheck, EqualSetupsPass) {
  const auto ms = Space("He4");
  std::string msg;
  EXPECT_EQ(CountThrowing(*ms, 3, msg), 0) << msg;
}

TEST(ReplicatedCheck, RankDependentOccupationsThrowEverywhere) {
  const Communicator& world = TestWorld();
  const auto ms = Space(world.rank() == 0 ? "He4" : "O16");
  std::string msg;
  EXPECT_EQ(CountThrowing(*ms, 3, msg), world.size());
  EXPECT_NE(msg.find("occupations"), std::string::npos) << msg;
  EXPECT_EQ(msg.find("thread"), std::string::npos) << msg;
}

TEST(ReplicatedCheck, RankDependentThreadCountThrowsEverywhere) {
  const Communicator& world = TestWorld();
  const auto ms = Space("He4");
  std::string msg;
  EXPECT_EQ(CountThrowing(*ms, 2 + world.rank(), msg), world.size());
  EXPECT_NE(msg.find("thread count"), std::string::npos) << msg;
  EXPECT_EQ(msg.find("occupations"), std::string::npos) << msg;
}

TEST(ReplicatedCheck, RuntimeConcurrencyOverloadPassesOnEqualRanks) {
  const auto ms = Space("He4");
  EXPECT_NO_THROW(CheckReplicatedSetup(*ms, TestWorld()));
}

}  // namespace
}  // namespace imsrg::driver::pipeline
