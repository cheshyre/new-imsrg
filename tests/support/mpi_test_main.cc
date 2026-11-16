// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/support/mpi_test_main.h"

#include <cstdint>
#include <cstdio>

#include <gtest/gtest.h>

#include "imsrg/comm/communicator.h"
#include "imsrg/comm/environment.h"
#include "imsrg/exec/runtime.h"

namespace imsrg::test_support {

namespace {

// Set by main() while MPI is up; the communicator is a cheap handle.
comm::Communicator& World() {
  static comm::Communicator* const kWorld = new comm::Communicator();
  return *kWorld;
}

/// Only rank 0 prints the GoogleTest report.
void SilenceOtherRanks(const comm::Communicator& world) {
  if (world.rank() == 0) return;
  auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
  delete listeners.Release(listeners.default_result_printer());
}

/// Runs all tests; the result is the largest failure count over the ranks.
int RunAllTestsOnWorld() {
  const comm::Communicator& world = World();
  SilenceOtherRanks(world);
  const int failed = RUN_ALL_TESTS();
  if (failed != 0 && world.rank() != 0) {
    std::fprintf(stderr, "mpi_test_main: tests failed on rank %d\n",
                 world.rank());
  }
  return static_cast<int>(
      world.AllreduceMax(static_cast<std::int64_t>(failed)));
}

}  // namespace

const comm::Communicator& TestWorld() { return World(); }

}  // namespace imsrg::test_support

/// MPI first, then Kokkos (design/10 section 4); the exit code is the largest
/// failure count over the ranks, so every rank exits non-zero if one failed.
int main(int argc, char** argv) {
  int exit_code = 0;
  {
    imsrg::comm::Environment environment(argc, argv);
    imsrg::test_support::World() = environment.world();
    ::testing::InitGoogleTest(&argc, argv);
    {
      imsrg::exec::Runtime runtime(argc, argv);
      exit_code = imsrg::test_support::RunAllTestsOnWorld();
    }
    imsrg::test_support::World() = imsrg::comm::Communicator::Serial();
  }
  return exit_code;
}
