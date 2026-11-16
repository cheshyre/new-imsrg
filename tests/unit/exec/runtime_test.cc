// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/exec/runtime.h"

#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"

namespace {

using imsrg::exec::Runtime;

TEST(ExecRuntime, IsInitializedUnderTestMain) {
  EXPECT_TRUE(Runtime::IsInitialized());
}

TEST(ExecRuntime, ExecutionSpaceNameMatchesConfiguredBackend) {
  EXPECT_EQ(Runtime::ExecutionSpaceName(),
            std::string(IMSRG_EXPECTED_KOKKOS_SPACE));
}

TEST(ExecRuntime, ConcurrencyIsPositive) {
  EXPECT_GE(Runtime::Concurrency(), 1);
}

TEST(ExecRuntime, SecondRuntimeThrows) {
  EXPECT_THROW(Runtime second, imsrg::RuntimeError);
}

// Options are validated before Kokkos sees them (Kokkos aborts on
// --kokkos-num-threads=0).
TEST(ExecRuntime, InvalidOptionsThrowConfigError) {
  imsrg::exec::RuntimeOptions zero_threads;
  zero_threads.num_threads = 0;
  EXPECT_THROW(Runtime r(zero_threads), imsrg::ConfigError);
  imsrg::exec::RuntimeOptions bad_device;
  bad_device.device_id = -2;
  int argc = 1;
  char name[] = "exec_runtime_test";
  char* argv[] = {name, nullptr};
  EXPECT_THROW(Runtime r(argc, argv, bad_device), imsrg::ConfigError);
}

TEST(ExecRuntime, MemoryAccessibilityMatchesBackend) {
  const std::string expected = IMSRG_EXPECTED_KOKKOS_SPACE;
  if (expected == "Serial" || expected == "OpenMP") {
    EXPECT_TRUE(imsrg::exec::DefaultMemoryIsHostAccessible());
  }
}

TEST(ExecRuntime, ProfilingRegionsDoNotThrow) {
  imsrg::exec::PushProfilingRegion("exec_runtime_test");
  imsrg::exec::PopProfilingRegion();
}

}  // namespace
