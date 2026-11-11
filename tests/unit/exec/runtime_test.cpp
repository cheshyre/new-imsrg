#include "imsrg/exec/runtime.hpp"

#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"

namespace {

using imsrg::exec::Runtime;

TEST(ExecRuntime, IsInitializedUnderTestMain) { EXPECT_TRUE(Runtime::is_initialized()); }

TEST(ExecRuntime, ExecutionSpaceNameMatchesConfiguredBackend) {
  EXPECT_EQ(Runtime::execution_space_name(), std::string(IMSRG_EXPECTED_KOKKOS_SPACE));
}

TEST(ExecRuntime, ConcurrencyIsPositive) { EXPECT_GE(Runtime::concurrency(), 1); }

TEST(ExecRuntime, SecondRuntimeThrows) { EXPECT_THROW(Runtime second, imsrg::RuntimeError); }

// Options are validated before Kokkos sees them (Kokkos aborts on --kokkos-num-threads=0).
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
    EXPECT_TRUE(imsrg::exec::default_memory_is_host_accessible());
  }
}

TEST(ExecRuntime, ProfilingRegionsDoNotThrow) {
  imsrg::exec::push_profiling_region("exec_runtime_test");
  imsrg::exec::pop_profiling_region();
}

}  // namespace
