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
