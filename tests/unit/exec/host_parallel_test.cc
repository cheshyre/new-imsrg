// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of exec::HostParallelForDynamic and exec::HostConcurrency (host
// lambdas only, no device code).
#include "imsrg/exec/host_parallel.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

using imsrg::exec::HostConcurrency;
using imsrg::exec::HostParallelForDynamic;
using imsrg::exec::InHostParallelRegion;

TEST(ExecHostParallel, ConcurrencyIsPositive) {
  EXPECT_GE(HostConcurrency(), 1);
}

TEST(ExecHostParallel, EachSlotWrittenExactlyOnce) {
  for (const std::int64_t n : {0, 1, 1000}) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(n), -1);
    std::vector<int> count(static_cast<std::size_t>(n), 0);
    HostParallelForDynamic("test::host_parallel", n, [&](std::int64_t i) {
      out[static_cast<std::size_t>(i)] = 3 * i + 7;
      ++count[static_cast<std::size_t>(i)];
    });
    for (std::int64_t i = 0; i < n; ++i) {
      ASSERT_EQ(count[static_cast<std::size_t>(i)], 1) << "slot " << i;
      ASSERT_EQ(out[static_cast<std::size_t>(i)], 3 * i + 7);
    }
  }
}

TEST(ExecHostParallel, InHostParallelRegionInsideTheLoopOnly) {
  EXPECT_FALSE(InHostParallelRegion());
  std::atomic<int> inside{0};
  HostParallelForDynamic("test::in_region", 64, [&](std::int64_t) {
    if (InHostParallelRegion()) ++inside;
  });
  // OpenMP reports an active region only with more than one thread.
  if (HostConcurrency() > 1) {
    EXPECT_EQ(inside.load(), 64);
  } else {
    EXPECT_EQ(inside.load(), 0);
  }
  EXPECT_FALSE(InHostParallelRegion());
}

}  // namespace
