// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "tests/unit/exec/kernels_impl.h"

namespace {

using imsrg::exec::DeviceBuffer;
namespace t = imsrg::exec::test;

TEST(ExecKernels, ParallelForWritesTwiceIndex) {
  DeviceBuffer<double> a(1000);
  t::FillTwiceIndex(a);
  const auto v = a.ToVector();
  for (std::size_t i = 0; i < v.size(); ++i) {
    ASSERT_EQ(v[i], 2.0 * static_cast<double>(i));
  }
}

TEST(ExecKernels, ParallelForDynamicWritesOddNumbers) {
  for (const std::int64_t n : {0, 12345}) {
    DeviceBuffer<std::int64_t> a(static_cast<std::size_t>(n));
    t::FillOddDynamic(a);
    const auto v = a.ToVector();
    ASSERT_EQ(v.size(), static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) {
      ASSERT_EQ(v[i], 2 * static_cast<std::int64_t>(i) + 1);
    }
  }
}

TEST(ExecKernels, ReduceSumIsExact) {
  const std::int64_t n = 10000;
  std::vector<std::int64_t> h(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    h[static_cast<std::size_t>(i)] = i + 1;
  }
  DeviceBuffer<std::int64_t> b(h.size());
  b.CopyFromHost(h);
  EXPECT_EQ(t::Sum(b), n * (n + 1) / 2);
  EXPECT_EQ(t::Sum(DeviceBuffer<std::int64_t>(0)), 0);
}

TEST(ExecKernels, ReduceMax) {
  std::vector<std::int64_t> h{3, -7, 42, 5, 41, -100};
  DeviceBuffer<std::int64_t> b(h.size());
  b.CopyFromHost(h);
  EXPECT_EQ(t::MaxValue(b), 42);
  EXPECT_EQ(t::MaxValue(DeviceBuffer<std::int64_t>(0)),
            std::numeric_limits<std::int64_t>::lowest());
}

TEST(ExecKernels, MdRangeFillsMatrix) {
  const std::int64_t n0 = 7, n1 = 11;
  DeviceBuffer<std::int64_t> m(static_cast<std::size_t>(n0 * n1));
  t::FillMatrix(m, n0, n1);
  const auto v = m.ToVector();
  for (std::int64_t i = 0; i < n0; ++i) {
    for (std::int64_t j = 0; j < n1; ++j) {
      ASSERT_EQ(v[static_cast<std::size_t>(i * n1 + j)], 100 * i + j);
    }
  }
}

TEST(ExecKernels, TeamRowSumsWithScratch) {
  const std::int64_t n0 = 9, n1 = 13;
  DeviceBuffer<std::int64_t> m(static_cast<std::size_t>(n0 * n1));
  t::FillMatrix(m, n0, n1);
  DeviceBuffer<std::int64_t> sums(static_cast<std::size_t>(n0));
  t::RowSumsTeam(m, n0, n1, sums);
  const auto s = sums.ToVector();
  for (std::int64_t i = 0; i < n0; ++i) {
    ASSERT_EQ(s[static_cast<std::size_t>(i)], 100 * i * n1 + n1 * (n1 - 1) / 2);
  }
}

TEST(ExecKernels, TeamForVisitsEveryLeagueIndexOnce) {
  for (const bool dynamic : {false, true}) {
    DeviceBuffer<std::int64_t> counts(1000);
    t::VisitLeague(counts, dynamic);
    for (const std::int64_t c : counts.ToVector()) ASSERT_EQ(c, 1);
  }
}

TEST(ExecKernels, ResolveScratchLevelBoundaries) {
  const std::size_t max0 = t::ScratchMax(0);
  const std::size_t max1 = t::ScratchMax(1);
  ASSERT_GT(max1, max0);
  EXPECT_EQ(t::ResolveLevel(0, 0), 0);
  EXPECT_EQ(t::ResolveLevel(0, max0), 0);
  EXPECT_EQ(t::ResolveLevel(0, max0 + 1), 1);
  EXPECT_EQ(t::ResolveLevel(0, max1), 1);
  EXPECT_THROW(t::ResolveLevel(0, max1 + 1), std::exception);
  EXPECT_EQ(t::ResolveLevel(1, 16), 0);
  EXPECT_THROW(t::ResolveLevel(1, max0 + 1), std::exception);
  EXPECT_EQ(t::ResolveLevel(2, 16), 1);
}

TEST(ExecKernels, TeamForWithLevel0AndLevel1Scratch) {
  const std::int64_t n0 = 9, n1 = 13;
  DeviceBuffer<std::int64_t> m(static_cast<std::size_t>(n0 * n1));
  t::FillMatrix(m, n0, n1);
  for (const int kind : {1, 2}) {
    DeviceBuffer<std::int64_t> sums(static_cast<std::size_t>(n0));
    t::RowSumsTeamLevel(m, n0, n1, sums, kind);
    const auto s = sums.ToVector();
    for (std::int64_t i = 0; i < n0; ++i) {
      ASSERT_EQ(s[static_cast<std::size_t>(i)],
                100 * i * n1 + n1 * (n1 - 1) / 2);
    }
  }
}

TEST(ExecKernels, TeamReduceMatchesSerialSumBitwise) {
  for (const bool dynamic : {false, true}) {
    for (const int n : {0, 1, 777}) {
      double expected = 0.0;
      for (int i = 0; i < n; ++i) expected += static_cast<double>(i) + 0.5;
      EXPECT_EQ(t::TeamReduceHalfIntegers(n, dynamic), expected);
    }
  }
}

TEST(ExecKernels, PodBufferInstantiatedInKernelTu) {
  DeviceBuffer<t::Pod> p(5, "pods");
  t::FillPods(p);
  const auto v = p.ToVector();
  ASSERT_EQ(v.size(), 5u);
  for (std::size_t k = 0; k < 5; ++k) {
    const int i = static_cast<int>(k);
    EXPECT_EQ(v[k].a, static_cast<double>(i));
    EXPECT_EQ(v[k].b, i);
    EXPECT_EQ(v[k].c, -i);
  }
}

}  // namespace
