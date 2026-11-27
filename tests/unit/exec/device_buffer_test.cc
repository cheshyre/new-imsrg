// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/exec/device_buffer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/exec/host_parallel.h"

namespace {

using imsrg::exec::DeviceBuffer;

TEST(ExecDeviceBuffer, DefaultIsEmpty) {
  DeviceBuffer<double> b;
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(b.size(), 0u);
  EXPECT_EQ(b.label(), "imsrg::DeviceBuffer");
  EXPECT_EQ(b.device_data(), nullptr);
  EXPECT_TRUE(b.host_span().empty());
  EXPECT_TRUE(b.ToVector().empty());
}

TEST(ExecDeviceBuffer, ZeroInitialized) {
  DeviceBuffer<double> b(17, "zeros");
  EXPECT_EQ(b.size(), 17u);
  EXPECT_EQ(b.label(), "zeros");
  for (double x : b.ToVector()) {
    EXPECT_EQ(x, 0.0);
  }
}

TEST(ExecDeviceBuffer, FillSetsAllElements) {
  DeviceBuffer<double> b(33);
  b.fill(2.5);
  for (double x : b.ToVector()) {
    EXPECT_EQ(x, 2.5);
  }
}

TEST(ExecDeviceBuffer, CopyIsDeepAndIndependent) {
  DeviceBuffer<std::int32_t> a(8, "orig");
  a.fill(7);
  DeviceBuffer<std::int32_t> c(a);
  EXPECT_EQ(c.label(), "orig");
  EXPECT_NE(c.device_data(), a.device_data());
  a.fill(1);
  for (auto x : c.ToVector()) {
    EXPECT_EQ(x, 7);
  }
  DeviceBuffer<std::int32_t> d;
  d = c;
  c.fill(3);
  for (auto x : d.ToVector()) {
    EXPECT_EQ(x, 7);
  }
}

TEST(ExecDeviceBuffer, MoveLeavesSourceEmpty) {
  DeviceBuffer<double> a(5, "mv");
  a.fill(4.0);
  const double* p = a.device_data();
  DeviceBuffer<double> b(std::move(a));
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.size(), 5u);
  EXPECT_EQ(b.device_data(), p);
  DeviceBuffer<double> c;
  c = std::move(b);
  EXPECT_TRUE(b.empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(c.device_data(), p);
  EXPECT_EQ(c.ToVector()[4], 4.0);
}

TEST(ExecDeviceBuffer, CopyFromAndToHost) {
  std::vector<double> src{1, 2, 3, 4};
  DeviceBuffer<double> b(4);
  b.CopyFromHost(src);
  std::vector<double> dst(4, -1.0);
  b.CopyToHost(dst);
  EXPECT_EQ(dst, src);
  EXPECT_EQ(b.ToVector(), src);
}

TEST(ExecDeviceBuffer, CopySizeMismatchThrows) {
  DeviceBuffer<double> b(4);
  std::vector<double> wrong(3);
  EXPECT_THROW(b.CopyFromHost(wrong), imsrg::InvariantError);
  EXPECT_THROW(b.CopyToHost(wrong), imsrg::InvariantError);
}

TEST(ExecDeviceBuffer, ResizeKeepsPrefix) {
  DeviceBuffer<std::int64_t> b(4);
  b.CopyFromHost(std::vector<std::int64_t>{1, 2, 3, 4});
  b.resize(6);
  EXPECT_EQ(b.ToVector(), (std::vector<std::int64_t>{1, 2, 3, 4, 0, 0}));
  b.resize(2);
  EXPECT_EQ(b.ToVector(), (std::vector<std::int64_t>{1, 2}));
  b.resize(0);
  EXPECT_TRUE(b.empty());
  b.resize(3);
  EXPECT_EQ(b.ToVector(), (std::vector<std::int64_t>{0, 0, 0}));
}

// Empty buffers hold no allocation (so they may outlive the Runtime) and report
// no device data.
TEST(ExecDeviceBuffer, EmptyBuffersHoldNoAllocation) {
  DeviceBuffer<double> zero(0, "zero");
  EXPECT_TRUE(zero.empty());
  EXPECT_EQ(zero.device_data(), nullptr);
  EXPECT_TRUE(zero.host_span().empty());
  DeviceBuffer<double> shrunk(5);
  shrunk.resize(0);
  EXPECT_EQ(shrunk.device_data(), nullptr);
  DeviceBuffer<double> grown;
  grown.resize(2);
  EXPECT_EQ(grown.ToVector(), (std::vector<double>{0.0, 0.0}));
}

TEST(ExecDeviceBuffer, SelfAssignmentKeepsData) {
  DeviceBuffer<double> b(3);
  b.fill(2.5);
  const DeviceBuffer<double>& alias = b;
  b = alias;
  EXPECT_EQ(b.ToVector(), (std::vector<double>{2.5, 2.5, 2.5}));
}

TEST(ExecDeviceBuffer, HostSpanSyncRoundTrip) {
  DeviceBuffer<double> b(6);
  auto h = b.host_span();
  ASSERT_EQ(h.size(), 6u);
  for (std::size_t i = 0; i < h.size(); ++i) {
    h[i] = 10.0 + static_cast<double>(i);
  }
  b.SyncToDevice();
  b.fill(0.0);  // device now zero
  for (double x : b.ToVector()) {
    EXPECT_EQ(x, 0.0);
  }
  b.SyncToHost();
  for (double x : b.host_span()) {
    EXPECT_EQ(x, 0.0);
  }
  b.host_span()[2] = 5.0;
  b.SyncToDevice();
  EXPECT_EQ(b.ToVector()[2], 5.0);
  const DeviceBuffer<double>& cb = b;
  EXPECT_EQ(cb.host_span().size(), 6u);
}

TEST(ExecDeviceBuffer, HostIsDeviceOnHostBackends) {
  DeviceBuffer<double> b(2);
  EXPECT_EQ(b.host_is_device(), b.host_span().data() == b.device_data());
}

template <class T>
void ExerciseType() {
  DeviceBuffer<T> b(9, "typed");
  b.fill(static_cast<T>(3));
  std::vector<T> v = b.ToVector();
  ASSERT_EQ(v.size(), 9u);
  for (T x : v) {
    EXPECT_EQ(x, static_cast<T>(3));
  }
  b.resize(11);
  EXPECT_EQ(b.ToVector()[10], static_cast<T>(0));
}

TEST(ExecDeviceBuffer, AllInstantiatedTypes) {
  ExerciseType<double>();
  ExerciseType<float>();
  ExerciseType<std::int32_t>();
  ExerciseType<std::int64_t>();
  ExerciseType<std::uint16_t>();
  ExerciseType<std::uint8_t>();
}

// Kept in a free function (no kernel lambda in a TEST body, design/12 s5).
void ReadHostMirrorInParallel(const DeviceBuffer<double>& b) {
  imsrg::exec::HostParallelForDynamic(
      "test::mirror_in_region", 16,
      [&b](std::int64_t) { static_cast<void>(b.host_span()); });
}

// Creating the host mirror inside a host-parallel region is the deadlock of
// the HIP build with an OpenMP host (global fence under the instance mutex);
// the debug guard turns it into a loud failure on every backend.
TEST(ExecDeviceBufferDeathTest, MirrorCreationInParallelRegionIsCaught) {
  if (!IMSRG_ASSERTS_ENABLED || imsrg::exec::HostConcurrency() < 2) {
    GTEST_SKIP() << "needs assertions and an OpenMP host with >= 2 threads";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        const DeviceBuffer<double> b(8);
        ReadHostMirrorInParallel(b);
      },
      "");
}

TEST(ExecDeviceBuffer, MirrorCreatedBeforeTheLoopIsReadInParallel) {
  DeviceBuffer<double> b(8);
  b.fill(1.5);
  b.SyncToHost();  // creates the mirror outside the region
  ReadHostMirrorInParallel(b);
  EXPECT_EQ(b.host_span()[7], 1.5);
}

}  // namespace
