#include "imsrg/exec/device_buffer.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"

namespace {

using imsrg::exec::DeviceBuffer;

TEST(ExecDeviceBuffer, DefaultIsEmpty) {
  DeviceBuffer<double> b;
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(b.size(), 0u);
  EXPECT_EQ(b.label(), "imsrg::DeviceBuffer");
  EXPECT_EQ(b.device_data(), nullptr);
  EXPECT_TRUE(b.host_span().empty());
  EXPECT_TRUE(b.to_vector().empty());
}

TEST(ExecDeviceBuffer, ZeroInitialized) {
  DeviceBuffer<double> b(17, "zeros");
  EXPECT_EQ(b.size(), 17u);
  EXPECT_EQ(b.label(), "zeros");
  for (double x : b.to_vector()) {
    EXPECT_EQ(x, 0.0);
  }
}

TEST(ExecDeviceBuffer, FillSetsAllElements) {
  DeviceBuffer<double> b(33);
  b.fill(2.5);
  for (double x : b.to_vector()) {
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
  for (auto x : c.to_vector()) {
    EXPECT_EQ(x, 7);
  }
  DeviceBuffer<std::int32_t> d;
  d = c;
  c.fill(3);
  for (auto x : d.to_vector()) {
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
  EXPECT_EQ(c.to_vector()[4], 4.0);
}

TEST(ExecDeviceBuffer, CopyFromAndToHost) {
  std::vector<double> src{1, 2, 3, 4};
  DeviceBuffer<double> b(4);
  b.copy_from_host(src);
  std::vector<double> dst(4, -1.0);
  b.copy_to_host(dst);
  EXPECT_EQ(dst, src);
  EXPECT_EQ(b.to_vector(), src);
}

TEST(ExecDeviceBuffer, CopySizeMismatchThrows) {
  DeviceBuffer<double> b(4);
  std::vector<double> wrong(3);
  EXPECT_THROW(b.copy_from_host(wrong), imsrg::InvariantError);
  EXPECT_THROW(b.copy_to_host(wrong), imsrg::InvariantError);
}

TEST(ExecDeviceBuffer, ResizeKeepsPrefix) {
  DeviceBuffer<std::int64_t> b(4);
  b.copy_from_host(std::vector<std::int64_t>{1, 2, 3, 4});
  b.resize(6);
  EXPECT_EQ(b.to_vector(), (std::vector<std::int64_t>{1, 2, 3, 4, 0, 0}));
  b.resize(2);
  EXPECT_EQ(b.to_vector(), (std::vector<std::int64_t>{1, 2}));
  b.resize(0);
  EXPECT_TRUE(b.empty());
  b.resize(3);
  EXPECT_EQ(b.to_vector(), (std::vector<std::int64_t>{0, 0, 0}));
}

TEST(ExecDeviceBuffer, HostSpanSyncRoundTrip) {
  DeviceBuffer<double> b(6);
  auto h = b.host_span();
  ASSERT_EQ(h.size(), 6u);
  for (std::size_t i = 0; i < h.size(); ++i) {
    h[i] = 10.0 + static_cast<double>(i);
  }
  b.sync_to_device();
  b.fill(0.0);  // device now zero
  for (double x : b.to_vector()) {
    EXPECT_EQ(x, 0.0);
  }
  b.sync_to_host();
  for (double x : b.host_span()) {
    EXPECT_EQ(x, 0.0);
  }
  b.host_span()[2] = 5.0;
  b.sync_to_device();
  EXPECT_EQ(b.to_vector()[2], 5.0);
  const DeviceBuffer<double>& cb = b;
  EXPECT_EQ(cb.host_span().size(), 6u);
}

TEST(ExecDeviceBuffer, HostIsDeviceOnHostBackends) {
  DeviceBuffer<double> b(2);
  EXPECT_EQ(b.host_is_device(), b.host_span().data() == b.device_data());
}

template <class T>
void exercise_type() {
  DeviceBuffer<T> b(9, "typed");
  b.fill(static_cast<T>(3));
  std::vector<T> v = b.to_vector();
  ASSERT_EQ(v.size(), 9u);
  for (T x : v) {
    EXPECT_EQ(x, static_cast<T>(3));
  }
  b.resize(11);
  EXPECT_EQ(b.to_vector()[10], static_cast<T>(0));
}

TEST(ExecDeviceBuffer, AllInstantiatedTypes) {
  exercise_type<double>();
  exercise_type<float>();
  exercise_type<std::int32_t>();
  exercise_type<std::int64_t>();
  exercise_type<std::uint16_t>();
  exercise_type<std::uint8_t>();
}

}  // namespace
