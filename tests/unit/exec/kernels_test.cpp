#include <cstdint>
#include <limits>
#include <vector>

#include "kernels_impl.hpp"
#include <gtest/gtest.h>

namespace {

using imsrg::exec::DeviceBuffer;
namespace t = imsrg::exec::test;

TEST(ExecKernels, ParallelForWritesTwiceIndex) {
  DeviceBuffer<double> a(1000);
  t::fill_twice_index(a);
  const auto v = a.to_vector();
  for (std::size_t i = 0; i < v.size(); ++i) {
    ASSERT_EQ(v[i], 2.0 * static_cast<double>(i));
  }
}

TEST(ExecKernels, ReduceSumIsExact) {
  const std::int64_t n = 10000;
  std::vector<std::int64_t> h(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    h[static_cast<std::size_t>(i)] = i + 1;
  }
  DeviceBuffer<std::int64_t> b(h.size());
  b.copy_from_host(h);
  EXPECT_EQ(t::sum(b), n * (n + 1) / 2);
  EXPECT_EQ(t::sum(DeviceBuffer<std::int64_t>(0)), 0);
}

TEST(ExecKernels, ReduceMax) {
  std::vector<std::int64_t> h{3, -7, 42, 5, 41, -100};
  DeviceBuffer<std::int64_t> b(h.size());
  b.copy_from_host(h);
  EXPECT_EQ(t::max_value(b), 42);
  EXPECT_EQ(t::max_value(DeviceBuffer<std::int64_t>(0)),
            std::numeric_limits<std::int64_t>::lowest());
}

TEST(ExecKernels, MdRangeFillsMatrix) {
  const std::int64_t n0 = 7, n1 = 11;
  DeviceBuffer<std::int64_t> m(static_cast<std::size_t>(n0 * n1));
  t::fill_matrix(m, n0, n1);
  const auto v = m.to_vector();
  for (std::int64_t i = 0; i < n0; ++i) {
    for (std::int64_t j = 0; j < n1; ++j) {
      ASSERT_EQ(v[static_cast<std::size_t>(i * n1 + j)], 100 * i + j);
    }
  }
}

TEST(ExecKernels, TeamRowSumsWithScratch) {
  const std::int64_t n0 = 9, n1 = 13;
  DeviceBuffer<std::int64_t> m(static_cast<std::size_t>(n0 * n1));
  t::fill_matrix(m, n0, n1);
  DeviceBuffer<std::int64_t> sums(static_cast<std::size_t>(n0));
  t::row_sums_team(m, n0, n1, sums);
  const auto s = sums.to_vector();
  for (std::int64_t i = 0; i < n0; ++i) {
    ASSERT_EQ(s[static_cast<std::size_t>(i)], 100 * i * n1 + n1 * (n1 - 1) / 2);
  }
}

TEST(ExecKernels, PodBufferInstantiatedInKernelTu) {
  DeviceBuffer<t::Pod> p(5, "pods");
  t::fill_pods(p);
  const auto v = p.to_vector();
  ASSERT_EQ(v.size(), 5u);
  for (std::size_t k = 0; k < 5; ++k) {
    const int i = static_cast<int>(k);
    EXPECT_EQ(v[k].a, static_cast<double>(i));
    EXPECT_EQ(v[k].b, i);
    EXPECT_EQ(v[k].c, -i);
  }
}

}  // namespace
