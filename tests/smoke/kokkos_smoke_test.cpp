/// @file
/// Smoke test of the Kokkos installation the build uses (tools/bootstrap_deps.sh or the in-tree
/// extern/kokkos fallback): the default execution space is the backend selected by
/// IMSRG_KOKKOS_BACKEND, and parallel_for/parallel_reduce give exact results on it.
/// Uses Kokkos directly because the imsrg::exec facade arrives in M2.

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>

namespace {

TEST(KokkosSmoke, DefaultExecutionSpaceIsConfiguredBackend) {
  const std::string space = Kokkos::DefaultExecutionSpace::name();
  fmt::print("Kokkos {}.{}.{}: default execution space {} with concurrency {}\n",
             KOKKOS_VERSION / 10000, KOKKOS_VERSION / 100 % 100, KOKKOS_VERSION % 100, space,
             Kokkos::DefaultExecutionSpace().concurrency());
  EXPECT_EQ(space, IMSRG_EXPECTED_KOKKOS_SPACE);
  EXPECT_GE(Kokkos::DefaultExecutionSpace().concurrency(), 1);
}

// Sum of 0..n-1 in 64-bit integers: exact, so any lost or duplicated iteration shows up.
TEST(KokkosSmoke, ParallelReduceSumsIndices) {
  constexpr std::int64_t n = 1'000'003;
  std::int64_t sum = 0;
  Kokkos::parallel_reduce(
      "imsrg::smoke::sum_indices", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, n),
      KOKKOS_LAMBDA(const std::int64_t i, std::int64_t& partial) { partial += i; }, sum);
  EXPECT_EQ(sum, n * (n - 1) / 2);
}

// Fill a View in the default memory space, reduce it there, and check a host mirror.
TEST(KokkosSmoke, ParallelForThenReduceOverView) {
  constexpr std::int64_t n = 4096;
  Kokkos::View<double*> values("imsrg::smoke::values", n);
  Kokkos::parallel_for(
      "imsrg::smoke::fill", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, n),
      KOKKOS_LAMBDA(const std::int64_t i) { values(i) = 0.5 * static_cast<double>(i); });
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "imsrg::smoke::sum_values", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, n),
      KOKKOS_LAMBDA(const std::int64_t i, double& partial) { partial += values(i); }, sum);
  // Every partial sum is a multiple of 0.5 below 2^53, so the result is exact in any order.
  EXPECT_EQ(sum, 0.25 * static_cast<double>(n) * static_cast<double>(n - 1));

  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  EXPECT_EQ(host(n - 1), 0.5 * static_cast<double>(n - 1));
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::ScopeGuard kokkos(argc, argv);
  return RUN_ALL_TESTS();
}
