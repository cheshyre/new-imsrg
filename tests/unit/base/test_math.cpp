#include <bit>
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "imsrg/base/math.hpp"

namespace {

namespace m = imsrg::math;

static_assert(m::phase(0) == 1 && m::phase(1) == -1 && m::phase(-1) == -1 && m::phase(-2) == 1);
static_assert(m::factorial(5) == 120.0);
static_assert(m::double_factorial(6) == 48.0);
static_assert(m::binomial(5, 2) == 10.0);

TEST(Math, Phase) {
  for (int x = -20; x <= 20; ++x) EXPECT_EQ(m::phase(x), (std::abs(x) % 2 == 0) ? 1 : -1) << x;
}

TEST(Math, FactorialExactUpTo20) {
  std::uint64_t p = 1;
  for (int n = 0; n <= 20; ++n) {
    if (n > 0) p *= static_cast<std::uint64_t>(n);
    EXPECT_EQ(m::factorial(n), static_cast<double>(p)) << n;
  }
}

TEST(Math, FactorialMatchesSuccessiveMultiplication) {
  double f = 1.0;
  for (int n = 0; n <= 170; ++n) {
    if (n > 1) f *= n;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(m::factorial(n)), std::bit_cast<std::uint64_t>(f)) << n;
  }
  EXPECT_TRUE(std::isfinite(m::factorial(170)));
}

TEST(Math, DoubleFactorial) {
  EXPECT_EQ(m::double_factorial(-1), 1.0);
  EXPECT_EQ(m::double_factorial(0), 1.0);
  EXPECT_EQ(m::double_factorial(1), 1.0);
  EXPECT_EQ(m::double_factorial(2), 2.0);
  EXPECT_EQ(m::double_factorial(3), 3.0);
  EXPECT_EQ(m::double_factorial(6), 48.0);
  EXPECT_EQ(m::double_factorial(7), 105.0);
  EXPECT_EQ(m::double_factorial(9), 945.0);
  // n!! = n (n-2)!!
  for (int n = 2; n <= 60; ++n) EXPECT_EQ(m::double_factorial(n), n * m::double_factorial(n - 2));
}

TEST(Math, Binomial) {
  EXPECT_EQ(m::binomial(0, 0), 1.0);
  EXPECT_EQ(m::binomial(5, 0), 1.0);
  EXPECT_EQ(m::binomial(5, 5), 1.0);
  EXPECT_EQ(m::binomial(5, 2), 10.0);
  EXPECT_EQ(m::binomial(10, 3), 120.0);
  EXPECT_EQ(m::binomial(52, 5), 2598960.0);
  EXPECT_EQ(m::binomial(5, -1), 0.0);
  EXPECT_EQ(m::binomial(5, 6), 0.0);
  EXPECT_EQ(m::binomial(-1, 0), 0.0);
  for (int n = 1; n <= 30; ++n) {
    for (int k = 1; k < n; ++k) {
      EXPECT_EQ(m::binomial(n, k), m::binomial(n - 1, k - 1) + m::binomial(n - 1, k));
    }
  }
}

TEST(Math, LogFactorial) {
  EXPECT_EQ(m::log_factorial(0), 0.0);
  for (int n = 1; n <= 170; ++n) {
    const double ref = std::log(m::factorial(n));
    EXPECT_NEAR(m::log_factorial(n), ref, 1e-14 * std::abs(ref)) << n;
  }
}

}  // namespace
