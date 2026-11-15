// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <bit>
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "imsrg/base/math.h"

namespace {

namespace m = imsrg::math;

static_assert(m::Phase(0) == 1 && m::Phase(1) == -1 && m::Phase(-1) == -1 &&
              m::Phase(-2) == 1);
static_assert(m::Factorial(5) == 120.0);
static_assert(m::DoubleFactorial(6) == 48.0);
static_assert(m::Binomial(5, 2) == 10.0);

TEST(Math, Phase) {
  for (int x = -20; x <= 20; ++x)
    EXPECT_EQ(m::Phase(x), (std::abs(x) % 2 == 0) ? 1 : -1) << x;
}

TEST(Math, FactorialExactUpTo20) {
  std::uint64_t p = 1;
  for (int n = 0; n <= 20; ++n) {
    if (n > 0) p *= static_cast<std::uint64_t>(n);
    EXPECT_EQ(m::Factorial(n), static_cast<double>(p)) << n;
  }
}

TEST(Math, FactorialMatchesSuccessiveMultiplication) {
  double f = 1.0;
  for (int n = 0; n <= 170; ++n) {
    if (n > 1) f *= n;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(m::Factorial(n)),
              std::bit_cast<std::uint64_t>(f))
        << n;
  }
  EXPECT_TRUE(std::isfinite(m::Factorial(170)));
}

TEST(Math, DoubleFactorial) {
  EXPECT_EQ(m::DoubleFactorial(-1), 1.0);
  EXPECT_EQ(m::DoubleFactorial(0), 1.0);
  EXPECT_EQ(m::DoubleFactorial(1), 1.0);
  EXPECT_EQ(m::DoubleFactorial(2), 2.0);
  EXPECT_EQ(m::DoubleFactorial(3), 3.0);
  EXPECT_EQ(m::DoubleFactorial(6), 48.0);
  EXPECT_EQ(m::DoubleFactorial(7), 105.0);
  EXPECT_EQ(m::DoubleFactorial(9), 945.0);
  // n!! = n (n-2)!!
  for (int n = 2; n <= 60; ++n)
    EXPECT_EQ(m::DoubleFactorial(n), n * m::DoubleFactorial(n - 2));
}

TEST(Math, Binomial) {
  EXPECT_EQ(m::Binomial(0, 0), 1.0);
  EXPECT_EQ(m::Binomial(5, 0), 1.0);
  EXPECT_EQ(m::Binomial(5, 5), 1.0);
  EXPECT_EQ(m::Binomial(5, 2), 10.0);
  EXPECT_EQ(m::Binomial(10, 3), 120.0);
  EXPECT_EQ(m::Binomial(52, 5), 2598960.0);
  EXPECT_EQ(m::Binomial(5, -1), 0.0);
  EXPECT_EQ(m::Binomial(5, 6), 0.0);
  EXPECT_EQ(m::Binomial(-1, 0), 0.0);
  for (int n = 1; n <= 30; ++n) {
    for (int k = 1; k < n; ++k) {
      EXPECT_EQ(m::Binomial(n, k),
                m::Binomial(n - 1, k - 1) + m::Binomial(n - 1, k));
    }
  }
}

TEST(Math, LogFactorial) {
  EXPECT_EQ(m::LogFactorial(0), 0.0);
  for (int n = 1; n <= 170; ++n) {
    const double ref = std::log(m::Factorial(n));
    EXPECT_NEAR(m::LogFactorial(n), ref, 1.0e-14 * std::abs(ref)) << n;
  }
}

}  // namespace
