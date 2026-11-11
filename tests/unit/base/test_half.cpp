#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "imsrg/base/half.hpp"

namespace {

using imsrg::float_to_half;
using imsrg::half_to_float;

static_assert(half_to_float(0x3C00) == 1.0f);
static_assert(float_to_half(1.0f) == 0x3C00);

// Independent decoder of non-NaN, non-inf half bit patterns using ldexp.
double decode(std::uint16_t h) {
  const int s = (h >> 15) ? -1 : 1;
  const int e = (h >> 10) & 0x1F;
  const int f = h & 0x3FF;
  if (e == 0) return s * std::ldexp(static_cast<double>(f), -24);
  return s * std::ldexp(1024.0 + f, e - 25);
}

TEST(Half, AllPatternsDecodeExactly) {
  for (std::uint32_t i = 0; i < 65536; ++i) {
    const auto h = static_cast<std::uint16_t>(i);
    const float v = half_to_float(h);
    const int e = (h >> 10) & 0x1F;
    const int f = h & 0x3FF;
    if (e == 0x1F && f != 0) {
      EXPECT_TRUE(std::isnan(v)) << i;
      EXPECT_EQ(std::signbit(v), (h >> 15) != 0) << i;
    } else if (e == 0x1F) {
      EXPECT_TRUE(std::isinf(v)) << i;
      EXPECT_EQ(std::signbit(v), (h >> 15) != 0) << i;
    } else {
      ASSERT_EQ(static_cast<double>(v), decode(h)) << i;
      EXPECT_EQ(std::signbit(v), (h >> 15) != 0) << i;  // also for -0
    }
  }
}

TEST(Half, RoundTripAllNonNaN) {
  for (std::uint32_t i = 0; i < 65536; ++i) {
    const auto h = static_cast<std::uint16_t>(i);
    if (std::isnan(half_to_float(h))) continue;
    ASSERT_EQ(float_to_half(half_to_float(h)), h) << i;
  }
}

TEST(Half, NaNMapsToQuietNaN) {
  const std::uint16_t h = float_to_half(std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(h & 0x7C00, 0x7C00);
  EXPECT_NE(h & 0x03FF, 0);
  EXPECT_TRUE(std::isnan(half_to_float(h)));
  const float neg_nan = std::bit_cast<float>(std::uint32_t{0xFFC00001u});
  EXPECT_EQ(float_to_half(neg_nan) & 0x8000, 0x8000);
}

TEST(Half, RoundingAndOverflow) {
  // Largest finite half and the overflow boundary (65520 is a tie to even -> inf).
  EXPECT_EQ(float_to_half(65504.0f), 0x7BFF);
  EXPECT_EQ(float_to_half(65519.0f), 0x7BFF);
  EXPECT_EQ(float_to_half(65520.0f), 0x7C00);
  EXPECT_EQ(float_to_half(1e10f), 0x7C00);
  EXPECT_EQ(float_to_half(-1e10f), 0xFC00);
  EXPECT_EQ(float_to_half(std::numeric_limits<float>::infinity()), 0x7C00);
  // Ties to even in the normal range: 1 + 2^-11 is halfway between 0x3C00 and 0x3C01.
  EXPECT_EQ(float_to_half(1.0f + std::ldexp(1.0f, -11)), 0x3C00);
  EXPECT_EQ(float_to_half(1.0f + 3 * std::ldexp(1.0f, -11)), 0x3C02);
  EXPECT_EQ(float_to_half(1.0f + std::ldexp(1.0f, -11) + std::ldexp(1.0f, -20)), 0x3C01);
  // Subnormals: smallest is 2^-24; 2^-25 is a tie to 0; just above rounds up to 2^-24.
  EXPECT_EQ(float_to_half(std::ldexp(1.0f, -24)), 0x0001);
  EXPECT_EQ(float_to_half(std::ldexp(1.0f, -25)), 0x0000);
  EXPECT_EQ(float_to_half(std::ldexp(1.5f, -25)), 0x0001);
  EXPECT_EQ(float_to_half(std::ldexp(1.0f, -26)), 0x0000);
  EXPECT_EQ(float_to_half(std::ldexp(3.0f, -25)), 0x0002);  // 1.5 units -> tie to even (2)
  EXPECT_EQ(float_to_half(-std::ldexp(1.0f, -30)), 0x8000);
  // Rounding up out of the subnormal range reaches the smallest normal.
  EXPECT_EQ(float_to_half(std::nextafterf(std::ldexp(1.0f, -14), 0.0f)), 0x0400);
  EXPECT_EQ(float_to_half(0.0f), 0x0000);
  EXPECT_EQ(float_to_half(-0.0f), 0x8000);
}

TEST(Half, MatchesNearestForRandomFloats) {
  // Compare against the nearest of the two neighbouring half values (ties excluded by
  // construction: the sampled floats are not exactly halfway).
  std::uint32_t state = 12345u;
  for (int n = 0; n < 200000; ++n) {
    state = state * 1664525u + 1013904223u;
    const float x = std::ldexp(static_cast<float>(state >> 8) / 16777216.0f - 0.5f,
                               static_cast<int>(state & 31u) - 20);
    const std::uint16_t h = float_to_half(x);
    const double err = std::abs(static_cast<double>(half_to_float(h)) - static_cast<double>(x));
    for (int d : {-1, 1}) {
      const std::uint16_t nb = static_cast<std::uint16_t>(h + d);
      if (std::isnan(half_to_float(nb)) || std::isinf(half_to_float(nb))) continue;
      if ((nb ^ h) & 0x8000) continue;
      ASSERT_LE(err, std::abs(static_cast<double>(half_to_float(nb)) - static_cast<double>(x)))
          << x;
    }
  }
}

}  // namespace
