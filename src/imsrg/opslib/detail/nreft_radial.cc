// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/opslib/detail/nreft_radial.h"

#include <cmath>
#include <cstdint>

#include "imsrg/base/assert.h"

namespace imsrg::opslib::detail {

namespace {

constexpr double kSqrtPi = 1.7724538509055160273;

double MinusOnePow(std::int32_t k) { return (k % 2 == 0) ? 1.0 : -1.0; }

/// sqrt((na-1)! (nb-1)! Gamma(na+la+1/2) Gamma(nb+lb+1/2)) with the shifted
/// radial numbers na, nb >= 1 (the common normalization of the three
/// integrals).
double Normalization(std::int32_t na, std::int32_t la, std::int32_t nb,
                     std::int32_t lb) {
  return std::sqrt(Factorial(nb - 1) * Factorial(na - 1) *
                   GammaOfHalf(2 * (na + la) + 1) *
                   GammaOfHalf(2 * (nb + lb) + 1));
}

/// The shared double sum of `jdmho` and `jdpho`; `second` is the coefficient
/// of the second 1F1 term: 2k for jdmho, 2 lb + 2k + 1 for jdpho.
template <class Second>
double DerivativeIntegral(std::int32_t na0, std::int32_t la, std::int32_t nb0,
                          std::int32_t lb, std::int32_t L, double y,
                          Second second) {
  const std::int32_t na = na0 + 1;
  const std::int32_t nb = nb0 + 1;
  const double f1 = std::pow(2.0, L - 1) / DoubleFactorial(2 * L + 1) *
                    std::pow(y, 0.5 * (L - 1)) * std::exp(-y) *
                    Normalization(na, la, nb, lb);
  const double c = L + 1.5;
  double sum = 0.0;
  for (std::int32_t k = 0; k < nb; ++k) {
    const double f2 = MinusOnePow(k) / Factorial(k) / Factorial(nb - 1 - k) /
                      GammaOfHalf(2 * lb + 2 * k + 3);
    double f3 = 0.0;
    for (std::int32_t k1 = 0; k1 < na; ++k1) {
      const std::int32_t s = L + lb + la + 2 * k + 2 * k1 + 2;  // even
      // 1F1 first arguments (L - lb - la - 2k - 2k1 -/+ 1) / 2 (integers).
      const std::int32_t a_minus = (L - lb - la - 2 * k - 2 * k1 - 1) / 2;
      const std::int32_t a_plus = a_minus + 1;
      f3 +=
          MinusOnePow(k1) / Factorial(k1) / Factorial(na - 1 - k1) *
          GammaOfHalf(s) / GammaOfHalf(2 * la + 2 * k1 + 3) *
          (-s * 0.5 * Hyp1F1(a_minus, c, y) + second(k) * Hyp1F1(a_plus, c, y));
    }
    sum += f2 * f3;
  }
  return f1 * sum;
}

}  // namespace

double Factorial(std::int32_t n) {
  IMSRG_ASSERT(n >= 0 && n <= 170);
  double f = 1.0;
  for (std::int32_t i = 2; i <= n; ++i) f *= i;
  return f;
}

double DoubleFactorial(std::int32_t n) {
  IMSRG_ASSERT(n >= -1 && n <= 300);
  double f = 1.0;
  for (std::int32_t i = n; i > 1; i -= 2) f *= i;
  return f;
}

double GammaOfHalf(std::int32_t twice_x) {
  IMSRG_ASSERT(twice_x >= 1 && twice_x <= 340);
  if (twice_x % 2 == 0) return Factorial(twice_x / 2 - 1);
  // x = m + 1/2: Gamma = (2m-1)!! sqrt(pi) / 2^m.
  const std::int32_t m = (twice_x - 1) / 2;
  return DoubleFactorial(2 * m - 1) * kSqrtPi / std::ldexp(1.0, m);
}

double Hyp1F1(std::int32_t a, double c, double y) {
  double term = 1.0;
  double sum = 1.0;
  if (a <= 0) {
    for (std::int32_t k = 0; k < -a; ++k) {
      term *= (a + k) / (c + k) * y / (k + 1);
      sum += term;
    }
    return sum;
  }
  for (std::int32_t k = 0; k < 2000; ++k) {
    term *= (a + k) / (c + k) * y / (k + 1);
    sum += term;
    if (term < 1.0e-17 * sum) break;
  }
  return sum;
}

double Jho(std::int32_t na0, std::int32_t la, std::int32_t nb0, std::int32_t lb,
           std::int32_t L, double y) {
  IMSRG_ASSERT((la + lb + L) % 2 == 0);
  const std::int32_t na = na0 + 1;
  const std::int32_t nb = nb0 + 1;
  const double f1 = std::pow(2.0, L) / DoubleFactorial(2 * L + 1) *
                    std::pow(y, 0.5 * L) * std::exp(-y) *
                    Normalization(na, la, nb, lb);
  const double c = L + 1.5;
  double sum = 0.0;
  for (std::int32_t k = 0; k < nb; ++k) {
    const double f2 = MinusOnePow(k) / Factorial(k) / Factorial(nb - 1 - k) /
                      GammaOfHalf(2 * lb + 2 * k + 3);
    double f3 = 0.0;
    for (std::int32_t k1 = 0; k1 < na; ++k1) {
      f3 += MinusOnePow(k1) / Factorial(k1) / Factorial(na - 1 - k1) *
            GammaOfHalf(L + lb + la + 2 * k + 2 * k1 + 3) /
            GammaOfHalf(2 * la + 2 * k1 + 3) *
            Hyp1F1((L - lb - la - 2 * k - 2 * k1) / 2, c, y);
    }
    sum += f2 * f3;
  }
  return f1 * sum;
}

double Jdmho(std::int32_t na, std::int32_t la, std::int32_t nb, std::int32_t lb,
             std::int32_t L, double y) {
  IMSRG_ASSERT((la + lb + L) % 2 == 1);
  return DerivativeIntegral(na, la, nb, lb, L, y,
                            [](std::int32_t k) { return 2.0 * k; });
}

double Jdpho(std::int32_t na, std::int32_t la, std::int32_t nb, std::int32_t lb,
             std::int32_t L, double y) {
  IMSRG_ASSERT((la + lb + L) % 2 == 1);
  return DerivativeIntegral(na, la, nb, lb, L, y, [lb](std::int32_t k) {
    return 2.0 * lb + 2.0 * k + 1.0;
  });
}

}  // namespace imsrg::opslib::detail
