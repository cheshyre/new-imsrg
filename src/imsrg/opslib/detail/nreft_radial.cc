// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The radial integrals evaluate the fork's alternating double sums in
// double-double arithmetic (about 32 significant digits): in double precision
// the sums lose up to 8 digits at emax 14 (terms up to 1e8 times the result
// for high radial n and large y), so a plain double port agrees with neither
// the exact value nor the fork (whose GSL 1F1 values are rounded differently).
// With double-double the integrals are exact to rounding of the final product.
#include "imsrg/opslib/detail/nreft_radial.h"

#include <cmath>
#include <cstdint>

#include "imsrg/base/assert.h"

namespace imsrg::opslib::detail {

namespace {

// ----- double-double arithmetic (Dekker / Knuth error-free transforms) -----

struct DD {
  double hi = 0.0;
  double lo = 0.0;
};

DD TwoSum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}
DD QuickTwoSum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}
DD TwoProd(double a, double b) {
  const double p = a * b;
  return {p, std::fma(a, b, -p)};
}
DD operator+(DD a, DD b) {
  DD s = TwoSum(a.hi, b.hi);
  const DD t = TwoSum(a.lo, b.lo);
  s.lo += t.hi;
  s = QuickTwoSum(s.hi, s.lo);
  s.lo += t.lo;
  return QuickTwoSum(s.hi, s.lo);
}
DD operator-(DD a) { return {-a.hi, -a.lo}; }
DD operator-(DD a, DD b) { return a + (-b); }
DD operator*(DD a, DD b) {
  DD p = TwoProd(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return QuickTwoSum(p.hi, p.lo);
}
DD operator*(DD a, double b) {
  DD p = TwoProd(a.hi, b);
  p.lo += a.lo * b;
  return QuickTwoSum(p.hi, p.lo);
}
DD operator/(DD a, DD b) {
  const double q1 = a.hi / b.hi;
  const DD r = a - b * q1;
  const double q2 = r.hi / b.hi;
  const DD r2 = r - b * q2;
  const double q3 = r2.hi / b.hi;
  return QuickTwoSum(q1, q2) + DD{q3, 0.0};
}

// sqrt(pi) to double-double precision (mpmath, 50 digits).
constexpr DD kSqrtPiDD{1.772453850905516, -7.666586499825799e-17};

DD FactorialDD(std::int32_t n) {
  DD f{1.0, 0.0};
  for (std::int32_t i = 2; i <= n; ++i) f = f * static_cast<double>(i);
  return f;
}

/// Gamma(twice_x / 2) in double-double.
DD GammaOfHalfDD(std::int32_t twice_x) {
  IMSRG_ASSERT(twice_x >= 1 && twice_x <= 340);
  if (twice_x % 2 == 0) return FactorialDD(twice_x / 2 - 1);
  const std::int32_t m = (twice_x - 1) / 2;
  DD f = kSqrtPiDD;
  for (std::int32_t i = 2 * m - 1; i > 1; i -= 2)
    f = f * static_cast<double>(i);
  return f * std::ldexp(1.0, -m);  // exact scaling
}

/// 1F1(a; c; y) for integer a and c = L + 3/2, in double-double. The term
/// ratio (a+k) y / ((c+k)(k+1)) is formed in double-double, so a terminating
/// alternating polynomial keeps its full relative precision as long as its
/// cancellation is below ~1e16.
DD Hyp1F1DD(std::int32_t a, double c, double y) {
  DD term{1.0, 0.0};
  DD sum{1.0, 0.0};
  const DD ydd{y, 0.0};
  const std::int32_t kmax = (a <= 0) ? -a : 4000;
  for (std::int32_t k = 0; k < kmax; ++k) {
    // c + k is exact in double (half-integers).
    term = term * ydd * static_cast<double>(a + k) /
           (DD{c + k, 0.0} * static_cast<double>(k + 1));
    sum = sum + term;
    if (a > 0 && term.hi < 1.0e-34 * sum.hi) break;
  }
  return sum;
}

/// sqrt((na-1)! (nb-1)! Gamma(na+la+1/2) Gamma(nb+lb+1/2)) with the shifted
/// radial numbers na, nb >= 1 (the common normalization of the integrals).
double Normalization(std::int32_t na, std::int32_t la, std::int32_t nb,
                     std::int32_t lb) {
  const DD p = FactorialDD(nb - 1) * FactorialDD(na - 1) *
               GammaOfHalfDD(2 * (na + la) + 1) *
               GammaOfHalfDD(2 * (nb + lb) + 1);
  return std::sqrt(p.hi + p.lo);
}

/// sign / (k! (n-1-k)!) as double-double.
DD BinomialWeight(std::int32_t k, std::int32_t n) {
  const DD w = DD{1.0, 0.0} / (FactorialDD(k) * FactorialDD(n - 1 - k));
  return (k % 2 == 0) ? w : -w;
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
  DD sum;
  for (std::int32_t k = 0; k < nb; ++k) {
    const DD f2 = BinomialWeight(k, nb) / GammaOfHalfDD(2 * lb + 2 * k + 3);
    DD f3;
    for (std::int32_t k1 = 0; k1 < na; ++k1) {
      const std::int32_t s = L + lb + la + 2 * k + 2 * k1 + 2;  // even
      // 1F1 first arguments (L - lb - la - 2k - 2k1 -/+ 1) / 2 (integers).
      const std::int32_t a_minus = (L - lb - la - 2 * k - 2 * k1 - 1) / 2;
      const DD bracket = Hyp1F1DD(a_minus, c, y) * (-0.5 * s) +
                         Hyp1F1DD(a_minus + 1, c, y) * second(k);
      f3 = f3 + BinomialWeight(k1, na) * GammaOfHalfDD(s) /
                    GammaOfHalfDD(2 * la + 2 * k1 + 3) * bracket;
    }
    sum = sum + f2 * f3;
  }
  return f1 * (sum.hi + sum.lo);
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
  const DD g = GammaOfHalfDD(twice_x);
  return g.hi + g.lo;
}

double Hyp1F1(std::int32_t a, double c, double y) {
  const DD h = Hyp1F1DD(a, c, y);
  return h.hi + h.lo;
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
  DD sum;
  for (std::int32_t k = 0; k < nb; ++k) {
    const DD f2 = BinomialWeight(k, nb) / GammaOfHalfDD(2 * lb + 2 * k + 3);
    DD f3;
    for (std::int32_t k1 = 0; k1 < na; ++k1) {
      f3 = f3 + BinomialWeight(k1, na) *
                    GammaOfHalfDD(L + lb + la + 2 * k + 2 * k1 + 3) /
                    GammaOfHalfDD(2 * la + 2 * k1 + 3) *
                    Hyp1F1DD((L - lb - la - 2 * k - 2 * k1) / 2, c, y);
    }
    sum = sum + f2 * f3;
  }
  return f1 * (sum.hi + sum.lo);
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
