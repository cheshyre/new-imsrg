// Private, host-only double-double (dd) arithmetic: a value is the unevaluated sum hi + lo with
// |lo| <= ulp(hi)/2, giving about 106 bits (~32 decimal digits). Built from the classic exact
// error-free transformations (Knuth two_sum, fma-based two_prod). Requires strict IEEE
// evaluation: do not compile with -ffast-math. Not part of the public API.
#pragma once

#include <cmath>

namespace imsrg::angmom::detail {

/// Unevaluated sum hi + lo (normalised: |lo| <= ulp(hi)/2).
struct DD {
  double hi = 0.0;
  double lo = 0.0;
};

/// Exact a + b = s + e with s = fl(a + b) (Knuth, no ordering precondition).
inline DD two_sum(double a, double b) noexcept {
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

/// Exact a + b = s + e, valid when |a| >= |b| or a == 0.
inline DD quick_two_sum(double a, double b) noexcept {
  const double s = a + b;
  return {s, b - (s - a)};
}

/// Exact a * b = p + e with p = fl(a * b), using one fused multiply-add.
inline DD two_prod(double a, double b) noexcept {
  const double p = a * b;
  return {p, std::fma(a, b, -p)};
}

/// Promotes a double (exact).
inline DD dd(double x) noexcept { return {x, 0.0}; }

/// Negation (exact).
inline DD operator-(DD a) noexcept { return {-a.hi, -a.lo}; }

/// Accurate dd addition (error about 2^-104 relative to |a| + |b|).
inline DD operator+(DD a, DD b) noexcept {
  DD s = two_sum(a.hi, b.hi);
  const DD t = two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

/// dd subtraction.
inline DD operator-(DD a, DD b) noexcept { return a + (-b); }

/// dd multiplication (relative error about 2^-104).
inline DD operator*(DD a, DD b) noexcept {
  DD p = two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return quick_two_sum(p.hi, p.lo);
}

/// dd division by long division: three quotient doubles with dd remainders. b must be nonzero.
inline DD operator/(DD a, DD b) noexcept {
  const double q1 = a.hi / b.hi;
  DD r = a - b * dd(q1);
  const double q2 = r.hi / b.hi;
  r = r - b * dd(q2);
  const double q3 = r.hi / b.hi;
  const DD q = quick_two_sum(q1, q2);
  return q + dd(q3);
}

/// Square root: s = sqrt(hi), then one dd Newton step s + (x - s^2) / (2 s). Returns 0 for x <= 0.
inline DD dd_sqrt(DD x) noexcept {
  if (x.hi <= 0.0) {
    return {};
  }
  const double s = std::sqrt(x.hi);
  const DD S = dd(s);
  return S + (x - S * S) / dd(2.0 * s);
}

/// Scales by 2^k exactly (barring over/underflow).
inline DD dd_ldexp(DD a, int k) noexcept { return {std::ldexp(a.hi, k), std::ldexp(a.lo, k)}; }

/// Rounds to the nearest double.
inline double to_double(DD a) noexcept { return a.hi + a.lo; }

}  // namespace imsrg::angmom::detail
