#pragma once
/// @file
/// Small integer/combinatorial math helpers usable in host and device code (no exceptions, no
/// allocation). Factorials use successive multiplication from 1.0, so results are bitwise
/// identical to the lists of imsrg++ (`AngMom.cc:21-67`).

#include <cmath>

#include "imsrg/base/device.hpp"

namespace imsrg::math {

/// (-1)^x. Correct for negative x (parity of the integer). O(1).
IMSRG_INLINE_FUNCTION constexpr int phase(int x) noexcept { return (x % 2 == 0) ? 1 : -1; }

/// n! as a double by successive multiplication (0! = 1! = 1). Precondition: 0 <= n <= 170
/// (171! overflows double); n < 0 returns 1.0. O(n).
IMSRG_INLINE_FUNCTION constexpr double factorial(int n) noexcept {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

/// n!! = n (n-2) (n-4) ... by successive multiplication, with 0!! = 1!! = (-1)!! = 1.
/// Precondition: n >= -1 (smaller values return 1.0) and n small enough not to overflow. O(n).
IMSRG_INLINE_FUNCTION constexpr double double_factorial(int n) noexcept {
  double f = 1.0;
  for (int i = (n % 2 == 0) ? 2 : 3; i <= n; i += 2) f *= i;
  return f;
}

/// Binomial coefficient C(n, k) as a double; 0 outside 0 <= k <= n. Computed multiplicatively
/// (exact while the intermediate values stay below 2^53). O(min(k, n-k)).
IMSRG_INLINE_FUNCTION constexpr double binomial(int n, int k) noexcept {
  if (k < 0 || n < 0 || k > n) return 0.0;
  if (k > n - k) k = n - k;
  double r = 1.0;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

/// ln(n!) via lgamma. Host only. Precondition: n >= 0. Relative accuracy about 1e-15.
inline double log_factorial(int n) { return std::lgamma(n + 1.0); }

}  // namespace imsrg::math
