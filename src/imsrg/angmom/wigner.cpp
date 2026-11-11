// Wigner 3j/6j/9j symbols and Clebsch-Gordan coefficients (see wigner.hpp).
//
// All internal arithmetic is on doubled integers; every factorial argument below is an exact
// integer by the parity checks done before. Racah's single-sum formulas are evaluated with the
// square roots taken factor by factor (so no product of factorials ever overflows) and the
// summands formed as a ratio of factorials divided one denominator group at a time.

#include "imsrg/angmom/wigner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace imsrg::angmom {

namespace {

constexpr int kMaxFactorial = 170;

constexpr std::array<double, kMaxFactorial + 1> make_factorials() {
  std::array<double, kMaxFactorial + 1> f{};
  f[0] = 1.0;
  for (int n = 1; n <= kMaxFactorial; ++n) {
    f[static_cast<std::size_t>(n)] = f[static_cast<std::size_t>(n - 1)] * n;
  }
  return f;
}

constexpr std::array<double, kMaxFactorial + 1> kFact = make_factorials();

/// n! from the table. Precondition: 0 <= n <= 170 (callers only pass non-negative integers).
inline double fact(int n) noexcept { return kFact[static_cast<std::size_t>(n)]; }

/// sqrt(n!).
inline double sqrt_fact(int n) noexcept { return std::sqrt(fact(n)); }

/// sqrt of the triangle coefficient (a! b! c!)/(s+1)!, with doubled arguments of a valid triad.
inline double sqrt_delta(int t1, int t2, int t3) noexcept {
  const int a = (t1 + t2 - t3) / 2;
  const int b = (t1 - t2 + t3) / 2;
  const int c = (-t1 + t2 + t3) / 2;
  const int s = (t1 + t2 + t3) / 2;
  return sqrt_fact(a) * sqrt_fact(b) * sqrt_fact(c) / sqrt_fact(s + 1);
}

/// (-1)^n for any integer n (including negative).
inline double sign_of(int n) noexcept { return (n % 2 == 0) ? 1.0 : -1.0; }

/// True if |m| <= j and j - m is an integer, for doubled arguments.
inline bool valid_projection(int tj, int tm) noexcept {
  return std::abs(tm) <= tj && (tj - tm) % 2 == 0;
}

/// Largest factorial argument that the 6j sum can reach, for the overflow guard.
inline bool within_table(int n) noexcept { return n <= kMaxFactorial; }

}  // namespace

double wigner_3j(TwoJ j1, TwoJ j2, TwoJ j3, TwoM m1, TwoM m2, TwoM m3) noexcept {
  const int t1 = j1.twice;
  const int t2 = j2.twice;
  const int t3 = j3.twice;
  const int u1 = m1.twice;
  const int u2 = m2.twice;
  const int u3 = m3.twice;
  if (!triangle(j1, j2, j3) || u1 + u2 + u3 != 0 || !valid_projection(t1, u1) ||
      !valid_projection(t2, u2) || !valid_projection(t3, u3)) {
    return 0.0;
  }
  if (!within_table((t1 + t2 + t3) / 2 + 1)) return 0.0;

  const int a = (t1 + t2 - t3) / 2;  // j1 + j2 - j3
  const int p1 = (t1 + u1) / 2;      // j1 + m1
  const int q1 = (t1 - u1) / 2;      // j1 - m1
  const int p2 = (t2 + u2) / 2;
  const int q2 = (t2 - u2) / 2;
  const int p3 = (t3 + u3) / 2;
  const int q3 = (t3 - u3) / 2;
  const int k1 = (t3 - t2 + u1) / 2;  // j3 - j2 + m1
  const int k2 = (t3 - t1 - u2) / 2;  // j3 - j1 - m2

  const int tmin = std::max({0, -k1, -k2});
  const int tmax = std::min({a, q1, p2});
  double sum = 0.0;
  for (int t = tmin; t <= tmax; ++t) {
    const double den1 = fact(t) * fact(t + k1) * fact(t + k2);
    const double den2 = fact(a - t) * fact(q1 - t) * fact(p2 - t);
    sum += sign_of(t) / den1 / den2;
  }
  const double pref = sqrt_delta(t1, t2, t3) * (sqrt_fact(p1) * sqrt_fact(q1)) *
                      (sqrt_fact(p2) * sqrt_fact(q2)) * (sqrt_fact(p3) * sqrt_fact(q3));
  return sign_of((t1 - t2 - u3) / 2) * pref * sum;
}

double clebsch_gordan(TwoJ j1, TwoM m1, TwoJ j2, TwoM m2, TwoJ J, TwoM M) noexcept {
  if (M.twice != m1.twice + m2.twice) return 0.0;
  const double w = wigner_3j(j1, j2, J, m1, m2, TwoM{-M.twice});
  if (w == 0.0) return 0.0;
  // j1 - j2 + M is an integer for a valid symbol.
  return sign_of((j1.twice - j2.twice + M.twice) / 2) *
         std::sqrt(static_cast<double>(J.twice + 1)) * w;
}

double wigner_6j(TwoJ j1, TwoJ j2, TwoJ j3, TwoJ j4, TwoJ j5, TwoJ j6) noexcept {
  const int t1 = j1.twice;
  const int t2 = j2.twice;
  const int t3 = j3.twice;
  const int t4 = j4.twice;
  const int t5 = j5.twice;
  const int t6 = j6.twice;
  if (!triangle(j1, j2, j3) || !triangle(j1, j5, j6) || !triangle(j4, j2, j6) ||
      !triangle(j4, j5, j3)) {
    return 0.0;
  }
  const int al1 = (t1 + t2 + t3) / 2;
  const int al2 = (t1 + t5 + t6) / 2;
  const int al3 = (t4 + t2 + t6) / 2;
  const int al4 = (t4 + t5 + t3) / 2;
  const int be1 = (t1 + t2 + t4 + t5) / 2;
  const int be2 = (t2 + t3 + t5 + t6) / 2;
  const int be3 = (t3 + t1 + t6 + t4) / 2;
  const int tmin = std::max({al1, al2, al3, al4});
  const int tmax = std::min({be1, be2, be3});
  if (!within_table(tmax + 1)) return 0.0;

  double sum = 0.0;
  for (int t = tmin; t <= tmax; ++t) {
    const double den1 = fact(t - al1) * fact(t - al2);
    const double den2 = fact(t - al3) * fact(t - al4);
    const double den3 = fact(be1 - t) * fact(be2 - t);
    sum += sign_of(t) * fact(t + 1) / den1 / den2 / den3 / fact(be3 - t);
  }
  const double pref = (sqrt_delta(t1, t2, t3) * sqrt_delta(t1, t5, t6)) *
                      (sqrt_delta(t4, t2, t6) * sqrt_delta(t4, t5, t3));
  return pref * sum;
}

double wigner_9j(TwoJ j11, TwoJ j12, TwoJ j13, TwoJ j21, TwoJ j22, TwoJ j23, TwoJ j31, TwoJ j32,
                 TwoJ j33) noexcept {
  if (!triangle(j11, j12, j13) || !triangle(j21, j22, j23) || !triangle(j31, j32, j33) ||
      !triangle(j11, j21, j31) || !triangle(j12, j22, j32) || !triangle(j13, j23, j33)) {
    return 0.0;
  }
  // x must satisfy the triangle conditions of all three 6j symbols (see header).
  const int lo = std::max({std::abs(j11.twice - j33.twice), std::abs(j21.twice - j32.twice),
                           std::abs(j12.twice - j23.twice)});
  const int hi = std::min({j11.twice + j33.twice, j21.twice + j32.twice, j12.twice + j23.twice});
  double sum = 0.0;
  for (int tx = lo; tx <= hi; ++tx) {
    if ((tx + j11.twice + j33.twice) % 2 != 0) continue;
    const TwoJ x{tx};
    const double s1 = wigner_6j(j11, j21, j31, j32, j33, x);
    if (s1 == 0.0) continue;
    const double s2 = wigner_6j(j12, j22, j32, j21, x, j23);
    if (s2 == 0.0) continue;
    const double s3 = wigner_6j(j13, j23, j33, x, j11, j12);
    sum += sign_of(tx) * (tx + 1) * s1 * s2 * s3;
  }
  return sum;
}

double norm_9j(TwoJ j11, TwoJ j12, TwoJ j13, TwoJ j21, TwoJ j22, TwoJ j23, TwoJ j31, TwoJ j32,
               TwoJ j33) noexcept {
  const double w = wigner_9j(j11, j12, j13, j21, j22, j23, j31, j32, j33);
  if (w == 0.0) return 0.0;
  const double f = static_cast<double>(j13.twice + 1) * static_cast<double>(j23.twice + 1) *
                   static_cast<double>(j31.twice + 1) * static_cast<double>(j32.twice + 1);
  return std::sqrt(f) * w;
}

}  // namespace imsrg::angmom
