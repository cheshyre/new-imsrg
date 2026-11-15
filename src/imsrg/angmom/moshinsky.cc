// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Moshinsky brackets and Talmi coefficients: a port of imsrg++
// `AngMom::Moshinsky` (AngMom.cc:163-241), `AngMom::TalmiB` (AngMom.cc:246-261)
// and `imsrg_util::TalmiI`. The Buck et al. sums suffer heavy cancellation, so
// `moshinsky` is evaluated entirely in double-double arithmetic
// (detail/double_double.h) and rounded once at the end.
#include "imsrg/angmom/moshinsky.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>

#include "imsrg/angmom/detail/double_double.h"

namespace imsrg::angmom {

namespace detail {
// Double-returning wrappers over the dd building blocks, exposed for tests
// only.
double DdCg0(int a, int b, int c);
double DdNinej(int a, int b, int c, int d, int e, int f, int g, int h, int i);
}  // namespace detail

namespace {

constexpr int kFactorialMax =
    170;  // 170! is the largest factorial that fits in a double
// Largest f1+f2 handled by moshinsky(); larger brackets return 0. The dd Racah
// sums reach factorial arguments of about 2E+2 <= 170.
constexpr int kEnergyMax = kMoshinskyMaxEnergy;

/// x! for 0 <= x <= 170 by successive multiplication (as imsrg++
/// `FillFactorialLists`).
constexpr std::array<double, kFactorialMax + 1> MakeFactorials() {
  std::array<double, kFactorialMax + 1> t{};
  t[0] = 1.0;
  for (int i = 1; i <= kFactorialMax; ++i) {
    t[static_cast<std::size_t>(i)] = i * t[static_cast<std::size_t>(i - 1)];
  }
  return t;
}

constexpr auto kFactorials = MakeFactorials();

double Factorial(int x) noexcept {
  return x <= kFactorialMax ? kFactorials[static_cast<std::size_t>(x)]
                            : std::numeric_limits<double>::infinity();
}

constexpr int Phase(int x) noexcept { return x % 2 == 0 ? 1 : -1; }

using detail::DD;
using detail::Dd;
using detail::DdLdexp;
using detail::DdSqrt;
using detail::ToDouble;

constexpr int kDdFactorialMax = 170;
constexpr int kDdDoubleFactorialMax = 2 * kEnergyMax + 1;

/// Tables of n!, 1/n!, n!! and 1/n!! in double-double, built by successive dd
/// multiplication, plus sqrt(2) and 2^(1/4). Built once on first use
/// (function-local static, immutable).
struct DdTables {
  std::array<DD, kDdFactorialMax + 1> fact{};
  std::array<DD, kDdFactorialMax + 1> inv_fact{};
  std::array<DD, kDdDoubleFactorialMax + 1> dfact{};
  std::array<DD, kDdDoubleFactorialMax + 1> inv_dfact{};
  DD r2{};
  DD r4{};
};

DdTables MakeDdTables() {
  DdTables t;
  t.fact[0] = Dd(1.0);
  for (int i = 1; i <= kDdFactorialMax; ++i) {
    t.fact[static_cast<std::size_t>(i)] =
        t.fact[static_cast<std::size_t>(i - 1)] * Dd(i);
  }
  t.dfact[0] = Dd(1.0);
  t.dfact[1] = Dd(1.0);
  for (int i = 2; i <= kDdDoubleFactorialMax; ++i) {
    t.dfact[static_cast<std::size_t>(i)] =
        t.dfact[static_cast<std::size_t>(i - 2)] * Dd(i);
  }
  for (std::size_t i = 0; i < t.fact.size(); ++i) {
    t.inv_fact[i] = Dd(1.0) / t.fact[i];
  }
  for (std::size_t i = 0; i < t.dfact.size(); ++i) {
    t.inv_dfact[i] = Dd(1.0) / t.dfact[i];
  }
  t.r2 = DdSqrt(Dd(2.0));
  t.r4 = DdSqrt(t.r2);
  return t;
}

const DdTables& Tab() {
  static const DdTables kTables = MakeDdTables();
  return kTables;
}

DD Dfa(int x) { return Tab().fact[static_cast<std::size_t>(x)]; }
DD Ifa(int x) { return Tab().inv_fact[static_cast<std::size_t>(x)]; }

/// 2^(k/2) for any integer k, in dd (exact power of two times sqrt(2) when k is
/// odd).
DD Pow2Half(int k) {
  const int q = k >= 0 ? k / 2 : -((-k + 1) / 2);
  const DD base = (k - 2 * q) == 0 ? Dd(1.0) : Tab().r2;
  return DdLdexp(base, q);
}

/// 2^(k/4) for any integer k, in dd.
DD Pow2Quarter(int k) {
  const int q = k >= 0 ? k / 4 : -((-k + 3) / 4);
  const int r = k - 4 * q;
  DD base = Dd(1.0);
  if (r == 1) {
    base = Tab().r4;
  } else if (r == 2) {
    base = Tab().r2;
  } else if (r == 3) {
    base = Tab().r2 * Tab().r4;
  }
  return DdLdexp(base, q);
}

bool Triangle(int a, int b, int c) noexcept {
  return c >= std::abs(a - b) && c <= a + b;
}

/// Squared triangle coefficient Delta^2(a b c) =
/// (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!.
DD Tri2(int a, int b, int c) {
  return Dfa(a + b - c) * Dfa(a - b + c) * Dfa(-a + b + c) * Ifa(a + b + c + 1);
}

/// <a 0 b 0 | c 0> from the closed product form (no sum): zero unless a+b+c is
/// even and the triangle holds; otherwise (-1)^(a-b+g) sqrt((2c+1) Delta^2)
/// g!/((g-a)!(g-b)!(g-c)!) with g = (a+b+c)/2.
DD DdCg0Impl(int a, int b, int c) {
  if (a < 0 || b < 0 || c < 0 || (a + b + c) % 2 != 0 || !Triangle(a, b, c)) {
    return {};
  }
  const int g = (a + b + c) / 2;
  const DD v = DdSqrt(Tri2(a, b, c) * Dd(2 * c + 1)) * Dfa(g) * Ifa(g - a) *
               Ifa(g - b) * Ifa(g - c);
  return ((a - b + g) % 2 == 0) ? v : -v;
}

/// 6j symbol {j1 j2 j3; j4 j5 j6} for integer arguments (Racah's single sum),
/// in dd.
DD DdSixjImpl(int j1, int j2, int j3, int j4, int j5, int j6) {
  if (!Triangle(j1, j2, j3) || !Triangle(j1, j5, j6) || !Triangle(j4, j2, j6) ||
      !Triangle(j4, j5, j3)) {
    return {};
  }
  const int a1 = j1 + j2 + j3, a2 = j1 + j5 + j6, a3 = j4 + j2 + j6,
            a4 = j4 + j5 + j3;
  const int b1 = j1 + j2 + j4 + j5, b2 = j2 + j3 + j5 + j6,
            b3 = j3 + j1 + j6 + j4;
  const int tmin = std::max(std::max(a1, a2), std::max(a3, a4));
  const int tmax = std::min(b1, std::min(b2, b3));
  DD sum{};
  for (int t = tmin; t <= tmax; ++t) {
    const DD term = Dfa(t + 1) * Ifa(t - a1) * Ifa(t - a2) * Ifa(t - a3) *
                    Ifa(t - a4) * Ifa(b1 - t) * Ifa(b2 - t) * Ifa(b3 - t);
    sum = (t % 2 == 0) ? sum + term : sum - term;
  }
  const DD delta = DdSqrt(Tri2(j1, j2, j3) * Tri2(j1, j5, j6) *
                          Tri2(j4, j2, j6) * Tri2(j4, j5, j3));
  return delta * sum;
}

/// 9j symbol {a b c; d e f; g h i} of integer arguments as the sum over x of
/// (2x+1) {a b c; f i x} {d e f; b x h} {g h i; x a d}, in dd.
DD DdNinejImpl(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
  const int xmin =
      std::max(std::abs(a - i), std::max(std::abs(b - f), std::abs(d - h)));
  const int xmax = std::min(a + i, std::min(b + f, d + h));
  DD sum{};
  for (int x = xmin; x <= xmax; ++x) {
    const DD s1 = DdSixjImpl(a, b, c, f, i, x);
    if (s1.hi == 0.0) {
      continue;
    }
    const DD s2 = DdSixjImpl(d, e, f, b, x, h);
    if (s2.hi == 0.0) {
      continue;
    }
    const DD s3 = DdSixjImpl(g, h, i, x, a, d);
    sum = sum + Dd(2 * x + 1) * s1 * s2 * s3;
  }
  return sum;
}

}  // namespace

namespace detail {

double DdCg0(int a, int b, int c) { return ToDouble(DdCg0Impl(a, b, c)); }

double DdNinej(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
  return ToDouble(DdNinejImpl(a, b, c, d, e, f, g, h, i));
}

}  // namespace detail

double Moshinsky(HoNL cm, HoNL rel, HoNL a, HoNL b, int lam) noexcept {
  const int N = cm.n, L = cm.l, n = rel.n, l = rel.l;
  const int n1 = a.n, l1 = a.l, n2 = b.n, l2 = b.l;
  if (N < 0 || L < 0 || n < 0 || l < 0 || n1 < 0 || l1 < 0 || n2 < 0 ||
      l2 < 0 || lam < 0) {
    return 0.0;
  }
  const int f1 = 2 * n1 + l1;
  const int f2 = 2 * n2 + l2;
  const int F = 2 * N + L;
  const int f = 2 * n + l;
  if (f1 + f2 != f + F) {
    return 0.0;
  }
  if (f1 + f2 > kEnergyMax) {
    return std::numeric_limits<double>::quiet_NaN();  // beyond the dd tables:
                                                      // fail visibly
  }
  // Parity and coupling of lambda: such brackets vanish identically (the 9j
  // below would be 0).
  if ((l1 + l2 + L + l) % 2 != 0 || lam < std::abs(l1 - l2) || lam > l1 + l2 ||
      lam < std::abs(L - l) || lam > L + l) {
    return 0.0;
  }

  // Buck et al. evaluated entirely in double-double. With cos B = sin B =
  // 2^(-1/2) every term of the innermost sum carries the same factor
  // 2^(-(f1+f2)/2) (the exponent 2(ia+ib+ic+id) + la+lb+lc+ld equals f1+f2),
  // applied once at the end.
  const DdTables& t = Tab();
  const auto u = [](int x) { return static_cast<std::size_t>(x); };
  DD mosh1 = Pow2Quarter(-(l1 + l2 + L + l));
  mosh1 = mosh1 *
          DdSqrt(t.fact[u(n1)] * t.fact[u(n2)] * t.fact[u(N)] * t.fact[u(n)]);
  mosh1 = mosh1 * DdSqrt(t.dfact[u(2 * (n1 + l1) + 1)]);
  mosh1 = mosh1 * DdSqrt(t.dfact[u(2 * (n2 + l2) + 1)]);
  mosh1 = mosh1 * DdSqrt(t.dfact[u(2 * (N + L) + 1)]);
  mosh1 = mosh1 * DdSqrt(t.dfact[u(2 * (n + l) + 1)]);
  if (((l1 + l2 + L + l) / 2) % 2 != 0) {
    mosh1 = -mosh1;
  }
  const auto idf = [&t, &u](int x) { return t.inv_dfact[u(x)]; };

  DD mosh2{};
  for (int la = 0; la <= std::min(f1, F); ++la) {
    for (int lb = (la + l1) % 2; lb <= std::min(l1 + la, f1 - la); lb += 2) {
      const DD cg_ab = DdCg0Impl(la, lb, l1);
      if (cg_ab.hi == 0.0) {
        continue;
      }
      const int amax = (f1 - la - lb) / 2;
      if (amax < 0) {
        continue;
      }
      for (int lc = (la + L) % 2; lc <= std::min(la + L, F - la); lc += 2) {
        const DD cg_ac = DdCg0Impl(la, lc, L);
        if (cg_ac.hi == 0.0) {
          continue;
        }
        const int ldmax =
            std::min(std::min(lb + l, l2 + lc), f2 - F + 2 * amax + la);
        for (int ld = (lb + l) % 2; ld <= ldmax; ld += 2) {
          const DD cg_bd = DdCg0Impl(lb, ld, l);
          if (cg_bd.hi == 0.0) {
            continue;
          }
          const DD cg_cd = DdCg0Impl(lc, ld, l2);
          if (cg_cd.hi == 0.0) {
            continue;
          }
          const DD ninej = DdNinejImpl(la, lb, l1, lc, ld, l2, L, l, lam);
          if (ninej.hi == 0.0) {
            continue;
          }
          DD pre = Pow2Half(la + lb + lc + ld) * ninej * cg_ab * cg_ac * cg_bd *
                   cg_cd *
                   Dd(static_cast<double>((2 * la + 1) * (2 * lb + 1)) *
                      static_cast<double>((2 * lc + 1) * (2 * ld + 1)));
          if ((la + lb + lc) % 2 != 0) {
            pre = -pre;
          }
          DD mosh3{};
          for (int ia = 0; ia <= amax; ++ia) {
            const int ib = (f1 - 2 * ia - la - lb) / 2;
            const int ic = (F - 2 * ia - la - lc) / 2;
            const int id = (f2 - F + 2 * ia + la - ld) / 2;
            if (ib < 0 || ic < 0 || id < 0) {
              continue;
            }
            mosh3 = mosh3 + Ifa(ia) * idf(2 * (ia + la) + 1) * Ifa(ib) *
                                idf(2 * (ib + lb) + 1) * Ifa(ic) *
                                idf(2 * (ic + lc) + 1) * Ifa(id) *
                                idf(2 * (id + ld) + 1);
          }
          mosh2 = mosh2 + mosh3 * pre;
        }
      }
    }
  }
  DD result = mosh1 * mosh2 * Pow2Half(-(f1 + f2));
  if (Phase(l + L + lam) < 0) {  // Buck et al. -> Moshinsky phase convention
    result = -result;
  }
  return ToDouble(result);
}

double TalmiB(HoNL a, HoNL b, int p) noexcept {
  const int n = a.n, l = a.l, nn = b.n, ll = b.l;
  if (n < 0 || nn < 0 || l < 0 || ll < 0 || (l + ll) % 2 != 0) {
    return 0.0;
  }
  const int h = (l + ll) / 2;
  if (p < h || p > h + n + nn || 2 * p + 1 > kFactorialMax ||
      2 * n + 2 * l + 1 > kFactorialMax ||
      2 * nn + 2 * ll + 1 > kFactorialMax ||
      2 * p - l + ll + 1 > kFactorialMax) {
    return 0.0;
  }
  double B = 0;
  const int alpha = std::max(0, p - h - nn);
  const int beta = std::min(n, p - h);
  for (int k = alpha; k <= beta; ++k) {
    B += Factorial(l + k) * Factorial(p - (l - ll) / 2 - k) /
         (Factorial(k) * Factorial(2 * l + 2 * k + 1) * Factorial(n - k) *
          Factorial(2 * p - l + ll - 2 * k + 1) * Factorial(nn - p + h + k) *
          Factorial(p - h - k));
  }
  B *= Phase(p - h) * Factorial(2 * p + 1) /
       (std::pow(2, n + nn) * Factorial(p)) *
       std::sqrt(Factorial(n) * Factorial(nn) * Factorial(2 * n + 2 * l + 1) *
                 Factorial(2 * nn + 2 * ll + 1) /
                 (Factorial(n + l) * Factorial(nn + ll)));
  return B;
}

double TalmiIntegral(int p, double k) noexcept {
  return std::exp(std::lgamma(p + 1.5 + 0.5 * k) - std::lgamma(p + 1.5));
}

}  // namespace imsrg::angmom
