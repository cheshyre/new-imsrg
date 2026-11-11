// Tests of the Talmi-Moshinsky bracket and Talmi coefficient/integral against exact values,
// imsrg++ samples, orthogonality and Talmi sum rules.
#include "imsrg/angmom/moshinsky.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/detail/double_double.hpp"
#include "imsrg/angmom/tables.hpp"
#include "imsrg/angmom/wigner.hpp"
#include "imsrg/base/error.hpp"

#ifndef IMSRG_MOSHINSKY_EXACT_FILE
#error "IMSRG_MOSHINSKY_EXACT_FILE must be defined by the build"
#endif
#ifndef IMSRG_MOSHINSKY_SAMPLES_FILE
#error "IMSRG_MOSHINSKY_SAMPLES_FILE must be defined by the build"
#endif

namespace imsrg::angmom::detail {
// Defined in moshinsky.cpp (test hooks, not in any public header).
double dd_cg0(int a, int b, int c);
double dd_ninej(int a, int b, int c, int d, int e, int f, int g, int h, int i);
}  // namespace imsrg::angmom::detail

namespace {

using namespace imsrg::angmom;
using imsrg::TwoJ;
using imsrg::TwoM;

bool within(double got, double exact) {
  return std::abs(got - exact) <= std::max(1e-12 * std::abs(exact), 1e-14);
}

struct Dev {
  double max_abs = 0.0;
  double max_rel = 0.0;
  long count = 0;
  long failures = 0;
  void add(double got, double exact) {
    const double d = std::abs(got - exact);
    max_abs = std::max(max_abs, d);
    if (exact != 0.0) {
      max_rel = std::max(max_rel, d / std::abs(exact));
    }
    ++count;
    if (!within(got, exact)) {
      ++failures;
    }
  }
};

TEST(AngmomMoshinsky, MatchesExactReference) {
  std::ifstream in(IMSRG_MOSHINSKY_EXACT_FILE);
  ASSERT_TRUE(in.good()) << "cannot open " << IMSRG_MOSHINSKY_EXACT_FILE;
  Dev mosh, talmi;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream s(line);
    std::string tag;
    s >> tag;
    if (tag == "talmib") {
      int na, la, nb, lb, p;
      double v;
      ASSERT_TRUE(static_cast<bool>(s >> na >> la >> nb >> lb >> p >> v)) << line;
      const double got = talmi_b(HoNL{na, la}, HoNL{nb, lb}, p);
      talmi.add(got, v);
      if (!within(got, v) && talmi.failures <= 20) {
        ADD_FAILURE() << line << " got " << got;
      }
      continue;
    }
    // Bracket lines: "N L n l n1 l1 n2 l2 lam value" (no tag) or tagged "mosh".
    std::istringstream s2(tag == "mosh" ? line.substr(4) : line);
    int N, L, n, l, n1, l1, n2, l2, lam;
    double v;
    ASSERT_TRUE(static_cast<bool>(s2 >> N >> L >> n >> l >> n1 >> l1 >> n2 >> l2 >> lam >> v))
        << line;
    const double got = moshinsky(HoNL{N, L}, HoNL{n, l}, HoNL{n1, l1}, HoNL{n2, l2}, lam);
    mosh.add(got, v);
    if (!within(got, v) && mosh.failures <= 20) {
      ADD_FAILURE() << line << " got " << got;
    }
  }
  std::cout << "[exact] brackets=" << mosh.count << " max_abs=" << mosh.max_abs
            << " max_rel=" << mosh.max_rel << " failures=" << mosh.failures
            << "; talmi_b=" << talmi.count << " max_abs=" << talmi.max_abs
            << " max_rel=" << talmi.max_rel << " failures=" << talmi.failures << "\n";
  EXPECT_GT(mosh.count, 1000);
  EXPECT_GT(talmi.count, 1000);
  EXPECT_EQ(mosh.failures, 0);
  EXPECT_EQ(talmi.failures, 0);
}

TEST(AngmomMoshinsky, MatchesImsrgppSamples) {
  std::ifstream in(IMSRG_MOSHINSKY_SAMPLES_FILE);
  ASSERT_TRUE(in.good()) << "cannot open " << IMSRG_MOSHINSKY_SAMPLES_FILE;
  Dev mosh, talmi;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream s(line);
    std::string tag;
    s >> tag;
    if (tag == "mosh") {
      int N, L, n, l, n1, l1, n2, l2, lam;
      double v;
      ASSERT_TRUE(static_cast<bool>(s >> N >> L >> n >> l >> n1 >> l1 >> n2 >> l2 >> lam >> v))
          << line;
      const double got = moshinsky(HoNL{N, L}, HoNL{n, l}, HoNL{n1, l1}, HoNL{n2, l2}, lam);
      mosh.add(got, v);
      if (!within(got, v)) {
        ADD_FAILURE() << line << " got " << got;
      }
    } else if (tag == "talmib") {
      int na, la, nb, lb, p;
      double v;
      ASSERT_TRUE(static_cast<bool>(s >> na >> la >> nb >> lb >> p >> v)) << line;
      const double got = talmi_b(HoNL{na, la}, HoNL{nb, lb}, p);
      talmi.add(got, v);
      if (!within(got, v)) {
        ADD_FAILURE() << line << " got " << got;
      }
    }
  }
  std::cout << "[moshinsky imsrg++] mosh entries=" << mosh.count << " max_abs=" << mosh.max_abs
            << " max_rel=" << mosh.max_rel << "; talmib entries=" << talmi.count
            << " max_abs=" << talmi.max_abs << " max_rel=" << talmi.max_rel << "\n";
  EXPECT_GT(mosh.count, 0);
  EXPECT_GT(talmi.count, 0);
}

TEST(AngmomMoshinsky, OrthogonalityExhaustiveUpToE12) {
  const int e2max = 12;
  const MoshinskyTable t(e2max);
  double worst = 0.0;
  long ones = 0;
  for (int E = 0; E <= e2max; ++E) {
    // Enumerate the tuples exactly like the table does.
    std::vector<std::array<int, 4>> tup;
    for (int l1 = 0; l1 <= E; ++l1) {
      for (int n1 = 0; 2 * n1 + l1 <= E; ++n1) {
        for (int l2 = 0; 2 * n1 + l1 + l2 <= E; ++l2) {
          const int rem = E - 2 * n1 - l1 - l2;
          if (rem % 2 == 0) {
            tup.push_back({n1, l1, rem / 2, l2});
          }
        }
      }
    }
    const std::size_t K = tup.size();
    std::vector<double> M(K * K);
    for (int lam = 0; lam <= E; ++lam) {
      for (std::size_t c = 0; c < K; ++c) {
        for (std::size_t u = 0; u < K; ++u) {
          M[c * K + u] = t(HoNL{tup[c][0], tup[c][1]}, HoNL{tup[c][2], tup[c][3]},
                           HoNL{tup[u][0], tup[u][1]}, HoNL{tup[u][2], tup[u][3]}, lam);
        }
      }
      // Columns (lab states): sum over cm-rel states.
      for (std::size_t u = 0; u < K; ++u) {
        const bool valid = lam >= std::abs(tup[u][1] - tup[u][3]) && lam <= tup[u][1] + tup[u][3];
        for (std::size_t w = u; w < K; ++w) {
          double s = 0.0;
          for (std::size_t c = 0; c < K; ++c) {
            s += M[c * K + u] * M[c * K + w];
          }
          const double expect = (u == w && valid) ? 1.0 : 0.0;
          worst = std::max(worst, std::abs(s - expect));
          if (u == w && valid) {
            ++ones;
          }
        }
      }
      // Rows (cm-rel states): sum over lab states.
      for (std::size_t c = 0; c < K; ++c) {
        const bool valid = lam >= std::abs(tup[c][1] - tup[c][3]) && lam <= tup[c][1] + tup[c][3];
        for (std::size_t d = c; d < K; ++d) {
          double s = 0.0;
          for (std::size_t u = 0; u < K; ++u) {
            s += M[c * K + u] * M[d * K + u];
          }
          const double expect = (c == d && valid) ? 1.0 : 0.0;
          worst = std::max(worst, std::abs(s - expect));
        }
      }
    }
  }
  std::cout << "[moshinsky orthogonality] E<=12 max |M M^T - 1| = " << worst
            << " (unit diagonals checked: " << ones << ")\n";
  EXPECT_GT(ones, 0);
  EXPECT_LT(worst, 1e-12);
}

TEST(AngmomMoshinsky, TalmiSumRules) {
  // B(a, b; p) carries the phase (-1)^(p - (la + lb)/2) and |B| reaches ~1e6, so the sums below
  // alternate and cancel. Rounding of a cancelling sum scales with the sum of the magnitudes of
  // its terms, so each identity is checked against 1e-13 * sum_p |B I_p| (+ 1e-14), not an
  // absolute tolerance. (talmi_b itself is compared with exact values to 1e-12 relative above.)
  const auto ok = [](double got, double expect, double mag) {
    return std::abs(got - expect) <= 1e-13 * mag + 1e-14;
  };
  // Per-identity worst deviation measured in units of the bound (<= 1 passes).
  double worst_sum = 0.0, worst_diag = 0.0, worst_off = 0.0, worst_off_flipped = 0.0;
  double worst_zero = 0.0;
  long bad = 0;
  const auto ratio = [](double got, double expect, double mag) {
    return std::abs(got - expect) / (1e-13 * mag + 1e-14);
  };
  for (int l = 0; l <= 12; ++l) {
    for (int n = 0; n <= 6; ++n) {
      for (int np = 0; np <= 6; ++np) {
        double s0 = 0.0, m0 = 0.0, s2 = 0.0, m2 = 0.0;
        for (int p = 0; p <= l + n + np; ++p) {
          const double B = talmi_b(HoNL{n, l}, HoNL{np, l}, p);
          const double I2 = talmi_integral(p, 2.0);
          s0 += B;
          m0 += std::abs(B);
          s2 += B * I2;
          m2 += std::abs(B * I2);
        }
        // sum rule: sum_p B = delta(n, n')
        const double e0 = n == np ? 1.0 : 0.0;
        worst_sum = std::max(worst_sum, ratio(s0, e0, m0));
        bad += !ok(s0, e0, m0);
        // <r^2>: sum_p B I_p(2), I_p(2) = p + 3/2.
        if (n == np) {
          const double e = 2.0 * n + l + 1.5;
          worst_diag = std::max(worst_diag, ratio(s2, e, m2));
          bad += !ok(s2, e, m2);
        } else if (n == np - 1) {
          const double mag = std::sqrt(np * (np + l + 0.5));
          worst_off = std::max(worst_off, ratio(s2, -mag, m2));
          worst_off_flipped = std::max(worst_off_flipped, ratio(s2, mag, m2));
          bad += !ok(s2, -mag, m2);
        } else if (std::abs(n - np) > 1) {
          worst_zero = std::max(worst_zero, ratio(s2, 0.0, m2));
          bad += !ok(s2, 0.0, m2);
        } else {  // n == np + 1: the matrix is symmetric
          const double mag = std::sqrt(n * (n + l + 0.5));
          worst_off = std::max(worst_off, ratio(s2, -mag, m2));
          bad += !ok(s2, -mag, m2);
        }
      }
    }
  }
  std::cout << "[talmi identities] worst deviation / bound: sum rule " << worst_sum
            << ", r^2 diagonal " << worst_diag << ", r^2 off-diagonal (-sqrt) " << worst_off
            << " (+sqrt: " << worst_off_flipped << "), |n-n'|>1 " << worst_zero << "\n";
  EXPECT_EQ(bad, 0);

  for (int p = 0; p <= 30; ++p) {
    EXPECT_NEAR(talmi_integral(p, 0.0), 1.0, 1e-13);
    EXPECT_NEAR(talmi_integral(p, 2.0), p + 1.5, 1e-12 * (p + 2));
  }
}

TEST(AngmomDoubleDouble, Arithmetic) {
  using namespace imsrg::angmom::detail;
  // (1/3)*3 == 1 to dd precision.
  const DD third = dd(1.0) / dd(3.0);
  const DD one = third * dd(3.0);
  EXPECT_LT(std::abs((one - dd(1.0)).hi), 1e-31);
  // sqrt(2)^2 - 2.
  const DD r2 = dd_sqrt(dd(2.0));
  EXPECT_LT(std::abs((r2 * r2 - dd(2.0)).hi), 1e-31);
  // 2^(1/4) to 1e-31: (2^(1/4))^4 == 2.
  const DD r4 = dd_sqrt(r2);
  const DD r4p = r4 * r4;
  EXPECT_LT(std::abs((r4p * r4p - dd(2.0)).hi), 1e-31);
  // 30! exactly against an integer product (30! < 2^108, use dd product of exact small factors
  // and compare with a __int128 product).
  DD f = dd(1.0);
  unsigned __int128 fi = 1;
  for (int i = 1; i <= 30; ++i) {
    f = f * dd(i);
    fi *= static_cast<unsigned>(i);
  }
  const double hi = static_cast<double>(fi);
  const unsigned __int128 hi_i = static_cast<unsigned __int128>(hi);
  const double lo = hi_i <= fi ? static_cast<double>(fi - hi_i) : -static_cast<double>(hi_i - fi);
  EXPECT_EQ(f.hi, hi);
  EXPECT_EQ(f.lo, lo);
}

TEST(AngmomDoubleDouble, ClebschGordanZeroMatchesDouble) {
  double worst = 0;
  for (int l1 = 0; l1 <= 32; ++l1) {
    for (int l2 = 0; l2 <= 32; ++l2) {
      for (int L = std::abs(l1 - l2); L <= l1 + l2; ++L) {
        const double ref =
            clebsch_gordan(TwoJ{2 * l1}, TwoM{0}, TwoJ{2 * l2}, TwoM{0}, TwoJ{2 * L}, TwoM{0});
        const double got = imsrg::angmom::detail::dd_cg0(l1, l2, L);
        worst = std::max(worst, std::abs(got - ref));
        // The double-precision reference is the limiting side here: its error reaches ~5e-12 at
        // l ~ 30 (exact values below, from Python big rationals), hence 1e-10.
        ASSERT_NEAR(got, ref, 1e-10) << l1 << " " << l2 << " " << L;
      }
    }
  }
  std::cout << "[dd] cg0 max abs deviation " << worst << "\n";
  // Exact spot checks (the dd values are accurate to ~1e-17 where the double ones are not).
  EXPECT_NEAR(imsrg::angmom::detail::dd_cg0(11, 29, 32), 0.24719909357682662719, 1e-16);
  EXPECT_NEAR(imsrg::angmom::detail::dd_cg0(27, 32, 37), -0.16559883667476296171, 1e-16);
  EXPECT_NEAR(imsrg::angmom::detail::dd_cg0(16, 29, 31), -0.20414884248844674639, 1e-16);
}

TEST(AngmomDoubleDouble, NineJMatchesDouble) {
  double worst = 0;
  std::uint64_t state = 12345;
  const auto rnd = [&state](int m) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<int>((state >> 33) % static_cast<std::uint64_t>(m));
  };
  int tested = 0;
  for (int it = 0; it < 200000 && tested < 3000; ++it) {
    const int a = rnd(9), b = rnd(9), d = rnd(9), e = rnd(9), g = rnd(9), h = rnd(9);
    const int c = std::abs(a - b) + rnd(std::min(2 * std::min(a, b) + 1, 9));
    const int f = std::abs(d - e) + rnd(std::min(2 * std::min(d, e) + 1, 9));
    const int i = std::abs(g - h) + rnd(std::min(2 * std::min(g, h) + 1, 9));
    const double ref = wigner_9j(TwoJ{2 * a}, TwoJ{2 * b}, TwoJ{2 * c}, TwoJ{2 * d}, TwoJ{2 * e},
                                 TwoJ{2 * f}, TwoJ{2 * g}, TwoJ{2 * h}, TwoJ{2 * i});
    const double got = imsrg::angmom::detail::dd_ninej(a, b, c, d, e, f, g, h, i);
    if (ref == 0.0 && got == 0.0) {
      continue;
    }
    ++tested;
    worst = std::max(worst, std::abs(got - ref));
    ASSERT_NEAR(got, ref, 1e-14) << a << b << c << d << e << f << g << h << i;
  }
  EXPECT_GT(tested, 500);
  std::cout << "[dd] 9j max abs deviation " << worst << " over " << tested << " samples\n";
}

// Beyond the supported energy the bracket fails visibly (NaN) and tables refuse to build.
TEST(Moshinsky, BeyondMaxEnergyIsNaNAndTableRefuses) {
  const int e = moshinsky_max_energy + 1;
  EXPECT_TRUE(std::isnan(moshinsky(HoNL{0, 0}, HoNL{0, e}, HoNL{0, e}, HoNL{0, 0}, e)));
  EXPECT_EQ(moshinsky(HoNL{0, 0}, HoNL{0, 2}, HoNL{0, 1}, HoNL{0, 0}, 1), 0.0);  // E not conserved
  EXPECT_THROW(MoshinskyTable table(e), imsrg::InvariantError);
}

}  // namespace
