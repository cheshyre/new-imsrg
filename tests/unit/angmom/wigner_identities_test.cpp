// Exhaustive internal identities of the Wigner symbols over the emax-8 ranges (design/30 section
// 2): orthogonality relations, symmetries, special values, invalid arguments, 9j reductions.
// Independent of any external reference data.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.hpp"

namespace {

using imsrg::TwoJ;
using imsrg::TwoM;
namespace am = imsrg::angmom;

constexpr int kEmax = 8;
constexpr int kTwoJMaxSp = 2 * kEmax + 1;        // largest single-particle 2j
constexpr int kTwoJMax2B = 2 * (2 * kEmax + 1);  // largest 2 * (2B J) = doubled 2B J

double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

/// Same acceptance as the reference test: max(1e-12 |exact|, 1e-14).
bool close(double got, double exact) {
  return std::fabs(got - exact) <= std::max(1e-12 * std::fabs(exact), 1e-14);
}

std::vector<int> sp_orbits() {
  std::vector<int> v;
  for (int t = 1; t <= kTwoJMaxSp; t += 2) v.push_back(t);
  return v;
}

std::vector<int> two_body_js() {
  std::vector<int> v;
  for (int t = 0; t <= kTwoJMax2B; t += 2) v.push_back(t);
  return v;
}

}  // namespace

TEST(WignerIdentities, ThreeJOrthogonalityOverM) {
  // sum_{m1 m2} (2J+1) (ja jb J; m1 m2 M)(ja jb J'; m1 m2 M') = delta_JJ' delta_MM'.
  double max_dev = 0.0;
  long checks = 0;
  for (int ta : sp_orbits()) {
    for (int tb : sp_orbits()) {
      for (int tJ = std::abs(ta - tb); tJ <= ta + tb; tJ += 2) {
        for (int tJp = std::abs(ta - tb); tJp <= ta + tb; tJp += 2) {
          for (int tM = -tJ; tM <= tJ; tM += 2) {
            // M' = M, plus one M' != M probe when it exists.
            for (int dM : {0, 2}) {
              const int tMp = tM + dM;
              if (std::abs(tMp) > tJp) continue;
              double sum = 0.0;
              for (int m1 = -ta; m1 <= ta; m1 += 2) {
                for (int m2 = -tb; m2 <= tb; m2 += 2) {
                  if (m1 + m2 != tM && m1 + m2 != tMp) continue;
                  sum +=
                      (tJ + 1) *
                      am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tJ}, TwoM{m1}, TwoM{m2}, TwoM{-tM}) *
                      am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tJp}, TwoM{m1}, TwoM{m2}, TwoM{-tMp});
                }
              }
              const double expect = (tJ == tJp && dM == 0) ? 1.0 : 0.0;
              max_dev = std::max(max_dev, std::fabs(sum - expect));
              ++checks;
            }
          }
        }
      }
    }
  }
  std::cout << "3j orthogonality: " << checks << " checks, max deviation " << max_dev << "\n";
  EXPECT_LT(max_dev, 1e-13);
}

TEST(WignerIdentities, ThreeJOrthogonalityOverJM) {
  // sum_{J M} (2J+1) (ja jb J; m1 m2 -M)(ja jb J; m1' m2' -M) = delta_m1m1' delta_m2m2'.
  double max_dev = 0.0;
  for (int ta : sp_orbits()) {
    for (int tb : sp_orbits()) {
      for (int m1 = -ta; m1 <= ta; m1 += 2) {
        for (int m2 = -tb; m2 <= tb; m2 += 2) {
          for (int n1 = -ta; n1 <= ta; n1 += 2) {
            const int n2 = m1 + m2 - n1;  // same M, so the sum can be non-trivial
            if (std::abs(n2) > tb || (tb - n2) % 2 != 0) continue;
            double sum = 0.0;
            for (int tJ = std::abs(ta - tb); tJ <= ta + tb; tJ += 2) {
              sum +=
                  (tJ + 1) *
                  am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tJ}, TwoM{m1}, TwoM{m2},
                                TwoM{-(m1 + m2)}) *
                  am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tJ}, TwoM{n1}, TwoM{n2}, TwoM{-(m1 + m2)});
            }
            const double expect = (m1 == n1) ? 1.0 : 0.0;
            max_dev = std::max(max_dev, std::fabs(sum - expect));
          }
        }
      }
    }
  }
  std::cout << "3j orthogonality over J,M: max deviation " << max_dev << "\n";
  EXPECT_LT(max_dev, 1e-13);
}

TEST(WignerIdentities, SixJOrthogonalityPandyaRange) {
  // sum_x (2x+1)(2j3+1) {j1 j2 x; j4 j5 j3}{j1 j2 x; j4 j5 j3'} = delta(j3, j3'); j1,j2,j4,j5
  // single-particle, j3 and j3' two-body J.
  double max_dev = 0.0;
  long checks = 0;
  const auto sp = sp_orbits();
  const auto js = two_body_js();
  for (int t1 : sp) {
    for (int t2 : sp) {
      for (int t4 : sp) {
        for (int t5 : sp) {
          for (int t3 : js) {
            if (!am::triangle(TwoJ{t1}, TwoJ{t5}, TwoJ{t3}) ||
                !am::triangle(TwoJ{t4}, TwoJ{t2}, TwoJ{t3}))
              continue;
            for (int t3p : js) {
              if (!am::triangle(TwoJ{t1}, TwoJ{t5}, TwoJ{t3p}) ||
                  !am::triangle(TwoJ{t4}, TwoJ{t2}, TwoJ{t3p}))
                continue;
              double sum = 0.0;
              for (int tx = 0; tx <= kTwoJMax2B; tx += 2) {
                sum += (tx + 1) * (t3 + 1) *
                       am::wigner_6j(TwoJ{t1}, TwoJ{t2}, TwoJ{tx}, TwoJ{t4}, TwoJ{t5}, TwoJ{t3}) *
                       am::wigner_6j(TwoJ{t1}, TwoJ{t2}, TwoJ{tx}, TwoJ{t4}, TwoJ{t5}, TwoJ{t3p});
              }
              max_dev = std::max(max_dev, std::fabs(sum - (t3 == t3p ? 1.0 : 0.0)));
              ++checks;
            }
          }
        }
      }
    }
  }
  std::cout << "6j orthogonality: " << checks << " checks, max deviation " << max_dev << "\n";
  EXPECT_GT(checks, 0);
  EXPECT_LT(max_dev, 1e-13);
}

TEST(WignerIdentities, SixJSymmetries) {
  // Column permutations and exchange of upper/lower entries in two columns, over the Pandya range
  // and the three-body recoupling range.
  long checks = 0, bad = 0;
  auto check = [&](int a, int b, int c, int d, int e, int f) {
    const double v = am::wigner_6j(TwoJ{a}, TwoJ{b}, TwoJ{c}, TwoJ{d}, TwoJ{e}, TwoJ{f});
    const double w[] = {
        am::wigner_6j(TwoJ{b}, TwoJ{a}, TwoJ{c}, TwoJ{e}, TwoJ{d}, TwoJ{f}),  // swap columns 1,2
        am::wigner_6j(TwoJ{a}, TwoJ{c}, TwoJ{b}, TwoJ{d}, TwoJ{f}, TwoJ{e}),  // swap columns 2,3
        am::wigner_6j(TwoJ{c}, TwoJ{b}, TwoJ{a}, TwoJ{f}, TwoJ{e}, TwoJ{d}),  // swap columns 1,3
        am::wigner_6j(TwoJ{d}, TwoJ{e}, TwoJ{c}, TwoJ{a}, TwoJ{b}, TwoJ{f}),  // flip columns 1,2
        am::wigner_6j(TwoJ{a}, TwoJ{e}, TwoJ{f}, TwoJ{d}, TwoJ{b}, TwoJ{c}),  // flip columns 2,3
        am::wigner_6j(TwoJ{d}, TwoJ{b}, TwoJ{f}, TwoJ{a}, TwoJ{e}, TwoJ{c}),  // flip columns 1,3
    };
    for (double x : w) {
      if (!close(x, v)) ++bad;
      ++checks;
    }
  };
  const auto sp = sp_orbits();
  const auto js = two_body_js();
  for (int ja : sp)
    for (int jb : sp)
      for (int jc : sp)
        for (int jd : sp)
          for (int J : js) {
            if (!am::triangle(TwoJ{ja}, TwoJ{jb}, TwoJ{J})) continue;
            for (int Jp : js) {
              if (am::triangle(TwoJ{ja}, TwoJ{jd}, TwoJ{Jp}) &&
                  am::triangle(TwoJ{jc}, TwoJ{jb}, TwoJ{Jp}) &&
                  am::triangle(TwoJ{jc}, TwoJ{jd}, TwoJ{J})) {
                check(ja, jb, J, jc, jd, Jp);
              }
            }
          }
  // Three-body range, sampled with a stride to bound the run time.
  long n = 0;
  for (int ja : sp)
    for (int jb : sp)
      for (int jc : sp)
        for (int Jab : js)
          for (int Jbc : js)
            for (int J = 1; J <= 3 * kTwoJMaxSp; J += 2) {
              if (am::triangle(TwoJ{ja}, TwoJ{jb}, TwoJ{Jab}) &&
                  am::triangle(TwoJ{ja}, TwoJ{J}, TwoJ{Jbc}) &&
                  am::triangle(TwoJ{jc}, TwoJ{jb}, TwoJ{Jbc}) &&
                  am::triangle(TwoJ{jc}, TwoJ{J}, TwoJ{Jab}) && (n++ % 7 == 0)) {
                check(ja, jb, Jab, jc, J, Jbc);
              }
            }
  std::cout << "6j symmetries: " << checks << " checks, " << bad << " outside tolerance\n";
  EXPECT_GT(checks, 0);
  EXPECT_EQ(bad, 0);
}

TEST(WignerIdentities, ThreeJSymmetries) {
  long checks = 0, bad = 0;
  for (int ta : sp_orbits()) {
    for (int tb : sp_orbits()) {
      for (int tc : two_body_js()) {
        if (!am::triangle(TwoJ{ta}, TwoJ{tb}, TwoJ{tc})) continue;
        const double ph = sgn((ta + tb + tc) / 2);
        for (int m1 = -ta; m1 <= ta; m1 += 2) {
          for (int m2 = -tb; m2 <= tb; m2 += 2) {
            const int m3 = -(m1 + m2);
            if (std::abs(m3) > tc) continue;
            const double v =
                am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tc}, TwoM{m1}, TwoM{m2}, TwoM{m3});
            // odd permutations: phase; cyclic: none; all m flipped: phase.
            const double chk[][2] = {
                {am::wigner_3j(TwoJ{tb}, TwoJ{ta}, TwoJ{tc}, TwoM{m2}, TwoM{m1}, TwoM{m3}), ph * v},
                {am::wigner_3j(TwoJ{ta}, TwoJ{tc}, TwoJ{tb}, TwoM{m1}, TwoM{m3}, TwoM{m2}), ph * v},
                {am::wigner_3j(TwoJ{tc}, TwoJ{tb}, TwoJ{ta}, TwoM{m3}, TwoM{m2}, TwoM{m1}), ph * v},
                {am::wigner_3j(TwoJ{tb}, TwoJ{tc}, TwoJ{ta}, TwoM{m2}, TwoM{m3}, TwoM{m1}), v},
                {am::wigner_3j(TwoJ{tc}, TwoJ{ta}, TwoJ{tb}, TwoM{m3}, TwoM{m1}, TwoM{m2}), v},
                {am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tc}, TwoM{-m1}, TwoM{-m2}, TwoM{-m3}),
                 ph * v},
            };
            for (const auto& c : chk) {
              if (!close(c[0], c[1])) ++bad;
              ++checks;
            }
          }
        }
      }
    }
  }
  std::cout << "3j symmetries: " << checks << " checks, " << bad << " outside tolerance\n";
  EXPECT_GT(checks, 0);
  EXPECT_EQ(bad, 0);
}

TEST(WignerIdentities, InvalidArgumentsGiveExactlyZero) {
  // Triangle violations.
  EXPECT_EQ(am::wigner_3j(TwoJ{2}, TwoJ{2}, TwoJ{6}, TwoM{0}, TwoM{0}, TwoM{0}), 0.0);
  // Odd sum of doubled j (j1+j2+j3 not an integer).
  EXPECT_EQ(am::wigner_3j(TwoJ{1}, TwoJ{2}, TwoJ{2}, TwoM{1}, TwoM{0}, TwoM{-1}), 0.0);
  // |m| > j.
  EXPECT_EQ(am::wigner_3j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoM{4}, TwoM{-2}, TwoM{-2}), 0.0);
  // m of the wrong integrality for j.
  EXPECT_EQ(am::wigner_3j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoM{1}, TwoM{-1}, TwoM{0}), 0.0);
  // Projections that do not sum to zero.
  EXPECT_EQ(am::wigner_3j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoM{2}, TwoM{0}, TwoM{0}), 0.0);
  // Negative 2j.
  EXPECT_EQ(am::wigner_3j(TwoJ{-2}, TwoJ{2}, TwoJ{0}, TwoM{0}, TwoM{0}, TwoM{0}), 0.0);
  EXPECT_EQ(am::wigner_6j(TwoJ{-2}, TwoJ{2}, TwoJ{0}, TwoJ{2}, TwoJ{2}, TwoJ{0}), 0.0);
  // CG: M != m1 + m2.
  EXPECT_EQ(am::clebsch_gordan(TwoJ{1}, TwoM{1}, TwoJ{1}, TwoM{-1}, TwoJ{2}, TwoM{2}), 0.0);
  // CG: triangle fails.
  EXPECT_EQ(am::clebsch_gordan(TwoJ{1}, TwoM{1}, TwoJ{1}, TwoM{1}, TwoJ{4}, TwoM{2}), 0.0);
  // 6j: triangle (j1 j2 j3) fails, and each of the other triads in turn.
  EXPECT_EQ(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{6}, TwoJ{2}, TwoJ{2}, TwoJ{2}), 0.0);
  EXPECT_EQ(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{6}, TwoJ{2}), 0.0);
  EXPECT_EQ(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{6}), 0.0);
  EXPECT_EQ(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{6}, TwoJ{2}, TwoJ{2}), 0.0);
  // 6j: odd doubled sum in one triad.
  EXPECT_EQ(am::wigner_6j(TwoJ{1}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}), 0.0);
  // 9j: a violated row and a violated column.
  EXPECT_EQ(am::wigner_9j(TwoJ{2}, TwoJ{2}, TwoJ{8}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2},
                          TwoJ{2}),
            0.0);
  EXPECT_EQ(am::wigner_9j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{8}, TwoJ{2},
                          TwoJ{2}),
            0.0);
  EXPECT_EQ(
      am::norm_9j(TwoJ{2}, TwoJ{2}, TwoJ{8}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}),
      0.0);
  // Exhaustive: all single-particle triples that violate the triangle give exactly 0.
  for (int ta : sp_orbits()) {
    for (int tb : sp_orbits()) {
      for (int tc = 0; tc <= kTwoJMax2B; tc += 2) {
        if (am::triangle(TwoJ{ta}, TwoJ{tb}, TwoJ{tc})) continue;
        EXPECT_EQ(am::wigner_3j(TwoJ{ta}, TwoJ{tb}, TwoJ{tc}, TwoM{ta}, TwoM{-ta}, TwoM{0}), 0.0);
        EXPECT_EQ(am::wigner_6j(TwoJ{ta}, TwoJ{tb}, TwoJ{tc}, TwoJ{ta}, TwoJ{tb}, TwoJ{tc}), 0.0);
      }
    }
  }
}

TEST(WignerIdentities, SpecialValues) {
  // Closed form {a b c; 0 c b} = (-1)^(a+b+c) / sqrt((2b+1)(2c+1)) over a wide range.
  double max_dev = 0.0;
  for (int ta = 0; ta <= 2 * kTwoJMaxSp; ++ta) {
    for (int tb = 0; tb <= kTwoJMaxSp; ++tb) {
      for (int tc = 0; tc <= kTwoJMaxSp; ++tc) {
        if (!am::triangle(TwoJ{ta}, TwoJ{tb}, TwoJ{tc})) continue;
        const double exact = sgn((ta + tb + tc) / 2) / std::sqrt(double((tb + 1) * (tc + 1)));
        const double got = am::wigner_6j(TwoJ{ta}, TwoJ{tb}, TwoJ{tc}, TwoJ{0}, TwoJ{tc}, TwoJ{tb});
        max_dev = std::max(max_dev, std::fabs(got - exact));
        EXPECT_TRUE(close(got, exact)) << ta << " " << tb << " " << tc;
      }
    }
  }
  std::cout << "6j {a b c; 0 c b} closed form: max deviation " << max_dev << "\n";

  // CG with J = 0: <j m j -m | 0 0> = (-1)^(j-m) / sqrt(2j+1); and the matching 3j.
  for (int tj = 0; tj <= 2 * kTwoJMaxSp; ++tj) {
    for (int tm = -tj; tm <= tj; tm += 2) {
      const double exact = sgn((tj - tm) / 2) / std::sqrt(double(tj + 1));
      EXPECT_TRUE(close(
          am::clebsch_gordan(TwoJ{tj}, TwoM{tm}, TwoJ{tj}, TwoM{-tm}, TwoJ{0}, TwoM{0}), exact));
      EXPECT_TRUE(close(am::wigner_3j(TwoJ{tj}, TwoJ{tj}, TwoJ{0}, TwoM{tm}, TwoM{-tm}, TwoM{0}),
                        sgn((tj - tm) / 2) / std::sqrt(double(tj + 1))));
    }
  }

  // Well-known values.
  EXPECT_TRUE(close(am::wigner_3j(TwoJ{1}, TwoJ{1}, TwoJ{2}, TwoM{1}, TwoM{-1}, TwoM{0}),
                    1.0 / std::sqrt(6.0)));
  EXPECT_TRUE(close(am::clebsch_gordan(TwoJ{1}, TwoM{1}, TwoJ{1}, TwoM{-1}, TwoJ{2}, TwoM{0}),
                    1.0 / std::sqrt(2.0)));
  EXPECT_TRUE(close(am::clebsch_gordan(TwoJ{1}, TwoM{1}, TwoJ{1}, TwoM{-1}, TwoJ{0}, TwoM{0}),
                    1.0 / std::sqrt(2.0)));
  EXPECT_TRUE(close(am::clebsch_gordan(TwoJ{1}, TwoM{-1}, TwoJ{1}, TwoM{1}, TwoJ{0}, TwoM{0}),
                    -1.0 / std::sqrt(2.0)));
  EXPECT_TRUE(
      close(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}, TwoJ{2}), 1.0 / 6.0));
  EXPECT_TRUE(
      close(am::wigner_6j(TwoJ{2}, TwoJ{2}, TwoJ{4}, TwoJ{2}, TwoJ{2}, TwoJ{4}), 1.0 / 30.0));
  EXPECT_TRUE(
      close(am::wigner_6j(TwoJ{1}, TwoJ{1}, TwoJ{2}, TwoJ{1}, TwoJ{1}, TwoJ{2}), 1.0 / 6.0));
  // Stretched CG is 1.
  EXPECT_TRUE(close(am::clebsch_gordan(TwoJ{5}, TwoM{5}, TwoJ{3}, TwoM{3}, TwoJ{8}, TwoM{8}), 1.0));
}

TEST(WignerIdentities, NineJWithZeroReducesToSixJ) {
  // {a b e; c d e; f f 0} = (-1)^(b+c+e+f) / sqrt((2e+1)(2f+1)) {a b e; d c f}.
  long checks = 0;
  double max_dev = 0.0;
  for (int ta = 0; ta <= 8; ++ta)
    for (int tb = 0; tb <= 8; ++tb)
      for (int tc = 0; tc <= 8; ++tc)
        for (int td = 0; td <= 8; ++td)
          for (int te = 0; te <= 8; ++te)
            for (int tf = 0; tf <= 8; ++tf) {
              if (!am::triangle(TwoJ{ta}, TwoJ{tb}, TwoJ{te}) ||
                  !am::triangle(TwoJ{tc}, TwoJ{td}, TwoJ{te}) ||
                  !am::triangle(TwoJ{ta}, TwoJ{tc}, TwoJ{tf}) ||
                  !am::triangle(TwoJ{tb}, TwoJ{td}, TwoJ{tf}))
                continue;
              const int ph2 = tb + tc + te + tf;
              ASSERT_EQ(ph2 % 2, 0);
              const double exact =
                  sgn(ph2 / 2) / std::sqrt(double((te + 1) * (tf + 1))) *
                  am::wigner_6j(TwoJ{ta}, TwoJ{tb}, TwoJ{te}, TwoJ{td}, TwoJ{tc}, TwoJ{tf});
              const double got = am::wigner_9j(TwoJ{ta}, TwoJ{tb}, TwoJ{te}, TwoJ{tc}, TwoJ{td},
                                               TwoJ{te}, TwoJ{tf}, TwoJ{tf}, TwoJ{0});
              max_dev = std::max(max_dev, std::fabs(got - exact));
              ASSERT_TRUE(close(got, exact)) << ta << tb << tc << td << te << tf;
              ++checks;
            }
  std::cout << "9j with a zero: " << checks << " checks, max deviation " << max_dev << "\n";
  EXPECT_GT(checks, 0);
}

TEST(WignerIdentities, NormalizedNineJLsJjUnitarity) {
  // The LS-jj recoupling coefficients form an orthogonal matrix: for fixed la, lb, L, S, J the
  // squares summed over ja, jb = l +- 1/2 give 1 (when the (L S J) channel exists).
  double max_dev = 0.0;
  long checks = 0;
  for (int la = 0; la <= kEmax; ++la) {
    for (int lb = 0; lb <= kEmax; ++lb) {
      for (int L = std::abs(la - lb); L <= la + lb; ++L) {
        for (int S : {0, 2}) {
          for (int J = std::abs(2 * L - S); J <= 2 * L + S; J += 2) {
            double sum = 0.0;
            for (int ja : {2 * la - 1, 2 * la + 1}) {
              for (int jb : {2 * lb - 1, 2 * lb + 1}) {
                if (ja < 0 || jb < 0) continue;
                const double c = am::norm_9j(TwoJ{2 * la}, TwoJ{1}, TwoJ{ja}, TwoJ{2 * lb}, TwoJ{1},
                                             TwoJ{jb}, TwoJ{2 * L}, TwoJ{S}, TwoJ{J});
                sum += c * c;
              }
            }
            // Channels with parity-forbidden (la lb L) combinations still sum to 1: the
            // identity holds for every J in the (L S) triangle.
            max_dev = std::max(max_dev, std::fabs(sum - 1.0));
            ++checks;
          }
        }
      }
    }
  }
  std::cout << "LS-jj unitarity: " << checks << " checks, max deviation " << max_dev << "\n";
  EXPECT_GT(checks, 0);
  EXPECT_LT(max_dev, 1e-13);
}
