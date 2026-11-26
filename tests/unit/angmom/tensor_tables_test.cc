// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The rank-lambda recoupling tables of the tensor commutators (M15) against
// the direct Wigner6j / Wigner9j evaluation, and the 9j skip threshold.
#include "imsrg/angmom/tensor_tables.h"

#include <cmath>
#include <cstdlib>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.h"

namespace imsrg::angmom {
namespace {

TEST(TensorTables, LambdaSixJMatchesWigner6j) {
  const int emax = 3;
  for (int L = 0; L <= 5; ++L) {
    const LambdaSixJTable t(emax, L);
    const LambdaSixJView v = t.host_view();
    for (int J1 = 0; J1 <= 2 * emax + 1; ++J1)
      for (int J2 = 0; J2 <= 2 * emax + 1; ++J2)
        for (int a = 1; a <= 2 * emax + 1; a += 2)
          for (int b = 1; b <= 2 * emax + 1; b += 2)
            for (int c = 1; c <= 2 * emax + 1; c += 2)
              ASSERT_EQ(
                  v(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{a}, TwoJ{b}, TwoJ{c}),
                  Wigner6j(TwoJ{2 * J1}, TwoJ{2 * J2}, TwoJ{2 * L}, TwoJ{a},
                           TwoJ{b}, TwoJ{c}));
    EXPECT_EQ(v(TwoJ{1}, TwoJ{0}, TwoJ{1}, TwoJ{1}, TwoJ{1}), 0.0);  // odd 2J
    EXPECT_EQ(v(TwoJ{0}, TwoJ{0}, TwoJ{9}, TwoJ{1}, TwoJ{1}), 0.0);  // 2j>7
  }
}

TEST(TensorTables, NineJMatchesWigner9jAndSkipsSmallValues) {
  const int emax = 2;
  for (int L = 0; L <= 3; ++L) {
    const NineJTable t(emax, L, 0.0);
    const NineJTable tc(emax, L, 1.0e-2);
    const NineJView v = t.host_view();
    const NineJView vc = tc.host_view();
    int nonzero = 0;
    int skipped = 0;
    for (int A = 1; A <= 5; A += 2)
      for (int B = 1; B <= 5; B += 2)
        for (int D = 1; D <= 5; D += 2)
          for (int E = 1; E <= 5; E += 2)
            for (int C = 0; C <= 10; C += 2)
              for (int F = 0; F <= 10; F += 2)
                for (int G = 0; G <= 10; G += 2)
                  for (int H = 0; H <= 10; H += 2) {
                    const double w =
                        Wigner9j(TwoJ{A}, TwoJ{B}, TwoJ{C}, TwoJ{D}, TwoJ{E},
                                 TwoJ{F}, TwoJ{G}, TwoJ{H}, TwoJ{2 * L});
                    const double x = v(TwoJ{A}, TwoJ{B}, TwoJ{C}, TwoJ{D},
                                       TwoJ{E}, TwoJ{F}, TwoJ{G}, TwoJ{H});
                    ASSERT_EQ(x, w);
                    const double y = vc(TwoJ{A}, TwoJ{B}, TwoJ{C}, TwoJ{D},
                                        TwoJ{E}, TwoJ{F}, TwoJ{G}, TwoJ{H});
                    ASSERT_EQ(y, std::abs(w) < 1.0e-2 ? 0.0 : w);
                    if (w != 0.0) ++nonzero;
                    if (w != 0.0 && y == 0.0) ++skipped;
                  }
    EXPECT_GT(nonzero, 0);
    EXPECT_GT(skipped, 0);
    EXPECT_EQ(t.lambda(), L);
    EXPECT_EQ(tc.skip_threshold(), 1.0e-2);
  }
}

}  // namespace
}  // namespace imsrg::angmom
