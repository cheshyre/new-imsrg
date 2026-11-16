// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The point-proton to charge-radius conversion (D35, design/13 section 8).
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/driver/pipeline/charge_radius.h"

namespace imsrg::driver::pipeline {
namespace {

TEST(ChargeRadius, FormulaFromTheTerms) {
  // O16: N = Z = 8, so (N / Z) r_n^2 = r_n^2.
  const double rp2 = 4.0;
  const double rso = 0.08;
  const double expected = 4.0 + 0.08 / 8.0 + 0.707 + (-0.1161) + 0.033172;
  EXPECT_NEAR(ChargeRadiusSquared(rp2, rso, 16, 8), expected, 1.0e-15);
  // Ca48: N / Z = 28 / 20.
  EXPECT_NEAR(ChargeRadiusSquared(11.0, -0.4, 48, 20),
              11.0 - 0.4 / 20.0 + 0.707 - 0.1161 * 28.0 / 20.0 + 0.033172,
              1.0e-14);
}

TEST(ChargeRadius, ReproducesImsrgppWithItsOldRp2) {
  // imsrg++.cc:1367 with the Rp2 of imsrg++ (= our Rp2 + Rso / Z, D35) and
  // PhysicalConstants.hh:29-33.
  const int a = 20;
  const int z = 10;
  const double rp2 = 6.1234;
  const double rso = -0.37;
  const double rp2_old = rp2 + rso / z;
  const double imsrgpp = rp2_old + 0.707 + -0.1161 * (a - z) / z + 0.033172;
  EXPECT_NEAR(ChargeRadiusSquared(rp2, rso, a, z), imsrgpp, 1.0e-14);
}

TEST(ChargeRadius, PristineImsrgppHartreeFockNumbers) {
  // He4 and O16 at emax 2 / 3: HF Rp2 of our driver (spin-orbit part zero for
  // a closed-LS core) -> the "HF charge radius" printed by the pristine
  // imsrg++ (8 decimals).
  const auto rch = [](double rp2, int a, int z) {
    return ChargeRadius(ChargeRadiusSquared(rp2, 0.0, a, z)).value();
  };
  EXPECT_NEAR(rch(2.1379209867597404, 4, 2), 1.66192448, 2.0e-8);
  EXPECT_NEAR(rch(4.3246980190283653, 16, 8), 2.22458311, 2.0e-8);
}

TEST(ChargeRadius, Constants) {
  EXPECT_EQ(ChargeRadiusSquared(0.0, 0.0, 2, 1),
            0.707 + (-0.1161) * 1 / 1.0 + 0.033172);
  EXPECT_EQ(constants::kProtonRch2, 0.707);
  EXPECT_EQ(constants::kNeutronRch2, -0.1161);
  EXPECT_EQ(constants::kDarwinFoldy, 0.033172);
}

TEST(ChargeRadius, RadiusIsTheRootOrNullopt) {
  EXPECT_DOUBLE_EQ(ChargeRadius(4.0).value(), 2.0);
  EXPECT_EQ(ChargeRadius(0.0).value(), 0.0);
  EXPECT_FALSE(ChargeRadius(-1.0e-3).has_value());
  EXPECT_FALSE(
      ChargeRadius(std::numeric_limits<double>::quiet_NaN()).has_value());
  EXPECT_FALSE(
      ChargeRadius(std::numeric_limits<double>::infinity()).has_value());
}

TEST(ChargeRadius, RejectsInvalidNucleusNumbers) {
  EXPECT_THROW(ChargeRadiusSquared(1.0, 0.0, 4, 0), InvariantError);
  EXPECT_THROW(ChargeRadiusSquared(1.0, 0.0, 1, 2), InvariantError);
}

}  // namespace
}  // namespace imsrg::driver::pipeline
