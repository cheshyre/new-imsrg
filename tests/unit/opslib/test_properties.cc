// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Structural properties of the opslib operators at emax 2 (O16 golden model
// space).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator_dump.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/kinetic.h"
#include "imsrg/opslib/radius.h"

namespace {

using namespace imsrg;
namespace fs = std::filesystem;

constexpr int kA = 16;
constexpr int kZ = 8;

std::shared_ptr<const basis::ModelSpace> GoldenSpace() {
  const fs::path dump =
      fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/T.dump";
  return op::ModelSpaceForDump(op::ReadOperatorDumpHeader(dump));
}

/// Largest |O_ij - O_ji| over the two-body blocks and one-body part (host
/// data).
double Asymmetry(op::Operator& o) {
  o.SyncToHost();
  double worst = 0.0;
  const auto& chans = o.model_space().two_body();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    auto block = o.two_body().host_block(ChannelIndex{c});
    for (std::int64_t i = 0; i < block.rows; ++i) {
      for (std::int64_t j = 0; j < i; ++j) {
        worst = std::max(worst, std::abs(block(i, j) - block(j, i)));
      }
    }
  }
  const std::int32_t n = o.model_space().num_orbits();
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < a; ++b) {
      worst = std::max(
          worst, std::abs(o.one_body().get(OrbitIndex{a}, OrbitIndex{b}) -
                          o.one_body().get(OrbitIndex{b}, OrbitIndex{a})));
    }
  }
  return worst;
}

TEST(OpslibProperties, AllOperatorsAreSymmetric) {
  const auto ms = GoldenSpace();
  op::Operator ops[] = {opslib::KineticEnergy(ms), opslib::TCm(ms, kA),
                        opslib::TRel(ms, kA),      opslib::R2Cm(ms, kA),
                        opslib::HCm(ms, kA),       opslib::Rp2(ms, kA, kZ),
                        opslib::Rn2(ms, kA, kZ),   opslib::Rm2(ms, kA)};
  for (op::Operator& o : ops) {
    EXPECT_TRUE(o.spec().hermiticity == Hermiticity::kHermitian);
    EXPECT_EQ(Asymmetry(o), 0.0);
  }
}

TEST(OpslibProperties, TrelIsKineticMinusTcmExactly) {
  const auto ms = GoldenSpace();
  const op::Operator diff = opslib::KineticEnergy(ms) - opslib::TCm(ms, kA);
  const op::PartMaxima d = op::MaxAbsDifference(diff, opslib::TRel(ms, kA));
  EXPECT_EQ(d.max(), 0.0);
  // T has no two-body part; Trel's two-body part is minus that of Tcm.
  EXPECT_EQ(op::MaxAbs(opslib::KineticEnergy(ms)).two_body, 0.0);
}

TEST(OpslibProperties, MatterRadiusAgainstSpeciesRadii) {
  const auto ms = GoldenSpace();
  // Without the spin-orbit term Z rp2 + N rn2 = A rm2 holds for the one-body
  // parts exactly; the two-body parts agree because R2TwoBody(proton) +
  // R2TwoBody(neutron) = R2TwoBody(matter) (the pn channels are split in
  // half). Print what actually holds.
  op::Operator lhs = opslib::Rp2(ms, kA, kZ, false);
  lhs *= static_cast<double>(kZ);
  op::Axpy(static_cast<double>(kA - kZ), opslib::Rn2(ms, kA, kZ), lhs);
  op::Operator rhs = opslib::Rm2(ms, kA);
  rhs *= static_cast<double>(kA);
  const op::PartMaxima d = op::MaxAbsDifference(lhs, rhs);
  std::cout << "[props] max |Z rp2 + N rn2 - A rm2| (no spin-orbit): one-body "
            << d.one_body << ", two-body " << d.two_body << "\n";
  EXPECT_LT(d.max(), 1e-10);
  // With the spin-orbit term included the relation is broken by that term only.
  op::Operator lhs_so = opslib::Rp2(ms, kA, kZ, true);
  const op::PartMaxima so =
      op::MaxAbsDifference(lhs_so, opslib::Rp2(ms, kA, kZ, false));
  const op::PartMaxima so_ref = op::MaxAbs(opslib::RpSpinOrbitCorrection(ms));
  EXPECT_NEAR(so.one_body, so_ref.one_body / kZ, 1e-12);
  EXPECT_EQ(so.two_body, 0.0);
}

TEST(OpslibProperties, SpeciesPartsAddUpToMatter) {
  const auto ms = GoldenSpace();
  op::Operator sum = opslib::R2TwoBody(ms, opslib::Species::kProton);
  sum += opslib::R2TwoBody(ms, opslib::Species::kNeutron);
  const op::PartMaxima d = op::MaxAbsDifference(
      sum, opslib::R2TwoBody(ms, opslib::Species::kMatter));
  EXPECT_LT(d.max(), 1e-12);
  op::Operator one = opslib::R2OneBody(ms, opslib::Species::kProton);
  one += opslib::R2OneBody(ms, opslib::Species::kNeutron);
  EXPECT_LT(
      op::MaxAbsDifference(one, opslib::R2OneBody(ms, opslib::Species::kMatter))
          .max(),
      1e-12);
}

TEST(OpslibProperties, HcmUsesTrapFrequencyOnlyInThePrefactor) {
  const auto ms = GoldenSpace();
  const double hw = ms->basis().hbar_omega();
  // H_cm(hw_trap) - H_cm(hw) = k (hw_trap^2 - hw^2) / hw^2 * (the hw-trap
  // part), so doubling the trap frequency changes the R^2 coefficient by
  // exactly 4.
  op::Operator h1 = opslib::HCm(ms, kA, hw);
  op::Operator h2 = opslib::HCm(ms, kA, 2.0 * hw);
  op::Operator d = h2 - h1;
  op::Operator expect = opslib::R2Cm(ms, kA);
  expect *= 3.0 * 0.5 * kA * constants::kMNucleon * hw * hw /
            (constants::kHbarc * constants::kHbarc);
  EXPECT_LT(op::MaxAbsDifference(d, expect).max(), 1e-10);
  // The default is the basis frequency.
  EXPECT_EQ(op::MaxAbsDifference(opslib::HCm(ms, kA), h1).max(), 0.0);
}

TEST(OpslibProperties, SpeciesStringsAndEdgeCases) {
  EXPECT_EQ(opslib::SpeciesFromString("proton"), opslib::Species::kProton);
  EXPECT_EQ(opslib::SpeciesFromString("neutron"), opslib::Species::kNeutron);
  EXPECT_EQ(opslib::SpeciesFromString("matter"), opslib::Species::kMatter);
  EXPECT_THROW(opslib::SpeciesFromString("charge"), InputError);
  const auto ms = GoldenSpace();
  EXPECT_EQ(op::MaxAbs(opslib::Rp2(ms, kA, 0)).max(), 0.0);
  EXPECT_EQ(op::MaxAbs(opslib::Rn2(ms, kA, kA)).max(), 0.0);
}

}  // namespace
