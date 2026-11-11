// Structural properties of the opslib operators at emax 2 (O16 golden model space).
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

#include "imsrg/base/constants.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator_dump.hpp"
#include "imsrg/opslib/center_of_mass.hpp"
#include "imsrg/opslib/kinetic.hpp"
#include "imsrg/opslib/radius.hpp"

namespace {

using namespace imsrg;
namespace fs = std::filesystem;

constexpr int kA = 16;
constexpr int kZ = 8;

std::shared_ptr<const basis::ModelSpace> golden_space() {
  const fs::path dump = fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16/T.dump";
  return op::model_space_for_dump(op::read_operator_dump_header(dump));
}

/// Largest |O_ij - O_ji| over the two-body blocks and one-body part (host data).
double asymmetry(op::Operator& o) {
  o.sync_to_host();
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
      worst = std::max(worst, std::abs(o.one_body().get(OrbitIndex{a}, OrbitIndex{b}) -
                                       o.one_body().get(OrbitIndex{b}, OrbitIndex{a})));
    }
  }
  return worst;
}

TEST(OpslibProperties, AllOperatorsAreSymmetric) {
  const auto ms = golden_space();
  op::Operator ops[] = {opslib::kinetic_energy(ms), opslib::t_cm(ms, kA), opslib::t_rel(ms, kA),
                        opslib::r2_cm(ms, kA),      opslib::h_cm(ms, kA), opslib::rp2(ms, kA, kZ),
                        opslib::rn2(ms, kA, kZ),    opslib::rm2(ms, kA)};
  for (op::Operator& o : ops) {
    EXPECT_TRUE(o.spec().hermiticity == Hermiticity::Hermitian);
    EXPECT_EQ(asymmetry(o), 0.0);
  }
}

TEST(OpslibProperties, TrelIsKineticMinusTcmExactly) {
  const auto ms = golden_space();
  const op::Operator diff = opslib::kinetic_energy(ms) - opslib::t_cm(ms, kA);
  const op::PartMaxima d = op::max_abs_difference(diff, opslib::t_rel(ms, kA));
  EXPECT_EQ(d.max(), 0.0);
  // T has no two-body part; Trel's two-body part is minus that of Tcm.
  EXPECT_EQ(op::max_abs(opslib::kinetic_energy(ms)).two_body, 0.0);
}

TEST(OpslibProperties, MatterRadiusAgainstSpeciesRadii) {
  const auto ms = golden_space();
  // Without the spin-orbit term Z rp2 + N rn2 = A rm2 holds for the one-body parts exactly; the
  // two-body parts agree because r2_two_body(proton) + r2_two_body(neutron) = r2_two_body(matter)
  // (the pn channels are split in half). Print what actually holds.
  op::Operator lhs = opslib::rp2(ms, kA, kZ, false);
  lhs *= static_cast<double>(kZ);
  op::axpy(static_cast<double>(kA - kZ), opslib::rn2(ms, kA, kZ), lhs);
  op::Operator rhs = opslib::rm2(ms, kA);
  rhs *= static_cast<double>(kA);
  const op::PartMaxima d = op::max_abs_difference(lhs, rhs);
  std::cout << "[props] max |Z rp2 + N rn2 - A rm2| (no spin-orbit): one-body " << d.one_body
            << ", two-body " << d.two_body << "\n";
  EXPECT_LT(d.max(), 1e-10);
  // With the spin-orbit term included the relation is broken by that term only.
  op::Operator lhs_so = opslib::rp2(ms, kA, kZ, true);
  const op::PartMaxima so = op::max_abs_difference(lhs_so, opslib::rp2(ms, kA, kZ, false));
  const op::PartMaxima so_ref = op::max_abs(opslib::rp_spin_orbit_correction(ms));
  EXPECT_NEAR(so.one_body, so_ref.one_body / kZ, 1e-12);
  EXPECT_EQ(so.two_body, 0.0);
}

TEST(OpslibProperties, SpeciesPartsAddUpToMatter) {
  const auto ms = golden_space();
  op::Operator sum = opslib::r2_two_body(ms, opslib::Species::Proton);
  sum += opslib::r2_two_body(ms, opslib::Species::Neutron);
  const op::PartMaxima d =
      op::max_abs_difference(sum, opslib::r2_two_body(ms, opslib::Species::Matter));
  EXPECT_LT(d.max(), 1e-12);
  op::Operator one = opslib::r2_one_body(ms, opslib::Species::Proton);
  one += opslib::r2_one_body(ms, opslib::Species::Neutron);
  EXPECT_LT(op::max_abs_difference(one, opslib::r2_one_body(ms, opslib::Species::Matter)).max(),
            1e-12);
}

TEST(OpslibProperties, HcmUsesTrapFrequencyOnlyInThePrefactor) {
  const auto ms = golden_space();
  const double hw = ms->basis().hbar_omega();
  // H_cm(hw_trap) - H_cm(hw) = k (hw_trap^2 - hw^2) / hw^2 * (the hw-trap part), so doubling the
  // trap frequency changes the R^2 coefficient by exactly 4.
  op::Operator h1 = opslib::h_cm(ms, kA, hw);
  op::Operator h2 = opslib::h_cm(ms, kA, 2.0 * hw);
  op::Operator d = h2 - h1;
  op::Operator expect = opslib::r2_cm(ms, kA);
  expect *= 3.0 * 0.5 * kA * constants::m_nucleon * hw * hw / (constants::hbarc * constants::hbarc);
  EXPECT_LT(op::max_abs_difference(d, expect).max(), 1e-10);
  // The default is the basis frequency.
  EXPECT_EQ(op::max_abs_difference(opslib::h_cm(ms, kA), h1).max(), 0.0);
}

TEST(OpslibProperties, SpeciesStringsAndEdgeCases) {
  EXPECT_EQ(opslib::species_from_string("proton"), opslib::Species::Proton);
  EXPECT_EQ(opslib::species_from_string("neutron"), opslib::Species::Neutron);
  EXPECT_EQ(opslib::species_from_string("matter"), opslib::Species::Matter);
  EXPECT_THROW(opslib::species_from_string("charge"), InputError);
  const auto ms = golden_space();
  EXPECT_EQ(op::max_abs(opslib::rp2(ms, kA, 0)).max(), 0.0);
  EXPECT_EQ(op::max_abs(opslib::rn2(ms, kA, kA)).max(), 0.0);
}

}  // namespace
