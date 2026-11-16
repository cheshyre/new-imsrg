// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of the reference factorized double commutator (D32): golden
// acceptance against imsrg++ dumps (design/30 section 2) and algebraic
// properties on random operators.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/commutator_ref/factorized_ref.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/operator.h"
#include "tests/support/commutator_test_support.h"

namespace {

using imsrg::Hermiticity;
using imsrg::op::Operator;
using imsrg::testing::FractionalModelSpace;
using imsrg::testing::GoldenModelSpace;
using imsrg::testing::MakeModelSpace;
using imsrg::testing::RandomOperator;
using imsrg::testing::ReadGolden;
using imsrg::testing::RelativeDeviation;
namespace cr = imsrg::commutator_ref;

constexpr double kGoldenTol = 1.0e-10;
constexpr double kTol = 1.0e-12;

struct GoldenCase {
  const char* dir;     // directory of the Z dump
  const char* in_dir;  // directory of the inputs
  const char* eta;
  const char* gamma;
};

const GoldenCase kGolden[] = {
    {"factorized/emax2_He6/", "factorized/emax2_He6/", "random_A1",
     "random_H1"},
    {"factorized/emax2_He6/", "factorized/emax2_He6/", "random_A1",
     "random_A2"},
    {"factorized/emax2_O16/", "commutator/emax2_O16/", "eta_white", "H"},
    {"factorized/emax2_O16/", "commutator/emax2_O16/", "random_A", "random_H"},
    {"factorized/emax2_O16/", "commutator/emax2_O16/", "eta_white", "random_A"},
};

void CheckGolden(const GoldenCase& g) {
  const std::string eta_path = std::string(g.in_dir) + g.eta + ".dump";
  const std::string gamma_path = std::string(g.in_dir) + g.gamma + ".dump";
  const auto ms = GoldenModelSpace(eta_path);
  const Operator eta = ReadGolden(eta_path, ms);
  const Operator gamma = ReadGolden(gamma_path, ms);
  Operator Z(ms, imsrg::op::OperatorSpec::Scalar(2, gamma.spec().hermiticity));
  cr::FactorizedDoubleCommutator(eta, gamma, Z);
  const Operator ref = ReadGolden(
      std::string(g.dir) + "Z_d32__" + g.eta + "__" + g.gamma + ".dump", ms);
  const double dev = RelativeDeviation(Z, ref);
  std::printf("[golden] d32 %s,%s: %.3e\n", g.eta, g.gamma, dev);
  EXPECT_LE(dev, kGoldenTol);
}

TEST(commutator_ref_factorized, golden_He6_A1_H1) { CheckGolden(kGolden[0]); }
TEST(commutator_ref_factorized, golden_He6_A1_A2) { CheckGolden(kGolden[1]); }
TEST(commutator_ref_factorized, golden_O16_white_H) { CheckGolden(kGolden[2]); }
TEST(commutator_ref_factorized, golden_O16_A_H) { CheckGolden(kGolden[3]); }
TEST(commutator_ref_factorized, golden_O16_white_A) { CheckGolden(kGolden[4]); }

using MS = std::shared_ptr<const imsrg::basis::ModelSpace>;

struct Case {
  std::string name;
  MS ms;
  std::uint64_t seed;
};

std::vector<Case> Cases() {
  return {{"O16", MakeModelSpace(2, "O16"), 31},
          {"He6", MakeModelSpace(2, "He6"), 41},
          {"O16_frac", FractionalModelSpace(2, "O16"), 51}};
}

constexpr Hermiticity kGammaHerm[] = {Hermiticity::kHermitian,
                                      Hermiticity::kAntiHermitian};

Operator Apply(const Operator& eta, const Operator& gamma) {
  Operator Z(eta.model_space_ptr(),
             imsrg::op::OperatorSpec::Scalar(2, gamma.spec().hermiticity));
  cr::FactorizedDoubleCommutator(eta, gamma, Z);
  return Z;
}

/// Largest |Z_ij - h Z_ji| / max(1, max|Z|) over the one-body and all two-body
/// blocks.
double HermiticityViolation(const Operator& Z, double h) {
  const auto& ms = Z.model_space();
  double maxabs = 1.0;
  double dev = 0.0;
  for (int i = 0; i < ms.num_orbits(); ++i) {
    for (const auto j : ms.basis().SameLjtz(imsrg::OrbitIndex{i})) {
      const double a = Z.one_body().get(imsrg::OrbitIndex{i}, j);
      const double b = Z.one_body().get(j, imsrg::OrbitIndex{i});
      maxabs = std::max(maxabs, std::abs(a));
      dev = std::max(dev, std::abs(a - h * b));
    }
  }
  const auto& chans = ms.two_body();
  for (std::int32_t c = 0; c < chans.num_channels(); ++c) {
    const imsrg::ChannelIndex ch{c};
    const int nk = chans.channel(ch).num_kets;
    for (int i = 0; i < nk; ++i) {
      for (int j = 0; j < nk; ++j) {
        const double a = Z.two_body().get(ch, ch, imsrg::LocalKetIndex{i},
                                          imsrg::LocalKetIndex{j});
        const double b = Z.two_body().get(ch, ch, imsrg::LocalKetIndex{j},
                                          imsrg::LocalKetIndex{i});
        maxabs = std::max(maxabs, std::abs(a));
        dev = std::max(dev, std::abs(a - h * b));
      }
    }
  }
  return dev / maxabs;
}

TEST(commutator_ref_factorized, hermiticity) {
  for (const auto& c : Cases()) {
    for (const auto hg : kGammaHerm) {
      const Operator eta =
          RandomOperator(c.ms, Hermiticity::kAntiHermitian, c.seed);
      const Operator gamma = RandomOperator(c.ms, hg, c.seed + 1000);
      const Operator Z = Apply(eta, gamma);
      const double h = (hg == Hermiticity::kHermitian) ? 1.0 : -1.0;
      const double dev = HermiticityViolation(Z, h);
      std::printf("[prop] hermiticity %s gamma %s: %.3e\n", c.name.c_str(),
                  h > 0 ? "H" : "A", dev);
      EXPECT_LE(dev, kTol) << c.name;
    }
  }
}

TEST(commutator_ref_factorized, quadratic_in_eta) {
  for (const auto& c : Cases()) {
    for (const auto hg : kGammaHerm) {
      const Operator eta =
          RandomOperator(c.ms, Hermiticity::kAntiHermitian, c.seed);
      const Operator gamma = RandomOperator(c.ms, hg, c.seed + 1000);
      const double dev =
          RelativeDeviation(Apply(2.0 * eta, gamma), 4.0 * Apply(eta, gamma));
      std::printf("[prop] quadratic %s: %.3e\n", c.name.c_str(), dev);
      EXPECT_LE(dev, kTol) << c.name;
    }
  }
}

TEST(commutator_ref_factorized, linear_in_gamma) {
  for (const auto& c : Cases()) {
    for (const auto hg : kGammaHerm) {
      const Operator eta =
          RandomOperator(c.ms, Hermiticity::kAntiHermitian, c.seed);
      const Operator g1 = RandomOperator(c.ms, hg, c.seed + 1000);
      const Operator g2 = RandomOperator(c.ms, hg, c.seed + 2000);
      const double dev = RelativeDeviation(
          Apply(eta, g1 + 2.0 * g2), Apply(eta, g1) + 2.0 * Apply(eta, g2));
      std::printf("[prop] linear %s: %.3e\n", c.name.c_str(), dev);
      EXPECT_LE(dev, kTol) << c.name;
    }
  }
}

TEST(commutator_ref_factorized, zero_for_one_body_eta) {
  const auto ms = MakeModelSpace(2, "O16");
  const Operator eta =
      RandomOperator(ms, Hermiticity::kAntiHermitian, 7, /*particle_rank=*/1);
  const Operator gamma = RandomOperator(ms, Hermiticity::kHermitian, 8);
  const Operator Z = Apply(eta, gamma);
  const Operator zero(ms, Z.spec());
  EXPECT_EQ(RelativeDeviation(Z, zero), 0.0);
}

TEST(commutator_ref_factorized, rejects_unsupported_operands) {
  const auto ms = MakeModelSpace(2, "O16");
  const Operator herm_eta = RandomOperator(ms, Hermiticity::kHermitian, 7);
  const Operator anti_eta = RandomOperator(ms, Hermiticity::kAntiHermitian, 7);
  const Operator gamma = RandomOperator(ms, Hermiticity::kHermitian, 8);
  const Operator nonherm = RandomOperator(ms, Hermiticity::kNonHermitian, 9);
  Operator Z(ms, imsrg::op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  EXPECT_THROW(cr::FactorizedDoubleCommutator(herm_eta, gamma, Z),
               imsrg::NotImplementedError);
  EXPECT_THROW(cr::FactorizedDoubleCommutator(anti_eta, nonherm, Z),
               imsrg::NotImplementedError);
  Operator Z_bad(
      ms, imsrg::op::OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
  EXPECT_THROW(cr::FactorizedDoubleCommutator(anti_eta, gamma, Z_bad),
               imsrg::InvariantError);
}

}  // namespace
