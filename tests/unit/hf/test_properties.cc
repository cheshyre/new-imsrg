// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Structural properties of the converged Hartree-Fock state (emax 2, O16, 2N).
#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>

#include "imsrg/base/error.h"
#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::hftest {

namespace fs = std::filesystem;
namespace {

struct State {
  op::Operator H;
  hf::HFResult r;
};

// Owned by the fixture so that the device buffers die before Kokkos is
// finalized.
std::unique_ptr<State> g_state;

const State& GetState() { return *g_state; }

class HfProperties : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    op::Operator H =
        LoadHamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
    hf::HFResult r = hf::Solve(H, nullptr);
    g_state = std::make_unique<State>(State{std::move(H), std::move(r)});
  }
  static void TearDownTestSuite() { g_state.reset(); }
};

const basis::Orbit& OrbitAt(std::int64_t i) {
  return GetState().H.model_space().basis().orbit(
      OrbitIndex{static_cast<std::int32_t>(i)});
}

TEST_F(HfProperties, DensityIsHermitianWithCorrectTrace) {
  const auto& r = GetState().r;
  const auto n = r.rho.rows();
  double trace = 0.0, asym = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    trace += (OrbitAt(i).j2.twice + 1.0) * r.rho(i, i);
    for (std::int64_t j = 0; j < n; ++j)
      asym = std::max(asym, std::abs(r.rho(i, j) - r.rho(j, i)));
  }
  EXPECT_LT(asym, 1.0e-13);
  EXPECT_NEAR(trace, 16.0, 1.0e-10);
}

TEST_F(HfProperties, DensityIsIdempotentAtConvergence) {
  const auto& r = GetState().r;
  const auto n = r.rho.rows();
  double dev = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::int64_t k = 0; k < n; ++k) s += r.rho(i, k) * r.rho(k, j);
      dev = std::max(dev, std::abs(s - r.rho(i, j)));
    }
  }
  std::cout << "[properties] max|rho^2 - rho| " << dev << "\n";
  EXPECT_LT(dev, 1.0e-10);
}

TEST_F(HfProperties, FockMatrixIsHermitian) {
  const auto& r = GetState().r;
  double asym = 0.0;
  for (std::int64_t i = 0; i < r.F.rows(); ++i) {
    for (std::int64_t j = 0; j < r.F.cols(); ++j) {
      asym = std::max(asym, std::abs(r.F(i, j) - r.F(j, i)));
    }
  }
  EXPECT_LT(asym, 1.0e-12);
}

TEST_F(HfProperties, ChannelEnergiesAreAscending) {
  const auto& r = GetState().r;
  const auto n = static_cast<std::int64_t>(r.spe.size());
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = i + 1; j < n; ++j) {
      const auto &a = OrbitAt(i), &b = OrbitAt(j);
      if (a.l == b.l && a.j2.twice == b.j2.twice &&
          a.tz2.twice == b.tz2.twice) {
        EXPECT_LE(r.spe[static_cast<std::size_t>(i)],
                  r.spe[static_cast<std::size_t>(j)]);
      }
    }
  }
}

TEST_F(HfProperties, CoefficientsAreOrthogonalWithPositiveDiagonal) {
  const auto& r = GetState().r;
  const auto n = r.C.rows();
  double dev = 0.0;
  for (std::int64_t a = 0; a < n; ++a) {
    EXPECT_GE(r.C(a, a), 0.0);
    for (std::int64_t b = 0; b < n; ++b) {
      double s = 0.0;
      for (std::int64_t i = 0; i < n; ++i) s += r.C(i, a) * r.C(i, b);
      dev = std::max(dev, std::abs(s - (a == b ? 1.0 : 0.0)));
    }
  }
  EXPECT_LT(dev, 1.0e-12);
}

TEST_F(HfProperties, EnergyEqualsSumOfParts) {
  const auto& r = GetState().r;
  EXPECT_DOUBLE_EQ(r.e_hf, r.e1 + r.e2 + r.e3);
}

// SolutionInBasis with the converged C and occupations reproduces the solved
// state: rho, F, energies and (since F is diagonal in its own eigenbasis) the
// single-particle energies.
TEST_F(HfProperties, SolutionInBasisReproducesTheSolvedState) {
  const auto& s = GetState();
  const hf::HFResult m =
      hf::SolutionInBasis(s.H, nullptr, s.r.C, s.r.occupations);
  EXPECT_TRUE(m.converged);
  EXPECT_EQ(m.iterations, 0);
  EXPECT_NEAR(m.e_hf, s.r.e_hf, 1.0e-7);
  EXPECT_NEAR(m.e1, s.r.e1, 1.0e-7);
  EXPECT_NEAR(m.e2, s.r.e2, 1.0e-7);
  double drho = 0.0, df = 0.0, de = 0.0;
  for (std::int64_t i = 0; i < m.rho.rows(); ++i) {
    for (std::int64_t j = 0; j < m.rho.cols(); ++j) {
      drho = std::max(drho, std::abs(m.rho(i, j) - s.r.rho(i, j)));
      df = std::max(df, std::abs(m.F(i, j) - s.r.F(i, j)));
    }
    de = std::max(de, std::abs(m.spe[static_cast<std::size_t>(i)] -
                               s.r.spe[static_cast<std::size_t>(i)]));
  }
  std::cout << "[SolutionInBasis] max dev rho " << drho << " F " << df
            << " spe " << de << "\n";
  EXPECT_LT(drho, 1.0e-8);  // HF is converged to 1e-8 only
  EXPECT_LT(df, 1.0e-7);
  EXPECT_LT(de, 1.0e-6);
}

// The occupations are those of the caller: with the oscillator basis (C = 1)
// and the reference occupations E_HF is the oscillator-basis reference energy,
// above the converged E_HF (variational principle).
TEST_F(HfProperties, SolutionInBasisOscillatorEnergyIsAboveHf) {
  const auto& s = GetState();
  const auto n = s.r.C.rows();
  linalg::Matrix<double> one(n, n);
  for (std::int64_t i = 0; i < n; ++i) one(i, i) = 1.0;
  const hf::HFResult m =
      hf::SolutionInBasis(s.H, nullptr, one, s.r.occupations);
  EXPECT_GT(m.e_hf, s.r.e_hf);
  EXPECT_NEAR(m.e_hf, m.e1 + m.e2 + m.e3, 1.0e-12);
  const linalg::Matrix<double> bad(n - 1, n - 1);
  EXPECT_THROW(hf::SolutionInBasis(s.H, nullptr, bad, s.r.occupations),
               imsrg::InvariantError);
}

}  // namespace
}  // namespace imsrg::hftest
