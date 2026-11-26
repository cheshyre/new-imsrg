// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Acceptance tests of pipeline stage 1 (prepare_hamiltonian /
// prepare_operator): O16 and Ne20 (sd-shell) at emax 4 against the HF and
// golden values. Skipped without the input files.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/operator/compare.h"

namespace imsrg::driver::pipeline {
namespace {

namespace fs = std::filesystem;

const fs::path kVnn = fs::path(IMSRG_TEST_DATA_DIR) /
                      "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
const fs::path kV3 =
    fs::path(IMSRG_TEST_DATA_DIR) /
    "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";

std::string Toml(const std::string& reference_block, bool with_3n,
                 const std::string& extra = "") {
  std::ostringstream t;
  t << "[basis]\nemax = 4\nhw = 16.0\n"
    << extra << "\n"
    << reference_block << "\n[interaction.two_body]\nfile = \"" << kVnn.string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n";
  if (with_3n) {
    t << "[interaction.three_body]\nfile = \"" << kV3.string()
      << "\"\ntype = \"no2b\"\ne1max = 6\ne2max = 12\ne3max = 18\n";
  }
  return t.str();
}

RunConfig Load(const std::string& text) {
  ConfigSources s;
  s.config_text = text;
  auto r = LoadConfig(s, LoadOptions{.check_files = false});
  if (!r.has_value()) {
    std::string msg;
    for (const auto& i : r.error()) msg += i.ToString() + "\n";
    throw std::runtime_error("config does not load:\n" + msg);
  }
  return r.value();
}

OperatorRequest Request(const std::string& spec) {
  auto r = ParseOperatorSpec(spec, ModelSpaceInfo{.emax = 4});
  if (!r.has_value()) throw std::runtime_error("bad operator spec " + spec);
  return r.value().front();
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

bool Have(const fs::path& p) { return fs::exists(p); }

constexpr char kO16[] = "[reference]\nnucleus = \"O16\"\n";

TEST(PipelineHamiltonian, O16TwoBody) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph = PrepareHamiltonian(Load(Toml(kO16, false)));
  std::cout << "[time] O16 2N: " << SecondsSince(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] O16 2N E_HF - golden = "
            << ph.hf->e_hf - -147.89357642845789 << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -147.89357642845789, 1.0e-6);
  EXPECT_NEAR(ph.hno.zero_body(), -147.89357642845789, 1.0e-6);
  EXPECT_NEAR(ph.e_reference, ph.hf->e_hf, 1.0e-12);
  EXPECT_EQ(ph.mass_number, 16);
  EXPECT_EQ(ph.proton_number, 8);
}

TEST(PipelineHamiltonian, HartreeFockMixingDiisAndLinearReachTheSameSolution) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const std::string head = "[hartree_fock]\nmixing = \"";
  const RunConfig diis = Load(Toml(kO16, false) + head + "diis\"\n");
  const RunConfig dflt = Load(Toml(kO16, false));
  const RunConfig lin = Load(Toml(kO16, false) + head + "linear\"\n");
  EXPECT_EQ(dflt.hartree_fock.mixing, HartreeFockMixing::kDiis);
  const PreparedHamiltonian pd = PrepareHamiltonian(diis);
  const PreparedHamiltonian pl = PrepareHamiltonian(lin);
  ASSERT_TRUE(pd.hf.has_value());
  ASSERT_TRUE(pl.hf.has_value());
  std::cout << "[dev] O16 2N E_HF diis - linear = " << pd.hf->e_hf - pl.hf->e_hf
            << ", iterations diis " << pd.hf->iterations << " linear "
            << pl.hf->iterations << "\n";
  EXPECT_NEAR(pd.hf->e_hf, pl.hf->e_hf, 1.0e-6);
  EXPECT_LT(pd.hf->iterations, pl.hf->iterations);
  EXPECT_TRUE(pd.hf->converged);
}

TEST(PipelineHamiltonian, O16WithNo2b) {
  if (!Have(kVnn) || !Have(kV3))
    GTEST_SKIP() << "missing input files in " << IMSRG_TEST_DATA_DIR;
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph =
      PrepareHamiltonian(Load(Toml(kO16, true, "e3max = 12")));
  std::cout << "[time] O16 2N+NO2B: " << SecondsSince(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] O16 NO2B E_HF - golden = "
            << ph.hf->e_hf - -313.16848141064463 << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -313.16848141064463, 1.0e-6);
}

TEST(PipelineHamiltonian, Ne20Sd) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const std::string ref =
      "[reference]\nnucleus = \"Ne20\"\n[valence]\nspace = \"sd-shell\"\n";
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph = PrepareHamiltonian(Load(Toml(ref, false)));
  std::cout << "[time] Ne20 2N: " << SecondsSince(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] Ne20 E_HF - IT-7 golden = " << ph.hf->e_hf - -170.4417504
            << " e1 = " << ph.hf->e1 << " e2 = " << ph.hf->e2
            << " it = " << ph.hf->iterations << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -170.4417504, 1.0e-6);
  EXPECT_NEAR(ph.hf->e1, 487.6161119, 1.0e-6);
  EXPECT_NEAR(ph.hf->e2, -658.0578624, 1.0e-6);
  EXPECT_EQ(ph.proton_number, 10);
}

TEST(PipelineHamiltonian, Rp2ZeroBodyMatchesIt6) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const RunConfig cfg = Load(Toml(kO16, false));
  const PreparedHamiltonian ph = PrepareHamiltonian(cfg);
  const op::Operator rp2 = PrepareOperator(Request("Rp2"), ph, cfg);
  std::cout << "[dev] Rp2 zero body - IT-6 golden = "
            << rp2.zero_body() - 4.294907 << " (" << rp2.zero_body() << ")\n";
  EXPECT_NEAR(rp2.zero_body(), 4.294907, 5.0e-7);
  EXPECT_EQ(rp2.model_space_ptr(), ph.hno.model_space_ptr());
}

// D35: Rso is its own catalog operator. Its reference expectation value for
// the closed-shell O16 vanishes (the spin-orbit pairs j = l +- 1/2 cancel in
// every filled shell, so the HF value of Rp2 is the one without Rso as well),
// while the operator itself is nonzero.
TEST(PipelineHamiltonian, RsoIsAnOperatorAndVanishesForClosedShells) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const RunConfig cfg = Load(Toml(kO16, false));
  const PreparedHamiltonian ph = PrepareHamiltonian(cfg);
  const op::Operator rso = PrepareOperator(Request("Rso"), ph, cfg);
  std::cout << "[dev] Rso zero body (O16) = " << rso.zero_body() << "\n";
  EXPECT_NEAR(rso.zero_body(), 0.0, 1.0e-10);
  EXPECT_GT(op::MaxAbs(rso).one_body, 0.05);
  EXPECT_EQ(rso.model_space_ptr(), ph.hno.model_space_ptr());
}

// FM0p_0 and FM0n_0 are the proton and neutron number operators (sqrt(4 pi)
// M_0(0)), so their reference expectation value in O16 is 8 each.
TEST(PipelineHamiltonian, NreftChargeAtZeroMomentumIsTheNumberOperator) {
  if (!Have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const RunConfig cfg = Load(Toml(kO16, false));
  const PreparedHamiltonian ph = PrepareHamiltonian(cfg);
  const op::Operator fp = PrepareOperator(Request("FM0p_0"), ph, cfg);
  const op::Operator fn = PrepareOperator(Request("FM0n_0"), ph, cfg);
  EXPECT_NEAR(fp.zero_body(), 8.0, 1.0e-8);
  EXPECT_NEAR(fn.zero_body(), 8.0, 1.0e-8);
}

TEST(PipelineHamiltonian, PlannedOptionsThrow) {
  const std::string base = Toml(kO16, false);
  const RunConfig nat = Load(Toml(kO16, false, "kind = \"nat\"") +
                             "[natural_orbitals]\nuse_occupations = true\n");
  EXPECT_THROW(PrepareHamiltonian(nat), ConfigError);
  try {
    PrepareHamiltonian(nat);
  } catch (const ConfigError& e) {
    EXPECT_NE(std::string(e.what()).find("E-PLANNED"), std::string::npos)
        << e.what();
  }
  EXPECT_THROW(PrepareHamiltonian(Load(Toml(kO16, false, "emax_unocc = 3"))),
               ConfigError);
}

}  // namespace
}  // namespace imsrg::driver::pipeline
