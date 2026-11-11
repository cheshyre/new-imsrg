// Acceptance tests of pipeline stage 1 (prepare_hamiltonian / prepare_operator): O16 and Ne20
// (sd-shell) at emax 4 against the HF and golden values. Skipped without the input files.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/driver/config/load_config.hpp"
#include "imsrg/driver/operators/operator_request.hpp"
#include "imsrg/driver/pipeline/hamiltonian.hpp"

namespace imsrg::driver::pipeline {
namespace {

namespace fs = std::filesystem;

const fs::path kVnn =
    fs::path(IMSRG_TEST_DATA_DIR) / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
const fs::path kV3 =
    fs::path(IMSRG_TEST_DATA_DIR) / "NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin";

std::string toml(const std::string& reference_block, bool with_3n, const std::string& extra = "") {
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

RunConfig load(const std::string& text) {
  ConfigSources s;
  s.config_text = text;
  auto r = load_config(s, LoadOptions{.check_files = false});
  if (!r.has_value()) {
    std::string msg;
    for (const auto& i : r.error()) msg += i.to_string() + "\n";
    throw std::runtime_error("config does not load:\n" + msg);
  }
  return r.value();
}

OperatorRequest request(const std::string& spec) {
  auto r = parse_operator_spec(spec, ModelSpaceInfo{.emax = 4});
  if (!r.has_value()) throw std::runtime_error("bad operator spec " + spec);
  return r.value().front();
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool have(const fs::path& p) { return fs::exists(p); }

const std::string kO16 = "[reference]\nnucleus = \"O16\"\n";

TEST(PipelineHamiltonian, O16TwoBody) {
  if (!have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph = prepare_hamiltonian(load(toml(kO16, false)));
  std::cout << "[time] O16 2N: " << seconds_since(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] O16 2N E_HF - golden = " << ph.hf->e_hf - -147.89357642845789 << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -147.89357642845789, 1e-6);
  EXPECT_NEAR(ph.hno.zero_body(), -147.89357642845789, 1e-6);
  EXPECT_NEAR(ph.e_reference, ph.hf->e_hf, 1e-12);
  EXPECT_EQ(ph.mass_number, 16);
  EXPECT_EQ(ph.proton_number, 8);
}

TEST(PipelineHamiltonian, O16WithNo2b) {
  if (!have(kVnn) || !have(kV3)) GTEST_SKIP() << "missing input files in " << IMSRG_TEST_DATA_DIR;
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph = prepare_hamiltonian(load(toml(kO16, true, "e3max = 12")));
  std::cout << "[time] O16 2N+NO2B: " << seconds_since(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] O16 NO2B E_HF - golden = " << ph.hf->e_hf - -313.16848141064463 << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -313.16848141064463, 1e-6);
}

TEST(PipelineHamiltonian, Ne20Sd) {
  if (!have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const std::string ref = "[reference]\nnucleus = \"Ne20\"\n[valence]\nspace = \"sd-shell\"\n";
  const auto t0 = std::chrono::steady_clock::now();
  const PreparedHamiltonian ph = prepare_hamiltonian(load(toml(ref, false)));
  std::cout << "[time] Ne20 2N: " << seconds_since(t0) << " s\n";
  ASSERT_TRUE(ph.hf.has_value());
  std::cout << "[dev] Ne20 E_HF - IT-7 golden = " << ph.hf->e_hf - -170.4417504
            << " e1 = " << ph.hf->e1 << " e2 = " << ph.hf->e2 << " it = " << ph.hf->iterations
            << "\n";
  EXPECT_NEAR(ph.hf->e_hf, -170.4417504, 1e-6);
  EXPECT_NEAR(ph.hf->e1, 487.6161119, 1e-6);
  EXPECT_NEAR(ph.hf->e2, -658.0578624, 1e-6);
  EXPECT_EQ(ph.proton_number, 10);
}

TEST(PipelineHamiltonian, Rp2ZeroBodyMatchesIt6) {
  if (!have(kVnn)) GTEST_SKIP() << "missing " << kVnn;
  const RunConfig cfg = load(toml(kO16, false));
  const PreparedHamiltonian ph = prepare_hamiltonian(cfg);
  const op::Operator rp2 = prepare_operator(request("Rp2"), ph, cfg);
  std::cout << "[dev] Rp2 zero body - IT-6 golden = " << rp2.zero_body() - 4.294907 << " ("
            << rp2.zero_body() << ")\n";
  EXPECT_NEAR(rp2.zero_body(), 4.294907, 5e-7);
  EXPECT_EQ(rp2.model_space_ptr(), ph.hno.model_space_ptr());
}

TEST(PipelineHamiltonian, PlannedOptionsThrow) {
  const std::string base = toml(kO16, false);
  const RunConfig nat = load(toml(kO16, false, "kind = \"nat\""));
  EXPECT_THROW(prepare_hamiltonian(nat), ConfigError);
  try {
    prepare_hamiltonian(nat);
  } catch (const ConfigError& e) {
    EXPECT_NE(std::string(e.what()).find("E-PLANNED"), std::string::npos) << e.what();
  }
  EXPECT_THROW(prepare_hamiltonian(load(toml(kO16, false, "emax_unocc = 3"))), ConfigError);
}

}  // namespace
}  // namespace imsrg::driver::pipeline
