// End-to-end Hartree-Fock inputs: the Hamiltonian is assembled from the real input files
// (me2j + NO2B + relative kinetic energy) instead of a dump, as in integration tests IT-1/IT-4.
#include <chrono>

#include "hf_test_support.hpp"

#include "imsrg/basis/model_space.hpp"
#include "imsrg/io/me2j.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/opslib/center_of_mass.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

fs::path me2j_file() {
  const fs::path p =
      fs::path(IMSRG_TEST_DATA_DIR) / "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  return fs::exists(p) ? p : fs::path{};
}

/// O16, emax 4, hw 16: H = T_rel + V from the me2j file. Stores the read time.
op::Operator build_hamiltonian(const fs::path& me2j, double* read_seconds) {
  const auto ms = basis::ModelSpace::from_names({.emax = 4}, 16.0, "O16", "O16");
  const auto t0 = std::chrono::steady_clock::now();
  op::Operator v = io::read_me2j(me2j, {.file = {14, 28, 14}}, ms);
  op::Operator h = opslib::t_rel(ms, 16) + v;
  *read_seconds = seconds_since(t0);
  return h;
}

void run_case(const op::No2bThreeBody* v3, const std::string& golden_dir, const std::string& tag,
              const double* it1_ehf) {
  const fs::path me2j = me2j_file();
  if (me2j.empty()) GTEST_SKIP() << "me2j file not found in " << IMSRG_TEST_DATA_DIR;
  double read_s = 0.0;
  const op::Operator H = build_hamiltonian(me2j, &read_s);
  const Golden g = load_golden(fs::path(IMSRG_GOLDEN_DIR) / "hf" / golden_dir / "hf.json");
  for (const hf::Mixing m : {hf::Mixing::Linear, hf::Mixing::Diis}) {
    const std::string name = tag + (m == hf::Mixing::Linear ? "_linear" : "_diis");
    hf::HFOptions opt;
    opt.mixing = m;
    const auto t0 = std::chrono::steady_clock::now();
    const hf::HFResult r = hf::solve(H, v3, opt);
    const double solve_s = seconds_since(t0);
    std::cout << "[" << name << "] read " << read_s << " s, solve " << solve_s << " s, iterations "
              << r.iterations << ", dE_HF " << r.e_hf - g.e_hf << ", max|dspe| "
              << max_spe_diff(r, g) << "\n";
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.e_hf, g.e_hf, 1e-4);
    EXPECT_LE(max_spe_diff(r, g), 1e-4);
    if (it1_ehf != nullptr) {
      std::cout << "[" << name << "] IT-1 result.json dE_HF " << r.e_hf - *it1_ehf << "\n";
      EXPECT_NEAR(r.e_hf, *it1_ehf, 1e-4);
    }
  }
}

TEST(HfIntegration, IT1_TwoNucleon) {
  const Json res = read_json(fs::path(IMSRG_GOLDEN_DIR) / "e2e/IT-1/result.json");
  const double ehf = res.at("hf").at("EHF").number;
  run_case(nullptr, "emax4_O16_2N", "IT1", &ehf);
}

TEST(HfIntegration, IT4_TwoPlusThreeNucleon) {
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = load_v3(file, {4, 8, 12, 4});
  run_case(&v3, "emax4_O16_2N3N", "IT4", nullptr);
}

}  // namespace
