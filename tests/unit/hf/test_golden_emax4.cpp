// Hartree-Fock at emax 4, O16, with and without the NO2B three-body interaction.
#include <chrono>

#include "hf_test_support.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

fs::path emax4_dir() { return fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16"; }

TEST(HfGoldenEmax4, TwoNucleon) {
  if (!fs::exists(emax4_dir() / "H.dump")) GTEST_SKIP() << "no generated golden in " << emax4_dir();
  const op::Operator H = load_hamiltonian(emax4_dir());
  const Golden g = load_golden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax4_O16_2N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::solve(H, nullptr);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "[emax4_2N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  compare_to_golden(r, g, "emax4_2N", 1e-8, 1e-8, 1e-7);
  EXPECT_NEAR(r.e_hf, -147.89357642845789, 1e-8);
  EXPECT_EQ(r.iterations, 30);
}

TEST(HfGoldenEmax4, TwoPlusThreeNucleon) {
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  if (!fs::exists(emax4_dir() / "H.dump")) GTEST_SKIP() << "no generated golden in " << emax4_dir();
  const op::Operator H = load_hamiltonian(emax4_dir());
  const op::No2bThreeBody v3 = load_v3(file, {4, 8, 12, 4});
  const Golden g = load_golden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax4_O16_2N3N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::solve(H, &v3);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "[emax4_2N3N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  compare_to_golden(r, g, "emax4_2N3N", 1e-8, 1e-8, 1e-7);
  EXPECT_NEAR(r.e_hf, -313.16848141064463, 1e-8);
  EXPECT_EQ(r.iterations, 25);
}

}  // namespace
