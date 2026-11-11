// Pulay DIIS mixing must reach the same Hartree-Fock solution as the imsrg++ linear mixing.
#include <functional>

#include "hf_test_support.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

void compare_modes(const op::Operator& H, const op::No2bThreeBody* v3, const std::string& tag) {
  const hf::HFResult lin = hf::solve(H, v3);
  hf::HFOptions opt;
  opt.mixing = hf::Mixing::Diis;
  const hf::HFResult diis = hf::solve(H, v3, opt);
  std::cout << "[" << tag << "] iterations: linear " << lin.iterations << ", diis "
            << diis.iterations << "; dE_HF " << diis.e_hf - lin.e_hf << "\n";
  ::testing::Test::RecordProperty(tag + "_iterations_linear", std::to_string(lin.iterations));
  ::testing::Test::RecordProperty(tag + "_iterations_diis", std::to_string(diis.iterations));
  EXPECT_TRUE(lin.converged);
  EXPECT_TRUE(diis.converged);
  EXPECT_NEAR(diis.e_hf, lin.e_hf, 1e-8);
  double dspe = 0.0;
  for (std::size_t i = 0; i < lin.spe.size(); ++i) {
    dspe = std::max(dspe, std::abs(diis.spe[i] - lin.spe[i]));
  }
  std::cout << "[" << tag << "] max|dspe| diis vs linear " << dspe << "\n";
  EXPECT_LE(dspe, 1e-6);
}

TEST(HfDiis, Emax2TwoNucleon) {
  const op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  compare_modes(H, nullptr, "emax2_2N");
}

TEST(HfDiis, Emax2TwoPlusThreeNucleon) {
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  const op::No2bThreeBody v3 = load_v3(file, {2, 4, 12, 2});
  compare_modes(H, &v3, "emax2_2N3N");
}

TEST(HfDiis, Emax4) {
  const fs::path dir = fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16";
  if (!fs::exists(dir / "H.dump")) GTEST_SKIP() << "no generated golden in " << dir;
  const op::Operator H = load_hamiltonian(dir);
  compare_modes(H, nullptr, "emax4_2N");
  const fs::path file = no2b_file();
  if (file.empty()) return;
  const op::No2bThreeBody v3 = load_v3(file, {4, 8, 12, 4});
  compare_modes(H, &v3, "emax4_2N3N");
}

}  // namespace
