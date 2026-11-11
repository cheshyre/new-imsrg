// Normal-ordered Hamiltonian in the HF basis and basis transformation against imsrg++ goldens.
#include <chrono>

#include "hf_test_support.hpp"

#include "imsrg/operator/compare.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

/// Compares `hno` with the dump `file`, prints the deviations and returns the maximum.
double compare_dump(const op::Operator& hno, const fs::path& file, const std::string& tag) {
  const auto ref = op::read_operator_dump(file, hno.model_space_ptr());
  const op::PartMaxima d = op::max_abs_difference(hno, ref);
  std::cout << "[" << tag << "] max|dHNO|: zero " << d.zero_body << " one " << d.one_body << " two "
            << d.two_body << "\n";
  return d.max();
}

void check_hno(const fs::path& ham_dir, const fs::path& hno_dump, const op::No2bThreeBody* v3,
               const std::string& tag, double tol) {
  const op::Operator H = load_hamiltonian(ham_dir);
  const hf::HFResult r = hf::solve(H, v3);
  const auto t0 = std::chrono::steady_clock::now();
  const op::Operator hno = hf::normal_ordered_hamiltonian(H, v3, r);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "[" << tag << "] HNO time " << secs << " s\n";
  EXPECT_LE(compare_dump(hno, hno_dump, tag), tol);
}

TEST(HfHno, Emax2TwoNucleon) {
  check_hno(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
            fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N/HNO.dump", nullptr, "emax2_2N", 1e-8);
}

TEST(HfHno, ToHfBasis) {
  const op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  const hf::HFResult r = hf::solve(H, nullptr);
  const op::Operator t = hf::to_hf_basis(H, r);
  const double dev =
      compare_dump(t, fs::path(IMSRG_GOLDEN_DIR) / "operator/emax2/H_HFbasis.dump", "H_HFbasis");
  EXPECT_LE(dev, 1e-10);
}

TEST(HfHnoIntegration, Emax2TwoPlusThreeNucleon) {
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = load_v3(file, {2, 4, 12, 2});
  check_hno(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16",
            fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N3N/HNO.dump", &v3, "emax2_2N3N", 1e-8);
}

TEST(HfHnoReference, Emax4TwoNucleon) {
  const fs::path gen = fs::path(IMSRG_GENERATED_GOLDEN_DIR);
  if (!fs::exists(gen / "hf/emax4_O16_2N/HNO.dump") ||
      !fs::exists(gen / "hamiltonian/emax4_O16/H.dump"))
    GTEST_SKIP() << "no generated golden in " << gen;
  check_hno(gen / "hamiltonian/emax4_O16", gen / "hf/emax4_O16_2N/HNO.dump", nullptr, "emax4_2N",
            1e-8);
}

TEST(HfHnoReference, Emax4TwoPlusThreeNucleon) {
  const fs::path gen = fs::path(IMSRG_GENERATED_GOLDEN_DIR);
  const fs::path file = no2b_file();
  if (file.empty()) GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  if (!fs::exists(gen / "hf/emax4_O16_2N3N/HNO.dump") ||
      !fs::exists(gen / "hamiltonian/emax4_O16/H.dump"))
    GTEST_SKIP() << "no generated golden in " << gen;
  const op::No2bThreeBody v3 = load_v3(file, {4, 8, 12, 4});
  check_hno(gen / "hamiltonian/emax4_O16", gen / "hf/emax4_O16_2N3N/HNO.dump", &v3, "emax4_2N3N",
            1e-8);
}

}  // namespace
