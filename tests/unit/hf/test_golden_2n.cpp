// Hartree-Fock (linear mixing, 2N only) against the imsrg++ harness output at emax 2, O16.
#include <chrono>

#include "hf_test_support.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

TEST(HfGolden2N, Emax2O16Linear) {
  const op::Operator H = load_hamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  const Golden g = load_golden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::solve(H, nullptr);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "[emax2_2N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  compare_to_golden(r, g, "emax2_2N", 1e-8, 1e-8, 1e-7);
  EXPECT_NEAR(r.e_hf, -134.03044657048093, 1e-8);
  EXPECT_NEAR(r.e1, 280.30704887012467, 1e-8);
  EXPECT_NEAR(r.e2, -414.33749544060561, 1e-8);
  EXPECT_EQ(r.e3, 0.0);
  EXPECT_EQ(r.iterations, 20);
}

}  // namespace
