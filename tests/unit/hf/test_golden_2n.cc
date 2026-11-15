// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Hartree-Fock (linear mixing, 2N only) against the imsrg++ harness output at
// emax 2, O16.
#include <chrono>

#include "hf_test_support.h"

namespace {

using namespace imsrg;
using namespace imsrg::hftest;

TEST(HfGolden2N, Emax2O16Linear) {
  const op::Operator H =
      LoadHamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  const Golden g =
      LoadGolden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::Solve(H, nullptr);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::cout << "[emax2_2N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  CompareToGolden(r, g, "emax2_2N", 1.0e-8, 1.0e-8, 1.0e-7);
  EXPECT_NEAR(r.e_hf, -134.03044657048093, 1.0e-8);
  EXPECT_NEAR(r.e1, 280.30704887012467, 1.0e-8);
  EXPECT_NEAR(r.e2, -414.33749544060561, 1.0e-8);
  EXPECT_EQ(r.e3, 0.0);
  EXPECT_EQ(r.iterations, 20);
}

}  // namespace
