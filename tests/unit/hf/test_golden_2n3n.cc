// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Hartree-Fock with the NO2B three-body interaction at emax 2, O16 (needs the
// NO2B file).
#include <chrono>
#include <iostream>

#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::hftest {

namespace fs = std::filesystem;
namespace {

TEST(HfGolden2N3N, Emax2O16Linear) {
  const fs::path file = No2bFile();
  if (file.empty())
    GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::Operator H =
      LoadHamiltonian(fs::path(IMSRG_GOLDEN_DIR) / "hamiltonian/emax2_O16");
  const op::No2bThreeBody v3 = LoadV3(file, {2, 4, 12, 2});
  const Golden g =
      LoadGolden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax2_O16_2N3N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::Solve(H, &v3);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::cout << "[emax2_2N3N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  CompareToGolden(r, g, "emax2_2N3N", 1.0e-8, 1.0e-8, 1.0e-7);
  EXPECT_NEAR(r.e_hf, -206.46003654384998, 1.0e-8);
  EXPECT_EQ(r.iterations, 22);
}

}  // namespace
}  // namespace imsrg::hftest
