// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Hartree-Fock at emax 4, O16, with and without the NO2B three-body
// interaction.
#include <chrono>
#include <iostream>

#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::hftest {

namespace fs = std::filesystem;
namespace {

fs::path Emax4Dir() {
  return fs::path(IMSRG_GENERATED_GOLDEN_DIR) / "hamiltonian/emax4_O16";
}

TEST(HfGoldenEmax4, TwoNucleon) {
  if (!fs::exists(Emax4Dir() / "H.dump"))
    GTEST_SKIP() << "no generated golden in " << Emax4Dir();
  const op::Operator H = LoadHamiltonian(Emax4Dir());
  const Golden g =
      LoadGolden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax4_O16_2N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::Solve(H, nullptr);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::cout << "[emax4_2N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  CompareToGolden(r, g, "emax4_2N", 1.0e-8, 1.0e-8, 1.0e-7);
  EXPECT_NEAR(r.e_hf, -147.89357642845789, 1.0e-8);
  EXPECT_EQ(r.iterations, 30);
}

TEST(HfGoldenEmax4, TwoPlusThreeNucleon) {
  const fs::path file = No2bFile();
  if (file.empty())
    GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  if (!fs::exists(Emax4Dir() / "H.dump"))
    GTEST_SKIP() << "no generated golden in " << Emax4Dir();
  const op::Operator H = LoadHamiltonian(Emax4Dir());
  const op::No2bThreeBody v3 = LoadV3(file, {4, 8, 12, 4});
  const Golden g =
      LoadGolden(fs::path(IMSRG_GOLDEN_DIR) / "hf/emax4_O16_2N3N/hf.json");
  const auto t0 = std::chrono::steady_clock::now();
  const hf::HFResult r = hf::Solve(H, &v3);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::cout << "[emax4_2N3N] solve time " << secs << " s\n";
  RecordProperty("solve_seconds", std::to_string(secs));
  CompareToGolden(r, g, "emax4_2N3N", 1.0e-8, 1.0e-8, 1.0e-7);
  EXPECT_NEAR(r.e_hf, -313.16848141064463, 1.0e-8);
  EXPECT_EQ(r.iterations, 25);
}

}  // namespace
}  // namespace imsrg::hftest
