// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// End-to-end Hartree-Fock inputs: the Hamiltonian is assembled from the real
// input files (me2j + NO2B + relative kinetic energy) instead of a dump, as in
// integration tests IT-1/IT-4.
#include <chrono>
#include <iostream>
#include <string>

#include "imsrg/basis/model_space.h"
#include "imsrg/io/me2j.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/opslib/center_of_mass.h"
#include "tests/unit/hf/hf_test_support.h"

namespace imsrg::hftest {
namespace {

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

fs::path Me2jFile() {
  const fs::path p = fs::path(IMSRG_TEST_DATA_DIR) /
                     "VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz";
  return fs::exists(p) ? p : fs::path{};
}

/// O16, emax 4, hw 16: H = T_rel + V from the me2j file. Stores the read time.
op::Operator BuildHamiltonian(const fs::path& me2j, double* read_seconds) {
  const auto ms = basis::ModelSpace::FromNames({.emax = 4}, 16.0, "O16", "O16");
  const auto t0 = std::chrono::steady_clock::now();
  op::Operator v = io::ReadMe2j(me2j, {.file = {14, 28, 14}}, ms);
  op::Operator h = opslib::TRel(ms, 16) + v;
  *read_seconds = SecondsSince(t0);
  return h;
}

void RunCase(const op::No2bThreeBody* v3, const std::string& golden_dir,
             const std::string& tag, const double* it1_ehf) {
  const fs::path me2j = Me2jFile();
  if (me2j.empty())
    GTEST_SKIP() << "me2j file not found in " << IMSRG_TEST_DATA_DIR;
  double read_s = 0.0;
  const op::Operator H = BuildHamiltonian(me2j, &read_s);
  const Golden g =
      LoadGolden(fs::path(IMSRG_GOLDEN_DIR) / "hf" / golden_dir / "hf.json");
  for (const hf::Mixing m : {hf::Mixing::kLinear, hf::Mixing::kDiis}) {
    const std::string name =
        tag + (m == hf::Mixing::kLinear ? "_linear" : "_diis");
    hf::HFOptions opt;
    opt.mixing = m;
    const auto t0 = std::chrono::steady_clock::now();
    const hf::HFResult r = hf::Solve(H, v3, opt);
    const double solve_s = SecondsSince(t0);
    std::cout << "[" << name << "] read " << read_s << " s, solve " << solve_s
              << " s, iterations " << r.iterations << ", dE_HF "
              << r.e_hf - g.e_hf << ", max|dspe| " << MaxSpeDiff(r, g) << "\n";
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.e_hf, g.e_hf, 1.0e-4);
    EXPECT_LE(MaxSpeDiff(r, g), 1.0e-4);
    if (it1_ehf != nullptr) {
      std::cout << "[" << name << "] IT-1 result.json dE_HF "
                << r.e_hf - *it1_ehf << "\n";
      EXPECT_NEAR(r.e_hf, *it1_ehf, 1.0e-4);
    }
  }
}

TEST(HfIntegration, IT1_TwoNucleon) {
  const Json res =
      ReadJson(fs::path(IMSRG_GOLDEN_DIR) / "e2e/IT-1/result.json");
  const double ehf = res.at("hf").at("EHF").number;
  RunCase(nullptr, "emax4_O16_2N", "IT1", &ehf);
}

TEST(HfIntegration, IT4_TwoPlusThreeNucleon) {
  const fs::path file = No2bFile();
  if (file.empty())
    GTEST_SKIP() << "NO2B file not found in " << IMSRG_TEST_DATA_DIR;
  const op::No2bThreeBody v3 = LoadV3(file, {4, 8, 12, 4});
  RunCase(&v3, "emax4_O16_2N3N", "IT4", nullptr);
}

}  // namespace
}  // namespace imsrg::hftest
