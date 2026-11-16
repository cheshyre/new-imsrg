// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IT-9 (design/30 section 4): the IMSRG(3)-N7 Magnus flow through the driver
// (`[imsrg3] in_flow = true, variant = "n7"`, legacy `IMSRG3=true
// imsrg3_n7=true`) against the imsrg++ goldens of tests/golden/e2e/IT-9. The
// emax-2 cases are in the integration target; the emax-3 cases (30 and 21
// minutes in imsrg++) are compiled for the slow label only. The flow must take
// the golden number of steps and end at the golden E0 within 1e-3 MeV (D19);
// the deviations and the wall time are printed. Skips without the 2N input
// file.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// One IT-9 case: `nucleus` at `emax`, atan, Magnus, linear HF mixing, N7.
void RunCase(const std::string& nucleus, int emax, const std::string& golden) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir("imsrg3n7_" + golden);
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n[reference]\nnucleus = \""
    << nucleus << "\"\n\n[interaction.two_body]\nfile = \""
    << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[hartree_fock]\nmixing = \"linear\"\n\n"
    << "[imsrg]\nmethod = \"magnus\"\n\n[imsrg.step]\nsmax = 200\n\n"
    << "[imsrg3]\nin_flow = true\nvariant = \"n7\"\n\n"
    << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  const auto t0 = std::chrono::steady_clock::now();
  const RunResult r = Execute(src, dir);
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  ASSERT_FALSE(r.planned) << r.note;

  const Json gold =
      ParseJson(Slurp(GoldenFile("IT-9/" + golden + "/result.json")));
  const Json& gflow = gold["flow"];
  const int steps_ref = static_cast<int>(gflow["n_steps"].number);
  const double e0_ref = gflow["final_E0"].number;
  const Json& stage = r.results["imsrg"]["stages"].items.at(0);
  const int steps = static_cast<int>(stage["steps"].number);
  const double e0 = r.results["imsrg"]["e0"].number;
  std::printf("[IT-9 %s] steps %d (golden %d)\n", golden.c_str(), steps,
              steps_ref);
  std::printf("[IT-9 %s] E0 %.12f golden %.9f dE0 %.3e  wall %.1f s\n",
              golden.c_str(), e0, e0_ref, e0 - e0_ref, wall);
  EXPECT_EQ(steps, steps_ref);
  EXPECT_LE(std::abs(e0 - e0_ref), 1.0e-3);  // D19

  // Last row of the flow file against the last row of the imsrg++ flow file.
  ASSERT_GT(r.flow.col("norm_omega_3b"), 0u);
  const std::size_t last = r.flow.rows.size() - 1;
  const double om3 = r.flow.at(last, "norm_omega_3b");
  const double om3_ref = gflow["last"]["norm_Omega3"].number;
  const double om2 = r.flow.at(last, "norm_omega_2b");
  const double om2_ref = gflow["last"]["norm_Omega2"].number;
  std::printf(
      "[IT-9 %s] final |Omega_2b| %.9f (golden %.9f), |Omega_3b| %.9f "
      "(golden %.9f)\n",
      golden.c_str(), om2, om2_ref, om3, om3_ref);
  EXPECT_NEAR(om2, om2_ref, 1.0e-5);
  EXPECT_NEAR(om3, om3_ref, 1.0e-5);
  EXPECT_GT(om3, 0.0);
}

#ifdef IMSRG_IT9_SLOW

TEST(Imsrg3N7, He4Emax3) { RunCase("He4", 3, "He4_emax3"); }

TEST(Imsrg3N7, O16Emax3) { RunCase("O16", 3, "O16_emax3"); }

#else

TEST(Imsrg3N7, He4Emax2) { RunCase("He4", 2, "He4_emax2"); }

TEST(Imsrg3N7, O16Emax2) { RunCase("O16", 2, "O16_emax2"); }

#endif

}  // namespace
}  // namespace imsrg::it
