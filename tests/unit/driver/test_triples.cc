// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// IT-8 (design/30 section 4): the perturbative-triples correction after the
// IMSRG(2) Magnus flow through the driver, against the imsrg++ goldens of
// tests/golden/e2e/IT-8: He4 emax 2 and O16 emax 3 (about 7 s each in the dev
// Debug build, so both are in the integration target). The flow must take the
// golden number of steps, dE agrees with the imsrg++ log (8 decimals) within
// 1e-4 MeV (D19) and with the 17-digit value of the reference harness
// (`harness_triples.json`) to a regression guard of 1e-7 (measured: 1e-15);
// the deviations are printed. Skips without the 2N input file.
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

/// One IT-8 case: `nucleus` at `emax`, atan, Magnus, linear HF mixing,
/// `[imsrg] perturbative_triples = true`, golden directory `IT-8/<golden>`.
void RunCase(const std::string& nucleus, int emax, const std::string& golden,
             int steps_ref) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir("triples_" + golden);
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n[reference]\nnucleus = \""
    << nucleus << "\"\n\n[interaction.two_body]\nfile = \""
    << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[hartree_fock]\nmixing = \"linear\"\n\n"
    << "[imsrg]\nmethod = \"magnus\"\nperturbative_triples = true\n\n"
    << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;

  const Json log =
      ParseJson(Slurp(GoldenFile("IT-8/" + golden + "/result.json")));
  const Json harness =
      ParseJson(Slurp(GoldenFile("IT-8/" + golden + "/harness_triples.json")));
  const Json& stage = r.results["imsrg"]["stages"].items.at(0);
  const int steps = static_cast<int>(stage["steps"].number);
  std::printf("[IT-8 %s] steps %d (golden %d)\n", golden.c_str(), steps,
              steps_ref);
  EXPECT_EQ(steps, steps_ref);
  EXPECT_EQ(static_cast<int>(harness["steps"].number), steps_ref);

  ASSERT_TRUE(r.results["imsrg"].has("perturbative_triples"));
  const Json& pt = r.results["imsrg"]["perturbative_triples"];
  const double de = pt["de"].number;
  const double e0 = r.results["imsrg"]["e0"].number;
  const double de_log = log["perturbative_triples"]["dE"].number;
  const double de_harness = harness["dE"].number;
  std::printf(
      "[IT-8 %s] dE %.12f  imsrg++ log %.8f (dev %.3e)  harness %.17g (dev "
      "%.3e)\n",
      golden.c_str(), de, de_log, de - de_log, de_harness, de - de_harness);
  std::printf("[IT-8 %s] E0 + dE %.9f  golden %.9f\n", golden.c_str(),
              pt["e0_plus_de"].number, harness["E0_plus_dE"].number);
  EXPECT_LE(std::abs(de - de_log), 1.0e-4);  // D19
  EXPECT_LE(std::abs(de - de_harness), 1.0e-7);
  EXPECT_NEAR(pt["e0_plus_de"].number, e0 + de, 1.0e-12);
  EXPECT_LT(de, 0.0);
}

TEST(Triples, IT8_He4_Emax2) { RunCase("He4", 2, "He4_emax2", 192); }

TEST(Triples, IT8_O16_Emax3) { RunCase("O16", 3, "emax3", 154); }

}  // namespace
}  // namespace imsrg::it
