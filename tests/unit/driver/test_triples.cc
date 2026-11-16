// SPDX-License-Identifier: GPL-3.0-only
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
/// `legacy` sets `imsrg3.legacy_omega_order` (imsrg++'s Omega-stack order, the
/// one the golden was made with, D49 / M10-Q40); only then the golden checks
/// apply. Returns dE.
double RunCase(const std::string& nucleus, int emax, const std::string& golden,
               int steps_ref, bool legacy) {
  const auto dir = FreshDir("triples_" + golden + (legacy ? "_legacy" : ""));
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n[reference]\nnucleus = \""
    << nucleus << "\"\n\n[interaction.two_body]\nfile = \""
    << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[hartree_fock]\nmixing = \"linear\"\n\n"
    << "[imsrg]\nmethod = \"magnus\"\nperturbative_triples = true\n\n"
    << (legacy ? "[imsrg3]\nlegacy_omega_order = true\n\n" : "")
    << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  const RunResult r = Execute(src, dir);
  EXPECT_FALSE(r.planned) << r.note;
  if (r.planned) return 0.0;

  const Json log =
      ParseJson(Slurp(GoldenFile("IT-8/" + golden + "/result.json")));
  const Json harness =
      ParseJson(Slurp(GoldenFile("IT-8/" + golden + "/harness_triples.json")));
  const Json& stage = r.results["imsrg"]["stages"].items.at(0);
  const int steps = static_cast<int>(stage["steps"].number);
  std::printf("[IT-8 %s%s] steps %d (golden %d)\n", golden.c_str(),
              legacy ? " legacy order" : "", steps, steps_ref);
  EXPECT_EQ(steps, steps_ref);
  EXPECT_EQ(static_cast<int>(harness["steps"].number), steps_ref);

  EXPECT_TRUE(r.results["imsrg"].has("perturbative_triples"));
  if (!r.results["imsrg"].has("perturbative_triples")) return 0.0;
  const Json& pt = r.results["imsrg"]["perturbative_triples"];
  const double de = pt["de"].number;
  const double e0 = r.results["imsrg"]["e0"].number;
  const double de_log = log["perturbative_triples"]["dE"].number;
  const double de_harness = harness["dE"].number;
  if (!legacy) {
    EXPECT_NEAR(pt["e0_plus_de"].number, e0 + de, 1.0e-12);
    EXPECT_LT(de, 0.0);
    return de;
  }
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
  return de;
}

/// The golden is imsrg++'s: the legacy order must reproduce it (D19 1e-4,
/// harness 1e-7). The transformation order (default) differs by the old-code
/// bug (design/30 M10-D49-2); the shift is logged and bounded (the flow is the
/// same, only the omega of the triples differs).
void RunBoth(const std::string& nucleus, int emax, const std::string& golden,
             int steps_ref) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const double de_old = RunCase(nucleus, emax, golden, steps_ref, true);
  const double de_new = RunCase(nucleus, emax, golden, steps_ref, false);
  std::printf(
      "[IT-8 %s] dE transformation order %.12f, imsrg++ order %.12f (shift "
      "%.3e)\n",
      golden.c_str(), de_new, de_old, de_new - de_old);
  EXPECT_LE(std::abs(de_new - de_old), 1.0e-2);
}

TEST(Triples, IT8_He4_Emax2) { RunBoth("He4", 2, "He4_emax2", 192); }

TEST(Triples, IT8_O16_Emax3) { RunBoth("O16", 3, "emax3", 154); }

}  // namespace
}  // namespace imsrg::it
