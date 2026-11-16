// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IT-9 (design/30 section 4): the IMSRG(3)-N7 Magnus flow through the driver
// (`[imsrg3] in_flow = true, variant = "n7"`, legacy `IMSRG3=true
// imsrg3_n7=true`) against the imsrg++ goldens of tests/golden/e2e/IT-9. The
// emax-2 cases are in the integration target; the emax-3 cases (30 and 21
// minutes in imsrg++) are compiled for the slow label only. The flow must take
// the golden number of steps and end at the golden E0 within 1e-3 MeV (D19);
// the deviations and the wall time are printed. The emax-2 cases also
// transform Rp2 with the IMSRG(3) commutators (imsrg++ `Operators=Rp2`, D49,
// M10-Q61: the residual three-body part is dropped before the output, a rank-2
// operator is transformed with the IMSRG(3) terms) and compare its zero-body
// part with the golden `intRp2.snt` of the patched imsrg++ (1e-6 fm^2 as
// IT-6). Skips without the 2N input file.
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"
#include "tests/unit/driver/integration/snt_compare.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// One IT-9 case: `nucleus` at `emax`, atan, Magnus, linear HF mixing, N7;
/// with `rp2` the point-proton radius operator is transformed as well.
void RunCase(const std::string& nucleus, int emax, const std::string& golden,
             bool rp2 = false) {
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
    << (rp2 ? "[operators]\nlist = [\"Rp2\"]\n\n" : "")
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

  if (rp2) {
    const auto snt = GoldenFile("IT-9/" + golden + "/intRp2.snt");
    ASSERT_TRUE(std::filesystem::exists(snt)) << snt;
    const SntData want = ReadSnt(snt);
    ASSERT_TRUE(r.results["operators"].has("Rp2"));
    const double rp2_hf = r.results["operators"]["Rp2"]["reference"].number;
    const double rp2_hf_ref = gold["operators"].items.at(0)["HF_value"].number;
    const double rp2_imsrg = r.results["operators"]["Rp2"]["imsrg"].number;
    std::printf(
        "[IT-9 %s] Rp2 HF %.8f (golden %.6f), IMSRG %.8f golden %.8f dev "
        "%.3e fm^2\n",
        golden.c_str(), rp2_hf, rp2_hf_ref, rp2_imsrg, want.zero_body,
        rp2_imsrg - want.zero_body);
    EXPECT_NEAR(rp2_hf, rp2_hf_ref, 1.0e-5);
    EXPECT_NEAR(rp2_imsrg, want.zero_body, 1.0e-6);
    // The core-only valence file of the operator has no one- or two-body part.
    const SntData got = ReadSnt(r.results["operators"]["Rp2"]["file"].text);
    EXPECT_NEAR(got.zero_body, want.zero_body, 1.0e-6);
    EXPECT_EQ(got.one_body.size(), want.one_body.size());
    EXPECT_EQ(got.two_body.size(), want.two_body.size());
  }
}

#ifdef IMSRG_IT9_SLOW

TEST(Imsrg3N7, He4Emax3) { RunCase("He4", 3, "He4_emax3"); }

TEST(Imsrg3N7, O16Emax3) { RunCase("O16", 3, "O16_emax3"); }

#else

TEST(Imsrg3N7, He4Emax2) { RunCase("He4", 2, "He4_emax2", true); }

TEST(Imsrg3N7, O16Emax2) { RunCase("O16", 2, "O16_emax2", true); }

/// Result of one He4 emax-2 IMSRG(3)-N7 run for the OccNat3Cut test.
struct CutRun {
  int steps = 0;
  double e0 = 0.0;
  std::int64_t kept = -1;   ///< Three-body rows kept ("3-body states kept").
  std::int64_t total = -1;  ///< Rows before the cuts ("out of").
};

/// Parses "3-body states kept: <kept> out of <total>" of the IMSRG(3)-N7 log
/// line in `log` into `run` (left at -1 if the line is missing).
void ParseKept(const std::string& log, CutRun& run) {
  const std::string key = "IMSRG(3)-N7:";
  const std::size_t at = log.find(key);
  if (at == std::string::npos) return;
  const std::string tail = log.substr(at);
  const std::size_t k = tail.find("3-body states kept: ");
  if (k == std::string::npos) return;
  std::int64_t kept = -1;
  std::int64_t total = -1;
  if (std::sscanf(tail.c_str() + k,
                  "3-body states kept: %" SCNd64 " out of %" SCNd64, &kept,
                  &total) == 2) {
    run.kept = kept;
    run.total = total;
  }
}

/// He4 emax 2, atan, Magnus, N7. `toml_cut` (> 0) adds `imsrg3.occ_nat3_cut`
/// to the TOML; empty means no cut (the key is absent; the legacy layer maps
/// `OccNat3Cut <= 0` to an absent key, test_legacy_parameters).
CutRun RunCut(const std::string& tag, const std::string& toml_cut) {
  const auto dir = FreshDir("imsrg3n7cut_" + tag);
  std::ostringstream t;
  t << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[hartree_fock]\nmixing = \"linear\"\n\n"
    << "[imsrg]\nmethod = \"magnus\"\n\n[imsrg.step]\nsmax = 200\n\n"
    << "[imsrg3]\nin_flow = true\nvariant = \"n7\"\n"
    << (toml_cut.empty() ? "" : "occ_nat3_cut = " + toml_cut + "\n")
    << "\n[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  testing::internal::CaptureStdout();
  RunResult r;
  try {
    r = Execute(src, dir);
  } catch (...) {
    (void)testing::internal::GetCapturedStdout();
    throw;
  }
  CutRun out;
  ParseKept(testing::internal::GetCapturedStdout(), out);
  if (r.planned) ADD_FAILURE() << r.note;
  out.steps = static_cast<int>(
      r.results["imsrg"]["stages"].items.at(0)["steps"].number);
  out.e0 = r.results["imsrg"]["e0"].number;
  return out;
}

/// imsrg3.occ_nat3_cut through the IMSRG(3) flow: no cut (absent key) is
/// reproducible bitwise; a positive cut
/// drops three-body kets (see the log line "3-body states kept") and moves E0
/// by a small amount.
TEST(Imsrg3N7, He4Emax2OccNat3Cut) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const CutRun cut = RunCut("cut", "1.0e-9");
  const CutRun uncut = RunCut("none", "");
  const CutRun again = RunCut("again", "");
  std::printf("[OccNat3Cut] uncut  steps %d E0 %.17g\n", uncut.steps, uncut.e0);
  std::printf("[OccNat3Cut] again  steps %d E0 %.17g\n", again.steps, again.e0);
  std::printf("[OccNat3Cut] cut    steps %d E0 %.17g  dE0 %.3e\n", cut.steps,
              cut.e0, cut.e0 - uncut.e0);
  EXPECT_EQ(again.steps, uncut.steps);
  EXPECT_EQ(again.e0, uncut.e0);
  EXPECT_NE(cut.e0, uncut.e0);
  std::printf("[OccNat3Cut] 3-body rows kept: cut %" PRId64 " of %" PRId64
              ", uncut %" PRId64 " of %" PRId64 "\n",
              cut.kept, cut.total, uncut.kept, uncut.total);
  ASSERT_GT(uncut.kept, 0) << "log line not found";
  ASSERT_GT(cut.kept, 0) << "log line not found";
  EXPECT_LT(cut.kept, uncut.kept);
  EXPECT_LE(std::abs(cut.e0 - uncut.e0), 1.0e-1);
}

#endif

}  // namespace
}  // namespace imsrg::it
