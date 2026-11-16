// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// IT-10 (design/30 section 4): IMSRG(3f2) through the driver against the
// imsrg++ goldens of tests/golden/e2e/IT-10. The config is `[imsrg3]
// factorized = true` only: the recipe supplies hunter-gatherer Magnus and
// omega_norm_max = 0.1 (D32). Step and gather counts must be identical, E0 is
// accepted within 1e-3 MeV (D19) and the whole trajectory row by row within
// 1e-8; deviations are printed. The O16 emax 3 case is compiled into the slow
// target (`IMSRG_IT10_SLOW`), the others into the integration target. Skips
// without the 2N input file.

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// One trajectory row: s, E0, ||eta||, ||Omega||.
struct Row {
  double s = 0.0, e0 = 0.0, norm_eta = 0.0, norm_omega = 0.0;
};

std::vector<Row> Rows(const FlowTable& t) {
  std::vector<Row> rows;
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    rows.push_back({t.at(i, "s"), t.at(i, "E0"), t.at(i, "norm_eta"),
                    t.at(i, "norm_omega")});
  }
  return rows;
}

/// Compares a flow with the golden trajectory row by row (bound `tol` on s,
/// E0, ||eta||, ||Omega||) and prints the maximal deviations and the first row
/// beyond `tol`.
void CompareTrajectory(const char* tag, const std::vector<Row>& got,
                       const std::vector<Row>& golden, double tol) {
  ASSERT_EQ(got.size(), golden.size()) << tag << ": number of rows";
  double ds = 0.0, de = 0.0, deta = 0.0, dom = 0.0;
  std::size_t first_bad = got.size();
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double a = std::abs(got[i].s - golden[i].s);
    const double b = std::abs(got[i].e0 - golden[i].e0);
    const double c = std::abs(got[i].norm_eta - golden[i].norm_eta);
    const double d = std::abs(got[i].norm_omega - golden[i].norm_omega);
    if (first_bad == got.size() && std::max({a, b, c, d}) > tol) first_bad = i;
    ds = std::max(ds, a);
    de = std::max(de, b);
    deta = std::max(deta, c);
    dom = std::max(dom, d);
  }
  std::printf(
      "[%s] trajectory (%zu rows) max dev: s %.3e  E0 %.3e  norm_eta %.3e  "
      "norm_omega %.3e\n",
      tag, got.size(), ds, de, deta, dom);
  if (first_bad < got.size()) {
    const std::size_t i = first_bad;
    std::printf(
        "[%s] first row beyond %.0e: %zu  got (%.17g %.17g %.17g %.17g)  "
        "golden (%.17g %.17g %.17g %.17g)\n",
        tag, tol, i, got[i].s, got[i].e0, got[i].norm_eta, got[i].norm_omega,
        golden[i].s, golden[i].e0, golden[i].norm_eta, golden[i].norm_omega);
  }
  EXPECT_LT(ds, tol);
  EXPECT_LT(de, tol);
  EXPECT_LT(deta, tol);
  EXPECT_LT(dom, tol);
}

/// Config text: `nucleus` at `emax`, hw 16, the 2N file, linear HF mixing,
/// atan generator (the default), smax 200, output into `dir`; `imsrg3` is the
/// body of `[imsrg3]` `step_extra` further `[imsrg.step]` keys and `extra`
/// further tables placed before it.
std::string Toml(const std::filesystem::path& dir, const std::string& nucleus,
                 int emax, const std::string& imsrg3,
                 const std::string& step_extra = "",
                 const std::string& extra = "") {
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n[reference]\nnucleus = \""
    << nucleus << "\"\n\n[interaction.two_body]\nfile = \""
    << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    // imsrg++'s linear HF mixing (the TOML default diis reaches the same
    // solution only to the HF tolerance, D37).
    << "[hartree_fock]\nmixing = \"linear\"\n\n[imsrg]\nmethod = "
       "\"magnus\"\n\n[imsrg.step]\nsmax = 200\n"
    << step_extra << "\n"
    << extra << "[imsrg3]\n"
    << imsrg3 << "\n[output]\ndirectory = \"" << dir.string() << "\"\n";
  return t.str();
}

bool HaveInput() { return std::filesystem::exists(DataFile(TwoBodyName())); }

struct Timed {
  RunResult result;
  double seconds = 0.0;
  int steps = 0;
};

Timed TimedExecute(const std::string& toml, const std::filesystem::path& dir) {
  driver::ConfigSources src;
  src.config_text = toml;
  const auto t0 = std::chrono::steady_clock::now();
  Timed out;
  out.result = Execute(src, dir);
  out.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  if (!out.result.planned) {
    out.steps = static_cast<int>(
        out.result.results["imsrg"]["stages"].items.at(0)["steps"].number);
  }
  return out;
}

/// One IT-10 case through the driver.
void RunCase(const std::string& nucleus, int emax, const std::string& golden,
             double e0_ref, int steps_ref, int gathers_ref) {
  if (!HaveInput()) GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir("imsrg3f2_" + golden);
  const Timed run =
      TimedExecute(Toml(dir, nucleus, emax, "factorized = true\n"), dir);
  const RunResult& r = run.result;
  ASSERT_FALSE(r.planned) << r.note;
  EXPECT_TRUE(r.config.imsrg3.factorized);
  EXPECT_TRUE(r.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(r.config.imsrg.step.omega_norm_max, 0.1);

  const Json gold =
      ParseJson(Slurp(GoldenFile("IT-10/" + golden + "/result.json")));
  EXPECT_DOUBLE_EQ(gold["E0"].number, e0_ref);
  const Json& stage = r.results["imsrg"]["stages"].items.at(0);
  const double e0 = r.results["imsrg"]["e0"].number;
  const double de0 = e0 - gold["E0"].number;
  std::printf("[IT-10 %s] steps %d (golden %d), gathers %.0f (golden %d)\n",
              golden.c_str(), run.steps, steps_ref, stage["num_gathers"].number,
              gathers_ref);
  std::printf(
      "[IT-10 %s] E0 %.12f golden %.12f dE0 %.3e  wall %.2f s (%.3f s/step)\n",
      golden.c_str(), e0, gold["E0"].number, de0, run.seconds,
      run.seconds / run.steps);
  EXPECT_EQ(run.steps, steps_ref);
  EXPECT_EQ(static_cast<int>(gold["steps"].number), steps_ref);
  EXPECT_EQ(static_cast<int>(stage["num_gathers"].number), gathers_ref);
  EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_gathers"].number),
            gathers_ref);
  EXPECT_EQ(static_cast<int>(gold["n_gathers"].number), gathers_ref);
  EXPECT_LE(std::abs(de0), 1.0e-3);  // D19
  CompareTrajectory(
      golden.c_str(), Rows(r.flow),
      Rows(ReadFlow(GoldenFile("IT-10/" + golden + "/trajectory.csv"))),
      1.0e-8);
}

#ifdef IMSRG_IT10_SLOW

TEST(Imsrg3f2, O16Emax3) {
  RunCase("O16", 3, "O16_emax3", -157.6737626968918, 150, 13);
}

#else

TEST(Imsrg3f2, O16Emax2) {
  RunCase("O16", 2, "O16_emax2", -145.69663666563525, 192, 24);
}

TEST(Imsrg3f2, He4Emax2) {
  RunCase("He4", 2, "He4_emax2", -25.180558243538115, 198, 8);
}

// With the one-body intermediates off the factorized correction vanishes, so
// the run is the plain hunter-gatherer flow (IT-2); also prints the time per
// step of plain hunter-gatherer and of 3f2 (He4, emax 2).
TEST(Imsrg3f2, SwitchesOffReproducePlainHunterGatherer) {
  if (!HaveInput()) GTEST_SKIP() << "missing 2N input file";
  const std::string rel = "IT-2/hunter_gatherer/He4_emax2_omax_0.10/";
  const Json gold = ParseJson(Slurp(GoldenFile(rel + "result.json")));
  EXPECT_DOUBLE_EQ(gold["E0"].number, -25.204890223641264);

  const auto dir = FreshDir("imsrg3f2_off");
  const Timed off = TimedExecute(
      Toml(dir, "He4", 2,
           "factorized = true\nfactorized_1b_intermediates = false\n"),
      dir);
  ASSERT_FALSE(off.result.planned) << off.result.note;
  const Json& stage = off.result.results["imsrg"]["stages"].items.at(0);
  const double e0 = off.result.results["imsrg"]["e0"].number;
  std::printf(
      "[IT-10 switches off] steps %d (golden %.0f), gathers %.0f (golden "
      "%.0f), dE0 %.3e\n",
      off.steps, gold["steps"].number, stage["num_gathers"].number,
      gold["n_gathers"].number, e0 - gold["E0"].number);
  EXPECT_EQ(off.steps, 196);
  EXPECT_EQ(static_cast<int>(stage["num_gathers"].number), 7);
  EXPECT_NEAR(e0, gold["E0"].number, 1.0e-9);
  CompareTrajectory("switches off", Rows(off.result.flow),
                    Rows(ReadFlow(GoldenFile(rel + "trajectory.csv"))), 1.0e-9);

  // The plain hunter-gatherer run (no factorized key) and a fresh 3f2 run for
  // the timing.
  const auto dir2 = FreshDir("imsrg3f2_plain");
  const Timed plain =
      TimedExecute(Toml(dir2, "He4", 2, "", "omega_norm_max = 0.1\n",
                        "[imsrg.omega]\nhunter_gatherer = true\n\n"),
                   dir2);
  ASSERT_FALSE(plain.result.planned) << plain.result.note;
  const auto dir3 = FreshDir("imsrg3f2_time");
  const Timed f3 =
      TimedExecute(Toml(dir3, "He4", 2, "factorized = true\n"), dir3);
  std::printf(
      "[IT-10 timing He4 emax2] plain HG: %d steps, %.2f s, %.4f s/step;  "
      "3f2: %d steps, %.2f s, %.4f s/step\n",
      plain.steps, plain.seconds, plain.seconds / plain.steps, f3.steps,
      f3.seconds, f3.seconds / f3.steps);
}

// The two-body intermediates are not implemented: E-PLANNED before any input
// is read.
TEST(Imsrg3f2, TwoBodyIntermediatesArePlanned) {
  if (!HaveInput()) GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir("imsrg3f2_2b");
  driver::ConfigSources src;
  src.config_text = Toml(
      dir, "He4", 2, "factorized = true\nfactorized_2b_intermediates = true\n");
  const RunResult r = Execute(src, dir);
  EXPECT_TRUE(r.planned);
  EXPECT_NE(r.note.find("factorized_2b_intermediates"), std::string::npos)
      << r.note;
}

#endif  // IMSRG_IT10_SLOW

}  // namespace
}  // namespace imsrg::it
