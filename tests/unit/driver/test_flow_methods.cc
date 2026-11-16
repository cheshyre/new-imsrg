// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The ODE flow methods `flow_rk4` (fixed-step RK4) and `flow` (odeint dopri5)
// against the imsrg++ goldens of tests/golden/e2e/IT-14 (imsrg++ `method=
// flow_RK4` / `method=flow` with Operators=Rp2): the solver level (He4, emax 2,
// the flow file line by line, including the 6 right-hand-side evaluations per
// try of the adaptive method) and the driver level (He4 emax 2; O16 emax 3 and
// 4 under the `slow` label: flow CSV, E0, and the Rp2 that flowed alongside H).
// Deviations are printed. Skips without the 2N input file.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/operators/operator_request.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/flow/ode.h"
#include "imsrg/mbpt/energies.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"
#include "tests/unit/driver/integration/snt_compare.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// One line of an imsrg++ flow file: the 13 columns after the step index.
struct GoldLine {
  int istep = 0;
  double s = 0, e0 = 0, norm_h = 0, eta1 = 0, eta2 = 0, e_mp2 = 0;
};

std::vector<GoldLine> ReadGoldFlow(const std::string& rel) {
  std::ifstream in(GoldenFile(rel));
  std::vector<GoldLine> lines;
  std::string text;
  while (std::getline(in, text)) {
    std::istringstream ls(text);
    GoldLine g;
    double err, om1, om2, om3, eta3, ncomm;
    if (ls >> g.istep >> g.s >> g.e0 >> g.norm_h >> err >> om1 >> om2 >> om3 >>
        g.eta1 >> g.eta2 >> eta3 >> ncomm >> g.e_mp2) {
      lines.push_back(g);
    }
  }
  return lines;
}

struct Deviations {
  double s = 0, e0 = 0, norm_h = 0, eta1 = 0, eta2 = 0, e_mp2 = 0;
};

void Report(const char* tag, std::size_t rows, const Deviations& d) {
  std::printf(
      "[%s] %zu flow lines, max dev: s %.2e  E0 %.2e  |H| %.2e  |eta1| %.2e  "
      "|eta2| %.2e  E_MP2 %.2e\n",
      tag, rows, d.s, d.e0, d.norm_h, d.eta1, d.eta2, d.e_mp2);
}

/// The goldens print 5 decimals of s and 9 of everything else; the norm of H
/// (about 385) differs by 4e-8 already at s = 0 (HF convergence).
void ExpectDeviationsSmall(const Deviations& d) {
  EXPECT_LT(d.s, 1.0e-5);
  EXPECT_LT(d.e0, 5.0e-8);
  EXPECT_LT(d.norm_h, 2.0e-7);  // |H| = 385; HF differs at 1e-10
  EXPECT_LT(d.eta1, 5.0e-8);
  EXPECT_LT(d.eta2, 5.0e-8);
  EXPECT_LT(d.e_mp2, 5.0e-8);
}

#ifndef IMSRG_FLOW_SLOW  // the fast executable
std::string He4Toml() {
  std::ostringstream t;
  t << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n";
  return t.str();
}

/// He4 emax 2 at the solver level: Rp2 flows alongside H; every status line is
/// compared with the golden flow file.
void SolverLevelHe4(flow::OdeMethod method, const char* golden_dir,
                    const char* tag) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  driver::ConfigSources src;
  src.config_text = He4Toml();
  auto loaded = driver::LoadConfig(src);
  ASSERT_TRUE(loaded.has_value());
  const driver::RunConfig& config = loaded.value();
  const auto ph = driver::pipeline::PrepareHamiltonian(config);
  auto request =
      driver::ParseOperatorSpec("Rp2", driver::ModelSpaceInfo{.emax = 2});
  ASSERT_TRUE(request.has_value());
  op::Operator rp2 =
      driver::pipeline::PrepareOperator(request.value().front(), ph, config);
  const double rp2_hf = rp2.zero_body();

  flow::SolverOptions so;  // imsrg++ executable defaults
  flow::GeneratorOptions go;
  flow::OdeSolver solver(method, ph.hno, {rp2}, so, 1.0e-6, go);
  std::vector<GoldLine> got;
  std::vector<double> mp2;
  const flow::SolveResult r = solver.Solve([&](const flow::FlowStatus& st) {
    GoldLine g;
    g.istep = st.istep;
    g.s = st.s;
    g.e0 = st.e0;
    g.norm_h = st.norm_h;
    g.eta1 = st.norm_eta_1b;
    g.eta2 = st.norm_eta_2b;
    op::Operator h = solver.hamiltonian();
    g.e_mp2 = mbpt::Mp2Energy(h);
    got.push_back(g);
  });

  const std::string dir = std::string("IT-14/") + golden_dir;
  const std::vector<GoldLine> gold = ReadGoldFlow(dir + "/flow.dat");
  const Json res = ParseJson(Slurp(GoldenFile(dir + "/result.json")));
  ASSERT_EQ(got.size(), gold.size()) << tag << ": number of flow lines";
  Deviations d;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].istep, gold[i].istep) << "line " << i;
    d.s = std::max(d.s, std::abs(got[i].s - gold[i].s));
    d.e0 = std::max(d.e0, std::abs(got[i].e0 - gold[i].e0));
    d.norm_h = std::max(d.norm_h, std::abs(got[i].norm_h - gold[i].norm_h));
    d.eta1 = std::max(d.eta1, std::abs(got[i].eta1 - gold[i].eta1));
    d.eta2 = std::max(d.eta2, std::abs(got[i].eta2 - gold[i].eta2));
    d.e_mp2 = std::max(d.e_mp2, std::abs(got[i].e_mp2 - gold[i].e_mp2));
  }
  Report(tag, got.size(), d);
  ExpectDeviationsSmall(d);

  // Final E0 (printed with 6 decimals) and the Rp2 that flowed with H.
  const double e0_gold = res["core_energy"].number;
  const double rp2_gold = res["flowing_operators"]["Rp2"]["value"].number;
  const double rp2_final = solver.flowing_operator(0).zero_body();
  std::printf(
      "[%s] steps %d, E0 %.9f (golden %.6f), Rp2 HF %.6f IMSRG %.8f "
      "(golden %.6f)\n",
      tag, r.steps, r.e0, e0_gold, rp2_hf, rp2_final, rp2_gold);
  EXPECT_NEAR(r.e0, e0_gold, 1.0e-6);
  EXPECT_NEAR(rp2_final, rp2_gold, 1.0e-6);
  EXPECT_NE(r.reason, flow::StopReason::kDiverged);
}

TEST(FlowMethods, He4Emax2Rk4MatchesImsrgFlowFile) {
  SolverLevelHe4(flow::OdeMethod::kRk4, "He4_emax2_rk4", "RK4 He4 emax2");
}

TEST(FlowMethods, He4Emax2AdaptiveMatchesImsrgFlowFile) {
  SolverLevelHe4(flow::OdeMethod::kDopri5, "He4_emax2_flow",
                 "dopri5 He4 emax2");
}

#endif  // IMSRG_FLOW_SLOW

/// The charge radii (fm) that the PRISTINE imsrg++ prints for `Operators=Rp2`
/// with a non-Magnus method ("HF charge radius", "IMSRG charge radius"; its
/// Rp2 still contained the spin-orbit term, so they must equal our
/// `charge_radius.rch` = sqrt(Rp2 + Rso/Z + nucleon terms), D35). Printed with
/// 8 (HF) and 6 (IMSRG) decimals. From the pristine build
/// (new-imsrg-reference/build), same command lines as the IT-14 goldens.
struct PristineRadius {
  double hf;
  double imsrg;
};

/// A full driver run of `method` (config key `imsrg.method`) for a nucleus at
/// the given emax with Rp2 requested, compared with the golden. With `radius`
/// the charge radius of results.json is compared with the pristine imsrg++.
void DriverLevel(const std::string& nucleus, const std::string& method,
                 int emax, const char* golden_dir, const char* tag,
                 const PristineRadius* radius = nullptr) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir(std::string("flow_") + golden_dir);
  std::ostringstream t;
  t << "[basis]\nemax = " << emax << "\nhw = 16.0\n\n"
    << "[reference]\nnucleus = \"" << nucleus << "\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[imsrg]\nmethod = \"" << method << "\"\n\n"
    << "[operators]\nlist = [\"Rp2\"]\n\n"
    << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;

  const std::string gdir = std::string("IT-14/") + golden_dir;
  const std::vector<GoldLine> gold = ReadGoldFlow(gdir + "/flow.dat");
  const Json res = ParseJson(Slurp(GoldenFile(gdir + "/result.json")));
  if (method == "flow_rk4") {
    ASSERT_EQ(r.flow.rows.size(), gold.size()) << tag << ": flow lines";
    Deviations d;
    for (std::size_t i = 0; i < gold.size(); ++i) {
      d.s = std::max(d.s, std::abs(r.flow.at(i, "s") - gold[i].s));
      d.e0 = std::max(d.e0, std::abs(r.flow.at(i, "E0") - gold[i].e0));
      d.norm_h =
          std::max(d.norm_h, std::abs(r.flow.at(i, "norm_H") - gold[i].norm_h));
      d.eta1 = std::max(d.eta1,
                        std::abs(r.flow.at(i, "norm_eta_1b") - gold[i].eta1));
      d.eta2 = std::max(d.eta2,
                        std::abs(r.flow.at(i, "norm_eta_2b") - gold[i].eta2));
      d.e_mp2 =
          std::max(d.e_mp2, std::abs(r.flow.at(i, "E_MP2") - gold[i].e_mp2));
    }
    Report(tag, gold.size(), d);
    ExpectDeviationsSmall(d);
  } else {
    // The charge radius needs Rso, which flows alongside H as one more state
    // component of the adaptive method and so changes its step-size control
    // (M8-Q50): the trajectory differs from the golden (generated with Rp2
    // alone; the solver-level tests above still compare it line by line). The
    // end point agrees, checked below.
    std::printf(
        "[%s] adaptive trajectory with Rso flowing: %zu rows (golden %zu)\n",
        tag, r.flow.rows.size(), gold.size());
  }

  const double e0 = r.results["imsrg"]["e0"].number;
  const double rp2 = r.results["operators"]["Rp2"]["imsrg"].number;
  const double rp2_gold = res["flowing_operators"]["Rp2"]["value"].number;
  std::printf("[%s] E0 %.9f (golden %.6f, dE0 %.2e)  Rp2 %.8f (golden %.6f)\n",
              tag, e0, res["core_energy"].number,
              e0 - res["core_energy"].number, rp2, rp2_gold);
  EXPECT_NEAR(e0, res["core_energy"].number, 1.0e-6);
  EXPECT_NEAR(rp2, rp2_gold, 1.0e-6);
  EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_omegas"].number), 1);
  // Rso was not requested: evaluated internally, no `operators` entry.
  EXPECT_FALSE(r.results["operators"].has("Rso"));
  if (radius != nullptr) {
    const double rch_hf = r.results["charge_radius"]["rch"]["reference"].number;
    const double rch = r.results["charge_radius"]["rch"]["imsrg"].number;
    std::printf(
        "[%s] charge radius HF %.8f (pristine %.8f, dev %.1e)  IMSRG %.8f "
        "(pristine %.6f, dev %.1e)\n",
        tag, rch_hf, radius->hf, rch_hf - radius->hf, rch, radius->imsrg,
        rch - radius->imsrg);
    EXPECT_NEAR(rch_hf, radius->hf, 2.0e-8);
    EXPECT_NEAR(rch, radius->imsrg, 2.0e-6);
  }
}

#ifndef IMSRG_FLOW_SLOW
TEST(FlowMethods, He4Emax2Rk4Driver) {
  const PristineRadius pristine{1.66192448, 1.690996};
  DriverLevel("He4", "flow_rk4", 2, "He4_emax2_rk4", "driver RK4 He4 emax2",
              &pristine);
}
TEST(FlowMethods, He4Emax2AdaptiveDriver) {
  const PristineRadius pristine{1.66192448, 1.691025};
  DriverLevel("He4", "flow", 2, "He4_emax2_flow", "driver dopri5 He4 emax2",
              &pristine);
}
TEST(FlowMethods, O16Emax3Rk4Driver) {
  const PristineRadius pristine{2.22458311, 2.269411};
  DriverLevel("O16", "flow_rk4", 3, "O16_emax3_rk4", "RK4 O16 emax3",
              &pristine);
}
TEST(FlowMethods, O16Emax3AdaptiveDriver) {
  const PristineRadius pristine{2.22458311, 2.269443};
  DriverLevel("O16", "flow", 3, "O16_emax3_flow", "dopri5 O16 emax3",
              &pristine);
}

// IT-15: two decoupling stages with an ODE method (He4 emax 2, p-shell valence
// space: core atan, then shell-model-atan). O16 emax 2 with the sd-shell is
// unstable in IMSRG(2) (design/30 IT-8 note), He4 converges. Compared with the
// imsrg++ golden tests/golden/e2e/IT-15 (patched build, D42): E0, the
// valence-space .snt elements (1e-3 MeV) and, for `flow_rk4`, the per-stage
// step counts and the flow file (s continues across the stages). imsrg++'s
// adaptive `flow` restarts s at 0 in stage 2 (M8-Q40), so only its final
// quantities are compared. Deviations are printed.
//
// The `_smax10` cases (D47) run with smax = 10 so that stage 1 stops at smax;
// stage 2 then integrates [s1, s1 + smax] for every method (final s = 20 for
// Magnus and `flow_rk4`; imsrg++ leaves a `flow_rk4` stage 2 empty and doubles
// smax for Magnus only, the patched reference does the same as we do). Magnus
// and RK4 flow files are compared line by line.
//
// `_smax10_eta` (D47): smax = 10, eta_criterion = 0.014, so that stage 1
// CONVERGES (s1 = 7.399) and stage 2 ends at s1 + smax = 17.399; a doubled smax
// would run to 20. `_hg` (D48): two-stage hunter-gatherer; stage 2 starts from
// the H rebuilt with the gathered Omega (E0 of its first flow row = the E0
// reported by stage 1); the flow logs carry one extra row per stage (the
// rebuilt H), so only steps, gathers and the stage-2 start are compared.
struct StageCase {
  std::string toml;  ///< extra TOML tables (before [output])
  std::string tag;   ///< directory suffix
  double s1 = -1.0;  ///< expected s after stage 1 (< 0: not checked)
  double s2 = -1.0;  ///< expected s after stage 2
  bool rows = true;  ///< compare the flow file row by row (RK4, Magnus)
  bool hg = false;   ///< hunter-gatherer case
};

void TwoStage(const std::string& method, const char* gdir, const char* tag,
              const StageCase& sc = {}) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const bool compare_flow = method != "flow";
  const auto dir = FreshDir("flow_two_stage_" + method + sc.tag);
  std::ostringstream t;
  t << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n\n"
    << "[valence]\nspace = \"p-shell\"\n\n"
    << "[imsrg]\nmethod = \"" << method << "\"\n\n"
    << sc.toml << "[output]\ndirectory = \"" << dir.string() << "\"\n";
  driver::ConfigSources src;
  src.config_text = t.str();
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;
  const Json res =
      ParseJson(Slurp(GoldenFile(std::string(gdir) + "/result.json")));
  const auto& stages = r.results["imsrg"]["stages"];
  ASSERT_EQ(stages.items.size(), 2u);
  EXPECT_EQ(stages.items[1]["generator"].text, "shell_model_atan");
  if (sc.s1 > 0.0) {  // D47: [s1, s1 + smax], s continues
    EXPECT_NEAR(stages.items[0]["s"].number, sc.s1, 1.0e-4);
    EXPECT_NEAR(stages.items[1]["s"].number, sc.s2, 1.0e-4);
  }
  std::printf(
      "[%s] steps per stage: %d %d; s at the end of the stages: %.4f %.4f\n",
      tag, static_cast<int>(stages.items[0]["steps"].number),
      static_cast<int>(stages.items[1]["steps"].number),
      stages.items[0]["s"].number, stages.items[1]["s"].number);

  if (compare_flow) {
    // Steps per stage of the golden: istep restarts at 0 for stage 2.
    const std::vector<GoldLine> gold =
        ReadGoldFlow(std::string(gdir) + "/flow.dat");
    std::vector<int> gold_steps;
    for (const GoldLine& g : gold) {
      if (g.istep == 0) gold_steps.push_back(0);
      gold_steps.back() = g.istep;
    }
    ASSERT_EQ(gold_steps.size(), 2u);
    std::printf("[%s] golden steps per stage: %d %d\n", tag, gold_steps[0],
                gold_steps[1]);
    EXPECT_EQ(static_cast<int>(stages.items[0]["steps"].number), gold_steps[0]);
    EXPECT_EQ(static_cast<int>(stages.items[1]["steps"].number), gold_steps[1]);

    if (sc.hg) {
      // Stage 2 starts from the rebuilt H: its istep-0 row has the E0 of the
      // golden's, and the reported E0 of stage 1 (D48).
      std::size_t g2 = 0, o2 = 0;
      for (std::size_t i = 0, n0 = 0; i < gold.size(); ++i)
        if (gold[i].istep == 0 && ++n0 == 2) g2 = i;
      for (std::size_t i = 0; i < r.flow.rows.size(); ++i)
        if (r.flow.at(i, "stage") == 2.0 && r.flow.at(i, "istep") == 0.0) {
          o2 = i;
          break;
        }
      ASSERT_GT(g2, 0u);
      ASSERT_GT(o2, 0u);
      std::printf("[%s] stage-2 start E0 %.9f golden %.9f; gathers %.0f\n", tag,
                  r.flow.at(o2, "E0"), gold[g2].e0,
                  r.results["imsrg"]["num_gathers"].number);
      EXPECT_NEAR(r.flow.at(o2, "E0"), gold[g2].e0, 1.0e-6);
      EXPECT_EQ(r.flow.at(o2, "E0"), stages.items[0]["e0"].number);
      // Gathers: the "gathering Omega." lines of the imsrg++ log.
      std::ifstream log(GoldenFile(std::string(gdir) + "/stdout.log"));
      int gathers_gold = 0;
      for (std::string line; std::getline(log, line);)
        if (line.find("gathering Omega") != std::string::npos) ++gathers_gold;
      EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_gathers"].number),
                gathers_gold);
      EXPECT_EQ(gathers_gold, 84);
    }
    // The flow file, both stages, row by row.
    if (sc.rows) {
      ASSERT_EQ(r.flow.rows.size(), gold.size());
      Deviations d;
      for (std::size_t i = 0; i < gold.size(); ++i) {
        d.s = std::max(d.s, std::abs(r.flow.at(i, "s") - gold[i].s));
        d.e0 = std::max(d.e0, std::abs(r.flow.at(i, "E0") - gold[i].e0));
        d.norm_h = std::max(d.norm_h,
                            std::abs(r.flow.at(i, "norm_H") - gold[i].norm_h));
        d.eta1 = std::max(d.eta1,
                          std::abs(r.flow.at(i, "norm_eta_1b") - gold[i].eta1));
        d.eta2 = std::max(d.eta2,
                          std::abs(r.flow.at(i, "norm_eta_2b") - gold[i].eta2));
        d.e_mp2 =
            std::max(d.e_mp2, std::abs(r.flow.at(i, "E_MP2") - gold[i].e_mp2));
      }
      Report(tag, gold.size(), d);
      EXPECT_LT(d.s, 1.0e-5);
      EXPECT_LT(d.e0, 1.0e-6);
      EXPECT_LT(d.norm_h, 1.0e-6);
      EXPECT_LT(d.eta1, 1.0e-6);
      EXPECT_LT(d.eta2, 1.0e-6);
    }
  }

  // Final E0 and the valence-space interaction.
  const double e0 = r.results["imsrg"]["e0"].number;
  const double e0_gold = res["flow"]["final_E0"].number;
  std::printf("[%s] E0 %.9f golden %.9f dev %.3e\n", tag, e0, e0_gold,
              e0 - e0_gold);
  // The golden's last flow line is the last hunter step, not the rebuilt H
  // (D48): the .snt (rebuilt H) is compared at 1e-9 instead.
  if (!sc.hg) EXPECT_NEAR(e0, e0_gold, 1.0e-3);
  const double tol = sc.hg ? 1.0e-9 : 1.0e-3;

  ASSERT_EQ(r.results["valence_interaction"].kind, Json::Kind::kString);
  const SntData got = ReadSnt(r.results["valence_interaction"].text);
  const SntData want = ReadSnt(GoldenFile(std::string(gdir) + "/int.snt"));
  const SntComparison c = CompareSnt(got, want, tol);
  std::printf(
      "[%s] .snt zero body dev %.3e, 1B max dev %.3e (%zu elements), 2B max "
      "dev %.3e (%zu elements), missing %zu, orbits equal %s\n",
      tag, c.zero_dev, c.max_dev_1b, c.n1b_a, c.max_dev_2b, c.n2b_a, c.missing,
      c.orbits_equal ? "yes" : "NO");
  EXPECT_TRUE(c.orbits_equal);
  EXPECT_LE(c.zero_dev, tol);
  EXPECT_LE(c.max_dev_1b, tol);
  EXPECT_LE(c.max_dev_2b, tol);
  EXPECT_EQ(c.missing, 0u);
}

TEST(FlowMethods, TwoStageRk4He4PShellDriver) {
  TwoStage("flow_rk4", "IT-15/He4_emax2_pshell_rk4", "IT-15 RK4 He4 p-shell");
}
TEST(FlowMethods, TwoStageAdaptiveHe4PShellDriver) {
  TwoStage("flow", "IT-15/He4_emax2_pshell_flow", "IT-15 dopri5 He4 p-shell");
}

// D47: stage 1 stops at smax = 10, stage 2 integrates [10, 20] (golden: the
// patched imsrg++), see `TwoStage`.
const char kSmax10[] = "[imsrg.step]\nsmax = 10.0\n\n";
TEST(FlowMethods, StageTwoRangeRk4Smax10) {
  TwoStage("flow_rk4", "IT-15/He4_emax2_pshell_rk4_smax10", "IT-15 RK4 smax 10",
           {kSmax10, "_smax10", 10.0, 20.0, true, false});
}
TEST(FlowMethods, StageTwoRangeAdaptiveSmax10) {
  TwoStage("flow", "IT-15/He4_emax2_pshell_flow_smax10", "IT-15 dopri5 smax 10",
           {kSmax10, "_smax10", 10.0, 20.0, true, false});
}
TEST(FlowMethods, StageTwoRangeMagnusSmax10) {
  TwoStage("magnus", "IT-15/He4_emax2_pshell_magnus_smax10",
           "IT-15 Magnus smax 10",
           {kSmax10, "_smax10", 10.0, 20.0, true, false});
}
// Stage 1 converges at s = 7.399: s1 + smax = 17.399, the doubled smax of
// pristine imsrg++ would end at 20 (98 steps, 16 + 82, in the golden).
TEST(FlowMethods, StageTwoRangeMagnusConvergedStage1) {
  TwoStage("magnus", "IT-15/He4_emax2_pshell_magnus_smax10_eta",
           "IT-15 Magnus smax 10 eta 0.014",
           {"[imsrg.step]\nsmax = 10.0\neta_criterion = 0.014\n\n", "_eta",
            7.39922, 17.39922, true, false});
}
// D48: two-stage hunter-gatherer, stage 2 starts from the rebuilt H.
TEST(FlowMethods, TwoStageHunterGathererHe4PShellDriver) {
  TwoStage("magnus", "IT-15/He4_emax2_pshell_hg", "IT-15 HG He4 p-shell",
           {"[hartree_fock]\nmixing = \"linear\"\n\n[imsrg.step]\n"
            "omega_norm_max = 0.1\n\n[imsrg.omega]\nhunter_gatherer = true\n\n",
            "_hg", -1.0, -1.0, false, true});
}
#else   // IMSRG_FLOW_SLOW: O16 emax 4, 7-14 s each in a debug build
TEST(FlowMethods, O16Emax4Rk4Driver) {
  DriverLevel("O16", "flow_rk4", 4, "O16_emax4_rk4", "RK4 O16 emax4");
}
TEST(FlowMethods, O16Emax4AdaptiveDriver) {
  DriverLevel("O16", "flow", 4, "O16_emax4_flow", "dopri5 O16 emax4");
}
#endif  // IMSRG_FLOW_SLOW

}  // namespace
}  // namespace imsrg::it
