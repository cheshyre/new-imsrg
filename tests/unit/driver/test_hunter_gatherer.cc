// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Hunter-gatherer Magnus (D32) against the imsrg++ goldens of
// tests/golden/e2e/IT-2/hunter_gatherer: the solver level (He4, emax 2, built
// through the pipeline and fed to `flow::MagnusSolver` directly) and the driver
// level (both O16 emax 4 cases through `driver::Run`). Step and gather counts
// must be identical; E0, |eta| and the trajectory are compared with a
// tolerance (the acceptance of D19 is 1e-3 MeV; the deviations are printed).
// Skips without the 2N input file.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/load_config.h"
#include "imsrg/driver/pipeline/hamiltonian.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/operator/norms.h"
#include "tests/unit/driver/integration/mini_json.h"
#include "tests/unit/driver/integration/run_support.h"

namespace imsrg::it {
namespace {

namespace driver = imsrg::driver;

/// One trajectory row: s, E0, ||eta||, ||hunter||.
struct Row {
  double s = 0.0, e0 = 0.0, norm_eta = 0.0, norm_omega = 0.0;
};

std::vector<Row> GoldenRows(const FlowTable& t) {
  std::vector<Row> rows;
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    rows.push_back({t.at(i, "s"), t.at(i, "E0"), t.at(i, "norm_eta"),
                    t.at(i, "norm_omega")});
  }
  return rows;
}

/// The flow log of a hunter-gatherer stage ends with one row for the H rebuilt
/// with the gathered Omega (same s as the row before, D48); removes it and
/// returns it, so the rest can be compared with the in-loop golden trajectory.
Row PopRebuiltRow(std::vector<Row>& rows) {
  EXPECT_GE(rows.size(), 2u);
  const Row rebuilt = rows.back();
  rows.pop_back();
  EXPECT_EQ(rebuilt.s, rows.back().s);
  return rebuilt;
}

/// Compares a flow with the golden trajectory row by row and prints the
/// maximal deviations (observed: 1e-13 or better).
void CompareTrajectory(const char* tag, const std::vector<Row>& got,
                       const std::vector<Row>& golden) {
  ASSERT_EQ(got.size(), golden.size()) << tag << ": number of rows";
  double ds = 0.0, de = 0.0, deta = 0.0, dom = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    ds = std::max(ds, std::abs(got[i].s - golden[i].s));
    de = std::max(de, std::abs(got[i].e0 - golden[i].e0));
    deta = std::max(deta, std::abs(got[i].norm_eta - golden[i].norm_eta));
    dom = std::max(dom, std::abs(got[i].norm_omega - golden[i].norm_omega));
  }
  std::printf(
      "[%s] trajectory (%zu rows) max dev: s %.3e  E0 %.3e  norm_eta %.3e  "
      "norm_omega %.3e\n",
      tag, got.size(), ds, de, deta, dom);
  EXPECT_LT(ds, 1.0e-9);
  EXPECT_LT(de, 1.0e-9);
  EXPECT_LT(deta, 1.0e-9);
  EXPECT_LT(dom, 1.0e-9);
}

std::filesystem::path HgGolden(const std::string& rel) {
  return GoldenFile("IT-2/hunter_gatherer/" + rel);
}

std::string He4Toml() {
  std::ostringstream t;
  t << "[basis]\nemax = 2\nhw = 16.0\n\n[reference]\nnucleus = \"He4\"\n\n"
    << "[interaction.two_body]\nfile = \"" << DataFile(TwoBodyName()).string()
    << "\"\ne1max = 14\ne2max = 28\nlmax = 14\n"
    << "\n[hartree_fock]\nmixing = \"linear\"\n";  // imsrg++ (D37)
  return t.str();
}

// Solver level: He4, emax 2, omega_norm_max 0.1 (= domega, so the 1e-4 bump of
// domega decides the splits too).
TEST(HunterGatherer, He4Emax2SolverMatchesGolden) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const std::string dir = "He4_emax2_omax_0.10/";
  driver::ConfigSources src;
  src.config_text = He4Toml();
  auto loaded = driver::LoadConfig(src);
  ASSERT_TRUE(loaded.has_value());
  const driver::pipeline::PreparedHamiltonian ph =
      driver::pipeline::PrepareHamiltonian(loaded.value());

  flow::SolverOptions so;  // imsrg++ executable defaults
  so.omega_norm_max = 0.1;
  so.hunter_gatherer = true;
  flow::GeneratorOptions go;  // atan
  go.kind = flow::GeneratorKind::kAtan;
  flow::MagnusSolver solver(ph.hno, so, go);
  std::vector<Row> got;
  const flow::SolveResult r = solver.Solve([&](const flow::FlowStatus& st) {
    got.push_back({st.s, st.e0, st.norm_eta, st.norm_omega});
  });

  const Row rebuilt = PopRebuiltRow(got);
  EXPECT_EQ(rebuilt.e0, r.e0);
  EXPECT_EQ(rebuilt.norm_eta, r.norm_eta);
  const Json gold = ParseJson(Slurp(HgGolden(dir + "result.json")));
  const double e0_gold = gold["E0"].number;
  std::printf(
      "[HG He4 emax2] steps %d (golden %.0f), gathers %d (golden %.0f)\n",
      r.steps, gold["steps"].number, r.num_gathers, gold["n_gathers"].number);
  std::printf("[HG He4 emax2] E0 %.12f golden %.12f dE0 %.3e  |eta| %.3e\n",
              r.e0, e0_gold, r.e0 - e0_gold, r.norm_eta);
  EXPECT_EQ(r.reason, flow::StopReason::kConverged);
  EXPECT_EQ(r.steps, static_cast<int>(gold["steps"].number));
  EXPECT_EQ(r.num_gathers, static_cast<int>(gold["n_gathers"].number));
  EXPECT_EQ(solver.num_gathers(), r.num_gathers);
  EXPECT_NEAR(r.e0, e0_gold, 1.0e-9);
  EXPECT_EQ(solver.omegas().size(), 2u);  // {gatherer, zero hunter}
  EXPECT_NEAR(op::Norm(solver.omegas().front()),
              gold["norm_omega_gathered"].number, 1.0e-6);
  EXPECT_EQ(op::Norm(solver.omegas().back()), 0.0);
  CompareTrajectory("HG He4 emax2", got,
                    GoldenRows(ReadFlow(HgGolden(dir + "trajectory.csv"))));
}

/// One O16 emax 4 case through the driver (config keys
/// `imsrg.omega.hunter_gatherer` and `imsrg.step.omega_norm_max`).
void RunO16(const std::string& golden_dir, const std::string& omax,
            const char* tag) {
  if (!std::filesystem::exists(DataFile(TwoBodyName())))
    GTEST_SKIP() << "missing 2N input file";
  const auto dir = FreshDir(std::string("hg_") + omax);
  driver::ConfigSources src;
  src.config_text = TomlText(dir, "O16",
                             "[imsrg.step]\nomega_norm_max = " + omax +
                                 "\n\n[imsrg.omega]\nhunter_gatherer = true\n");
  const RunResult r = Execute(src, dir);
  ASSERT_FALSE(r.planned) << r.note;
  EXPECT_TRUE(r.config.imsrg.omega.hunter_gatherer);
  EXPECT_DOUBLE_EQ(r.config.imsrg.step.omega_norm_max, std::stod(omax));

  const Json gold = ParseJson(Slurp(HgGolden(golden_dir + "/result.json")));
  const Json& stage = r.results["imsrg"]["stages"].items.at(0);
  const double e0 = r.results["imsrg"]["e0"].number;
  std::printf("[HG %s] steps %.0f (golden %.0f), gathers %.0f (golden %.0f)\n",
              tag, stage["steps"].number, gold["steps"].number,
              stage["num_gathers"].number, gold["n_gathers"].number);
  std::printf("[HG %s] E0 %.12f golden %.12f dE0 %.3e\n", tag, e0,
              gold["E0"].number, e0 - gold["E0"].number);
  EXPECT_EQ(static_cast<int>(stage["steps"].number),
            static_cast<int>(gold["steps"].number));
  EXPECT_EQ(static_cast<int>(stage["num_gathers"].number),
            static_cast<int>(gold["n_gathers"].number));
  EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_gathers"].number),
            static_cast<int>(gold["n_gathers"].number));
  EXPECT_NEAR(e0, gold["E0"].number, 1.0e-9);
  // D48: eta is that of the rebuilt H (1e-3 here, not < eta_criterion); the
  // flow log ends with the rebuilt row, whose E0 is the reported one.
  EXPECT_EQ(stage["norm_eta"].number,
            r.flow.at(r.flow.rows.size() - 1, "norm_eta"));
  EXPECT_GT(stage["norm_eta"].number, 1.0e-6);
  EXPECT_EQ(r.flow.at(r.flow.rows.size() - 1, "E0"), e0);
  EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_omegas"].number), 2);
  std::vector<Row> rows = GoldenRows(r.flow);
  PopRebuiltRow(rows);
  CompareTrajectory(
      tag, rows,
      GoldenRows(ReadFlow(HgGolden(golden_dir + "/trajectory.csv"))));
}

TEST(HunterGatherer, O16Emax4Omax025) {
  RunO16("omax_0.25", "0.25", "omax 0.25");
}
TEST(HunterGatherer, O16Emax4Omax010) {
  RunO16("omax_0.10", "0.1", "omax 0.10");
}

}  // namespace
}  // namespace imsrg::it
