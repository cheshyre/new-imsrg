// SPDX-License-Identifier: GPL-3.0-or-later
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

/// A full driver run of `method` (config key `imsrg.method`) for a nucleus at
/// the given emax with Rp2 requested, compared with the golden.
void DriverLevel(const std::string& nucleus, const std::string& method,
                 int emax, const char* golden_dir, const char* tag) {
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
  ASSERT_EQ(r.flow.rows.size(), gold.size()) << tag << ": flow lines";
  Deviations d;
  for (std::size_t i = 0; i < gold.size(); ++i) {
    d.s = std::max(d.s, std::abs(r.flow.at(i, "s") - gold[i].s));
    d.e0 = std::max(d.e0, std::abs(r.flow.at(i, "E0") - gold[i].e0));
    d.norm_h =
        std::max(d.norm_h, std::abs(r.flow.at(i, "norm_H") - gold[i].norm_h));
    d.eta1 =
        std::max(d.eta1, std::abs(r.flow.at(i, "norm_eta_1b") - gold[i].eta1));
    d.eta2 =
        std::max(d.eta2, std::abs(r.flow.at(i, "norm_eta_2b") - gold[i].eta2));
    d.e_mp2 =
        std::max(d.e_mp2, std::abs(r.flow.at(i, "E_MP2") - gold[i].e_mp2));
  }
  Report(tag, gold.size(), d);
  ExpectDeviationsSmall(d);

  const double e0 = r.results["imsrg"]["e0"].number;
  const double rp2 = r.results["operators"]["Rp2"]["imsrg"].number;
  const double rp2_gold = res["flowing_operators"]["Rp2"]["value"].number;
  std::printf("[%s] E0 %.9f (golden %.6f, dE0 %.2e)  Rp2 %.8f (golden %.6f)\n",
              tag, e0, res["core_energy"].number,
              e0 - res["core_energy"].number, rp2, rp2_gold);
  EXPECT_NEAR(e0, res["core_energy"].number, 1.0e-6);
  EXPECT_NEAR(rp2, rp2_gold, 1.0e-6);
  EXPECT_EQ(static_cast<int>(r.results["imsrg"]["num_omegas"].number), 1);
}

#ifndef IMSRG_FLOW_SLOW
TEST(FlowMethods, He4Emax2Rk4Driver) {
  DriverLevel("He4", "flow_rk4", 2, "He4_emax2_rk4", "driver RK4 He4 emax2");
}
TEST(FlowMethods, He4Emax2AdaptiveDriver) {
  DriverLevel("He4", "flow", 2, "He4_emax2_flow", "driver dopri5 He4 emax2");
}
TEST(FlowMethods, O16Emax3Rk4Driver) {
  DriverLevel("O16", "flow_rk4", 3, "O16_emax3_rk4", "RK4 O16 emax3");
}
TEST(FlowMethods, O16Emax3AdaptiveDriver) {
  DriverLevel("O16", "flow", 3, "O16_emax3_flow", "dopri5 O16 emax3");
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
