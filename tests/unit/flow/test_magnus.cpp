// Magnus solver (M8 task T3): a short single-reference flow of the emax-2 O16 HF Hamiltonian
// against the imsrg++ flow file (tests/golden/e2e/IT-8/emax2, same Hamiltonian as
// tests/golden/hf/emax2_O16_2N/HNO.dump), the Omega-stack replay, stage switching and the stop
// reasons.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/flow/magnus.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/norms.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace {

using namespace imsrg;

/// One line of an imsrg++ flow file (columns istep s E0 |H| err |Om1| |Om2| |Om3| |Eta1| |Eta2|).
struct GoldenLine {
  int istep = 0;
  double s = 0, e0 = 0, norm_h = 0, om1 = 0, om2 = 0, eta1 = 0, eta2 = 0;
};

std::vector<GoldenLine> read_flow_file(const std::string& path) {
  std::ifstream in(path);
  std::vector<GoldenLine> lines;
  std::string text;
  while (std::getline(in, text)) {
    std::istringstream ls(text);
    GoldenLine g;
    double err = 0, om3 = 0;
    if (ls >> g.istep >> g.s >> g.e0 >> g.norm_h >> err >> g.om1 >> g.om2 >> om3 >> g.eta1 >>
        g.eta2) {
      lines.push_back(g);
    }
  }
  return lines;
}

op::Operator emax2_hno() {
  const std::string path = std::string(IMSRG_GOLDEN_DIR) + "/hf/emax2_O16_2N/HNO.dump";
  const auto ms = op::model_space_for_dump(op::read_operator_dump_header(path));
  return op::read_operator_dump(path, ms);
}

flow::SolverOptions short_flow(double smax) {
  flow::SolverOptions o;  // imsrg++ executable defaults
  o.smax = smax;
  return o;
}

// imsrg++ prints 9 decimals; the codes agree far better than the 1e-8 used here.
constexpr double kTol = 1e-8;

TEST(MagnusSolver, ShortFlowMatchesImsrgFlowFile) {
  const auto golden =
      read_flow_file(std::string(IMSRG_GOLDEN_DIR) + "/e2e/IT-8/emax2/flow.dat");
  ASSERT_GT(golden.size(), 30u);
  const double smax = 4.0;
  flow::MagnusSolver solver(emax2_hno(), short_flow(smax), flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r = solver.solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  EXPECT_EQ(r.reason, flow::StopReason::SmaxReached);
  EXPECT_DOUBLE_EQ(r.s, smax);
  double max_de0 = 0, max_dnorm = 0;
  int compared = 0;
  for (const auto& st : lines) {
    ASSERT_LT(static_cast<std::size_t>(st.istep), golden.size());
    const GoldenLine& g = golden[static_cast<std::size_t>(st.istep)];
    ASSERT_EQ(g.istep, st.istep);
    if (g.s > smax - 1e-5) break;  // the last step is cut at smax
    EXPECT_NEAR(st.s, g.s, 6e-6) << "step " << st.istep;
    const double de0 = std::abs(st.e0 - g.e0);
    const double dnorm = std::max({std::abs(st.norm_h - g.norm_h),
                                   std::abs(st.norm_omega_1b - g.om1),
                                   std::abs(st.norm_omega_2b - g.om2),
                                   std::abs(st.norm_eta_1b - g.eta1),
                                   std::abs(st.norm_eta_2b - g.eta2)});
    EXPECT_LT(de0, kTol) << "step " << st.istep;
    EXPECT_LT(dnorm, kTol) << "step " << st.istep;
    max_de0 = std::max(max_de0, de0);
    max_dnorm = std::max(max_dnorm, dnorm);
    ++compared;
  }
  EXPECT_GE(compared, 15);
  std::printf("[magnus emax2 O16] %d steps compared: max|dE0| %.3e, max|d norm| %.3e, %zu Omegas\n",
              compared, max_de0, max_dnorm, solver.omegas().size());
  EXPECT_GT(solver.omegas().size(), 3u);
}

TEST(MagnusSolver, ReplayOfOmegaStackReproducesH) {
  op::Operator h0 = emax2_hno();
  flow::MagnusSolver solver(h0, short_flow(1.5), flow::GeneratorOptions{});
  solver.solve();
  ASSERT_GT(solver.omegas().size(), 2u);
  const op::Operator replayed = solver.transform(h0);
  const op::PartMaxima d = op::max_abs_difference(replayed, solver.hamiltonian());
  EXPECT_LT(d.max(), 1e-12) << d.zero_body << " " << d.one_body << " " << d.two_body;
}

TEST(MagnusSolver, SecondStageOpensNewOmegaAndContinuesS) {
  flow::MagnusSolver solver(emax2_hno(), short_flow(0.8), flow::GeneratorOptions{});
  const flow::SolveResult r1 = solver.solve();
  const std::size_t n1 = solver.omegas().size();
  ASSERT_GT(op::norm(solver.omegas().back()), 1e-6);
  solver.set_generator(flow::GeneratorOptions{});
  EXPECT_EQ(solver.omegas().size(), n1 + 1);
  EXPECT_EQ(op::norm(solver.omegas().back()), 0.0);
  solver.set_smax(1.6);
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r2 = solver.solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(lines.front().stage, 2);
  EXPECT_EQ(lines.front().istep, 0);
  EXPECT_DOUBLE_EQ(lines.front().s, r1.s);
  EXPECT_DOUBLE_EQ(r2.s, 1.6);
  EXPECT_LT(r2.e0, r1.e0);
}

TEST(MagnusSolver, StopsWhenConverged) {
  flow::SolverOptions o = short_flow(50.0);
  o.eta_criterion = 0.5;
  flow::MagnusSolver solver(emax2_hno(), o, flow::GeneratorOptions{});
  const flow::SolveResult r = solver.solve();
  EXPECT_EQ(r.reason, flow::StopReason::Converged);
  EXPECT_LT(r.norm_eta, 0.5);
  EXPECT_GT(r.steps, 0);
  EXPECT_LT(r.s, 50.0);
}

}  // namespace
