// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Magnus solver (M8 task T3): a short single-reference flow of the emax-2 O16
// HF Hamiltonian against the imsrg++ flow file (tests/golden/e2e/IT-8/emax2,
// same Hamiltonian as tests/golden/hf/emax2_O16_2N/HNO.dump), the Omega-stack
// replay, stage switching and the stop reasons.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"

namespace imsrg {
namespace {

/// One line of an imsrg++ flow file (columns istep s E0 |H| err |Om1| |Om2|
/// |Om3| |Eta1| |Eta2|).
struct GoldenLine {
  int istep = 0;
  double s = 0, e0 = 0, norm_h = 0, om1 = 0, om2 = 0, eta1 = 0, eta2 = 0;
};

std::vector<GoldenLine> ReadFlowFile(const std::string& path) {
  std::ifstream in(path);
  std::vector<GoldenLine> lines;
  std::string text;
  while (std::getline(in, text)) {
    std::istringstream ls(text);
    GoldenLine g;
    double err = 0, om3 = 0;
    if (ls >> g.istep >> g.s >> g.e0 >> g.norm_h >> err >> g.om1 >> g.om2 >>
        om3 >> g.eta1 >> g.eta2) {
      lines.push_back(g);
    }
  }
  return lines;
}

op::Operator Emax2Hno() {
  const std::string path =
      std::string(IMSRG_GOLDEN_DIR) + "/hf/emax2_O16_2N/HNO.dump";
  const auto ms = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(path));
  return op::ReadOperatorDump(path, ms);
}

flow::SolverOptions ShortFlow(double smax) {
  flow::SolverOptions o;  // imsrg++ executable defaults
  o.smax = smax;
  return o;
}

// imsrg++ prints 9 decimals; the codes agree far better than the 1e-8 used
// here.
constexpr double kTol = 1.0e-8;

TEST(MagnusSolver, ShortFlowMatchesImsrgFlowFile) {
  const auto golden =
      ReadFlowFile(std::string(IMSRG_GOLDEN_DIR) + "/e2e/IT-8/emax2/flow.dat");
  ASSERT_GT(golden.size(), 30u);
  const double smax = 4.0;
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(smax),
                            flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r =
      solver.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  EXPECT_EQ(r.reason, flow::StopReason::kSmaxReached);
  EXPECT_DOUBLE_EQ(r.s, smax);
  double max_de0 = 0, max_dnorm = 0;
  int compared = 0;
  for (const auto& st : lines) {
    ASSERT_LT(static_cast<std::size_t>(st.istep), golden.size());
    const GoldenLine& g = golden[static_cast<std::size_t>(st.istep)];
    ASSERT_EQ(g.istep, st.istep);
    if (g.s > smax - 1.0e-5) break;  // the last step is cut at smax
    EXPECT_NEAR(st.s, g.s, 6.0e-6) << "step " << st.istep;
    const double de0 = std::abs(st.e0 - g.e0);
    const double dnorm = std::max(
        {std::abs(st.norm_h - g.norm_h), std::abs(st.norm_omega_1b - g.om1),
         std::abs(st.norm_omega_2b - g.om2), std::abs(st.norm_eta_1b - g.eta1),
         std::abs(st.norm_eta_2b - g.eta2)});
    EXPECT_LT(de0, kTol) << "step " << st.istep;
    EXPECT_LT(dnorm, kTol) << "step " << st.istep;
    max_de0 = std::max(max_de0, de0);
    max_dnorm = std::max(max_dnorm, dnorm);
    ++compared;
  }
  EXPECT_GE(compared, 15);
  std::printf(
      "[magnus emax2 O16] %d steps compared: max|dE0| %.3e, max|d norm| %.3e, "
      "%zu Omegas\n",
      compared, max_de0, max_dnorm, solver.omegas().size());
  EXPECT_GT(solver.omegas().size(), 3u);
}

TEST(MagnusSolver, ReplayOfOmegaStackReproducesH) {
  op::Operator h0 = Emax2Hno();
  flow::MagnusSolver solver(h0, ShortFlow(1.5), flow::GeneratorOptions{});
  solver.Solve();
  ASSERT_GT(solver.omegas().size(), 2u);
  const op::Operator replayed = solver.Transform(h0);
  const op::PartMaxima d = op::MaxAbsDifference(replayed, solver.hamiltonian());
  EXPECT_LT(d.max(), 1.0e-12)
      << d.zero_body << " " << d.one_body << " " << d.two_body;
}

// Hunter-gatherer (imsrg++ GatherOmega semantics, D32): the stack is
// {gatherer, hunter}, the final gather leaves H(s) untouched, so H(s) differs
// from the replay of the gathered Omega on H_0 at the truncation level of the
// BCH product/transform (unlike the split stack, see the test above).
TEST(MagnusSolver, HunterGathererDocumentsImsrgSemantics) {
  op::Operator h0 = Emax2Hno();
  flow::SolverOptions o = ShortFlow(1.5);
  o.hunter_gatherer = true;
  flow::MagnusSolver solver(h0, o, flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r =
      solver.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_GT(lines.size(), 3u);
  // Gather count: the loop gathers (whenever the hunter was > omega_norm_max)
  // plus the final one.
  int loop_gathers = 0;
  for (std::size_t i = 1; i < lines.size(); ++i) {
    if (lines[i].norm_omega_2b + lines[i].norm_omega_1b > 0 &&
        lines[i].norm_omega < lines[i - 1].norm_omega) {
      ++loop_gathers;
    }
  }
  EXPECT_GE(r.num_gathers, 2);
  EXPECT_EQ(solver.num_gathers(), r.num_gathers);
  EXPECT_EQ(loop_gathers + 1, r.num_gathers);
  // {gatherer, zero hunter}; the Omega stack never grows beyond two.
  EXPECT_EQ(solver.omegas().size(), 2u);
  for (const auto& st : lines) EXPECT_LE(st.num_omegas, 2u);
  EXPECT_EQ(op::Norm(solver.omegas().back()), 0.0);
  EXPECT_GT(op::Norm(solver.omegas().front()), 0.25);
  // H(s) is untouched by the final gather.
  EXPECT_EQ(solver.hamiltonian().zero_body(), lines.back().e0);
  EXPECT_EQ(r.e0, lines.back().e0);
  // Replay of the gathered Omega: close to, but not equal to, H(s).
  const op::Operator replayed = solver.Transform(h0);
  const op::PartMaxima d = op::MaxAbsDifference(replayed, solver.hamiltonian());
  std::printf(
      "[magnus HG] %d gathers; replay vs H(s): E0 %.3e, 1b %.3e, 2b %.3e\n",
      r.num_gathers, d.zero_body, d.one_body, d.two_body);
  EXPECT_LT(d.max(), 1.0e-2);
  // A second stage gathers nothing in set_generator (the hunter is zero) and
  // keeps the stack at two entries.
  solver.set_generator(flow::GeneratorOptions{});
  EXPECT_EQ(solver.omegas().size(), 2u);
  EXPECT_EQ(solver.num_gathers(), r.num_gathers);
}

// Review item 5: with omega_norm_max == domega the imsrg++ solver raises domega
// by 1e-4 (once; the bump persists).
TEST(MagnusSolver, DomegaBumpWhenEqualToOmegaNormMax) {
  flow::SolverOptions o = ShortFlow(0.2);
  o.omega_norm_max = 0.1;
  o.domega = 0.1;
  flow::MagnusSolver solver(Emax2Hno(), o, flow::GeneratorOptions{});
  solver.Solve();
  EXPECT_DOUBLE_EQ(solver.solver_options().domega, 0.1 + 1.0e-4);
  solver.set_smax(0.4);
  solver.Solve();
  EXPECT_DOUBLE_EQ(solver.solver_options().domega, 0.1 + 1.0e-4);
  // Different values: no bump.
  flow::SolverOptions o2 = ShortFlow(0.1);
  flow::MagnusSolver other(Emax2Hno(), o2, flow::GeneratorOptions{});
  other.Solve();
  EXPECT_DOUBLE_EQ(other.solver_options().domega, 0.1);
}

// Review item 5: set_generator on a zero Omega does not open a new one.
TEST(MagnusSolver, SetGeneratorWithoutOmegaKeepsStack) {
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(0.5),
                            flow::GeneratorOptions{});
  solver.set_generator(flow::GeneratorOptions{});
  EXPECT_EQ(solver.omegas().size(), 1u);
}

TEST(MagnusSolver, SecondStageOpensNewOmegaAndContinuesS) {
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(0.8),
                            flow::GeneratorOptions{});
  const flow::SolveResult r1 = solver.Solve();
  const std::size_t n1 = solver.omegas().size();
  ASSERT_GT(op::Norm(solver.omegas().back()), 1.0e-6);
  solver.set_generator(flow::GeneratorOptions{});
  EXPECT_EQ(solver.omegas().size(), n1 + 1);
  EXPECT_EQ(op::Norm(solver.omegas().back()), 0.0);
  solver.set_smax(1.6);
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r2 =
      solver.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(lines.front().stage, 2);
  EXPECT_EQ(lines.front().istep, 0);
  EXPECT_DOUBLE_EQ(lines.front().s, r1.s);
  EXPECT_DOUBLE_EQ(r2.s, 1.6);
  EXPECT_LT(r2.e0, r1.e0);
}

// ---- flow review items 1, 2, 5, 8: validation, NaN, ds cap, stage-2 options.

TEST(MagnusSolver, InvalidStepOptionsThrowInputError) {
  using Mutator = void (*)(flow::SolverOptions&);
  const Mutator bad[] = {
      [](flow::SolverOptions& o) { o.dsmax = 0.0; },
      [](flow::SolverOptions& o) { o.dsmax = -1.0; },
      [](flow::SolverOptions& o) { o.domega = 0.0; },
      [](flow::SolverOptions& o) { o.ds_0 = 0.0; },
      [](flow::SolverOptions& o) { o.omega_norm_max = -0.1; },
      [](flow::SolverOptions& o) {
        o.dsmax = std::numeric_limits<double>::quiet_NaN();
      },
      [](flow::SolverOptions& o) {
        o.smax = std::numeric_limits<double>::infinity();
      },
  };
  for (const Mutator m : bad) {
    flow::SolverOptions o = ShortFlow(0.5);
    m(o);
    EXPECT_THROW(flow::MagnusSolver(Emax2Hno(), o, flow::GeneratorOptions{}),
                 InputError);
  }
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(0.5),
                            flow::GeneratorOptions{});
  flow::SolverOptions o = ShortFlow(0.5);
  o.domega = 0.0;
  EXPECT_THROW(solver.set_solver_options(o), InputError);
  EXPECT_DOUBLE_EQ(solver.solver_options().domega, 0.1);  // unchanged
}

TEST(MagnusSolver, NonPositiveDenominatorCutoffThrows) {
  flow::GeneratorOptions g;
  g.denominator_cutoff = 0.0;
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(0.5), g);
  EXPECT_THROW(solver.Solve(), InputError);
}

TEST(MagnusSolver, NanInHamiltonianIsDivergenceNotSmax) {
  // NaN zero-body energy: caught by the E0 guard before any step.
  {
    op::Operator h0 = Emax2Hno();
    h0.zero_body() = std::numeric_limits<double>::quiet_NaN();
    flow::MagnusSolver solver(std::move(h0), ShortFlow(2.0),
                              flow::GeneratorOptions{});
    const flow::SolveResult r = solver.Solve();
    EXPECT_EQ(r.reason, flow::StopReason::kDiverged);
    EXPECT_EQ(r.steps, 0);
  }
  // NaN one-body element: eta is NaN, ||eta|| fails every comparison.
  {
    op::Operator h0 = Emax2Hno();
    h0.one_body().set(OrbitIndex{0}, OrbitIndex{0},
                      std::numeric_limits<double>::quiet_NaN());
    flow::MagnusSolver solver(std::move(h0), ShortFlow(2.0),
                              flow::GeneratorOptions{});
    const flow::SolveResult r = solver.Solve();
    EXPECT_EQ(r.reason, flow::StopReason::kDiverged);
    EXPECT_NE(r.reason, flow::StopReason::kSmaxReached);
    EXPECT_LE(r.steps, 1);
  }
}

TEST(MagnusSolver, NonAdaptiveStepIsCappedByDsmax) {
  // imsrg++ SetDsmax caps the member ds (IMSRGSolver.cc:201-205).
  flow::SolverOptions o = ShortFlow(0.3);
  o.magnus_adaptive = false;
  o.ds_0 = 0.5;
  o.dsmax = 0.05;
  flow::MagnusSolver solver(Emax2Hno(), o, flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> lines;
  solver.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_GT(lines.size(), 4u);
  EXPECT_DOUBLE_EQ(lines[1].ds, 0.05);
  for (const auto& st : lines) EXPECT_LE(st.ds, 0.05);
  // ds_0 below dsmax is kept.
  o.ds_0 = 0.02;
  flow::MagnusSolver small(Emax2Hno(), o, flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> small_lines;
  small.Solve([&](const flow::FlowStatus& st) { small_lines.push_back(st); });
  EXPECT_DOUBLE_EQ(small_lines[1].ds, 0.02);
}

TEST(MagnusSolver, StageTwoOptionsReachTheSolver) {
  // As the driver does for a stiff valence generator (imsrg++.cc:1130-1140):
  // dsmax 1e-2 for stage 2 only.
  flow::MagnusSolver solver(Emax2Hno(), ShortFlow(0.3),
                            flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> stage1;
  solver.Solve([&](const flow::FlowStatus& st) { stage1.push_back(st); });
  double max_ds1 = 0.0;
  for (const auto& st : stage1) max_ds1 = std::max(max_ds1, st.ds);
  EXPECT_GT(max_ds1, 1.0e-2);  // without the cap the steps are larger
  flow::SolverOptions o2 = solver.solver_options();
  o2.smax = 0.4;
  o2.ds_0 = 1.0e-4;
  o2.dsmax = 1.0e-2;
  solver.set_generator(flow::GeneratorOptions{});
  solver.set_solver_options(o2);
  std::vector<flow::FlowStatus> stage2;
  solver.Solve([&](const flow::FlowStatus& st) { stage2.push_back(st); });
  ASSERT_GT(stage2.size(), 5u);
  for (const auto& st : stage2) EXPECT_LE(st.ds, 1.0e-2);
}

TEST(MagnusSolver, StopsWhenConverged) {
  flow::SolverOptions o = ShortFlow(50.0);
  o.eta_criterion = 0.5;
  flow::MagnusSolver solver(Emax2Hno(), o, flow::GeneratorOptions{});
  const flow::SolveResult r = solver.Solve();
  EXPECT_EQ(r.reason, flow::StopReason::kConverged);
  EXPECT_LT(r.norm_eta, 0.5);
  EXPECT_GT(r.steps, 0);
  EXPECT_LT(r.s, 50.0);
}

}  // namespace
}  // namespace imsrg
