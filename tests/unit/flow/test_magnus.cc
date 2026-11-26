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

// Hunter-gatherer (imsrg++ GatherOmega semantics, D32, and D48): the stack is
// {gatherer, hunter}; after the final gather H(s) is the H rebuilt with the
// gathered Omega, so H(s) == Transform(H_0) with the final Omega (imsrg++ left
// H(s) at the last hunter step; design/30 section 6). A gather inside the loop
// leaves H(s) alone (the status lines are those of the loop).
TEST(MagnusSolver, HunterGathererHamiltonianIsRebuiltWithGatheredOmega) {
  op::Operator h0 = Emax2Hno();
  flow::SolverOptions o = ShortFlow(1.5);
  o.hunter_gatherer = true;
  flow::MagnusSolver solver(h0, o, flow::GeneratorOptions{});
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r =
      solver.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_GT(lines.size(), 4u);
  // The last status line is the rebuilt H at the same s (istep = steps + 1).
  const flow::FlowStatus rebuilt = lines.back();
  lines.pop_back();
  EXPECT_EQ(rebuilt.istep, r.steps + 1);
  EXPECT_EQ(rebuilt.s, lines.back().s);
  EXPECT_EQ(rebuilt.ds, 0.0);
  EXPECT_EQ(rebuilt.e0, r.e0);
  // eta(s) belongs to the rebuilt H (not the discarded hunter-step H).
  EXPECT_EQ(rebuilt.norm_eta, r.norm_eta);
  EXPECT_EQ(r.norm_eta, op::Norm(solver.eta()));
  EXPECT_NE(r.norm_eta, lines.back().norm_eta);
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
  // D48: H(s) is the replay of the gathered Omega on H_0 (the loop's last line
  // still shows the hunter-step H; the rebuilt one differs from it).
  const op::Operator replayed = solver.Transform(h0);
  const op::PartMaxima d = op::MaxAbsDifference(replayed, solver.hamiltonian());
  const double scale = std::max(1.0, op::Norm(solver.hamiltonian()));
  std::printf(
      "[magnus HG] %d gathers; replay vs H(s): E0 %.3e, 1b %.3e, 2b %.3e; "
      "E0 %.9f (last loop line %.9f, shift %.3e)\n",
      r.num_gathers, d.zero_body, d.one_body, d.two_body, r.e0, lines.back().e0,
      r.e0 - lines.back().e0);
  EXPECT_LT(d.max(), 1.0e-10 * scale);
  EXPECT_EQ(r.e0, solver.hamiltonian().zero_body());
  EXPECT_NE(r.e0, lines.back().e0);  // the rebuild moves E0 (BCH truncation)
  EXPECT_LT(std::abs(r.e0 - lines.back().e0), 1.0e-1);
  // A second stage gathers nothing in set_generator (the hunter is zero) and
  // keeps the stack at two entries.
  solver.set_generator(flow::GeneratorOptions{});
  EXPECT_EQ(solver.omegas().size(), 2u);
  EXPECT_EQ(solver.num_gathers(), r.num_gathers);
}

// D47: eta_criterion is a double. A threshold one ulp above ||eta|| of the
// initial H stops the flow at once; rounded through float (imsrg++) the
// threshold would fall below ||eta|| (a float has 24 mantissa bits) and the
// solver would take steps.
TEST(MagnusSolver, EtaCriterionIsADouble) {
  const op::Operator h0 = Emax2Hno();
  flow::MagnusSolver probe(h0, ShortFlow(0.0), flow::GeneratorOptions{});
  probe.Solve();  // smax 0: only computes eta
  const double norm_eta = op::Norm(probe.eta());
  ASSERT_GT(norm_eta, 0.0);
  const double above = std::nextafter(norm_eta, 1.0e300);
  const double below = std::nextafter(norm_eta, 0.0);
  // Precondition of the test: a round trip through float moves one of the two
  // thresholds to the other side of ||eta|| (float has 24 mantissa bits).
  ASSERT_TRUE(static_cast<double>(static_cast<float>(above)) <= norm_eta ||
              static_cast<double>(static_cast<float>(below)) >= norm_eta);
  for (const double crit : {above, below}) {
    flow::SolverOptions o = ShortFlow(1.5);
    o.eta_criterion = crit;
    flow::MagnusSolver solver(h0, o, flow::GeneratorOptions{});
    const flow::SolveResult r = solver.Solve();
    if (crit > norm_eta) {
      EXPECT_EQ(r.reason, flow::StopReason::kConverged);
      EXPECT_EQ(r.steps, 0);
    } else {
      EXPECT_GT(r.steps, 0);
    }
  }
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
    h0.SyncToDevice();
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
    h0.SyncToDevice();
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

namespace imsrg {
namespace {

// A captured checkpoint: loop state plus copies of the operators.
struct Snapshot {
  flow::MagnusLoopState state;
  std::vector<op::Operator> omegas;
  op::Operator h_saved;
  op::Operator h;
};

void ExpectSameOperator(const op::Operator& a, const op::Operator& b) {
  EXPECT_EQ(a.zero_body(), b.zero_body());
  EXPECT_EQ(op::MaxAbsDifference(a, b).max(), 0.0);
}

// Checkpoint/restart (design/50 C7): a two-stage flow with Omega splits,
// restored from a step in stage 1 and from a step in stage 2, continues bit
// for bit to the same result as the uninterrupted flow.
TEST(MagnusSolver, RestoreContinuesBitForBit) {
  flow::SolverOptions o = ShortFlow(0.8);
  o.omega_norm_max = 0.1;  // several Omegas per stage
  const op::Operator h0 = Emax2Hno();
  const flow::GeneratorOptions g1{};
  flow::GeneratorOptions g2{};
  g2.kind = flow::GeneratorKind::kWhite;

  std::vector<Snapshot> snaps;
  auto capture = [&](const flow::MagnusSolver& s) {
    return [&snaps, &s](const flow::MagnusLoopState& st) {
      if (st.istep == 3) {
        snaps.push_back(
            {st, s.omegas(), s.saved_hamiltonian(), s.hamiltonian()});
      }
    };
  };
  flow::MagnusSolver a(h0, o, g1);
  const flow::SolveResult a1 = a.Solve({}, capture(a));
  a.set_generator(g2);
  a.set_smax(1.6);
  const flow::SolveResult a2 = a.Solve({}, capture(a));
  ASSERT_EQ(snaps.size(), 2u);
  ASSERT_GT(snaps[0].omegas.size(), 1u);  // a split happened before step 3
  ASSERT_GT(a2.steps, 3);

  // Restart in stage 1.
  {
    flow::MagnusSolver b(h0, o, g1);
    b.Restore(snaps[0].state, g1, snaps[0].omegas, snaps[0].h_saved,
              snaps[0].h);
    const flow::SolveResult b1 = b.Solve();
    EXPECT_EQ(b1.steps, a1.steps);
    EXPECT_EQ(b1.s, a1.s);
    EXPECT_EQ(b1.e0, a1.e0);
    b.set_generator(g2);
    b.set_smax(1.6);
    const flow::SolveResult b2 = b.Solve();
    EXPECT_EQ(b2.steps, a2.steps);
    EXPECT_EQ(b2.e0, a2.e0);
    ASSERT_EQ(b.omegas().size(), a.omegas().size());
    for (std::size_t i = 0; i < a.omegas().size(); ++i)
      ExpectSameOperator(b.omegas()[i], a.omegas()[i]);
    ExpectSameOperator(b.hamiltonian(), a.hamiltonian());
  }
  // Restart in stage 2 (the generator and smax come from the checkpoint).
  {
    flow::MagnusSolver c(h0, o, g1);
    c.Restore(snaps[1].state, g2, snaps[1].omegas, snaps[1].h_saved,
              snaps[1].h);
    std::vector<flow::FlowStatus> lines;
    const flow::SolveResult c2 =
        c.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.front().stage, 2);
    EXPECT_EQ(lines.front().istep, 4);
    EXPECT_EQ(c2.steps, a2.steps);
    EXPECT_EQ(c2.s, a2.s);
    EXPECT_EQ(c2.e0, a2.e0);
    ExpectSameOperator(c.hamiltonian(), a.hamiltonian());
    ASSERT_EQ(c.omegas().size(), a.omegas().size());
    ExpectSameOperator(c.omegas().back(), a.omegas().back());
  }
}

TEST(MagnusSolver, RestoreRejectsHunterGatherer) {
  flow::SolverOptions o = ShortFlow(0.5);
  o.hunter_gatherer = true;
  flow::MagnusSolver s(Emax2Hno(), o, flow::GeneratorOptions{});
  flow::MagnusLoopState st;
  st.stage = 1;
  st.solver = o;
  EXPECT_THROW(s.Restore(st, flow::GeneratorOptions{}, s.omegas(),
                         op::Operator(), s.hamiltonian()),
               InputError);
}

}  // namespace
}  // namespace imsrg
