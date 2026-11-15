// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The ODE flow solvers (flow_rk4, flow = dopri5) on the emax-2 O16 HF
// Hamiltonian: convergence order, odeint step control, operators flowing
// alongside H, the odeint error norm, stages, validation and NaN handling. The
// comparison with imsrg++ flow files is in tests/unit/driver/
// test_flow_methods.cc (needs the 2N interaction file).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/flow/ode.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/elementwise.h"
#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"

namespace imsrg {
namespace {

op::Operator Emax2Hno() {
  const std::string path =
      std::string(IMSRG_GOLDEN_DIR) + "/hf/emax2_O16_2N/HNO.dump";
  const auto ms = op::ModelSpaceForDump(op::ReadOperatorDumpHeader(path));
  return op::ReadOperatorDump(path, ms);
}

flow::SolverOptions Options(double smax, double dsmax = 0.5) {
  flow::SolverOptions o;  // imsrg++ executable defaults
  o.smax = smax;
  o.dsmax = dsmax;
  return o;
}

flow::OdeSolver Make(flow::OdeMethod m, const flow::SolverOptions& o,
                     std::vector<op::Operator> ops = {}, double tol = 1.0e-6) {
  return flow::OdeSolver(m, Emax2Hno(), std::move(ops), o, tol,
                         flow::GeneratorOptions{});
}

double E0Rk4(double ds) {
  flow::OdeSolver s = Make(flow::OdeMethod::kRk4, Options(0.4, ds));
  return s.Solve().e0;
}

TEST(OdeRk4, IsFourthOrderAccurate) {
  const double ref = E0Rk4(0.0125 / 2);
  const double e1 = E0Rk4(0.1), e2 = E0Rk4(0.05), e3 = E0Rk4(0.025);
  const double d1 = std::abs(e1 - ref), d2 = std::abs(e2 - ref),
               d3 = std::abs(e3 - ref);
  std::printf("[rk4 order] errors %.3e %.3e %.3e, ratios %.2f %.2f\n", d1, d2,
              d3, d1 / d2, d2 / d3);
  EXPECT_GT(d1 / d2, 10.0);
  EXPECT_LT(d1 / d2, 25.0);
  EXPECT_GT(d2 / d3, 10.0);
  EXPECT_LT(d2 / d3, 25.0);
}

TEST(OdeRk4, StepIsMinOfDsmaxAndRemainder) {
  flow::OdeSolver s = Make(flow::OdeMethod::kRk4, Options(0.35, 0.1));
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r =
      s.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  ASSERT_EQ(lines.size(), 5u);  // istep 0 + 4 steps
  EXPECT_EQ(r.steps, 4);
  EXPECT_EQ(lines[0].istep, 0);
  EXPECT_DOUBLE_EQ(lines[1].ds, 0.1);
  EXPECT_NEAR(lines[4].ds, 0.05, 1.0e-15);
  EXPECT_DOUBLE_EQ(r.s, 0.35);
  EXPECT_EQ(r.reason, flow::StopReason::kSmaxReached);
}

// d O / ds = [eta, O] is linear in O: an operator that starts as H follows H.
TEST(OdeFlow, OperatorStartingAsHFollowsH) {
  for (const auto method : {flow::OdeMethod::kRk4, flow::OdeMethod::kDopri5}) {
    flow::OdeSolver s = Make(method, Options(0.5, 0.1), {Emax2Hno()});
    s.Solve();
    const op::PartMaxima d =
        op::MaxAbsDifference(s.flowing_operator(0), s.hamiltonian());
    EXPECT_LT(d.max(), 1.0e-12)
        << d.zero_body << " " << d.one_body << " " << d.two_body;
    EXPECT_EQ(s.num_operators(), 1u);
  }
}

TEST(OdeDopri5, EveryTryEvaluatesTheRightHandSideSixTimes) {
  // A huge first step is rejected, so the trial count exceeds the step count.
  flow::SolverOptions o = Options(4.0, 4.0);
  o.ds_0 = 4.0;
  flow::OdeSolver s = Make(flow::OdeMethod::kDopri5, o);
  std::vector<flow::FlowStatus> lines;
  const flow::SolveResult r =
      s.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
  // 1 initial line, 1 evaluation at the start, 6 per try.
  ASSERT_GE(lines.size(), 8u);
  EXPECT_EQ((lines.size() - 2) % 6, 0u);
  const std::size_t trials = (lines.size() - 2) / 6;
  EXPECT_GT(trials, static_cast<std::size_t>(r.steps));
  EXPECT_GT(r.steps, 3);
  EXPECT_DOUBLE_EQ(lines[0].norm_eta, 0.0);  // initial line: eta shown as 0
  EXPECT_DOUBLE_EQ(lines[0].s, 0.0);
  EXPECT_DOUBLE_EQ(lines[1].s, 0.0);
  EXPECT_DOUBLE_EQ(r.s, 4.0);
  // First trial: stage times are dt * (1/5, 3/10, 4/5, 8/9, 1, 1).
  EXPECT_NEAR(lines[2].s, 4.0 / 5.0, 1.0e-14);
  EXPECT_NEAR(lines[3].s, 4.0 * 3.0 / 10.0, 1.0e-14);
  EXPECT_NEAR(lines[7].s, 4.0, 1.0e-14);
}

TEST(OdeDopri5, TighterToleranceApproachesTheFineRk4Solution) {
  const double ref = E0Rk4(0.00625);
  double previous = 1.0;
  for (const double tol : {1.0e-4, 1.0e-6, 1.0e-8}) {
    flow::OdeSolver s = Make(flow::OdeMethod::kDopri5, Options(0.4), {}, tol);
    const double dev = std::abs(s.Solve().e0 - ref);
    std::printf("[dopri5] tol %.0e: |E0 - E0(rk4 fine)| = %.3e\n", tol, dev);
    EXPECT_LT(dev, previous);
    previous = dev;
  }
  EXPECT_LT(previous, 1.0e-8);
}

TEST(OdeDopri5, ScaledStateErrorProperties) {
  const op::Operator h = Emax2Hno();
  op::Operator e = h;
  op::Scale(e, 1.0e-3);
  op::Operator z = h;
  op::Scale(z, 0.0);
  const std::vector<op::Operator> xerr = {e};
  const std::vector<op::Operator> zero = {z};
  // x = 0, dxdt = 0: the denominator is eps_abs everywhere.
  const double err0 =
      flow::ScaledStateError(xerr, zero, zero, 0.1, 1.0e-6, 1.0e-6);
  EXPECT_NEAR(err0, op::Norm(e) / 1.0e-6, 1.0e-9 * err0);
  // Linear in |xerr|.
  std::vector<op::Operator> e2 = xerr;
  op::Scale(e2[0], 3.0);
  EXPECT_NEAR(flow::ScaledStateError(e2, zero, zero, 0.1, 1.0e-6, 1.0e-6),
              3.0 * err0, 1.0e-9 * err0);
  // A nonzero state enlarges the denominator and lowers the error; the
  // zero-body part is excluded from the norm.
  const std::vector<op::Operator> x = {h};
  const double err1 =
      flow::ScaledStateError(xerr, x, zero, 0.1, 1.0e-6, 1.0e-6);
  EXPECT_LT(err1, err0);
  std::vector<op::Operator> x_e0 = x;
  x_e0[0].zero_body() += 100.0;
  EXPECT_DOUBLE_EQ(
      flow::ScaledStateError(xerr, x_e0, zero, 0.1, 1.0e-6, 1.0e-6), err1);
  // The sum over the operators of the state.
  const std::vector<op::Operator> two = {e, e};
  const std::vector<op::Operator> two_zero = {z, z};
  EXPECT_NEAR(
      flow::ScaledStateError(two, two_zero, two_zero, 0.1, 1.0e-6, 1.0e-6),
      2.0 * err0, 1.0e-9 * err0);
}

TEST(OdeFlow, ConvergenceStopsRk4AndDopri5Reports) {
  flow::SolverOptions o = Options(50.0);
  o.eta_criterion = 0.5;
  flow::OdeSolver rk4 = Make(flow::OdeMethod::kRk4, o);
  const flow::SolveResult r = rk4.Solve();
  EXPECT_EQ(r.reason, flow::StopReason::kConverged);
  EXPECT_GT(r.steps, 0);
  EXPECT_LT(r.s, 50.0);
  // dopri5 goes on to smax with dxdt = 0 once ||eta|| < eta_criterion
  // (imsrg++ has no stop) and reports the convergence.
  flow::OdeSolver dopri = Make(flow::OdeMethod::kDopri5, o);
  const flow::SolveResult d = dopri.Solve();
  EXPECT_EQ(d.reason, flow::StopReason::kConverged);
  EXPECT_DOUBLE_EQ(d.s, 50.0);
  EXPECT_LT(d.norm_eta, 0.5);
}

TEST(OdeFlow, SecondStageContinuesS) {
  for (const auto method : {flow::OdeMethod::kRk4, flow::OdeMethod::kDopri5}) {
    flow::OdeSolver s = Make(method, Options(0.3), {Emax2Hno()});
    const flow::SolveResult r1 = s.Solve();
    flow::SolverOptions o2 = s.solver_options();
    o2.smax = 0.6;
    o2.ds_0 = 1.0e-3;
    o2.dsmax = 1.0e-2;
    s.set_generator(flow::GeneratorOptions{});
    s.set_solver_options(o2);
    std::vector<flow::FlowStatus> lines;
    const flow::SolveResult r2 =
        s.Solve([&](const flow::FlowStatus& st) { lines.push_back(st); });
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.front().stage, 2);
    EXPECT_DOUBLE_EQ(lines.front().s, r1.s);
    EXPECT_DOUBLE_EQ(r2.s, 0.6);
    EXPECT_LT(r2.e0, r1.e0);
    if (method == flow::OdeMethod::kRk4) {
      for (const auto& st : lines) EXPECT_LE(st.ds, 1.0e-2);
    }
  }
}

TEST(OdeFlow, ValidationThrowsInputError) {
  flow::SolverOptions hg = Options(1.0);
  hg.hunter_gatherer = true;
  EXPECT_THROW(Make(flow::OdeMethod::kRk4, hg), InputError);
  flow::SolverOptions bad = Options(1.0);
  bad.dsmax = 0.0;
  EXPECT_THROW(Make(flow::OdeMethod::kDopri5, bad), InputError);
  EXPECT_THROW(Make(flow::OdeMethod::kDopri5, Options(1.0), {}, 0.0),
               InputError);
  EXPECT_THROW(Make(flow::OdeMethod::kDopri5, Options(1.0), {}, std::nan("")),
               InputError);
  flow::OdeSolver s = Make(flow::OdeMethod::kRk4, Options(1.0));
  EXPECT_THROW(s.set_solver_options(bad), InputError);
  EXPECT_THROW(s.set_solver_options(hg), InputError);
}

TEST(OdeFlow, NanIsDivergenceNotSmax) {
  for (const auto method : {flow::OdeMethod::kRk4, flow::OdeMethod::kDopri5}) {
    op::Operator h0 = Emax2Hno();
    h0.one_body().set(OrbitIndex{0}, OrbitIndex{0},
                      std::numeric_limits<double>::quiet_NaN());
    flow::OdeSolver s(method, std::move(h0), {}, Options(2.0), 1.0e-6,
                      flow::GeneratorOptions{});
    const flow::SolveResult r = s.Solve();
    EXPECT_EQ(r.reason, flow::StopReason::kDiverged);
  }
}

}  // namespace
}  // namespace imsrg
