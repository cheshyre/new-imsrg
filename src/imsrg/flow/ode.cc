// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The ODE flow solvers (design/13 section 7; imsrg++ IMSRGSolver.cc:550-683
// and 682-933, boost odeint 1.92 `integrate_adaptive` with the controlled
// `runge_kutta_dopri5` stepper).
#include "imsrg/flow/ode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/commutator/commutator.h"
#include "imsrg/flow/bch.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/elementwise.h"
#include "imsrg/operator/norms.h"

namespace imsrg::flow {

namespace {

using State = std::vector<op::Operator>;

double NowSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// out = c_1 t_1 + c_2 t_2 + ... evaluated left to right with every term
/// scaled before it is added, as odeint's `scale_sum` on the imsrg++ state
/// algebra (`operator*(double, deque)`, `operator+(deque, deque)`).
struct Term {
  double c;
  const State* x;
};
void LinComb(State& out, std::initializer_list<Term> terms) {
  const Term* it = terms.begin();
  const State& first = *it->x;
  out.resize(first.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    out[i] = first[i];
    op::Scale(out[i], it->c);
  }
  for (++it; it != terms.end(); ++it) {
    for (std::size_t i = 0; i < first.size(); ++i) {
      op::Axpy(it->c, (*it->x)[i], out[i]);
    }
  }
}

// Adds ds/6 (K1 + 2 K2 + 2 K3 + K4) to `state`, summed in imsrg++'s order.
void Rk4Combine(State& state, const State& k1, const State& k2, const State& k3,
                const State& k4, double ds) {
  for (std::size_t i = 0; i < state.size(); ++i) {
    op::Operator sum = k1[i];
    op::Axpy(2.0, k2[i], sum);
    op::Axpy(2.0, k3[i], sum);
    op::Axpy(1.0, k4[i], sum);
    op::Axpy(ds / 6.0, sum, state[i]);
  }
}

// Dormand-Prince coefficients exactly as boost/numeric/odeint/stepper/
// runge_kutta_dopri5.hpp.
constexpr double kA2 = 1.0 / 5.0, kA3 = 3.0 / 10.0, kA4 = 4.0 / 5.0,
                 kA5 = 8.0 / 9.0;
constexpr double kB21 = 1.0 / 5.0;
constexpr double kB31 = 3.0 / 40.0, kB32 = 9.0 / 40.0;
constexpr double kB41 = 44.0 / 45.0, kB42 = -56.0 / 15.0, kB43 = 32.0 / 9.0;
constexpr double kB51 = 19372.0 / 6561.0, kB52 = -25360.0 / 2187.0,
                 kB53 = 64448.0 / 6561.0, kB54 = -212.0 / 729.0;
constexpr double kB61 = 9017.0 / 3168.0, kB62 = -355.0 / 33.0,
                 kB63 = 46732.0 / 5247.0, kB64 = 49.0 / 176.0,
                 kB65 = -5103.0 / 18656.0;
constexpr double kC1 = 35.0 / 384.0, kC3 = 500.0 / 1113.0, kC4 = 125.0 / 192.0,
                 kC5 = -2187.0 / 6784.0, kC6 = 11.0 / 84.0;
constexpr double kDc1 = kC1 - 5179.0 / 57600.0;
constexpr double kDc3 = kC3 - 7571.0 / 16695.0;
constexpr double kDc4 = kC4 - 393.0 / 640.0;
constexpr double kDc5 = kC5 + 92097.0 / 339200.0;
constexpr double kDc6 = kC6 - 187.0 / 2100.0;
constexpr double kDc7 = -1.0 / 40.0;

// The stage vectors of one Dormand-Prince step (dxdt is k1).
struct DopriStages {
  State dxdt, dxdt_new, xnew, xerr, xtmp, k2, k3, k4, k5, k6;
};

// One Dormand-Prince step from (x, t) with step dt, given dxdt = f(x, t) in
// `w`: fills w.xnew, w.dxdt_new (FSAL) and the error estimate w.xerr. `rhs`
// is called as rhs(x, dxdt, t) in odeint's stage order.
template <typename RhsFn>
void DopriTrialStep(RhsFn&& rhs, const State& x, double t, double dt,
                    DopriStages& w) {
  LinComb(w.xtmp, {{1.0, &x}, {dt * kB21, &w.dxdt}});
  rhs(w.xtmp, w.k2, t + dt * kA2);
  LinComb(w.xtmp, {{1.0, &x}, {dt * kB31, &w.dxdt}, {dt * kB32, &w.k2}});
  rhs(w.xtmp, w.k3, t + dt * kA3);
  LinComb(w.xtmp, {{1.0, &x},
                   {dt * kB41, &w.dxdt},
                   {dt * kB42, &w.k2},
                   {dt * kB43, &w.k3}});
  rhs(w.xtmp, w.k4, t + dt * kA4);
  LinComb(w.xtmp, {{1.0, &x},
                   {dt * kB51, &w.dxdt},
                   {dt * kB52, &w.k2},
                   {dt * kB53, &w.k3},
                   {dt * kB54, &w.k4}});
  rhs(w.xtmp, w.k5, t + dt * kA5);
  LinComb(w.xtmp, {{1.0, &x},
                   {dt * kB61, &w.dxdt},
                   {dt * kB62, &w.k2},
                   {dt * kB63, &w.k3},
                   {dt * kB64, &w.k4},
                   {dt * kB65, &w.k5}});
  rhs(w.xtmp, w.k6, t + dt);
  LinComb(w.xnew, {{1.0, &x},
                   {dt * kC1, &w.dxdt},
                   {dt * kC3, &w.k3},
                   {dt * kC4, &w.k4},
                   {dt * kC5, &w.k5},
                   {dt * kC6, &w.k6}});
  rhs(w.xnew, w.dxdt_new, t + dt);  // the new derivative (FSAL)
  LinComb(w.xerr, {{dt * kDc1, &w.dxdt},
                   {dt * kDc3, &w.k3},
                   {dt * kDc4, &w.k4},
                   {dt * kDc5, &w.k5},
                   {dt * kDc6, &w.k6},
                   {dt * kDc7, &w.dxdt_new}});
}

/// Shrinks `dt` after a rejected step with scaled error `err` > 1
/// (odeint default_step_adjuster::decrease_step, error order 4).
void DecreaseStep(double& dt, double err) {
  dt *= std::max(0.9 * std::pow(err, -1.0 / 3.0), 1.0 / 5.0);
}

/// Grows `dt` after an accepted step with scaled error `err` (odeint
/// default_step_adjuster::increase_step, stepper order 5); unchanged for
/// err >= 0.5.
void IncreaseStep(double& dt, double err) {
  if (err < 0.5) {
    const double e = std::max(std::pow(5.0, -5.0), err);
    dt *= 0.9 * std::pow(e, -1.0 / 5.0);
  }
}

}  // namespace

double ScaledStateError(const State& x_err, const State& x, const State& dxdt,
                        double dt, double eps_abs, double eps_rel) {
  IMSRG_CHECK(x_err.size() == x.size() && x.size() == dxdt.size(),
              "ScaledStateError: states differ in size");
  IMSRG_CHECK(eps_abs > 0.0, "ScaledStateError: eps_abs must be positive");
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    op::Operator num = x_err[i];
    op::ElementwiseAbs(num);
    // den = eps_abs + eps_rel * (a_x |x| + (a_dxdt |dt|) |dxdt|), a_x = a_dxdt
    // = 1 (odeint's default_error_checker).
    op::Operator den = x[i];
    op::ElementwiseAbs(den);
    op::Operator b = dxdt[i];
    op::ElementwiseAbs(b);
    op::Scale(b, std::abs(dt));
    op::Axpy(1.0, b, den);
    op::Scale(den, eps_rel);
    op::AddToEveryElement(den, eps_abs);
    op::ElementwiseDivide(num, den);
    sum += op::Norm(num);
  }
  return sum;
}

OdeSolver::OdeSolver(OdeMethod method, op::Operator H0,
                     std::vector<op::Operator> operators,
                     const SolverOptions& solver, double ode_tolerance,
                     const GeneratorOptions& generator,
                     const commutator::CommutatorOptions& comm)
    : solver_(solver),
      generator_(generator),
      comm_(comm),
      method_(method),
      t_start_(NowSeconds()) {
  ValidateSolverOptions(solver_);
  if (solver_.hunter_gatherer) {
    throw InputError("OdeSolver: hunter_gatherer applies to Magnus only");
  }
  if (!(ode_tolerance > 0.0) || !std::isfinite(ode_tolerance)) {
    throw InputError("OdeSolver: ode_tolerance must be positive and finite");
  }
  eps_ = ode_tolerance;  // double (D47; imsrg++ rounds it through float)
  IMSRG_CHECK(!H0.empty(), "OdeSolver: empty Hamiltonian");
  IMSRG_CHECK(
      H0.spec().is_scalar() && H0.spec().is_number_conserving() &&
          H0.spec().particle_rank == 2,
      "OdeSolver: H0 must be a scalar, number-conserving two-body operator");
  ws_ = std::make_unique<commutator::Workspace>(H0.model_space_ptr());
  eta_ = op::Operator(H0.model_space_ptr(),
                      GeneratorSpec(H0.spec(), generator_.kind));
  state_.push_back(std::move(H0));
  for (op::Operator& o : operators) {
    if (o.empty() || !o.spec().is_scalar() ||
        !o.spec().is_number_conserving()) {
      throw InputError(
          "OdeSolver: the flowing operators must be scalar and number "
          "conserving");
    }
    state_.push_back(WithParticleRank(o, 2));
  }
}

OdeSolver::~OdeSolver() = default;
OdeSolver::OdeSolver(OdeSolver&&) noexcept = default;
OdeSolver& OdeSolver::operator=(OdeSolver&&) noexcept = default;

void OdeSolver::set_generator(const GeneratorOptions& generator) {
  generator_ = generator;
}

void OdeSolver::set_solver_options(const SolverOptions& solver) {
  ValidateSolverOptions(solver);
  if (solver.hunter_gatherer) {
    throw InputError("OdeSolver: hunter_gatherer applies to Magnus only");
  }
  solver_ = solver;
}

const op::Operator& OdeSolver::hamiltonian() const noexcept {
  return eval_h_ != nullptr ? *eval_h_ : state_.front();
}

FlowStatus OdeSolver::MakeStatus(int istep, double ds, double s,
                                 const op::Operator& h, bool zero_eta) const {
  FlowStatus st;
  st.stage = stage_;
  st.istep = istep;
  st.s = s;
  st.ds = ds;
  st.e0 = h.zero_body();
  st.norm_h = op::Norm(h);
  if (!zero_eta) {
    st.norm_eta_1b = op::OneBodyNorm(eta_);
    st.norm_eta_2b = op::TwoBodyNorm(eta_);
    st.norm_eta = op::Norm(eta_);
  }
  st.num_omegas = 1;  // no Omega stack in the ODE methods
  st.wall_seconds = NowSeconds() - t_start_;
  return st;
}

void OdeSolver::Rhs(const State& x, State& dxdt, double t) {
  dxdt.resize(x.size());
  if (diverged_) {  // no more work after a runaway
    for (std::size_t i = 0; i < x.size(); ++i) {
      dxdt[i] = x[i];
      op::Scale(dxdt[i], 0.0);
    }
    return;
  }
  ComputeGenerator(x[0], eta_, generator_);
  if (on_status_) {
    eval_h_ = &x[0];
    on_status_(MakeStatus(0, 0.0, t, x[0], false));
    eval_h_ = nullptr;
  }
  const double norm_eta = op::Norm(eta_);
  // imsrg++ `ode_*` has no guard; ours is NaN-safe (see the class comment).
  if (!(norm_eta <= 1.0e12) || !(std::abs(x[0].zero_body()) <= 1.0e9)) {
    log::Error("the IMSRG flow diverged: ||eta|| = {}, E0 = {} at s = {}",
               norm_eta, x[0].zero_body(), t);
    diverged_ = true;
  }
  // D47: a double (imsrg++ rounds it through float, quirk P7).
  const double eta_criterion = solver_.eta_criterion;
  if (norm_eta < eta_criterion || diverged_) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      dxdt[i] = x[i];
      op::Scale(dxdt[i], 0.0);
    }
  } else {
    for (std::size_t i = 0; i < x.size(); ++i) {
      dxdt[i] = commutator::Commutator(eta_, x[i], comm_, *ws_);
    }
  }
}

SolveResult OdeSolver::Solve(const StatusCallback& on_status) {
  ++stage_;
  on_status_ = on_status;
  diverged_ = false;
  SolveResult result = method_ == OdeMethod::kRk4 ? SolveRk4() : SolveDopri5();
  on_status_ = nullptr;
  eval_h_ = nullptr;
  result.s = s_;
  result.e0 = state_.front().zero_body();
  result.norm_eta = op::Norm(eta_);
  const double eta_criterion = solver_.eta_criterion;
  if (diverged_ || !std::isfinite(result.norm_eta) ||
      !std::isfinite(result.e0)) {
    result.reason = StopReason::kDiverged;
  } else if (result.norm_eta < eta_criterion) {
    result.reason = StopReason::kConverged;
  } else {
    result.reason = StopReason::kSmaxReached;
  }
  return result;
}

SolveResult OdeSolver::SolveRk4() {
  SolveResult result;
  const bool shell_atan = generator_.kind == GeneratorKind::kShellModelAtan;
  const double user_cutoff = generator_.denominator_cutoff;
  if (shell_atan) generator_.denominator_cutoff = 1.0;
  ComputeGenerator(state_[0], eta_, generator_);
  const double eta_criterion = solver_.eta_criterion;
  if (on_status_) on_status_(MakeStatus(0, 0.0, s_, state_[0], false));

  const std::size_t n = state_.size();
  State k1(n), k2(n), k3(n), k4(n), tmp(n);
  for (int istep = 1; s_ < solver_.smax; ++istep) {
    const double norm_eta = op::Norm(eta_);
    if (norm_eta < eta_criterion) break;
    if (!(norm_eta <= 1.0e12) || !(std::abs(state_[0].zero_body()) <= 1.0e9)) {
      log::Error("the IMSRG flow diverged: ||eta|| = {}, E0 = {} at s = {}",
                 norm_eta, state_[0].zero_body(), s_);
      diverged_ = true;
      break;
    }
    const double ds = std::min(solver_.dsmax, solver_.smax - s_);
    s_ += ds;
    auto commutators = [&](const State& x, State& k) {
      for (std::size_t i = 0; i < n; ++i) {
        k[i] = commutator::Commutator(eta_, x[i], comm_, *ws_);
      }
    };
    commutators(state_, k1);
    LinComb(tmp, {{1.0, &state_}, {0.5 * ds, &k1}});
    ComputeGenerator(tmp[0], eta_, generator_);
    commutators(tmp, k2);
    LinComb(tmp, {{1.0, &state_}, {0.5 * ds, &k2}});
    ComputeGenerator(tmp[0], eta_, generator_);
    commutators(tmp, k3);
    LinComb(tmp, {{1.0, &state_}, {1.0 * ds, &k3}});
    ComputeGenerator(tmp[0], eta_, generator_);
    commutators(tmp, k4);
    Rk4Combine(state_, k1, k2, k3, k4, ds);
    if (shell_atan && norm_eta < 1.0) generator_.denominator_cutoff = 1.0e-6;
    ComputeGenerator(state_[0], eta_, generator_);
    if (on_status_) on_status_(MakeStatus(istep, ds, s_, state_[0], false));
    result.steps = istep;
  }
  if (shell_atan) generator_.denominator_cutoff = user_cutoff;
  return result;
}

SolveResult OdeSolver::SolveDopri5() {
  constexpr double kEps = std::numeric_limits<double>::epsilon();

  SolveResult result;
  // imsrg++ writes the status once before odeint starts (eta not computed).
  if (on_status_) on_status_(MakeStatus(0, 0.0, s_, state_[0], true));

  State& x = state_;
  DopriStages w;
  const auto rhs = [this](const State& xs, State& dxs, double ts) {
    Rhs(xs, dxs, ts);
  };
  double t = s_;
  double dt = std::min(solver_.ds_0, solver_.dsmax);
  const double end = solver_.smax;
  int failed = 0;
  bool first_call = true;

  // odeint detail::integrate_adaptive (controlled stepper) with
  // less_with_sign(t1, t2, dt) = t2 - t1 > eps.
  while (end - t > kEps) {
    if (t + dt - end > kEps) dt = end - t;
    bool accepted = false;
    do {
      if (first_call) {
        Rhs(x, w.dxdt, t);
        first_call = false;
      }
      DopriTrialStep(rhs, x, t, dt, w);  // try_step with error estimate
      if (diverged_) return result;
      const double err = ScaledStateError(w.xerr, x, w.dxdt, dt, eps_, eps_);
      if (err > 1.0) {
        // default_step_adjuster::decrease_step (error_order 4).
        DecreaseStep(dt, err);
        if (++failed >= 500) {
          throw NumericalError(
              "OdeSolver: max number of iterations exceeded (500): a new step "
              "size was not found");
        }
      } else {
        t += dt;
        // default_step_adjuster::increase_step (stepper_order 5).
        IncreaseStep(dt, err);
        std::swap(x, w.xnew);
        std::swap(w.dxdt, w.dxdt_new);
        accepted = true;
      }
    } while (!accepted);
    failed = 0;
    ++result.steps;
    s_ = t;
  }
  s_ = t;
  return result;
}

}  // namespace imsrg::flow
