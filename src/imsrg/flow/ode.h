// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_ODE_H_
#define IMSRG_FLOW_ODE_H_
/// @file
/// The IMSRG flow equation solvers that integrate dH/ds = [eta(H), H] directly
/// (imsrg++ methods `flow_RK4` and `flow`/`flow_adaptive`, design/13 section 7,
/// design/reference/03 sections 2.2-2.3): the fixed-step classical RK4 of
/// `IMSRGSolver::Solve_flow_RK4` (IMSRGSolver.cc:550-683) and the adaptive
/// Dormand-Prince 5(4) of boost odeint's `integrate_adaptive` with a
/// `make_controlled<runge_kutta_dopri5>` stepper (`Solve_ode_adaptive`,
/// IMSRGSolver.cc:819-933, error control of the imsrg++ element-wise state
/// algebra, :682-807). The operators of the run evolve alongside H with the
/// same eta (imsrg++.cc:1074-1077): d O / ds = [eta, O].
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/flow/generator.h"
#include "imsrg/flow/magnus.h"
#include "imsrg/operator/operator.h"

namespace imsrg::flow {

/// The integrator of `OdeSolver`.
enum class OdeMethod : std::int8_t {
  kRk4,     ///< `flow_RK4`: fixed step ds = min(dsmax, smax - s).
  kDopri5,  ///< `flow`: adaptive Dormand-Prince 5(4), odeint step control.
};

/// odeint's `default_error_checker` on the imsrg++ state algebra: for every
/// operator i of the state, with the element-wise arithmetic of
/// `op/elementwise.h`,
///   norm_i = || |xerr_i| / (eps_abs + eps_rel (|x_i| + |dt| |dxdt_i|)) ||
/// (`op::Norm`: the zero-body part, i.e. E0, is excluded) and the result is
/// the SUM of the norm_i (imsrg++ `vector_space_norm_inf`,
/// IMSRGSolver.cc:798-807). A step is accepted if the result is <= 1.
/// Preconditions: the four states have the same size and layout; eps_abs > 0.
/// Complexity O(size of the state).
double ScaledStateError(const std::vector<op::Operator>& x_err,
                        const std::vector<op::Operator>& x,
                        const std::vector<op::Operator>& dxdt, double dt,
                        double eps_abs, double eps_rel);

/// Solver state of the ODE flow methods: H(s) and the flowing operators.
/// Owns a commutator `Workspace`; movable, not copyable.
///
/// Differences from imsrg++ (design/QUESTIONS M8-Q40): (a) the flow parameter
/// continues in a second stage (imsrg++'s odeint path integrates a copy of the
/// solver, so its `s` and `ds` never advance and stage 2 restarts at s = 0, the
/// same range length as the [s1, s1 + smax] of D47);
/// (b) non-finite or runaway ||eta|| / E0 ends the solve as `kDiverged` (the
/// odeint path of imsrg++ has no such guard).
class OdeSolver {
 public:
  /// Callback receiving every flow-log line (see `Solve`).
  using StatusCallback = std::function<void(const FlowStatus&)>;

  /// Sets up the solver for H0 (scalar, Hermitian, particle_rank 2, normal
  /// ordered; copied) and the operators that flow with it (scalar, number
  /// conserving; copied and promoted to particle_rank 2). `ode_tolerance` is
  /// the absolute and relative odeint tolerance of `kDopri5`, used as a
  /// double (D47; imsrg++ `SetODETolerance(float)` rounds it). Builds the
  /// commutator workspace with `workspace` (communicator of the ranks that
  /// share the commutator work, serial by default) (needs an initialized
  /// exec::Runtime).
  /// @throws InputError for invalid step options (see `ValidateSolverOptions`),
  ///   `solver.hunter_gatherer`, ode_tolerance <= 0, or an operator that is not
  ///   a scalar number-conserving one.
  OdeSolver(OdeMethod method, op::Operator H0,
            std::vector<op::Operator> operators, const SolverOptions& solver,
            double ode_tolerance, const GeneratorOptions& generator,
            const commutator::CommutatorOptions& comm = {},
            const commutator::WorkspaceOptions& workspace = {});
  ~OdeSolver();
  OdeSolver(OdeSolver&&) noexcept;
  OdeSolver& operator=(OdeSolver&&) noexcept;
  OdeSolver(const OdeSolver&) = delete;
  OdeSolver& operator=(const OdeSolver&) = delete;

  /// Changes the generator for the next `Solve` (imsrg++ `SetGenerator`).
  void set_generator(const GeneratorOptions& generator);
  /// Replaces the step options for the next `Solve` (imsrg++.cc:1130-1140).
  /// @throws InputError as the constructor.
  void set_solver_options(const SolverOptions& solver);

  /// Integrates from the current s to smax (or until converged, `kRk4`).
  ///
  /// `kRk4` (Solve_flow_RK4): eta = gen(H); loop while s < smax: stop if
  /// ||eta|| < eta_criterion (a double, D47); ds = min(dsmax, smax -
  /// s); classical RK4 on every operator with K_j = [eta_j, X_j], eta_j =
  /// gen(X_0 at stage j) (stage times s, s + ds/2, s + ds/2, s + ds), then
  /// X += ds/6 (K1 + 2 K2 + 2 K3 + K4) and eta = gen(H). For the
  /// "shell-model-atan" generator the denominator cutoff is 1.0 until
  /// ||eta|| < 1, then 1e-6. `on_status` receives the istep-0 line and one
  /// line per step.
  ///
  /// `kDopri5` (Solve_ode_adaptive): odeint `integrate_adaptive` with initial
  /// dt = min(ds_0, dsmax) from s to smax: every try of a step evaluates the
  /// right-hand side 6 times (FSAL), rejected steps (error > 1) shrink dt by
  /// max(0.9 err^(-1/3), 0.2), accepted steps with error < 0.5 grow it by
  /// 0.9 max(err, 5^-5)^(-1/5); dxdt = 0 where ||eta|| < eta_criterion (so,
  /// as in imsrg++, the integration runs on to smax in a few growing steps
  /// after convergence; more than 500 consecutive rejections throw
  /// `NumericalError`). `on_status` receives one line for the initial state
  /// (eta shown as 0) and one line for EVERY right-hand-side evaluation (as
  /// the imsrg++ flow file; istep 0, s = the stage time, H and eta of that
  /// stage). `SolveResult::steps` counts accepted steps.
  ///
  /// The stop reason is `kConverged` if the final ||eta|| < eta_criterion,
  /// `kSmaxReached` otherwise, `kDiverged` as for Magnus.
  ///
  /// Collective over the Workspace communicator: every rank calls it with
  /// identical inputs.
  SolveResult Solve(const StatusCallback& on_status = {});

  /// H(s); during `Solve` (inside a status callback) the Hamiltonian that line
  /// refers to.
  const op::Operator& hamiltonian() const noexcept;
  /// Number of flowing operators (without H).
  std::size_t num_operators() const noexcept { return state_.size() - 1; }
  /// The flowing operator O_i(s), in the order given, i < num_operators().
  const op::Operator& flowing_operator(std::size_t i) const {
    return state_.at(i + 1);
  }
  /// eta of the last right-hand-side evaluation / step.
  const op::Operator& eta() const noexcept { return eta_; }
  /// Current flow parameter.
  double s() const noexcept { return s_; }
  /// Current step options.
  const SolverOptions& solver_options() const noexcept { return solver_; }
  /// The commutator workspace.
  commutator::Workspace& workspace() noexcept { return *ws_; }

 private:
  using State = std::vector<op::Operator>;

  /// dxdt_i = [eta(x_0), x_i]; sets eta, reports a status line, flags
  /// divergence. dxdt_i = 0 if ||eta|| < eta_criterion (imsrg++ operator()).
  void Rhs(const State& x, State& dxdt, double t);
  FlowStatus MakeStatus(int istep, double ds, double s, const op::Operator& h,
                        bool zero_eta) const;
  SolveResult SolveRk4();
  SolveResult SolveDopri5();

  SolverOptions solver_;
  GeneratorOptions generator_;
  commutator::CommutatorOptions comm_;
  std::unique_ptr<commutator::Workspace> ws_;
  OdeMethod method_;
  double eps_ = 1.0e-6;
  std::vector<op::Operator> state_;  // H first, then the operators
  op::Operator eta_;
  const op::Operator* eval_h_ = nullptr;  // H of the line being reported
  double s_ = 0.0;
  int stage_ = 0;
  double t_start_ = 0.0;
  bool diverged_ = false;
  StatusCallback on_status_;
};

}  // namespace imsrg::flow

#endif  // IMSRG_FLOW_ODE_H_
