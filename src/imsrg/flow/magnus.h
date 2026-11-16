// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_MAGNUS_H_
#define IMSRG_FLOW_MAGNUS_H_
/// @file
/// The Magnus IMSRG solver (design/13 section 7, design/reference/03 section
/// 2.1, imsrg++ `IMSRGSolver::Solve_magnus_euler`, IMSRGSolver.cc:280-366):
/// Euler steps on Omega through the BCH product, H(s) = e^Omega H_saved
/// e^-Omega, an Omega stack split at `omega_norm_max`, and the imsrg++
/// step-size rule. The step rule, the split criterion and the start condition
/// are numerics-defining (design/00 D12) and reproduce imsrg++ exactly.
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/flow/bch.h"
#include "imsrg/flow/generator.h"
#include "imsrg/operator/operator.h"

namespace imsrg::flow {

/// Step control of the Magnus solver. Defaults are those of the imsrg++
/// executable (`Parameters.hh`), not of the `IMSRGSolver` constructor.
struct SolverOptions {
  double smax = 200.0;  ///< Stop when s >= smax.
  double dsmax = 0.5;   ///< Upper bound of ds.
  double ds_0 = 0.5;    ///< Initial ds; unused by Magnus (imsrg++ quirk P3).
  double domega = 0.1;  ///< Bound of ||eta ds|| (and of ||eta ds|| ||Omega||).
  double omega_norm_max = 0.25;   ///< Split Omega when ||Omega|| exceeds this.
  double eta_criterion = 1.0e-6;  ///< Converged when ||eta|| < this (a double,
                                  ///< D47; imsrg++: float).
  bool magnus_adaptive =
      true;  ///< Recompute ds every step (imsrg++ `magnus_adaptive`).
  /// Hunter-gatherer Omega (imsrg++ `hunter_gatherer`, D32): keep two Omegas,
  /// the hunter (the one being built) and the gatherer, and gather instead of
  /// splitting when ||hunter|| > omega_norm_max; gather once more at the end
  /// of every `Solve`. See `MagnusSolver::Solve`.
  bool hunter_gatherer = false;
};

/// Checks the step options: dsmax, domega, ds_0 and omega_norm_max positive and
/// finite, smax finite, eta_criterion not NaN.
/// @throws InputError otherwise (ds would be 0 or NaN and the loops endless).
void ValidateSolverOptions(const SolverOptions& options);

/// One line of the flow log: the state after the step (istep 0 = the initial
/// state of a stage).
struct FlowStatus {
  int stage = 1;  ///< 1-based number of the `solve` call.
  int istep =
      0;  ///< Step within the stage (restarts at 0 per stage, as imsrg++).
  double s = 0.0;            ///< Flow parameter (continues across stages).
  double ds = 0.0;           ///< Step just taken (0 for istep 0).
  double e0 = 0.0;           ///< Zero-body part of H(s).
  double norm_h = 0.0;       ///< ||H(s)|| (op::norm, zero-body excluded).
  double norm_eta_1b = 0.0;  ///< One-body norm of eta(s).
  double norm_eta_2b = 0.0;  ///< Two-body norm of eta(s).
  double norm_eta = 0.0;     ///< ||eta(s)||.
  double norm_eta_3b = 0.0;  ///< Three-body norm of eta(s) (IMSRG(3); else 0).
  double norm_omega_1b = 0.0;  ///< One-body norm of the current (last) Omega.
  double norm_omega_2b = 0.0;  ///< Two-body norm of the current Omega.
  double norm_omega_3b = 0.0;  ///< Three-body norm of the current Omega.
  double norm_omega = 0.0;     ///< ||current Omega||.
  std::size_t num_omegas = 1;  ///< Size of the Omega stack.
  double wall_seconds = 0.0;   ///< Wall time since the solver was constructed.
};

/// Why `solve` stopped.
enum class StopReason : std::int8_t {
  kConverged,    ///< ||eta|| < eta_criterion.
  kSmaxReached,  ///< s >= smax.
  kDiverged      ///< ||eta|| > 1e12, |E0| > 1e9 or a NaN/inf norm or energy
                 ///< (imsrg++ guard, made NaN-safe; imsrg++ then fills H with
                 ///< inf, we leave H as it is).
};

/// Summary of one `solve` call.
struct SolveResult {
  int steps = 0;    ///< Steps taken in this stage.
  double s = 0.0;   ///< Final s.
  double e0 = 0.0;  ///< Final zero-body part of H(s).
  double norm_eta = 0.0;
  /// Omega gathers of this call (hunter-gatherer only, else 0), counted as the
  /// `gathering Omega.` lines of imsrg++: the gathers inside the loop plus the
  /// final one.
  int num_gathers = 0;
  StopReason reason = StopReason::kSmaxReached;
};

/// Magnus solver state: H_0, H(s), eta(s), the Omega stack and s (imsrg++
/// `IMSRGSolver` with method magnus, Omega kept in memory, hunter-gatherer
/// optional). Owns a commutator `Workspace` for H_0's model space. Not copyable
/// (the workspace is not); movable.
class MagnusSolver {
 public:
  /// Callback receiving every flow-log line (istep 0 and each step).
  using StatusCallback = std::function<void(const FlowStatus&)>;

  /// Sets up the solver for H0 (copied; scalar, Hermitian, particle_rank 2 or
  /// 3, normal ordered with respect to the reference of its model space). With
  /// `comm.terms` selecting IMSRG(3) terms H0 is promoted to particle_rank 3
  /// (zero three-body part) and eta, every Omega and H(s) have rank 3: the
  /// model space of H0 then needs a three-body space (InputError otherwise).
  /// A rank-3 H0 needs IMSRG(3) terms in `comm` as well (else the commutator
  /// results would not match its rank; InputError): s = 0,
  /// Omega stack = {0}, eta not computed yet. Builds the commutator workspace
  /// (needs an initialized exec::Runtime) with `workspace` (its communicator
  /// is the set of ranks that share the commutator work; the default is
  /// serial; construction is not collective).
  /// @throws InputError if `solver.dsmax`, `domega`, `ds_0` or
  ///   `omega_norm_max` is not a positive finite number, `smax` is not finite
  ///   or `eta_criterion` is NaN (ds would be 0 or NaN and the loop endless).
  MagnusSolver(op::Operator H0, const SolverOptions& solver,
               const GeneratorOptions& generator, const BchOptions& bch = {},
               const commutator::CommutatorOptions& comm = {},
               const commutator::WorkspaceOptions& workspace = {});
  ~MagnusSolver();
  MagnusSolver(MagnusSolver&&) noexcept;
  MagnusSolver& operator=(MagnusSolver&&) noexcept;
  MagnusSolver(const MagnusSolver&) = delete;
  MagnusSolver& operator=(const MagnusSolver&) = delete;

  /// Changes the generator for the next `solve` (imsrg++ `SetGenerator`): if
  /// ||last Omega|| > 1e-6 a new (zero) Omega is pushed and H_saved = H(s), so
  /// a second stage starts a new Omega (hunter-gatherer: the Omega is gathered
  /// instead; after a `Solve` the hunter is zero and nothing happens).
  void set_generator(const GeneratorOptions& generator);
  /// Changes smax for the next `solve` (the driver sets s1 + smax for stage 2,
  /// D47; imsrg++ doubled it for the second Magnus stage only).
  void set_smax(double smax) noexcept { solver_.smax = smax; }
  /// Replaces the step options for the next `solve` (imsrg++.cc:1130-1140
  /// sets Ds, Dsmax and Smax before the second stage). The bump of `domega`
  /// already applied is NOT undone: pass options derived from
  /// `solver_options()`.
  /// @throws InputError for options that would loop forever or give NaN steps
  ///   (see the constructor).
  void set_solver_options(const SolverOptions& solver);

  /// Runs the Magnus loop until convergence, s >= smax or divergence (imsrg++
  /// `Solve_magnus_euler`):
  ///   eta = generator(H);
  ///   bump domega by 1e-4 if |omega_norm_max - domega| < 1e-6;
  ///   loop: stop if ||eta|| < eta_criterion (converged) or ||eta|| > 1e12
  ///         or |E0_last| > 1e9;
  ///     if ||Omega_last|| > omega_norm_max:
  ///       push a new zero Omega, H_saved = H, ||Omega|| = 0
  ///       (hunter-gatherer: GatherOmega, see below);
  ///     ds = min(domega/||eta||, domega/||eta||/(||Omega||+1e-9),
  ///              (omega_norm_max+1e-5)/||eta||, dsmax) (if magnus_adaptive);
  ///     ds = min(ds, smax - s); s += ds;
  ///     Omega_last = BchProduct(eta ds, Omega_last);
  ///     H = BchTransform(stack size < 2 ? H_0 : H_saved, Omega_last);
  ///     eta = generator(H).
  /// Hunter-gatherer (`SolverOptions::hunter_gatherer`, imsrg++ `GatherOmega`,
  /// IMSRGSolver.cc:98-122 and :236-239), reproduced exactly: the stack is
  /// {gatherer, hunter}. GatherOmega with a one-entry stack appends a zero
  /// hunter (the old Omega becomes the gatherer); otherwise
  /// gatherer = BchProduct(hunter, gatherer) if ||hunter|| > 1e-12; then the
  /// hunter is zeroed and H_saved = H_0 transformed by the gatherer. A gather
  /// inside the loop leaves H(s) alone (the flow goes on from it). `Solve`
  /// gathers once more after the loop (also after divergence), counts that in
  /// `SolveResult::num_gathers` and, unless the flow diverged, sets
  /// H(s) = H_saved = e^Omega H_0 e^-Omega with the gathered Omega (D48: one
  /// Omega for E0 and every transformed operator, and stage 2 starts from it;
  /// imsrg++ leaves H(s) at the last hunter step, which differs from this by
  /// up to tens of keV in E0). `result.e0` and `hamiltonian()` are the
  /// rebuilt ones, so `Transform(H_0)` reproduces H(s) at round-off level.
  /// eta(s) is recomputed from the rebuilt H (`eta()`, `result.norm_eta`), a
  /// warning is logged if its norm exceeds `eta_criterion` (it usually does,
  /// by 1e3: the flow does not continue, `result.reason` stays that of the
  /// loop), and one more status line (istep = steps + 1, ds = 0, same s)
  /// reports the rebuilt H, so the flow log ends with the reported E0.
  /// `on_status` (optional) receives the istep-0 line and one line per step.
  ///
  /// Collective over the Workspace communicator: every rank calls it with
  /// identical inputs.
  SolveResult Solve(const StatusCallback& on_status = {});

  /// e^Omega_n ... e^Omega_1 O e^-Omega_1 ... e^-Omega_n: replays the Omega
  /// stack in chronological order with `BchTransform` (imsrg++
  /// `IMSRGSolver::Transform`). O must live on the solver's model space.
  /// Collective over the Workspace communicator: every rank calls it with
  /// identical inputs.
  op::Operator Transform(const op::Operator& O);
  /// `Transform` with the BCH options `bch` instead of the solver's own
  /// (collective like `Transform`; e.g.
  /// `skip_first_commutator` for the perturbative triples); the commutator
  /// options and the workspace are the solver's.
  op::Operator Transform(const op::Operator& O, const BchOptions& bch);

  /// H_0, the Hamiltonian the solver was constructed with.
  const op::Operator& initial_hamiltonian() const noexcept { return h0_; }
  /// The BCH options of the flow.
  const BchOptions& bch_options() const noexcept { return bch_; }
  /// H(s).
  const op::Operator& hamiltonian() const noexcept { return h_; }
  /// eta(s) (zero before the first `solve`).
  const op::Operator& eta() const noexcept { return eta_; }
  /// The Omega stack, oldest first; the last entry is the one being built.
  const std::vector<op::Operator>& omegas() const noexcept { return omegas_; }
  /// Number of Omega gathers over the solver's lifetime (hunter-gatherer).
  int num_gathers() const noexcept { return num_gathers_; }
  /// Current flow parameter.
  double s() const noexcept { return s_; }
  /// Current step options.
  const SolverOptions& solver_options() const noexcept { return solver_; }
  /// Current generator options.
  const GeneratorOptions& generator_options() const noexcept {
    return generator_;
  }
  /// The commutator workspace (shared with callers that transform operators
  /// themselves).
  commutator::Workspace& workspace() noexcept { return *ws_; }

 private:
  FlowStatus Status(int istep, double ds) const;
  void PushOmega();
  /// imsrg++ `GatherOmega`; see `Solve`.
  void GatherOmega();
  /// A split (`PushOmega`) or, in hunter-gatherer mode, a gather.
  void StartNewPiece();

  SolverOptions solver_;
  GeneratorOptions generator_;
  BchOptions bch_;
  commutator::CommutatorOptions comm_;
  std::unique_ptr<commutator::Workspace> ws_;
  op::Operator h0_;
  op::Operator h_saved_;
  op::Operator h_;
  op::Operator eta_;
  std::vector<op::Operator> omegas_;
  double s_ = 0.0;
  int num_gathers_ = 0;
  int stage_ = 0;
  double t_start_ = 0.0;
};

}  // namespace imsrg::flow

#endif  // IMSRG_FLOW_MAGNUS_H_
