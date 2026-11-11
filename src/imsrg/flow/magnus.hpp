#pragma once
/// @file
/// The Magnus IMSRG solver (design/13 section 7, design/reference/03 section 2.1, imsrg++
/// `IMSRGSolver::Solve_magnus_euler`, IMSRGSolver.cc:280-366): Euler steps on Omega through the
/// BCH product, H(s) = e^Omega H_saved e^-Omega, an Omega stack split at `omega_norm_max`, and the
/// imsrg++ step-size rule. The step rule, the split criterion and the start condition are
/// numerics-defining (design/00 D12) and reproduce imsrg++ exactly.
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "imsrg/commutator/options.hpp"
#include "imsrg/commutator/workspace.hpp"
#include "imsrg/flow/bch.hpp"
#include "imsrg/flow/generator.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::flow {

/// Step control of the Magnus solver. Defaults are those of the imsrg++ executable
/// (`Parameters.hh`), not of the `IMSRGSolver` constructor.
struct SolverOptions {
  double smax = 200.0;            ///< Stop when s >= smax.
  double dsmax = 0.5;             ///< Upper bound of ds.
  double ds_0 = 0.5;              ///< Initial ds; unused by Magnus (imsrg++ quirk P3).
  double domega = 0.1;            ///< Bound of ||eta ds|| (and of ||eta ds|| ||Omega||).
  double omega_norm_max = 0.25;   ///< Split Omega when ||Omega|| exceeds this.
  double eta_criterion = 1e-6;    ///< Converged when ||eta|| < this (compared as a float, P7).
  bool magnus_adaptive = true;    ///< Recompute ds every step (imsrg++ `magnus_adaptive`).
};

/// One line of the flow log: the state after the step (istep 0 = the initial state of a stage).
struct FlowStatus {
  int stage = 1;                ///< 1-based number of the `solve` call.
  int istep = 0;                ///< Step within the stage (restarts at 0 per stage, as imsrg++).
  double s = 0.0;               ///< Flow parameter (continues across stages).
  double ds = 0.0;              ///< Step just taken (0 for istep 0).
  double e0 = 0.0;              ///< Zero-body part of H(s).
  double norm_h = 0.0;          ///< ||H(s)|| (op::norm, zero-body excluded).
  double norm_eta_1b = 0.0;     ///< One-body norm of eta(s).
  double norm_eta_2b = 0.0;     ///< Two-body norm of eta(s).
  double norm_eta = 0.0;        ///< ||eta(s)||.
  double norm_omega_1b = 0.0;   ///< One-body norm of the current (last) Omega.
  double norm_omega_2b = 0.0;   ///< Two-body norm of the current Omega.
  double norm_omega = 0.0;      ///< ||current Omega||.
  std::size_t num_omegas = 1;   ///< Size of the Omega stack.
  double wall_seconds = 0.0;    ///< Wall time since the solver was constructed.
};

/// Why `solve` stopped.
enum class StopReason : std::uint8_t {
  Converged,    ///< ||eta|| < eta_criterion.
  SmaxReached,  ///< s >= smax.
  Diverged      ///< ||eta|| > 1e12 or |E0| > 1e9 (imsrg++ guard; imsrg++ then fills H with inf).
};

/// Summary of one `solve` call.
struct SolveResult {
  int steps = 0;        ///< Steps taken in this stage.
  double s = 0.0;       ///< Final s.
  double e0 = 0.0;      ///< Final zero-body part of H(s).
  double norm_eta = 0.0;
  StopReason reason = StopReason::SmaxReached;
};

/// Magnus solver state: H_0, H(s), eta(s), the Omega stack and s (imsrg++ `IMSRGSolver` with
/// method magnus, Omega kept in memory, no hunter-gatherer). Owns a commutator `Workspace` for
/// H_0's model space. Not copyable (the workspace is not); movable.
class MagnusSolver {
 public:
  /// Callback receiving every flow-log line (istep 0 and each step).
  using StatusCallback = std::function<void(const FlowStatus&)>;

  /// Sets up the solver for H0 (copied; scalar, Hermitian, particle_rank 2, normal ordered with
  /// respect to the reference of its model space): s = 0, Omega stack = {0}, eta not computed
  /// yet. Builds the commutator workspace (needs an initialized exec::Runtime).
  MagnusSolver(op::Operator H0, const SolverOptions& solver, const GeneratorOptions& generator,
               const BchOptions& bch = {}, const commutator::CommutatorOptions& comm = {});
  ~MagnusSolver();
  MagnusSolver(MagnusSolver&&) noexcept;
  MagnusSolver& operator=(MagnusSolver&&) noexcept;
  MagnusSolver(const MagnusSolver&) = delete;
  MagnusSolver& operator=(const MagnusSolver&) = delete;

  /// Changes the generator for the next `solve` (imsrg++ `SetGenerator`): if ||last Omega|| >
  /// 1e-6 a new (zero) Omega is pushed and H_saved = H(s), so a second stage starts a new Omega.
  void set_generator(const GeneratorOptions& generator);
  /// Changes smax for the next `solve` (imsrg++ doubles it for the second Magnus stage; the
  /// driver does that).
  void set_smax(double smax) noexcept { solver_.smax = smax; }

  /// Runs the Magnus loop until convergence, s >= smax or divergence (imsrg++
  /// `Solve_magnus_euler`):
  ///   eta = generator(H); bump domega by 1e-4 if |omega_norm_max - domega| < 1e-6;
  ///   loop: stop if ||eta|| < eta_criterion (converged) or ||eta|| > 1e12 or |E0_last| > 1e9;
  ///     if ||Omega_last|| > omega_norm_max: push a new zero Omega, H_saved = H, ||Omega|| = 0;
  ///     ds = min(domega/||eta||, domega/||eta||/(||Omega||+1e-9), (omega_norm_max+1e-5)/||eta||,
  ///              dsmax) (if magnus_adaptive); ds = min(ds, smax - s); s += ds;
  ///     Omega_last = bch_product(eta ds, Omega_last);
  ///     H = bch_transform(stack size < 2 ? H_0 : H_saved, Omega_last); eta = generator(H).
  /// `on_status` (optional) receives the istep-0 line and one line per step.
  SolveResult solve(const StatusCallback& on_status = {});

  /// e^Omega_n ... e^Omega_1 O e^-Omega_1 ... e^-Omega_n: replays the Omega stack in
  /// chronological order with `bch_transform` (imsrg++ `IMSRGSolver::Transform`). O must live on
  /// the solver's model space.
  op::Operator transform(const op::Operator& O);

  /// H(s).
  const op::Operator& hamiltonian() const noexcept { return h_; }
  /// eta(s) (zero before the first `solve`).
  const op::Operator& eta() const noexcept { return eta_; }
  /// The Omega stack, oldest first; the last entry is the one being built.
  const std::vector<op::Operator>& omegas() const noexcept { return omegas_; }
  /// Current flow parameter.
  double s() const noexcept { return s_; }
  /// Current step options.
  const SolverOptions& solver_options() const noexcept { return solver_; }
  /// Current generator options.
  const GeneratorOptions& generator_options() const noexcept { return generator_; }
  /// The commutator workspace (shared with callers that transform operators themselves).
  commutator::Workspace& workspace() noexcept { return *ws_; }

 private:
  FlowStatus status(int istep, double ds) const;
  void push_omega();

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
  int stage_ = 0;
  double t_start_ = 0.0;
};

}  // namespace imsrg::flow
