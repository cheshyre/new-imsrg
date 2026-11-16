// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The Magnus IMSRG solver (design/13 section 7, imsrg++ IMSRGSolver.cc:20-122
// and 280-366).
#include "imsrg/flow/magnus.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/log.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"

namespace imsrg::flow {

namespace {

double NowSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// A zero Omega: scalar, anti-Hermitian, two-body (imsrg++ copies Eta and
/// erases it).
op::Operator ZeroOmega(const std::shared_ptr<const basis::ModelSpace>& ms,
                       std::int32_t particle_rank) {
  return op::Operator(
      ms, op::OperatorSpec::Scalar(particle_rank, Hermiticity::kAntiHermitian));
}

// Recomputes eta(s) for the H rebuilt with the gathered Omega and warns when
// the flow stopped on the eta criterion of the last hunter step but the
// rebuilt ||eta|| exceeds it. The flow does NOT continue (steps and gathers
// stay as decided in D48); ||eta|| of the rebuilt H is usually far above
// eta_criterion (1e-3 against 1e-6 in the IT-2 hunter-gatherer cases).
void RebuiltEtaCheck(const op::Operator& h, op::Operator& eta,
                     const GeneratorOptions& generator, double eta_criterion,
                     StopReason reason) {
  // eta(s) and the reported norms belong to the rebuilt H as well.
  ComputeGenerator(h, eta, generator);
  const double norm_eta_rebuilt = op::Norm(eta);
  if (reason == StopReason::kConverged && !(norm_eta_rebuilt < eta_criterion)) {
    log::Warn(
        "hunter-gatherer: ||eta|| = {} of the H rebuilt with the gathered "
        "Omega exceeds eta_criterion = {} (the flow stopped on the "
        "||eta|| of the last hunter step)",
        norm_eta_rebuilt, eta_criterion);
  }
}

// The last step may have produced a NaN that the loop condition never saw:
// flags the result as diverged when ||eta|| or E0 is not finite.
void MarkNonFiniteDiverged(SolveResult& result, double s) {
  if (result.reason != StopReason::kDiverged &&
      (!std::isfinite(result.norm_eta) || !std::isfinite(result.e0))) {
    log::Error("the IMSRG flow diverged: non-finite ||eta|| or E0 at s = {}",
               s);
    result.reason = StopReason::kDiverged;
  }
}

}  // namespace

void ValidateSolverOptions(const SolverOptions& o) {
  auto positive = [](double x) { return std::isfinite(x) && x > 0.0; };
  if (!positive(o.dsmax))
    throw InputError("SolverOptions: dsmax must be positive and finite");
  if (!positive(o.domega))
    throw InputError("SolverOptions: domega must be positive and finite");
  if (!positive(o.ds_0))
    throw InputError("SolverOptions: ds_0 must be positive and finite");
  if (!positive(o.omega_norm_max))
    throw InputError(
        "SolverOptions: omega_norm_max must be positive and finite");
  if (!std::isfinite(o.smax))
    throw InputError("SolverOptions: smax must be finite");
  if (std::isnan(o.eta_criterion))
    throw InputError("SolverOptions: eta_criterion must not be NaN");
}

void MagnusSolver::set_solver_options(const SolverOptions& solver) {
  ValidateSolverOptions(solver);
  solver_ = solver;
}

MagnusSolver::MagnusSolver(op::Operator H0, const SolverOptions& solver,
                           const GeneratorOptions& generator,
                           const BchOptions& bch,
                           const commutator::CommutatorOptions& comm)
    : solver_(solver),
      generator_(generator),
      bch_(bch),
      comm_(comm),
      h0_(std::move(H0)),
      t_start_(NowSeconds()) {
  ValidateSolverOptions(solver_);
  IMSRG_CHECK(!h0_.empty(), "MagnusSolver: empty Hamiltonian");
  IMSRG_CHECK(
      h0_.spec().is_scalar() && h0_.spec().is_number_conserving() &&
          (h0_.spec().particle_rank == 2 || h0_.spec().particle_rank == 3),
      "MagnusSolver: H0 must be a scalar, number-conserving operator of "
      "particle rank 2 or 3");
  const bool imsrg3 = !comm_.terms.ThreeBodyTerms().empty();
  if (imsrg3 && h0_.spec().particle_rank < 3) {
    if (h0_.model_space().three_body_space() == nullptr) {
      throw InputError(
          "MagnusSolver: IMSRG(3) terms need a model space with a three-body "
          "space (ModelSpace::WithThreeBody)");
    }
    h0_ = WithParticleRank(h0_, 3);
  }
  if (!imsrg3 && h0_.spec().particle_rank == 3) {
    throw InputError(
        "MagnusSolver: a particle-rank-3 H0 needs IMSRG(3) commutator terms");
  }
  const std::int32_t rank = h0_.spec().particle_rank;
  ws_ = std::make_unique<commutator::Workspace>(h0_.model_space_ptr());
  h_ = h0_;
  eta_ = op::Operator(h0_.model_space_ptr(),
                      GeneratorSpec(h0_.spec(), generator_.kind));
  omegas_.push_back(ZeroOmega(h0_.model_space_ptr(), rank));
}

MagnusSolver::~MagnusSolver() = default;
MagnusSolver::MagnusSolver(MagnusSolver&&) noexcept = default;
MagnusSolver& MagnusSolver::operator=(MagnusSolver&&) noexcept = default;

void MagnusSolver::PushOmega() {
  // imsrg++ NewOmega (IMSRGSolver.cc:43-82), Omega kept in memory: H_saved =
  // H(s), new zero Omega.
  h_saved_ = h_;
  log::Info("pushing back another Omega. Omega.size = {}", omegas_.size());
  omegas_.push_back(ZeroOmega(h0_.model_space_ptr(), h0_.spec().particle_rank));
}

void MagnusSolver::GatherOmega() {
  // imsrg++ GatherOmega (IMSRGSolver.cc:98-122). With a one-entry stack the
  // single Omega becomes the gatherer and the hunter is a fresh zero Omega
  // (imsrg++ copies and erases the last one; the hunter is zero either way).
  log::Info("gathering Omega.");
  ++num_gathers_;
  if (omegas_.size() < 2) {
    omegas_.push_back(
        ZeroOmega(h0_.model_space_ptr(), h0_.spec().particle_rank));
  }
  op::Operator& hunter = omegas_.back();
  op::Operator& gatherer = omegas_[omegas_.size() - 2];
  if (op::Norm(hunter) > 1.0e-12) {
    gatherer = BchProduct(hunter, gatherer, bch_, comm_, *ws_);
  }
  hunter.set_zero();
  h_saved_ = h0_;
  for (std::size_t i = 0; i + 1 < omegas_.size(); ++i) {
    h_saved_ = BchTransform(h_saved_, omegas_[i], bch_, comm_, *ws_);
  }
}

void MagnusSolver::StartNewPiece() {
  if (solver_.hunter_gatherer) {
    GatherOmega();
  } else {
    PushOmega();
  }
}

void MagnusSolver::set_generator(const GeneratorOptions& generator) {
  // imsrg++ SetGenerator (IMSRGSolver.cc:157-178): a new Omega if the current
  // one is not ~0.
  generator_ = generator;
  if (op::Norm(omegas_.back()) > 1.0e-6) {
    eta_.set_zero();
    StartNewPiece();
  }
}

FlowStatus MagnusSolver::Status(int istep, double ds) const {
  FlowStatus st;
  st.stage = stage_;
  st.istep = istep;
  st.s = s_;
  st.ds = ds;
  st.e0 = h_.zero_body();
  st.norm_h = op::Norm(h_);
  st.norm_eta_1b = op::OneBodyNorm(eta_);
  st.norm_eta_2b = op::TwoBodyNorm(eta_);
  st.norm_eta_3b = op::ThreeBodyNorm(eta_);
  st.norm_eta = op::Norm(eta_);
  st.norm_omega_1b = op::OneBodyNorm(omegas_.back());
  st.norm_omega_2b = op::TwoBodyNorm(omegas_.back());
  st.norm_omega_3b = op::ThreeBodyNorm(omegas_.back());
  st.norm_omega = op::Norm(omegas_.back());
  st.num_omegas = omegas_.size();
  st.wall_seconds = NowSeconds() - t_start_;
  return st;
}

SolveResult MagnusSolver::Solve(const StatusCallback& on_status) {
  ++stage_;
  ComputeGenerator(h_, eta_, generator_);
  // imsrg++ bumps domega to avoid machine-dependent splitting
  // (IMSRGSolver.cc:290); the bump persists, as the member is modified there.
  if (std::abs(solver_.omega_norm_max - solver_.domega) < 1.0e-6) {
    solver_.domega += 1.0e-4;
    log::Info(
        "adjusting domega to {} to avoid numerical trouble, since "
        "omega_norm_max = {}",
        solver_.domega, solver_.omega_norm_max);
  }
  // D47: a double (imsrg++ rounds it through float, quirk P7).
  const double eta_criterion = solver_.eta_criterion;
  double e_last = h0_.zero_body();
  // Used only if magnus_adaptive is off: imsrg++'s member ds is set to ds_0
  // and capped by SetDsmax (IMSRGSolver.cc:201-205).
  double ds = std::min(solver_.ds_0, solver_.dsmax);
  if (on_status) on_status(Status(0, 0.0));

  int loop_gathers = 0;
  SolveResult result;
  result.reason = StopReason::kSmaxReached;
  for (int istep = 1; s_ < solver_.smax; ++istep) {
    const double norm_eta = op::Norm(eta_);
    if (norm_eta < eta_criterion) {
      result.reason = StopReason::kConverged;
      break;
    }
    // NaN-safe: a NaN norm or energy fails the "<=" and counts as divergence.
    if (!(norm_eta <= 1.0e12) || !(std::abs(e_last) <= 1.0e9)) {
      log::Error("the IMSRG flow diverged: ||eta|| = {}, E0 = {} at s = {}",
                 norm_eta, e_last, s_);
      result.reason = StopReason::kDiverged;
      break;
    }
    double norm_omega = op::Norm(omegas_.back());
    if (norm_omega > solver_.omega_norm_max) {
      StartNewPiece();
      if (solver_.hunter_gatherer) ++loop_gathers;
      norm_omega = 0.0;
    }
    if (solver_.magnus_adaptive) {
      ds = std::min({solver_.domega / norm_eta,
                     solver_.domega / norm_eta / (norm_omega + 1.0e-9),
                     (solver_.omega_norm_max + 1.0e-5) / norm_eta,
                     solver_.dsmax});
    }
    ds = std::min(ds, solver_.smax - s_);
    s_ += ds;
    op::Scale(eta_, ds);
    // exp(Omega_new) = exp(eta ds) exp(Omega_old).
    omegas_.back() = BchProduct(eta_, omegas_.back(), bch_, comm_, *ws_);
    // The first Omega acts on H_0, later ones on H at the time they were
    // opened.
    const op::Operator& base = omegas_.size() < 2 ? h0_ : h_saved_;
    h_ = BchTransform(base, omegas_.back(), bch_, comm_, *ws_);
    ComputeGenerator(h_, eta_, generator_);
    if (on_status) on_status(Status(istep, ds));
    result.steps = istep;
    e_last = h_.zero_body();
  }
  // imsrg++ Solve() gathers once more at the end (IMSRGSolver.cc:236-239).
  if (solver_.hunter_gatherer) {
    GatherOmega();
    result.num_gathers = loop_gathers + 1;
    // D48 (owner, M8-Q10): H(s) becomes the H rebuilt with the gathered Omega
    // (h_saved_ = e^Omega H_0 e^-Omega, already computed by the gather), so E0
    // and every transformed operator come from one Omega and a second stage
    // starts from it. imsrg++ leaves H(s) at the last hunter step, which
    // differs from this by up to tens of keV in E0 (design/30 section 6). Not
    // after a divergence (the rebuilt H could be non-finite as well).
    if (result.reason != StopReason::kDiverged) {
      h_ = h_saved_;
      RebuiltEtaCheck(h_, eta_, generator_, eta_criterion, result.reason);
      // Last flow-log row: the rebuilt H at the same s (istep = steps + 1,
      // ds = 0), so the log ends with the reported E0.
      if (on_status) on_status(Status(result.steps + 1, 0.0));
    }
  }
  result.s = s_;
  result.e0 = h_.zero_body();
  result.norm_eta = op::Norm(eta_);
  MarkNonFiniteDiverged(result, s_);
  return result;
}

op::Operator MagnusSolver::Transform(const op::Operator& O) {
  return Transform(O, bch_);
}

op::Operator MagnusSolver::Transform(const op::Operator& O,
                                     const BchOptions& bch) {
  // imsrg++ Transform (IMSRGSolver.cc:937-1000), in-memory stack, chronological
  // order.
  op::Operator out = WithParticleRank(O, 2);
  for (const op::Operator& omega : omegas_) {
    out = BchTransform(out, omega, bch, comm_, *ws_);
  }
  return out;
}

}  // namespace imsrg::flow
