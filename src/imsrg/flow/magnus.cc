// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The Magnus IMSRG solver (design/13 section 7, imsrg++ IMSRGSolver.cc:20-122
// and 280-366).
#include "imsrg/flow/magnus.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "imsrg/base/assert.h"
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
op::Operator ZeroOmega(const std::shared_ptr<const basis::ModelSpace>& ms) {
  return op::Operator(ms,
                      op::OperatorSpec::Scalar(2, Hermiticity::kAntiHermitian));
}

}  // namespace

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
  IMSRG_CHECK(!h0_.empty(), "MagnusSolver: empty Hamiltonian");
  IMSRG_CHECK(
      h0_.spec().is_scalar() && h0_.spec().is_number_conserving() &&
          h0_.spec().particle_rank == 2,
      "MagnusSolver: H0 must be a scalar, number-conserving two-body operator");
  ws_ = std::make_unique<commutator::Workspace>(h0_.model_space_ptr());
  h_ = h0_;
  eta_ = op::Operator(h0_.model_space_ptr(),
                      GeneratorSpec(h0_.spec(), generator_.kind));
  omegas_.push_back(ZeroOmega(h0_.model_space_ptr()));
}

MagnusSolver::~MagnusSolver() = default;
MagnusSolver::MagnusSolver(MagnusSolver&&) noexcept = default;
MagnusSolver& MagnusSolver::operator=(MagnusSolver&&) noexcept = default;

void MagnusSolver::PushOmega() {
  // imsrg++ NewOmega (IMSRGSolver.cc:43-82), Omega kept in memory: H_saved =
  // H(s), new zero Omega.
  h_saved_ = h_;
  log::Info("pushing back another Omega. Omega.size = {}", omegas_.size());
  omegas_.push_back(ZeroOmega(h0_.model_space_ptr()));
}

void MagnusSolver::set_generator(const GeneratorOptions& generator) {
  // imsrg++ SetGenerator (IMSRGSolver.cc:157-178): a new Omega if the current
  // one is not ~0.
  generator_ = generator;
  if (op::Norm(omegas_.back()) > 1.0e-6) {
    eta_.set_zero();
    PushOmega();
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
  st.norm_eta = op::Norm(eta_);
  st.norm_omega_1b = op::OneBodyNorm(omegas_.back());
  st.norm_omega_2b = op::TwoBodyNorm(omegas_.back());
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
  // imsrg++ SetEtaCriterion takes a float (quirk P7).
  const double eta_criterion =
      static_cast<double>(static_cast<float>(solver_.eta_criterion));
  double e_last = h0_.zero_body();
  double ds =
      solver_.ds_0;  // used only if magnus_adaptive is off (imsrg++ member ds)
  if (on_status) on_status(Status(0, 0.0));

  SolveResult result;
  result.reason = StopReason::kSmaxReached;
  for (int istep = 1; s_ < solver_.smax; ++istep) {
    const double norm_eta = op::Norm(eta_);
    if (norm_eta < eta_criterion) {
      result.reason = StopReason::kConverged;
      break;
    }
    if (norm_eta > 1.0e12 || std::abs(e_last) > 1.0e9) {
      log::Error("the IMSRG flow diverged: ||eta|| = {}, E0 = {} at s = {}",
                 norm_eta, e_last, s_);
      result.reason = StopReason::kDiverged;
      break;
    }
    double norm_omega = op::Norm(omegas_.back());
    if (norm_omega > solver_.omega_norm_max) {
      PushOmega();
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
  result.s = s_;
  result.e0 = h_.zero_body();
  result.norm_eta = op::Norm(eta_);
  return result;
}

op::Operator MagnusSolver::Transform(const op::Operator& O) {
  // imsrg++ Transform (IMSRGSolver.cc:937-1000), in-memory stack, chronological
  // order.
  op::Operator out = WithParticleRank(O, 2);
  for (const op::Operator& omega : omegas_) {
    out = BchTransform(out, omega, bch_, comm_, *ws_);
  }
  return out;
}

}  // namespace imsrg::flow
