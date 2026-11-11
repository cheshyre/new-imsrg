// The Magnus IMSRG solver (design/13 section 7, imsrg++ IMSRGSolver.cc:20-122 and 280-366).
#include "imsrg/flow/magnus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/log.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/norms.hpp"

namespace imsrg::flow {

namespace {

double now_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// A zero Omega: scalar, anti-Hermitian, two-body (imsrg++ copies Eta and erases it).
op::Operator zero_omega(const std::shared_ptr<const basis::ModelSpace>& ms) {
  return op::Operator(ms, op::OperatorSpec::scalar(2, Hermiticity::AntiHermitian));
}

}  // namespace

MagnusSolver::MagnusSolver(op::Operator H0, const SolverOptions& solver,
                           const GeneratorOptions& generator, const BchOptions& bch,
                           const commutator::CommutatorOptions& comm)
    : solver_(solver),
      generator_(generator),
      bch_(bch),
      comm_(comm),
      h0_(std::move(H0)),
      t_start_(now_seconds()) {
  IMSRG_CHECK(!h0_.empty(), "MagnusSolver: empty Hamiltonian");
  IMSRG_CHECK(h0_.spec().is_scalar() && h0_.spec().is_number_conserving() &&
                  h0_.spec().particle_rank == 2,
              "MagnusSolver: H0 must be a scalar, number-conserving two-body operator");
  ws_ = std::make_unique<commutator::Workspace>(h0_.model_space_ptr());
  h_ = h0_;
  eta_ = op::Operator(h0_.model_space_ptr(), generator_spec(h0_.spec(), generator_.kind));
  omegas_.push_back(zero_omega(h0_.model_space_ptr()));
}

MagnusSolver::~MagnusSolver() = default;
MagnusSolver::MagnusSolver(MagnusSolver&&) noexcept = default;
MagnusSolver& MagnusSolver::operator=(MagnusSolver&&) noexcept = default;

void MagnusSolver::push_omega() {
  // imsrg++ NewOmega (IMSRGSolver.cc:43-82), Omega kept in memory: H_saved = H(s), new zero Omega.
  h_saved_ = h_;
  log::info("pushing back another Omega. Omega.size = {}", omegas_.size());
  omegas_.push_back(zero_omega(h0_.model_space_ptr()));
}

void MagnusSolver::set_generator(const GeneratorOptions& generator) {
  // imsrg++ SetGenerator (IMSRGSolver.cc:157-178): a new Omega if the current one is not ~0.
  generator_ = generator;
  if (op::norm(omegas_.back()) > 1e-6) {
    eta_.set_zero();
    push_omega();
  }
}

FlowStatus MagnusSolver::status(int istep, double ds) const {
  FlowStatus st;
  st.stage = stage_;
  st.istep = istep;
  st.s = s_;
  st.ds = ds;
  st.e0 = h_.zero_body();
  st.norm_h = op::norm(h_);
  st.norm_eta_1b = op::one_body_norm(eta_);
  st.norm_eta_2b = op::two_body_norm(eta_);
  st.norm_eta = op::norm(eta_);
  st.norm_omega_1b = op::one_body_norm(omegas_.back());
  st.norm_omega_2b = op::two_body_norm(omegas_.back());
  st.norm_omega = op::norm(omegas_.back());
  st.num_omegas = omegas_.size();
  st.wall_seconds = now_seconds() - t_start_;
  return st;
}

SolveResult MagnusSolver::solve(const StatusCallback& on_status) {
  ++stage_;
  compute_generator(h_, eta_, generator_);
  // imsrg++ bumps domega to avoid machine-dependent splitting (IMSRGSolver.cc:290); the bump
  // persists, as the member is modified there.
  if (std::abs(solver_.omega_norm_max - solver_.domega) < 1e-6) {
    solver_.domega += 1e-4;
    log::info("adjusting domega to {} to avoid numerical trouble, since omega_norm_max = {}",
              solver_.domega, solver_.omega_norm_max);
  }
  // imsrg++ SetEtaCriterion takes a float (quirk P7).
  const double eta_criterion = static_cast<double>(static_cast<float>(solver_.eta_criterion));
  double e_last = h0_.zero_body();
  double ds = solver_.ds_0;  // used only if magnus_adaptive is off (imsrg++ member ds)
  if (on_status) on_status(status(0, 0.0));

  SolveResult result;
  result.reason = StopReason::SmaxReached;
  for (int istep = 1; s_ < solver_.smax; ++istep) {
    const double norm_eta = op::norm(eta_);
    if (norm_eta < eta_criterion) {
      result.reason = StopReason::Converged;
      break;
    }
    if (norm_eta > 1e12 || std::abs(e_last) > 1e9) {
      log::error("the IMSRG flow diverged: ||eta|| = {}, E0 = {} at s = {}", norm_eta, e_last, s_);
      result.reason = StopReason::Diverged;
      break;
    }
    double norm_omega = op::norm(omegas_.back());
    if (norm_omega > solver_.omega_norm_max) {
      push_omega();
      norm_omega = 0.0;
    }
    if (solver_.magnus_adaptive) {
      ds = std::min({solver_.domega / norm_eta, solver_.domega / norm_eta / (norm_omega + 1.0e-9),
                     (solver_.omega_norm_max + 1e-5) / norm_eta, solver_.dsmax});
    }
    ds = std::min(ds, solver_.smax - s_);
    s_ += ds;
    op::scale(eta_, ds);
    // exp(Omega_new) = exp(eta ds) exp(Omega_old).
    omegas_.back() = bch_product(eta_, omegas_.back(), bch_, comm_, *ws_);
    // The first Omega acts on H_0, later ones on H at the time they were opened.
    const op::Operator& base = omegas_.size() < 2 ? h0_ : h_saved_;
    h_ = bch_transform(base, omegas_.back(), bch_, comm_, *ws_);
    compute_generator(h_, eta_, generator_);
    if (on_status) on_status(status(istep, ds));
    result.steps = istep;
    e_last = h_.zero_body();
  }
  result.s = s_;
  result.e0 = h_.zero_body();
  result.norm_eta = op::norm(eta_);
  return result;
}

op::Operator MagnusSolver::transform(const op::Operator& O) {
  // imsrg++ Transform (IMSRGSolver.cc:937-1000), in-memory stack, chronological order.
  op::Operator out = with_particle_rank(O, 2);
  for (const op::Operator& omega : omegas_) {
    out = bch_transform(out, omega, bch_, comm_, *ws_);
  }
  return out;
}

}  // namespace imsrg::flow
