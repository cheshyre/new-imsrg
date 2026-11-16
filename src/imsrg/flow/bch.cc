// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/flow/bch.h"

#include <array>
#include <cmath>
#include <optional>
#include <utility>

#include "imsrg/base/log.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/operator/norms.h"

namespace imsrg::flow {

namespace {

/// The IMSRG(3f2) correction F(eta, gamma) for one eta (the Omega of a
/// transform, the Y of a product) and the gammas of a series: eta's
/// intermediates (`commutator::FactorizedEta`) are computed at the first
/// evaluation and reused.
class FactorizedTerm {
 public:
  FactorizedTerm(const op::Operator& eta,
                 const commutator::FactorizedOptions& opts,
                 commutator::Workspace& ws)
      : eta_(eta), opts_(opts), ws_(ws) {}
  FactorizedTerm(const FactorizedTerm&) = delete;
  FactorizedTerm& operator=(const FactorizedTerm&) = delete;
  FactorizedTerm(FactorizedTerm&&) = delete;
  FactorizedTerm& operator=(FactorizedTerm&&) = delete;
  ~FactorizedTerm() = default;

  /// corr = F(eta, gamma); returns false, leaving corr untouched, if F
  /// vanishes identically because eta or gamma has no two-body part or no
  /// topology is selected (so a series of rank-1 operators stays rank 1).
  bool Evaluate(const op::Operator& gamma, op::Operator& corr) {
    if (!eta_.two_body().allocated() || !gamma.two_body().allocated())
      return false;
    if (!opts_.one_body_intermediates && !opts_.two_body_intermediates)
      return false;
    if (!pre_) pre_.emplace(eta_, ws_);
    commutator::FactorizedDoubleCommutator(*pre_, gamma, corr, opts_, ws_);
    return true;
  }

 private:
  const op::Operator& eta_;
  const commutator::FactorizedOptions& opts_;
  commutator::Workspace& ws_;
  std::optional<commutator::FactorizedEta> pre_;
};

/// Particle rank of the commutators selected by `copts` on `ws`: 3 if they
/// include an IMSRG(3) term and the workspace has a three-body context.
std::int32_t CommutatorRank(const commutator::CommutatorOptions& copts,
                            const commutator::Workspace& ws) {
  return (!copts.terms.ThreeBodyTerms().empty() &&
          ws.three_body_context() != nullptr)
             ? 3
             : 2;
}

}  // namespace

op::Operator WithParticleRank(const op::Operator& op,
                              std::int32_t particle_rank) {
  if (op.spec().particle_rank >= particle_rank) return op;
  op::OperatorSpec spec = op.spec();
  spec.particle_rank = particle_rank;
  op::Operator out(op.model_space_ptr(), spec);
  op::Axpy(1.0, op, out);  // zero-, one- and (if present) two-body parts
  return out;
}

op::Operator BchTransform(const op::Operator& X, const op::Operator& Omega,
                          const BchOptions& opts,
                          const commutator::CommutatorOptions& copts,
                          commutator::Workspace& ws, BchReport* report) {
  if (report != nullptr) *report = BchReport{};
  const double nx = op::Norm(X);
  const double ny = op::Norm(Omega);
  op::Operator out = WithParticleRank(X, CommutatorRank(copts, ws));
  if (!(nx > opts.transform_threshold)) return out;

  // Two alternating buffers hold the nested commutators (no allocation after
  // the first one); a third holds the factorized correction (IMSRG(3f2)).
  op::Operator nested = out;
  op::Operator next;
  op::Operator corr;
  double factorial_denom = 1.0;
  // ny == 0 gives +inf under IEEE rules, as in imsrg++ (one commutator, then
  // stop).
  double epsilon = nx * std::exp(-2 * ny) * opts.transform_threshold / (2 * ny);
  int count = 0;
  int corrections = 0;
  bool converged = false;
  FactorizedTerm f_omega(Omega, opts.factorized_terms, ws);
  // With skip_first_commutator the loop index i runs one ahead of the nested
  // commutator index j = i - 1 (imsrg++ BCH.cc:163): the i = 1 pass does
  // nothing at all (no factorial division, no epsilon update), so the term
  // added at i is nested_{i-1} / i!.
  const int first = opts.skip_first_commutator ? 2 : 1;
  for (int i = first; i <= opts.max_iterations; ++i) {
    // IMSRG(3f2): nested_j += F(Omega, nested_{j-2}) for j >= 2, i.e. for
    // i >= first + 1. `next` holds the (corrected) nested_{j-2}, nested_0 =
    // out at j = 2, until the commutator below overwrites it.
    const bool factorized =
        opts.factorized && i >= first + 1 && f_omega.Evaluate(next, corr);
    commutator::Commutator(Omega, nested, next, copts, ws);
    std::swap(nested, next);
    if (factorized) {
      op::Axpy(1.0, corr, nested);
      ++corrections;
    }
    ++count;
    factorial_denom /= i;
    op::Axpy(factorial_denom, nested, out);
    epsilon *= i + 1;
    const double norm_nested = op::Norm(nested);
    if (!std::isfinite(norm_nested)) {  // NaN/inf: stop, no 40 commutators
      log::Warn("BCH transform: non-finite nested commutator, stopping");
      break;
    }
    if (norm_nested < epsilon) {
      converged = true;
      break;
    }
    if (i == opts.warn_iterations)
      log::Warn("BCH transform not converged after {} nested commutators",
                opts.warn_iterations);
    else if (i == opts.max_iterations)
      log::Warn("BCH transform did not converge after {} nested commutators",
                opts.max_iterations);
  }
  if (report != nullptr) {
    report->nested_commutators = count;
    report->converged = converged;
    report->factorized_corrections = corrections;
  }
  return out;
}

op::Operator BchProduct(const op::Operator& X, const op::Operator& Y,
                        const BchOptions& opts,
                        const commutator::CommutatorOptions& copts,
                        commutator::Workspace& ws) {
  constexpr std::array<double, 9> bernoulli = {
      1.0, -0.5, 1.0 / 6, 0.0, -1.0 / 30, 0.0, 1.0 / 42, 0.0, -1.0 / 30};
  constexpr std::array<double, 9> factorial = {
      1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0};
  const double thr = opts.product_threshold;
  const double nx = op::Norm(X);
  const double ny = op::Norm(Y);

  op::Operator Z = X + Y;
  if (CommutatorRank(copts, ws) == 3) Z = WithParticleRank(Z, 3);
  op::Operator nested = commutator::Commutator(Y, X, copts, ws);  // [Y, X]
  op::Operator next;
  op::Operator corr;  // factorized correction (IMSRG(3f2))
  double nxy = op::Norm(nested);
  if (nxy * nx > thr) {
    commutator::Commutator(nested, X, next, copts, ws);
    op::Axpy(1.0 / 12, next, Z);
    if (opts.factorized) {  // imsrg++ BCH.cc:312-319: eta = X, gamma = Y
      FactorizedTerm f_x(X, opts.factorized_terms, ws);
      if (f_x.Evaluate(Y, corr)) op::Axpy(1.0 / 12, corr, Z);
    }
  }
  FactorizedTerm f_y(Y, opts.factorized_terms, ws);
  std::size_t k = 1;
  while (nxy > thr) {
    if (k < 2 || k % 2 == 0) op::Axpy(bernoulli[k] / factorial[k], nested, Z);
    ++k;
    if (k >= bernoulli.size()) break;
    if (2 * ny * nxy < thr) break;
    // IMSRG(3f2): N_k += F(Y, N_{k-2}) with N_0 = X (imsrg++ BCH.cc:321-359);
    // for k >= 3 `next` holds N_{k-2} until the commutator overwrites it.
    const bool factorized =
        opts.factorized && f_y.Evaluate(k == 2 ? X : next, corr);
    commutator::Commutator(Y, nested, next, copts, ws);
    std::swap(nested, next);
    if (factorized) op::Axpy(1.0, corr, nested);
    nxy = op::Norm(nested);
  }
  return Z;
}

}  // namespace imsrg::flow
