#include "imsrg/flow/bch.hpp"

#include <array>
#include <cmath>
#include <utility>

#include "imsrg/base/log.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/operator/norms.hpp"

namespace imsrg::flow {

op::Operator with_particle_rank(const op::Operator& op, std::int32_t particle_rank) {
  if (op.spec().particle_rank >= particle_rank) return op;
  op::OperatorSpec spec = op.spec();
  spec.particle_rank = particle_rank;
  op::Operator out(op.model_space_ptr(), spec);
  op::axpy(1.0, op, out);  // zero-, one- and (if present) two-body parts
  return out;
}

op::Operator bch_transform(const op::Operator& X, const op::Operator& Omega, const BchOptions& opts,
                           const commutator::CommutatorOptions& copts, commutator::Workspace& ws,
                           BchReport* report) {
  if (report != nullptr) *report = BchReport{};
  const double nx = op::norm(X);
  const double ny = op::norm(Omega);
  op::Operator out = with_particle_rank(X, 2);
  if (!(nx > opts.transform_threshold)) return out;

  // Two alternating buffers hold the nested commutators (no allocation after the first one).
  op::Operator nested = out;
  op::Operator next;
  double factorial_denom = 1.0;
  // ny == 0 gives +inf under IEEE rules, as in imsrg++ (one commutator, then stop).
  double epsilon = nx * std::exp(-2 * ny) * opts.transform_threshold / (2 * ny);
  int count = 0;
  bool converged = false;
  for (int i = 1; i <= opts.max_iterations; ++i) {
    commutator::commutator(Omega, nested, next, copts, ws);
    std::swap(nested, next);
    ++count;
    factorial_denom /= i;
    op::axpy(factorial_denom, nested, out);
    epsilon *= i + 1;
    if (op::norm(nested) < epsilon) {
      converged = true;
      break;
    }
    if (i == opts.warn_iterations)
      log::warn("BCH transform not converged after {} nested commutators", opts.warn_iterations);
    else if (i == opts.max_iterations)
      log::warn("BCH transform did not converge after {} nested commutators", opts.max_iterations);
  }
  if (report != nullptr) {
    report->nested_commutators = count;
    report->converged = converged;
  }
  return out;
}

op::Operator bch_product(const op::Operator& X, const op::Operator& Y, const BchOptions& opts,
                         const commutator::CommutatorOptions& copts, commutator::Workspace& ws) {
  constexpr std::array<double, 9> bernoulli = {1.0, -0.5,    1. / 6, 0.0,     -1. / 30,
                                               0.0, 1. / 42, 0.0,    -1. / 30};
  constexpr std::array<double, 9> factorial = {1.0, 1.0, 2.0, 6.0, 24., 120., 720., 5040., 40320.};
  const double thr = opts.product_threshold;
  const double nx = op::norm(X);
  const double ny = op::norm(Y);

  op::Operator Z = X + Y;
  op::Operator nested = commutator::commutator(Y, X, copts, ws);  // [Y, X]
  op::Operator next;
  double nxy = op::norm(nested);
  if (nxy * nx > thr) {
    commutator::commutator(nested, X, next, copts, ws);
    op::axpy(1. / 12, next, Z);
  }
  std::size_t k = 1;
  while (nxy > thr) {
    if (k < 2 || k % 2 == 0) op::axpy(bernoulli[k] / factorial[k], nested, Z);
    ++k;
    if (k >= bernoulli.size()) break;
    if (2 * ny * nxy < thr) break;
    commutator::commutator(Y, nested, next, copts, ws);
    std::swap(nested, next);
    nxy = op::norm(nested);
  }
  return Z;
}

}  // namespace imsrg::flow
