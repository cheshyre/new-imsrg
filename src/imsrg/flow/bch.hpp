#pragma once
/// @file
/// Baker-Campbell-Hausdorff machinery of the Magnus solver (design/13 section 7,
/// design/reference/03 section 3, imsrg++ `BCH.cc`): the similarity transformation
/// e^Omega X e^-Omega and the product Z with e^Z = e^X e^Y. Commutators are the truncated
/// (IMSRG(2)) commutators of the `commutator` module, so both series act on the 0+1+2-body space.
/// The thresholds and the truncation are numerics-defining (design/00 D12): they reproduce
/// imsrg++ exactly.
#include "imsrg/commutator/commutator.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::flow {

/// Options of `bch_transform` and `bch_product` (imsrg++ `BCH::` file-scope defaults).
struct BchOptions {
  /// `bch_transform`: X with ||X|| <= this is returned unchanged; also scales the termination
  /// epsilon (imsrg++ `bch_transform_threshold`).
  double transform_threshold = 1e-9;
  /// `bch_product` series threshold (imsrg++ `bch_product_threshold`).
  double product_threshold = 1e-4;
  /// Maximum number of nested commutators in `bch_transform`.
  int max_iterations = 40;
  /// A warning is logged if `bch_transform` reaches this many nested commutators.
  int warn_iterations = 12;
};

/// Diagnostics of one `bch_transform` call.
struct BchReport {
  int nested_commutators = 0;  ///< Number of nested commutators evaluated.
  bool converged = true;       ///< False if the series hit `max_iterations`.
};

/// Z = e^Omega X e^-Omega = X + [Omega, X] + 1/2! [Omega, [Omega, X]] + ... (imsrg++
/// `Standard_BCH_Transform`, BCH.cc:101-218):
/// * if ||X|| <= transform_threshold, Z = X (promoted to particle_rank 2);
/// * otherwise with eps = ||X|| exp(-2 ||Omega||) transform_threshold / (2 ||Omega||), for
///   i = 1, 2, ..., max_iterations: nested_i = [Omega, nested_{i-1}] (nested_0 = X),
///   Z += nested_i / i!, eps *= (i + 1), stop when ||nested_i|| < eps.
/// Norms are `op::norm` (zero-body part excluded). X is promoted to particle_rank 2 (imsrg++
/// `SetParticleRank(2)`) so Z has a two-body part. Omega must be anti-Hermitian for a unitary
/// transformation, but any hermiticity the commutator accepts works.
/// Preconditions: X and Omega scalar, number conserving, on `ws`'s model space.
/// Complexity: (number of nested commutators) x one commutator.
op::Operator bch_transform(const op::Operator& X, const op::Operator& Omega, const BchOptions& opts,
                           const commutator::CommutatorOptions& copts, commutator::Workspace& ws,
                           BchReport* report = nullptr);

/// Z with e^Z = e^X e^Y for small X (imsrg++ `BCH_Product`, BCH.cc:261-352): with
/// B = {1, -1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30}, k! = {1, 1, 2, 6, ...}:
///   Z = X + Y; N = [Y, X]; n = ||N||;
///   if n ||X|| > product_threshold: Z += 1/12 [N, X];
///   k = 1; while n > product_threshold:
///     if k < 2 or k even: Z += B_k / k! N;   k += 1;
///     if k >= 9 or 2 ||Y|| n < product_threshold: stop;
///     N = [Y, N]; n = ||N||.
/// This is the series in powers of ad_Y to 8th order plus the single 1/12 [[Y,X],X] term; the
/// -1/24 [Y,[X,[X,Y]]] term of the exact series is NOT included (a truncation choice of
/// imsrg++, reproduced on purpose; design/13 section 7).
/// Preconditions: X and Y scalar, number conserving, on `ws`'s model space; Z gets the larger
/// particle_rank of X and Y.
/// Complexity: up to 9 commutators.
op::Operator bch_product(const op::Operator& X, const op::Operator& Y, const BchOptions& opts,
                         const commutator::CommutatorOptions& copts, commutator::Workspace& ws);

/// Returns `op` with at least the given particle rank (a copy with zero higher-body parts added
/// if needed; `op` itself if it already has that rank).
op::Operator with_particle_rank(const op::Operator& op, std::int32_t particle_rank);

}  // namespace imsrg::flow
