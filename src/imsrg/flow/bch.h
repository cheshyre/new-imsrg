// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_BCH_H_
#define IMSRG_FLOW_BCH_H_
/// @file
/// Baker-Campbell-Hausdorff machinery of the Magnus solver (design/13 section
/// 7, design/reference/03 section 3, imsrg++ `BCH.cc`): the similarity
/// transformation e^Omega X e^-Omega and the product Z with e^Z = e^X e^Y.
/// Commutators are the truncated (IMSRG(2)) commutators of the `commutator`
/// module, so both series act on the 0+1+2-body space. The thresholds and the
/// truncation are numerics-defining (design/00 D12): they reproduce imsrg++
/// exactly.
#include "imsrg/commutator/commutator.h"
#include "imsrg/commutator/factorized.h"
#include "imsrg/operator/operator.h"

namespace imsrg::flow {

/// Options of `BchTransform` and `BchProduct` (imsrg++ `BCH::` file-scope
/// defaults).
struct BchOptions {
  /// `BchTransform`: X with ||X|| <= this is returned unchanged; also scales
  /// the termination epsilon (imsrg++ `bch_transform_threshold`).
  double transform_threshold = 1.0e-9;
  /// `BchProduct` series threshold (imsrg++ `bch_product_threshold`).
  double product_threshold = 1.0e-4;
  /// Maximum number of nested commutators in `BchTransform`.
  int max_iterations = 40;
  /// A warning is logged if `BchTransform` reaches this many nested
  /// commutators.
  int warn_iterations = 12;
  /// IMSRG(3f2) (design/14 section 4 item 3, owner decision D32; imsrg++
  /// `BCH::SetUseFactorizedCorrection`, which switches on the transform and
  /// the product correction together): every nested commutator from the
  /// second on gets the factorized double commutator F of the nested
  /// commutator two steps back, in `BchTransform` and in `BchProduct` (see
  /// there). Needs an anti-Hermitian Omega (resp. X and Y of the product).
  bool factorized = false;
  /// Topologies of F (`commutator::FactorizedDoubleCommutator`); the defaults
  /// are the D32 recipe (one-body intermediates on, two-body off).
  commutator::FactorizedOptions factorized_terms;
  /// `BchTransform` only (imsrg++ `BCH::SetBCHSkipiEq1`, used by the
  /// perturbative triples): the series starts at the second nested
  /// commutator with the weights of the next order, Z = X + sum_{k>=1}
  /// ad_Omega^k(X) / (k+1)!, so that [Omega, Z] is the full transform of the
  /// triples' Wbar. Exactly the imsrg++ loop: the i = 1 pass is skipped before
  /// any bookkeeping, hence epsilon is not multiplied by 2 and the counter of
  /// the factorial runs one ahead of the nested commutators.
  bool skip_first_commutator = false;
};

/// Diagnostics of one `BchTransform` call.
struct BchReport {
  int nested_commutators = 0;  ///< Number of nested commutators evaluated.
  bool converged = true;       ///< False if the series hit `max_iterations`.
  int factorized_corrections = 0;  ///< Non-vanishing factorized corrections
                                   ///< added.
};

/// Z = e^Omega X e^-Omega = X + [Omega, X] + 1/2! [Omega, [Omega, X]] + ...
/// (imsrg++ `Standard_BCH_Transform`, BCH.cc:101-218):
/// * if ||X|| <= transform_threshold, Z = X (promoted to particle_rank 2);
/// * otherwise with eps = ||X|| exp(-2 ||Omega||) transform_threshold / (2
///   ||Omega||), for i = 1, 2, ..., max_iterations: nested_i = [Omega,
///   nested_{i-1}] (nested_0 = X), Z += nested_i / i!, eps *= (i + 1), stop
///   when ||nested_i|| < eps.
/// With `opts.factorized` (IMSRG(3f2), imsrg++ BCH.cc:153-157 and 179-191),
/// nested_i = [Omega, nested_{i-1}] + F(Omega, nested_{i-2}) for i >= 2, with
/// the corrected nested_{i-2}; the correction is added before the series
/// update and the epsilon test (Omega's intermediates are computed once per
/// call, `commutator::FactorizedEta`).
/// With `opts.skip_first_commutator` (imsrg++ `BCH::SetBCHSkipiEq1`) the term
/// i = 1 is skipped as a whole: the pass i = 1 is empty (no division by i,
/// no epsilon update), the passes i >= 2 add nested_{i-1} / i! and multiply
/// epsilon by (i + 1); the factorized correction F(Omega, nested_{j-2}) then
/// applies from nested_2 on, as without the skip (imsrg++ `i_min_factorized =
/// 2`).
/// Norms are `op::norm` (zero-body part excluded, three-body part included). X
/// is promoted to particle_rank 2 (imsrg++ `SetParticleRank(2)`) so Z has a
/// two-body part, and to particle_rank 3 if `copts` select IMSRG(3) terms and
/// `ws` has a three-body context (the commutators then return rank 3).
/// Omega must be anti-Hermitian for a unitary transformation; without
/// `factorized` any hermiticity the commutator accepts works, with it Omega
/// must be anti-Hermitian (NotImplementedError otherwise). Preconditions: X and
/// Omega scalar, number conserving, on `ws`'s model space. Complexity: (number
/// of nested commutators) x one commutator, plus one factorized double
/// commutator per nested term from the second on with `factorized` (about half
/// a commutator each).
op::Operator BchTransform(const op::Operator& X, const op::Operator& Omega,
                          const BchOptions& opts,
                          const commutator::CommutatorOptions& copts,
                          commutator::Workspace& ws,
                          BchReport* report = nullptr);

/// Z with e^Z = e^X e^Y for small X (imsrg++ `BCH_Product`, BCH.cc:261-352):
/// with B = {1, -1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30}, k! = {1, 1, 2, 6,
/// ...}:
///   Z = X + Y; N = [Y, X]; n = ||N||;
///   if n ||X|| > product_threshold: Z += 1/12 [N, X];
///   k = 1; while n > product_threshold:
///     if k < 2 or k even: Z += B_k / k! N;   k += 1;
///     if k >= 9 or 2 ||Y|| n < product_threshold: stop;
///     N = [Y, N]; n = ||N||.
/// This is the series in powers of ad_Y to 8th order plus the single 1/12
/// [[Y,X],X] term; the -1/24 [Y,[X,[X,Y]]] term of the exact series is NOT
/// included (a truncation choice of imsrg++, reproduced on purpose; design/13
/// section 7). With `opts.factorized` (imsrg++ BCH.cc:312-359): the 1/12 term
/// adds F(X, Y) / 12, and N_k = [Y, N_{k-1}] + F(Y, N_{k-2}) for k >= 2 with
/// N_0 = X, N_1 = [Y, X] (F of `commutator::FactorizedDoubleCommutator`; a
/// term whose operands lack a two-body part vanishes and is skipped; with
/// `factorized`, X and Y must be anti-Hermitian).
/// Preconditions: X and Y scalar, number conserving, on `ws`'s
/// model space; Z gets the larger particle_rank of X and Y, and rank 3 if
/// `copts` select IMSRG(3) terms and `ws` has a three-body context. Complexity:
/// up to 9 commutators, plus up to 8 factorized double commutators with
/// `factorized`.
op::Operator BchProduct(const op::Operator& X, const op::Operator& Y,
                        const BchOptions& opts,
                        const commutator::CommutatorOptions& copts,
                        commutator::Workspace& ws);

/// Returns `op` with at least the given particle rank (a copy with zero
/// higher-body parts added if needed; `op` itself if it already has that rank).
op::Operator WithParticleRank(const op::Operator& op,
                              std::int32_t particle_rank);

}  // namespace imsrg::flow

#endif  // IMSRG_FLOW_BCH_H_
