// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_TENSOR_H_
#define IMSRG_COMMUTATOR_TENSOR_H_

/// @file
/// Scalar-tensor IMSRG(2) commutators Z = [X, Y] with X a scalar (e.g. the
/// Magnus Omega) and Y a tensor of rank lambda (milestone M15, capability C3 of
/// design/50; design/reference/02 section 6.1). The formulas, phases and the
/// triangle each term computes are those of imsrg-devel
/// `Commutator/TensorCommutators.cc` (comm111st, comm121st, comm122st,
/// comm222_pp_hh_221st, comm222_phst), so every term compares element by
/// element with the M15 goldens (tests/golden/tensor/).
///
/// Conventions: X is a scalar, number-conserving operator (not reduced); Y
/// and Z are tensors (`OperatorSpec::is_scalar()` false) with REDUCED elements
/// in imsrg++'s gauge (`op::OperatorSpec`). Z has Y's rank, parity, delta_Tz
/// and particle rank, no zero-body part, and hermiticity hZ = -hX hY (both X
/// and Y must be Hermitian or anti-Hermitian). Elements of Y with ch_bra >
/// ch_ket are read through Y's hermiticity. Sums run over all orbits with the
/// reference occupations n_a of the workspace's model space (fractional
/// occupations allowed); "holes" are the orbits with n_a >= 1e-6 (imsrg++
/// OCC_CUT).
///
/// The 9j symbols of the particle-hole term are skipped below
/// `TensorOptions::ninej_threshold` (imsrg-devel 1e-10, the 79e9330 fork
/// 1e-8; design/50 section 5).

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator {

/// Options of the scalar-tensor commutators. Value type.
struct TensorOptions {
  /// 9j symbols with |9j| below this are treated as zero in the particle-hole
  /// term (imsrg-devel TensorCommutators.cc: 1e-10; fork 79e9330: 1e-8).
  double ninej_threshold = 1.0e-10;
  /// Reproduce the 79e9330 fork's particle-hole term for ODD rank L: when it
  /// reads Zbar^{J3 J4}_{(il),(kj)} through the stored (min, max) pair it
  /// omits the factor (-1)^L that imsrg-devel includes (Commutator.cc:
  /// 12396-12405 vs TensorCommutators.cc phase(Lambda + J3 + J4 + ...)); the
  /// affected elements get the opposite sign. No effect for even L.
  bool fork_ph_missing_rank_phase = false;
  /// Evaluate comm111st.
  bool comm111 = true;
  /// Evaluate comm121st.
  bool comm121 = true;
  /// Evaluate comm122st.
  bool comm122 = true;
  /// Evaluate the one-body part of comm222_pp_hh_221st (imsrg++ comm221st).
  bool comm221 = true;
  /// Evaluate the two-body part of comm222_pp_hh_221st (imsrg++
  /// comm222_pp_hhst).
  bool comm222_pp_hh = true;
  /// Evaluate comm222_phst.
  bool comm222_ph = true;
};

namespace detail {
struct TensorContext;
}  // namespace detail

/// Device tables of the scalar-tensor commutators for one model space: the
/// orbit and ket tables, one-body connection lists, the cross-coupled pair
/// structure, the 6j tables and, per (lambda, 9j threshold), the rank-lambda
/// 6j and 9j tables (built on first use and kept). Not thread safe; move-only.
/// Needs an initialized `exec::Runtime`.
class TensorWorkspace {
 public:
  /// Builds the lambda-independent tables of `ms` (non-null).
  explicit TensorWorkspace(std::shared_ptr<const basis::ModelSpace> ms);
  TensorWorkspace(TensorWorkspace&&) noexcept;
  TensorWorkspace& operator=(TensorWorkspace&&) noexcept;
  TensorWorkspace(const TensorWorkspace&) = delete;
  TensorWorkspace& operator=(const TensorWorkspace&) = delete;
  ~TensorWorkspace();

  /// The model space.
  const basis::ModelSpace& model_space() const noexcept { return *ms_; }
  /// True if `op` lives on this workspace's model space (same object or
  /// structurally equal).
  bool Compatible(const op::Operator& op) const noexcept;

  /// Internal data of the kernels (defined in `detail/tensor_impl.h`).
  detail::TensorContext& context() noexcept { return *ctx_; }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::unique_ptr<detail::TensorContext> ctx_;
};

/// Spec of [X, Y] for scalar X and tensor Y: Y's symmetry labels and particle
/// rank, hermiticity -hX hY.
/// @throws NotImplementedError unless X is scalar and number conserving, Y is
///         a number-conserving tensor, and both have definite hermiticity.
op::OperatorSpec TensorCommutatorSpec(const op::OperatorSpec& x,
                                      const op::OperatorSpec& y);

/// Z = [X, Y] for scalar X and tensor Y with the terms selected in `opts`,
/// in imsrg++'s order (111, 121, 122, 222_pp_hh_221, 222_ph). Z is replaced
/// by a zero operator of `TensorCommutatorSpec` first (reused without
/// allocation if it already has that spec). Preconditions (InvariantError):
/// X and Y on the workspace's model space; Z is neither X nor Y.
/// @throws NotImplementedError from `TensorCommutatorSpec`.
/// Complexity: dominated by comm222_phst, O(sum over cross-coupled channel
/// pairs of N3 N4 (Nph3 + Nph4)) plus its Pandya and inverse transforms with
/// 9j sums.
void TensorCommutator(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, const TensorOptions& opts,
                      TensorWorkspace& ws);

/// Allocating form of `TensorCommutator`.
op::Operator TensorCommutator(const op::Operator& X, const op::Operator& Y,
                              const TensorOptions& opts, TensorWorkspace& ws);

/// The individual terms. Each ADDS its contribution to Z (spec
/// `TensorCommutatorSpec(X, Y)`), computes the triangle imsrg++ computes and
/// mirrors it with the tensor hermiticity phase of Z, as imsrg++ does.
namespace tensor_terms {

/// Z1 += X1 Y1 - Y1 X1 (imsrg++ comm111st). O(norb^3).
void Comm111st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws);

/// [X1, Y2] + [X2, Y1] -> Z1 (imsrg++ comm121st, TensorCommutators.cc:46-155):
///   z_ij = sum_{a hole, b} n_a nbar_b sum_{J1 J2} (-1)^(j_j + j_a + J1 + L)
///          hat(J1) hat(J2) {J1 J2 L; j_j j_i j_a}
///          (X_ab Y^{J1 J2}_{bi,aj} - X_ba Y^{J1 J2}_{ai,bj})
///        - sum_{a hole, b} n_a nbar_b [Y_ab Xbar_ijab - Y_ba Xbar_ijba],
///   Xbar_ijab = - sum_J1 (2J1+1) {j_i j_j L; j_a j_b J1} X^{J1}_{ib,aj}
/// (unnormalized two-body elements), for j >= i, mirrored with hZ (-1)^(j_i -
/// j_j). O(norb^2 holes norb J^2).
void Comm121st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws);

/// [X1, Y2] + [X2, Y1] -> Z2 (imsrg++ comm122st, TensorCommutators.cc:
/// 158-330) for every stored block, the lower triangle of diagonal blocks
/// mirrored with hZ. O(sum over blocks of N_bra N_ket orbits per one-body
/// channel).
void Comm122st(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               TensorWorkspace& ws);

/// The ladder terms (imsrg++ comm222_pp_hh_221st, TensorCommutators.cc:
/// 336-504): per stored block Mpp = X Pbar Y - Y Pbar X, Mhh = X P Y - Y P X
/// (P = n_a n_b, Pbar = (1 - n_a)(1 - n_b) on the kets); `two_body`: Z2 += Mpp
/// - Mhh; `one_body`: Z1_ij += sum_c sum_{J1 J2} hat(J1) hat(J2) {J1 J2 L; j_j
/// j_i j_c} (-1)^(j_j + j_c + J1 + L) (n_c Mpp + nbar_c Mhh)^{J1 J2}_{ci,cj}
/// for holes c and Mhh for particles c, j >= i, mirrored. O(sum over blocks of
/// N^3).
void Comm222PpHh221st(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, TensorWorkspace& ws, bool one_body,
                      bool two_body);

/// The particle-hole term (imsrg++ comm222_phst, TensorCommutators.cc:
/// 1099-1413 with the Pandya transforms 536-1000), evaluated over all ordered
/// cross-coupled pairs:
///   Xbar^{J3}_{(il),(ab)} = - sum_J' (2J'+1) {j_i j_l J3; j_a j_b J'}
///                           X^{J'}_{ib,al},
///   Ybar^{J3 J4}_{(ab),(cd)} = - sum_{J1 J2} hat(J1 J2 J3 J4)
///       (-1)^(j_b + j_d + J4 + J2) {j_a j_d J1; j_b j_c J2; J3 J4 L}
///       Y^{J1 J2}_{ad,cb},
///   Zbar^{J3 J4} = Xbar^{J3} N Ybar^{J3 J4} - Ybar^{J3 J4} N Xbar^{J4},
///       N = diag(n_a - n_b) over the intermediate pairs (a, b),
///   Z^{J1 J2}_{ijkl} += sum_{J3 J4} hat(J1 J2 J3 J4) [ (-1)^(j_j + j_l + J2 +
///       J4) {j_i j_l J3; j_j j_k J4; J1 J2 L} Zbar_{(il),(kj)} - (-1)^(j_i +
///       j_j - J1) (-1)^(j_i + j_l + J2 + J4) {j_j j_l J3; j_i j_k J4; J1 J2 L}
///       Zbar_{(jl),(ki)} ] / sqrt((1 + d_ij)(1 + d_kl)).
/// 9j symbols below `ninej_threshold` are skipped.
void Comm222Phst(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 TensorWorkspace& ws, double ninej_threshold,
                 bool fork_ph_missing_rank_phase = false);

}  // namespace tensor_terms

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_TENSOR_H_
