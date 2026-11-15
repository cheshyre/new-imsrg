// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_THREE_BODY_H_
#define IMSRG_COMMUTATOR_THREE_BODY_H_

/// @file
/// The [2,2]->3 commutator term (imsrg++ `comm223ss`,
/// IMSRG3Commutators.cc:5012-5376) and the streamed perturbative-triples
/// energy (`comm223ss` with `perturbative_triples = true`), both through the
/// pair-third tiles of the layout benchmark (kernel K1 of
/// design/benchmarks/imsrg3-layout.md section 3) with the production changes
/// of design/14 section 4 item 0: per-tile gathers of the two-body legs, the
/// internal orbit restricted to its 6j-allowed j range, team-level
/// parallelism with team scratch, and packed-triangle writes.
///
/// Formula (layout A, D43). For a channel (2J, parity, 2Tz) of the kept
/// canonical space of the context's model space, a bra triple T, a ket triple
/// T', the pair-third states r = (P, z, J1') of T and s = (Q, w, J2') of T'
/// (`op::PairThirdSpace`):
///
///   W[r, s] = Hat(J1') Hat(J2') sum_a {j_w j_a J1'; j_z J J2'}
///             (X_{P,(w a)} Y_{(z a),Q} - Y_{P,(w a)} X_{(z a),Q}),
///   Z[T, T'] = R_T W R_T'^T,
///
/// with Hat(J) = sqrt(2J + 1), two-body elements normalized in the pairs P
/// and Q (their sqrt(2) is in the recoupling map R) and unnormalized in the
/// pairs (w a) and (z a) (times sqrt(2) for w == a, z == a), exchange phases
/// for any orbit order (`op::TwoBody::get_normalized`). Both legs are read as
/// stored (the ket leg X_{(z a),Q} is the element with bra (z a) and ket Q),
/// so no hermiticity of X or Y is assumed in W. Z values are imsrg++'s
/// unnormalized antisymmetrized canonical elements (`op::ThreeBody`).
///
/// Operands: scalar, number-conserving X and Y with particle rank >= 2 (only
/// their two-body parts enter), on a model space with the same basis,
/// reference occupations and partition as the context's (the three-body data
/// may differ, e.g. operators built on the model space before
/// `ModelSpace::WithThreeBody`). Deviation from imsrg++: its early return for
/// ||X_2|| ||Y_2|| < 1e-6 (IMSRG3Commutators.cc:5020) is not reproduced; the
/// term is always evaluated.

#include <cstdint>
#include <memory>
#include <span>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::commutator {

/// Device tables of the [2,2]->3 tiles for one model space with a three-body
/// space: the basis device tables, the 3B 6j table (`angmom::SixJ3Table`),
/// the pair-third states with the recoupling map R (`op::PairThirdSpace`),
/// the kept canonical space and the orbit groups by (parity, 2tz) sorted by
/// (2j, index) on the device, the largest tile extents (team scratch size) and
/// the team work orders (bra triples by estimated tile work, largest first).
/// Built once; read-only afterwards (no lazy caches, D10), so concurrent
/// calls may share a context. The occupations of the model space's reference
/// select the triples work (a new reference needs a new context). Needs an
/// initialized `exec::Runtime`; destroy it before the Runtime. Movable, not
/// copyable.
class ThreeBodyContext {
 public:
  /// Builds the tables for `ms3`. Cost: the 6j table (O(emax^6) symbols) and
  /// the pair-third states with their maps (O(states x J range)).
  /// @throws InputError if `ms3` is null or has no three-body space (build it
  ///         with `ModelSpace::WithThreeBody`).
  explicit ThreeBodyContext(std::shared_ptr<const basis::ModelSpace> ms3);

  ThreeBodyContext(ThreeBodyContext&&) noexcept;
  ThreeBodyContext& operator=(ThreeBodyContext&&) noexcept;
  ThreeBodyContext(const ThreeBodyContext&) = delete;
  ThreeBodyContext& operator=(const ThreeBodyContext&) = delete;
  ~ThreeBodyContext();

  /// The model space (with its three-body space) the tables were built for.
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr()
      const noexcept {
    return ms_;
  }

  /// Bytes of the device tables owned by the context (basis tables, 6j table,
  /// pair-third states with R and E, canonical-space tables, orbit groups,
  /// leg gather tables, work orders), for reports. 0 for a moved-from
  /// context.
  std::int64_t TableBytes() const noexcept;

  /// Bytes of team scratch one team of the tile kernels requests (sized from
  /// the largest tile extents: the bra-leg cache, the ket legs, the tile W,
  /// the half-transformed tile and the per-state index tables), for reports.
  /// 0 for a moved-from context.
  std::int64_t ScratchBytesPerTeam() const noexcept;

  /// Internal tables (defined in three_body.cc).
  struct Impl;
  /// Internal tables, for the functions of this header. Precondition: not
  /// moved from (IMSRG_CHECK).
  const Impl& impl() const;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::unique_ptr<Impl> impl_;
};

/// The [2,2]->3 part of Z = [X, Y] (imsrg++ `comm223ss`; formula in the file
/// comment) as a new `op::ThreeBody` on the context's model space. Its
/// hermiticity is -h_X h_Y for X and Y of definite hermiticity (only the
/// tiles T' <= T are computed and the packed lower triangle is written
/// directly, the anti-Hermitian diagonal as 0), NonHermitian if X or Y is
/// non-Hermitian (all tiles, dense blocks). Device data of the result are
/// current (memory model of `op::Operator`).
/// @throws NotImplementedError for non-scalar or non-number-conserving X or Y.
/// @throws InputError if X or Y is empty or has no two-body part (particle
///         rank < 2).
/// @throws InvariantError if X or Y lives on a model space whose basis,
///         reference or partition differs from the context's, or the context
///         was moved from.
/// Complexity: sum over channels and tiles (T, T' <= T) of m m' n_a
/// multiply-adds for W (m, m' pair-third states of T and T', n_a the
/// 6j-allowed internal orbits) plus the gathers O((m + m') n_a) and the
/// epilogue O(nnz(R_T') m + nnz(R_T) k') per tile; twice the tiles for a
/// non-Hermitian result. One team per bra triple.
op::ThreeBody Comm223(const op::Operator& X, const op::Operator& Y,
                      const ThreeBodyContext& ctx);

/// The perturbative-triples energy of imsrg++ `comm223ss` with
/// `perturbative_triples = true` (IMSRG3Commutators.cc:5048-5070,
/// 5123-5124, 5157-5158, 5343-5373), without three-body storage:
///
///   E = sum_ch (2J + 1) sum_{bra > ket} (sym_bra / 6) (sym_ket / 6) z^2
///       (n_bra nbar_ket - n_ket nbar_bra) / (e_bra - e_ket),
///
/// z = Z[bra, ket] of `Comm223` (the lower-triangle element, any hermiticity),
/// bra > ket in the channel-local row order, sym = 6, 3, 1 for a triple of
/// distinct, two equal, three equal orbits, n_abc = n_a n_b n_c and nbar_abc
/// = (1 - n_a)(1 - n_b)(1 - n_c) with the reference occupations of the
/// context's model space, and e_abc = epsilon[a] + epsilon[b] + epsilon[c]
/// (MeV, one value per orbit; imsrg++ uses the one-body diagonal of Z).
/// imsrg++'s skips: bras with n_bra < 1e-8 and nbar_bra < 1e-8, pairs with
/// |n_bra nbar_ket - n_ket nbar_bra| < 1e-8. Only the tiles T' < T with a
/// nonzero occupation factor are computed (no storage). The sum is a parallel
/// reduction: the last digits depend on the thread count.
/// @throws InputError if `epsilon.size()` is not the number of orbits, and as
///         `Comm223` for X, Y and the context.
/// Complexity: as `Comm223` restricted to the hhh-ppp-type tiles (2 % of the
/// time of the full term for O16 at emax 3).
double PerturbativeTriplesEnergy(const op::Operator& X, const op::Operator& Y,
                                 std::span<const double> epsilon,
                                 const ThreeBodyContext& ctx);

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_THREE_BODY_H_
