#pragma once
/// @file
/// Normal ordering of scalar operators with respect to a reference with (possibly fractional)
/// occupations, its inverse, and the normal ordering of a three-nucleon force given as NO2B input
/// (design/13 section 2, design/reference/01 section 3.3). The formulas and summation conventions
/// are those of imsrg++ `Operator::DoNormalOrdering2/3` (Operator.cc:476-674), so results agree
/// element by element (tested against imsrg-devel goldens at 1e-10).
#include <cstdint>
#include <span>

#include "imsrg/operator/no2b.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::op {

/// Direction of a normal-ordering transformation: imsrg++ `sign` = +1 (vacuum -> reference) or
/// -1 (reference -> vacuum).
enum class NormalOrderDirection : std::uint8_t { ToReference, ToVacuum };

/// Normal orders the 0B + 1B + 2B parts of a scalar operator w.r.t. the occupations n_a (one per
/// orbit). The sums run over the "occupied" orbits n_a >= `basis::kOccupationCut` (imsrg++
/// holes), weighted by n_a; s = +1 (ToReference) or -1 (ToVacuum):
/// * E0 += s sum_k (2j_k + 1) n_k O_kk + sum_{k <= l} sum_J (2J + 1) n_k n_l V^J_{kl,kl}
///   (normalized V; the two-body term has no s, which makes ToVacuum the exact inverse);
/// * O_ab += s sum_h sum_J (2J + 1)/(2j_a + 1) n_h Vt^J_{ah,bh} for a, b with equal (l, 2j, tz2)
///   (Vt unnormalized, imsrg++ `GetTBME`), with O_ba = h O_ab for (anti-)Hermitian operators;
/// * the two-body part is unchanged.
/// The result is on the same model space with the same spec.
/// @throws InvariantError if `occupations.size()` differs from the number of orbits.
/// @throws NotImplementedError for non-scalar operators (by construction none exist yet).
Operator normal_order(const Operator& op, std::span<const double> occupations,
                      NormalOrderDirection direction);

/// Normal orders a vacuum-normal-ordered operator w.r.t. the reference of its model space
/// (imsrg++ `DoNormalOrdering` for particle_rank <= 2).
Operator normal_order(const Operator& op);

/// Inverse of `normal_order(op)`: from the reference of the model space back to the vacuum
/// (imsrg++ `UndoNormalOrdering`).
Operator undo_normal_order(const Operator& op);

/// imsrg++ `DoNormalOrdering3` (Operator.cc:582-674) for an operator whose 0B + 1B + 2B parts
/// are `op` and whose three-body part is the NO2B input `v3`: the rank-2 operator
///   NO2(op) + [NO2(Gamma) with its zero-body part scaled by 1/3 and its one-body part by 1/2],
/// where NO2 is `normal_order(., occupations, direction)` and Gamma is the two-body operator
///   Gamma^J_{ij,kl} = s sum_a n_a W_pn(ija, kla; J) / ((2J + 1) sqrt((1 + d_ij)(1 + d_kl)))
/// (sum over the occupied orbits a; W_pn = `v3.get_pn`; terms where (i j a) or (k l a) exceed
/// the e3max of `v3` are skipped); Gamma is computed for bra <= ket and mirrored. The result has
/// particle_rank 2 and is Hermitian.
/// @throws InvariantError if `occupations.size()` differs from the number of orbits or `op` is
///         not Hermitian (the NO2B storage is symmetric, so only Hermitian three-body parts can
///         be represented).
Operator normal_order(const Operator& op, const No2bThreeBody& v3,
                      std::span<const double> occupations, NormalOrderDirection direction);

/// `normal_order(op, v3, occupations, ToReference)` with the reference occupations of the
/// model space: the normal-ordered Hamiltonian in the NO2B approximation.
Operator normal_order(const Operator& op, const No2bThreeBody& v3);

}  // namespace imsrg::op
