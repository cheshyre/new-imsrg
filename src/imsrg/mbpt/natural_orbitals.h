// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_MBPT_NATURAL_ORBITALS_H_
#define IMSRG_MBPT_NATURAL_ORBITALS_H_
/// @file
/// MBPT(2) one-body density matrix and natural orbitals (imsrg++
/// `HFMBPT::GetNaturalOrbitals`, HFMBPT.cc:421-777). All matrices are indexed
/// by orbit index (row = old orbit, column = new orbit). Host only. mbpt does
/// not depend on hf: the caller (driver / test) assembles the NAT-basis
/// Hamiltonian from `ReferenceDensityHo` and the C matrices (design/13).

#include <cstdint>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/linalg/matrix_view.h"
#include "imsrg/operator/operator.h"

namespace imsrg::mbpt {

/// Natural orbitals of one density matrix.
struct NaturalOrbitals {
  linalg::Matrix<double> c_hf2nat;  ///< Row = HF orbit, column = NAT orbit;
                                    ///< block diagonal in (l,j,tz).
  std::vector<double> occupations;  ///< Occupation of each NAT orbit
                                    ///< (descending within a channel).
};

/// MBPT(2) one-body density matrix in the HF basis (norb x norb), rho =
/// occupations on the diagonal for holes plus the PP, HH and PH second-order
/// corrections with the imsrg++ energy-denominator regularization
/// (HFMBPT.cc:421-777). H must be the normal-ordered Hamiltonian in the HF
/// basis (f diagonal). Syncs H to host. Complexity: O(N_p^2 N_p N_h^2 J_max)
/// within each 1B channel.
linalg::Matrix<double> Mbpt2DensityMatrix(op::Operator& H);

/// Order of the natural orbitals inside one (l,j,tz) channel.
enum class NatSorting : std::int8_t {
  /// Descending signed occupation (imsrg-devel `DiagonalizeRho`; only for
  /// comparisons with imsrg-devel, config `natural_orbitals.sort =
  /// "signed_occupation"`).
  kSignedDescending,
  /// Descending |occupation| (the default, owner rule D72; the fork imsrg++
  /// 79e9330 re-sorting, HFMBPT.cc:131-182): MBPT(2)
  /// occupations can be negative, and a negative one with a large magnitude
  /// moves to a lower slot (lower n label), which changes which orbits survive
  /// a later emax truncation. Ties in |occupation| keep the signed order
  /// (stable sort; the fork's std::sort leaves them unspecified).
  kAbsDescending,
};

/// Diagonalizes each (l,j,tz) block of `rho_hf`, orders eigenvalues as
/// `sorting` says (column k of a channel = its k-th orbit in index order),
/// fixes the column signs so that C(i,i) >= 0 (after the sorting, as imsrg++
/// `ReorderHFMBPTCoefficients`) and checks sum (2j+1) n = A to 1e-6. Throws
/// InvariantError if the trace check fails. Complexity O(sum n_ch^3).
NaturalOrbitals ComputeNaturalOrbitals(
    const linalg::Matrix<double>& rho_hf, const basis::ModelSpace& ms,
    NatSorting sorting = NatSorting::kAbsDescending);

/// Resets the occupied natural orbitals to the HF orbitals (fork imsrg++
/// 79e9330 `HFMBPT::UseHFForHoleStates`, HFMBPT.cc:892-1136, enabled by
/// `use_HF_reference_in_NAT`), in place on `nat.c_hf2nat` (HF basis):
///   1. for every orbit i with reference occupation > 1e-8 (in `ms`): column
///      i := e_i and row i := 0 in every other column (removes the HF-hole
///      components from the NAT virtuals);
///   2. Gram-Schmidt inside each (l,j,tz) channel, orbits i in increasing
///      index, against every channel orbit j < i in increasing index, with
///      the coefficient <i|j>/<j|j> of the current (not yet normalized)
///      columns, sums over the channel's rows in index order (the fork's loop
///      order, so the result agrees to the last bits);
///   3. every column normalized; then the sign convention C(i,i) >= 0.
/// The occupations are not changed. The hole columns stay unit vectors, so
/// the mean-field state built from the result (`hf::SolutionInBasis`) is the
/// HF state: E_ref(NAT) = E_HF. Complexity O(sum n_ch^3).
void UseHfForHoleStates(NaturalOrbitals& nat, const basis::ModelSpace& ms);

/// HO-basis reference density of the NAT basis: rho(a,b) = sum over reference
/// holes alpha of n_alpha C_ho2nat(a,alpha) C_ho2nat(b,alpha) (imsrg++ default,
/// use_NAT_occupations = false). `c_ho2nat` is norb x norb. Complexity O(norb^2
/// N_h).
linalg::Matrix<double> ReferenceDensityHo(
    const linalg::Matrix<double>& c_ho2nat, const basis::ModelSpace& ms);

/// Matrix product a * b of two square matrices (used for C_ho2nat = C_ho2hf *
/// C_hf2nat). Complexity O(n^3).
linalg::Matrix<double> Matmul(const linalg::Matrix<double>& a,
                              const linalg::Matrix<double>& b);

/// Transforms the one- and two-body parts of `H` (HF basis) to the NAT basis
/// with `op::TransformBasis` and the given C_hf2nat.
op::Operator ToNatBasis(op::Operator& H,
                        const linalg::Matrix<double>& c_hf2nat);

}  // namespace imsrg::mbpt

#endif  // IMSRG_MBPT_NATURAL_ORBITALS_H_
