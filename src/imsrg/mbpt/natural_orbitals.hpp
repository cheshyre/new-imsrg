#pragma once
/// @file
/// MBPT(2) one-body density matrix and natural orbitals (imsrg++ `HFMBPT::GetNaturalOrbitals`,
/// HFMBPT.cc:421-777). All matrices are indexed by orbit index (row = old orbit, column = new
/// orbit). Host only. mbpt does not depend on hf: the caller (driver / test) assembles the
/// NAT-basis Hamiltonian from `reference_density_ho` and the C matrices (design/13).

#include <vector>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/linalg/matrix_view.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::mbpt {

/// Natural orbitals of one density matrix.
struct NaturalOrbitals {
  linalg::Matrix<double> c_hf2nat;  ///< Row = HF orbit, column = NAT orbit; block diagonal in (l,j,tz).
  std::vector<double> occupations;  ///< Occupation of each NAT orbit (descending within a channel).
};

/// MBPT(2) one-body density matrix in the HF basis (norb x norb), rho = occupations on the
/// diagonal for holes plus the PP, HH and PH second-order corrections with the imsrg++
/// energy-denominator regularization (HFMBPT.cc:421-777). H must be the normal-ordered
/// Hamiltonian in the HF basis (f diagonal). Syncs H to host. Complexity:
/// O(N_p^2 N_p N_h^2 J_max) within each 1B channel.
linalg::Matrix<double> mbpt2_density_matrix(op::Operator& H);

/// Diagonalizes each (l,j,tz) block of `rho_hf`, orders eigenvalues descending, fixes the column
/// signs so that C(i,i) >= 0 and checks sum (2j+1) n = A to 1e-6. Throws InvariantError if the
/// trace check fails. Complexity O(sum n_ch^3).
NaturalOrbitals natural_orbitals(const linalg::Matrix<double>& rho_hf,
                                 const basis::ModelSpace& ms);

/// HO-basis reference density of the NAT basis: rho(a,b) = sum over reference holes alpha of
/// n_alpha C_ho2nat(a,alpha) C_ho2nat(b,alpha) (imsrg++ default, use_NAT_occupations = false).
/// `c_ho2nat` is norb x norb. Complexity O(norb^2 N_h).
linalg::Matrix<double> reference_density_ho(const linalg::Matrix<double>& c_ho2nat,
                                            const basis::ModelSpace& ms);

/// Matrix product a * b of two square matrices (used for C_ho2nat = C_ho2hf * C_hf2nat).
/// Complexity O(n^3).
linalg::Matrix<double> matmul(const linalg::Matrix<double>& a, const linalg::Matrix<double>& b);

/// Transforms the one- and two-body parts of `H` (HF basis) to the NAT basis with
/// `op::transform_basis` and the given C_hf2nat.
op::Operator to_nat_basis(op::Operator& H, const linalg::Matrix<double>& c_hf2nat);

}  // namespace imsrg::mbpt
