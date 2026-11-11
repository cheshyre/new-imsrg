#pragma once

/// @file
/// Scalar IMSRG(2) commutators Z = [X, Y] (design/13 section 6, design/reference/02 sections 2-4)
/// with the conventions of imsrg++, so that every term compares element by element with
/// imsrg-devel (golden data in tests/golden/commutator/).
///
/// Notation (all J-coupled, scalar operators): n_a is the reference occupation of orbit a,
/// nbar_a = 1 - n_a; X_ab is the one-body part; X^J_abcd is the UNNORMALIZED antisymmetrized
/// two-body matrix element (`op::TwoBody::get_unnormalized`, imsrg++ `GetTBME_J`), defined for
/// orbits in any order; Z^J_ijkl is written into the stored NORMALIZED element by dividing by
/// sqrt((1 + delta_ij)(1 + delta_kl)). Sums over a, b, c run over all orbits.
///
/// Every function here supports X and Y that are each Hermitian or anti-Hermitian (the
/// symmetry tricks of the fast kernels need it); a NonHermitian input throws
/// NotImplementedError. The test-only module `commutator_ref` evaluates the same expressions
/// without symmetry tricks, for any hermiticity.

#include "imsrg/commutator/options.hpp"
#include "imsrg/commutator/workspace.hpp"
#include "imsrg/operator/operator.hpp"
#include "imsrg/operator/operator_spec.hpp"

namespace imsrg::commutator {

/// Spec of Z = [X, Y] for scalar, number-conserving X and Y: scalar, particle_rank
/// max(x.particle_rank, y.particle_rank), hermiticity AntiHermitian if X and Y have the same
/// (definite) hermiticity, Hermitian if they have opposite ones, NonHermitian otherwise
/// (imsrg++ `CommutatorScalarScalar`, Commutator.cc:337-342).
/// @throws NotImplementedError for non-scalar or non-number-conserving specs and for
///         particle_rank 3 (IMSRG(3), milestone M10).
op::OperatorSpec commutator_spec(const op::OperatorSpec& x, const op::OperatorSpec& y);

/// Z = [X, Y] restricted to the terms in `opts.terms` (IMSRG(2) truncation: Z has no 3B part).
/// Z is overwritten: it is reset to zero if it already has spec `commutator_spec(X, Y)` and the
/// workspace's model space (no allocation, the flow-loop case), otherwise it is replaced by a
/// new operator of that spec. Terms that need a part X or Y lacks (particle_rank 1) are skipped.
/// The 222 ladder and the 221 term share their intermediates when both are selected (imsrg++
/// `comm222_pp_hh_221ss`).
/// Preconditions (InvariantError): X, Y on the workspace's model space (`ws.compatible`); Z is
/// neither X nor Y.
/// @throws NotImplementedError for NonHermitian X or Y and for specs `commutator_spec` rejects.
/// Complexity: dominated by the 2B terms, O(sum over channels of N^3) GEMMs.
void commutator(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                const CommutatorOptions& opts, Workspace& ws);

/// Allocating form of `commutator`: returns [X, Y] restricted to `opts.terms`.
op::Operator commutator(const op::Operator& X, const op::Operator& Y, const CommutatorOptions& opts,
                        Workspace& ws);

/// The individual terms. Each ADDS its contribution to Z (imsrg++ semantics) and computes every
/// stored element of the part it writes (both triangles; no final symmetrization step).
/// Common preconditions (InvariantError): X, Y and Z on the workspace's model space, Z with
/// spec `commutator_spec(X.spec(), Y.spec())` (or at least the part written allocated and the
/// same hermiticity). Each throws NotImplementedError for NonHermitian X or Y. A term whose
/// input parts are missing (particle_rank 1) adds nothing.
namespace terms {

/// Z0 += sum_ab (2 j_a + 1) n_a nbar_b (X_ab Y_ba - Y_ab X_ba). Adds nothing if Z is
/// anti-Hermitian (the term vanishes; imsrg++ returns early). O(norb^2).
void comm110ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z0 += 1/4 sum_abcd sum_J (2J+1) n_a n_b nbar_c nbar_d (X^J_abcd Y^J_cdab - Y^J_abcd X^J_cdab);
/// on normalized kets per channel: sum_J (2J+1) sum_{i,k} nn_i nbarnbar_k (X_ik Y_ki - Y_ik X_ki).
/// Adds nothing if Z is anti-Hermitian. O(sum over channels of N^2).
void comm220ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z_ij += sum_a (X_ia Y_aj - Y_ia X_aj). O(norb^3).
void comm111ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z_ij += sum_ab sum_J (2J+1)/(2 j_i + 1) (n_a - n_b) (X_ab Y^J_biaj - Y_ab X^J_biaj), for i, j
/// with equal (l, 2j, 2tz). O(norb^2 * orbits per one-body channel * J range).
void comm121ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z_ij += 1/2 sum_abc sum_J (2J+1)/(2 j_i + 1) (n_a n_b nbar_c + nbar_a nbar_b n_c)
///         (X^J_ciab Y^J_abcj - Y^J_ciab X^J_abcj),
/// evaluated from the ladder intermediates Mpp = X P_pp Y - Y P_pp X, Mhh = X P_hh Y - Y P_hh X
/// (P = diag(nbar_a nbar_b), diag(n_a n_b) over the kets of a channel) as
/// Z_ij += sum_c sum_J (2J+1)/(2 j_i + 1) (n_c Mpp^J_cicj + nbar_c Mhh^J_cicj) (unnormalized M).
/// O(sum over channels of N^3).
void comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z^J_ijkl += sum_a (X_ia Y^J_ajkl + X_ja Y^J_iakl - Y^J_ijal X_ak - Y^J_ijka X_al) - (X <-> Y).
/// O(sum over channels of N^2 * orbits per one-body channel).
void comm122ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// Z^J_ijkl += 1/2 sum_ab (nbar_a nbar_b - n_a n_b) (X^J_ijab Y^J_abkl - Y^J_ijab X^J_abkl),
/// i.e. Z2 += Mpp - Mhh per channel (normalized kets). O(sum over channels of N^3).
void comm222_pp_hhss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// The particle-hole term (imsrg++ `comm222_phss`):
///   Z^J_ijkl += sum_J' (2J'+1) [ {j_i j_j J; j_k j_l J'} Zbar^J'_ilkj
///                               - (-1)^(j_i + j_j - J) {j_j j_i J; j_k j_l J'} Zbar^J'_jlki ],
///   Zbar^J'_ilkj = sum_ab (n_a - n_b) (Xbar^J'_ilab Ybar^J'_abkj - Ybar^J'_ilab Xbar^J'_abkj),
///   Xbar^J'_ilab = - sum_J'' (2J''+1) {j_i j_l J'; j_a j_b J''} X^J''_ibal (Pandya transform).
/// Evaluated with the Pandya transform into the cross-coupled channels, one GEMM per channel and
/// the inverse transform (design/12 section 4.2). O(sum over cc channels of Ncc^2 Nph).
void comm222_phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z, Workspace& ws);

/// comm222_pp_hhss + comm221ss with shared intermediates (imsrg++ `comm222_pp_hh_221ss`).
void comm222_pp_hh_221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                         Workspace& ws);

}  // namespace terms

}  // namespace imsrg::commutator
