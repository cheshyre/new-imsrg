// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_COMMUTATOR_H_
#define IMSRG_COMMUTATOR_COMMUTATOR_H_

/// @file
/// Scalar IMSRG(2) commutators Z = [X, Y] (design/13 section 6,
/// design/reference/02 sections 2-4) with the conventions of imsrg++, so that
/// every term compares element by element with imsrg-devel (golden data in
/// tests/golden/commutator/).
///
/// Notation (all J-coupled, scalar operators): n_a is the reference occupation
/// of orbit a, nbar_a = 1 - n_a; X_ab is the one-body part; X^J_abcd is the
/// UNNORMALIZED antisymmetrized two-body matrix element
/// (`op::TwoBody::get_unnormalized`, imsrg++ `GetTBME_J`), defined for orbits
/// in any order; Z^J_ijkl is written into the stored NORMALIZED element by
/// dividing by sqrt((1 + delta_ij)(1 + delta_kl)). Sums over a, b, c run over
/// all orbits.
///
/// Every function here supports X and Y that are each Hermitian, anti-Hermitian
/// or general (NonHermitian). Z is general if either input is. The
/// (anti-)Hermitian inputs take the fast paths that use X^T = hX X (half the
/// work in the ladder and particle-hole terms, one triangle per two-body
/// block); a NonHermitian input takes the general paths, which evaluate the
/// explicit products X P Y - Y P X and every stored element independently (the
/// ladder terms need a second product accumulated into the first, the
/// particle-hole term works with the Pandya matrices of all 2 Ncc ordered
/// pairs: about 2x and 4x the flops, with its extra scratch, at most twice the
/// Hermitian one, allocated at the first general call). The test-only module
/// `commutator_ref` evaluates the same expressions without symmetry tricks and
/// is the oracle of the general paths (design/00 D36, M7-Q2).

#include "imsrg/commutator/options.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator {

/// Spec of Z = [X, Y] for scalar, number-conserving X and Y: scalar,
/// particle_rank max(x.particle_rank, y.particle_rank), hermiticity
/// AntiHermitian if X and Y have the same (definite) hermiticity, Hermitian if
/// they have opposite ones, NonHermitian if either is NonHermitian (imsrg++
/// `CommutatorScalarScalar`, Commutator.cc:337-342).
/// @throws NotImplementedError for non-scalar or non-number-conserving specs.
op::OperatorSpec CommutatorSpec(const op::OperatorSpec& x,
                                const op::OperatorSpec& y);

/// Z = [X, Y] restricted to the terms in `opts.terms`. Truncation: IMSRG(2)
/// (Z has no 3B part) unless `opts.terms` holds an IMSRG(3) term, then
/// IMSRG(3) (Z has particle rank 3; imsrg++ `use_imsrg3`). Z is overwritten:
/// it is reset to zero if it already has spec `CommutatorSpec(X, Y)` (rank 3
/// for IMSRG(3)) and the workspace's model space (no allocation, the
/// flow-loop case), otherwise it is replaced by a new operator of that spec.
/// Terms that need a part X or Y lacks are skipped. The 222 ladder and the
/// 221 term share their intermediates when both are selected (imsrg++
/// `comm222_pp_hh_221ss`). The IMSRG(3) terms (imsrg3.h) follow the IMSRG(2)
/// ones in imsrg++'s order (133, 330, 223, 232, 331, 231, 132), only if
/// ||X|| and ||Y|| exceed `opts.threebody_threshold`. Preconditions
/// (InvariantError): X, Y on the workspace's model space (`ws.compatible`); Z
/// is neither X nor Y.
/// MPI (D14 phase 1): on a workspace whose communicator has more than one
/// rank (`WorkspaceOptions::comm`) this is a collective: every rank calls it
/// with the same X and Y (identical bits), computes its share of the terms
/// (the expensive IMSRG(2) terms over its work items, the cheap ones and the
/// N7 terms on rank 0), and the shares are combined
/// (`detail::CombinePartials`), so every rank returns the same Z, bit for
/// bit. The result equals the serial one up to the reordering of sums.
/// @throws NotImplementedError for X or Y with a three-body part and an
///         IMSRG(2) term set (no silent drop), for the N8/N9 terms of the
///         full IMSRG(3), and for specs `CommutatorSpec` rejects.
/// @throws InputError for IMSRG(3) terms on a workspace without a three-body
///         space (`Workspace::three_body_context()` null).
/// Complexity: IMSRG(2): dominated by the 2B terms, O(sum over channels of
/// N^3) GEMMs; IMSRG(3): see the terms in imsrg3.h.
void Commutator(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                const CommutatorOptions& opts, Workspace& ws);

/// Allocating form of `commutator`: returns [X, Y] restricted to `opts.terms`.
op::Operator Commutator(const op::Operator& X, const op::Operator& Y,
                        const CommutatorOptions& opts, Workspace& ws);

/// The individual terms. Each ADDS its contribution to Z (imsrg++ semantics)
/// and computes every stored element of the part it writes (both triangles; no
/// final symmetrization step). Common preconditions (InvariantError): X, Y and
/// Z on the workspace's model space, Z with spec `CommutatorSpec(X.spec(),
/// Y.spec())` (or at least the part written allocated and the same
/// hermiticity). A term whose input parts are missing (particle_rank 1) adds
/// nothing. On a distributed workspace (more than one part, see
/// `detail::AssignWork`) the comm122, ladder (pp/hh, 221) and particle-hole
/// terms add only this part's share and nothing combines it; use `Commutator`
/// for the full result.
namespace terms {

/// Z0 += sum_ab (2 j_a + 1) n_a nbar_b (X_ab Y_ba - Y_ab X_ba). Adds nothing if
/// Z is anti-Hermitian (the term vanishes; imsrg++ returns early); a general Z
/// gets the full sum. O(norb^2).
void Comm110ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z0 += 1/4 sum_abcd sum_J (2J+1) n_a n_b nbar_c nbar_d (X^J_abcd Y^J_cdab -
/// Y^J_abcd X^J_cdab); on normalized kets per channel: sum_J (2J+1) sum_{i,k}
/// nn_i nbarnbar_k (X_ik Y_ki - Y_ik X_ki). Adds nothing if Z is
/// anti-Hermitian (a general Z gets the full sum). O(sum over channels of N^2).
void Comm220ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z_ij += sum_a (X_ia Y_aj - Y_ia X_aj). O(norb^3).
void Comm111ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z_ij += sum_ab sum_J (2J+1)/(2 j_i + 1) (n_a - n_b) (X_ab Y^J_biaj - Y_ab
/// X^J_biaj), for i, j with equal (l, 2j, 2tz). O(E norb c Jr): E = norb c
/// one-body-channel entries (i, j), the sum over a runs over norb orbits, b
/// over the c orbits of a's one-body channel, J over Jr <= jmax + 1 values; X1
/// Y2 and X2 Y1 are each evaluated only if the two-body operand exists.
void Comm121ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z_ij += 1/2 sum_abc sum_J (2J+1)/(2 j_i + 1) (n_a n_b nbar_c + nbar_a nbar_b
/// n_c)
///         (X^J_ciab Y^J_abcj - Y^J_ciab X^J_abcj),
/// evaluated from the ladder intermediates Mpp = X P_pp Y - Y P_pp X, Mhh = X
/// P_hh Y - Y P_hh X (P = diag(nbar_a nbar_b), diag(n_a n_b) over the kets of a
/// channel) as Z_ij += sum_c sum_J (2J+1)/(2 j_i + 1) (n_c Mpp^J_cicj + nbar_c
/// Mhh^J_cicj) (unnormalized M). O(sum over channels of N^3).
void Comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z^J_ijkl += sum_a (X_ia Y^J_ajkl + X_ja Y^J_iakl - Y^J_ijal X_ak - Y^J_ijka
/// X_al) - (X <-> Y). O(sum over channels of N^2 * orbits per one-body
/// channel).
void Comm122ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               Workspace& ws);

/// Z^J_ijkl += 1/2 sum_ab (nbar_a nbar_b - n_a n_b) (X^J_ijab Y^J_abkl -
/// Y^J_ijab X^J_abkl), i.e. Z2 += Mpp - Mhh per channel (normalized kets).
/// O(sum over channels of N^3).
void Comm222PpHhss(const op::Operator& X, const op::Operator& Y,
                   op::Operator& Z, Workspace& ws);

/// The particle-hole term (imsrg++ `comm222_phss`):
///   Z^J_ijkl += sum_J' (2J'+1) [ {j_i j_j J; j_k j_l J'} Zbar^J'_ilkj
///                               - (-1)^(j_i + j_j - J) {j_j j_i J; j_k j_l J'}
///                                 Zbar^J'_jlki ],
///   Zbar^J'_ilkj = sum_ab (n_a - n_b) (Xbar^J'_ilab Ybar^J'_abkj -
///   Ybar^J'_ilab Xbar^J'_abkj), Xbar^J'_ilab = - sum_J'' (2J''+1) {j_i j_l J';
///   j_a j_b J''} X^J''_ibal (Pandya transform).
/// Evaluated with the Pandya transform into the cross-coupled channels, one
/// GEMM per channel and the inverse transform (design/12 section 4.2). O(sum
/// over cc channels of Ncc^2 Nph).
void Comm222Phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 Workspace& ws);

/// comm222_pp_hhss + comm221ss with shared intermediates (imsrg++
/// `Comm222PpHh221ss`). O(sum over channels of N^3), the cost of one ladder
/// pair (the 221 part is a lower-order contraction of the same intermediates).
void Comm222PpHh221ss(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z, Workspace& ws);

}  // namespace terms

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_COMMUTATOR_H_
