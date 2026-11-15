// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_FACTORIZED_H_
#define IMSRG_COMMUTATOR_FACTORIZED_H_

/// @file
/// The factorized double commutator of IMSRG(3f2) (design/14 section 4 item 3,
/// owner decision D32): the part of [eta, [eta, gamma]_3b]_{1b,2b} that imsrg++
/// evaluates with one-body intermediates (`FactorizedDoubleCommutator.cc`
/// `comm223_231_chi1b` and `comm223_232_chi1b`), without storing the
/// three-body operator [eta, gamma]_3b. It is not a `TermMask` term: the BCH
/// series adds it to its nested commutators (`flow::BchOptions::factorized`,
/// imsrg++ `BCH::SetUseFactorizedCorrection`).
///
/// Notation as in `commutator.h` (unnormalized X~^J_abcd, n_a, nbar_a = 1 -
/// n_a; jhat_a^2 = 2 j_a + 1). Per two-body channel, over the normalized kets K
/// = (k, l), P_pp = diag(nbar_k nbar_l) and P_hh = diag(n_k n_l). The
/// intermediates are the ladder products
///   W_x = eta P_x eta,   V_x = gamma P_x eta   (x = pp, hh),
/// contracted like comm221ss, C[M; w]_de = jhat_d^-2 sum_c w_c sum_J (2J+1)
/// M~^J_cdce, into the one-body matrices
///   A_pp = C[W_pp; n], A_hh = C[W_hh; nbar], B_pp = C[V_pp; n],
///   B_hh = C[V_hh; nbar],
///   chi1 = A_pp + A_hh,   chi2 = B_pp + B_hh,
///   chia_de = 2 (n_d + n_e) A_pp,de - 2 (nbar_d + nbar_e) A_hh,de,
///   chib_de = 2 nbar_e B_hh,de - 2 n_e B_pp,de,
///   chic_de = chib_de + s chib_ed,   s = h_eta h_Z = -h_Z.
/// The result (h_Z = h_gamma) is
///   Z_pq += 1 / (2 jhat_p^2) sum_de sum_J (2J+1)
///           (chia_de gamma~^J_epdq + chic_de eta~^J_epdq)          [231: I,
///           IIIa, IIIb]
///   Z~^J_pqrs += sum_b (chi1_pb gamma~_bqrs + chi1_qb gamma~_pbrs
///                       + gamma~_pqbs chi1_br + gamma~_pqrb chi1_bs)
///              - sum_b (s chi2_bp eta~_bqrs + s chi2_bq eta~_pbrs
///                       + eta~_pqbs chi2_br + eta~_pqrb chi2_bs)
///                                               [232: Ia, Ib, IVa, IVb]
/// which is imsrg++'s expression (the GEMM form reorders its orbit sums; the
/// skip thresholds of imsrg++ on occupation products, below 1e-6, are not
/// applied).

#include "imsrg/commutator/workspace.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator {

/// Which topologies of the factorized double commutator are evaluated (pyIMSRG
/// `FactorizedDoubleCommutator::SetUse_1b_Intermediates` /
/// `SetUse_2b_Intermediates`). The defaults are the owner's IMSRG(3f2) recipe
/// (D32); imsrg++'s defaults are both true.
struct FactorizedOptions {
  /// Topologies with one-body intermediates (231: I, IIIa, IIIb; 232: Ia, Ib,
  /// IVa, IVb).
  bool one_body_intermediates = true;
  /// Topologies with two-body intermediates (imsrg++ `comm223_231_chi2b`,
  /// `comm223_232_chi2b`). Not implemented (D32 switches them off).
  bool two_body_intermediates = false;
};

/// Z = the factorized double commutator F(eta, gamma) restricted to the
/// topologies selected by `opts` (see the file comment for the expression).
/// Z is overwritten like the output of `Commutator`: reset to zero if it
/// already is a scalar rank-2 operator with gamma's hermiticity on the
/// workspace's model space, otherwise replaced by a new one. Z has no zero-body
/// part (it stays 0). Z = 0 if no topology is selected or if eta or gamma has
/// no two-body part (every topology needs eta_2 and gamma_2; one-body parts do
/// not enter).
/// Preconditions (InvariantError): eta and gamma on the workspace's model
/// space; Z is neither of them.
/// @throws NotImplementedError if eta or gamma is not scalar, number conserving
///         and of particle_rank <= 2, if eta is not anti-Hermitian or gamma is
///         NonHermitian (imsrg++ assumes both, design/reference/02 P-10), or if
///         `opts.two_body_intermediates` is set.
/// Complexity: two ladder-product pairs (four grouped GEMMs of sum over
/// channels N^2 (|Kpp| + |Khh|) flops), a comm221ss-like contraction per
/// intermediate, a comm121ss-like one-body pass and a comm122ss-like two-body
/// pass; about a third of an IMSRG(2) commutator.
void FactorizedDoubleCommutator(const op::Operator& eta,
                                const op::Operator& gamma, op::Operator& Z,
                                const FactorizedOptions& opts, Workspace& ws);

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_FACTORIZED_H_
