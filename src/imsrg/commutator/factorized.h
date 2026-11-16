// SPDX-License-Identifier: GPL-3.0-only
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
/// applied, design/30 section 6). Like imsrg++, no three-body truncation
/// (E3max, dE3max) is applied to the implied [eta, gamma]_3b.

#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/device_buffer.h"
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

/// The eta-only intermediates chi1 and chia (from W_x = eta P_x eta) of the
/// factorized double commutator, for evaluating F(eta, gamma) with one eta
/// and several gamma: the BCH series needs F(Omega, nested_{i-2}) for every
/// nested term with the same Omega, and these intermediates are half of the
/// GEMM work of each F. An explicit value, not a cache inside the Workspace
/// (design/00 D10). Keeps a pointer to eta, which must outlive the object
/// unchanged. Movable, not copyable.
class FactorizedEta {
 public:
  /// Checks eta as `FactorizedDoubleCommutator` does and computes chi1 and
  /// chia with `ws` (both zero if eta has no two-body part). Complexity: one
  /// ladder-product pair and a comm221ss-like contraction.
  /// @throws InvariantError if eta is not on the workspace's model space,
  ///         NotImplementedError for an eta `FactorizedDoubleCommutator`
  ///         rejects.
  FactorizedEta(const op::Operator& eta, Workspace& ws);
  ~FactorizedEta() = default;
  FactorizedEta(FactorizedEta&&) noexcept = default;
  FactorizedEta& operator=(FactorizedEta&&) noexcept = default;
  FactorizedEta(const FactorizedEta&) = delete;
  FactorizedEta& operator=(const FactorizedEta&) = delete;

  /// The operator the intermediates belong to.
  const op::Operator& eta() const noexcept { return *eta_; }
  /// chi1 = A_pp + A_hh: dense column-major num_orbits x num_orbits device
  /// matrix (only entries of equal (l, 2j, 2tz) are non-zero).
  const exec::DeviceBuffer<double>& chi1() const noexcept { return chi1_; }
  /// chia, same layout.
  const exec::DeviceBuffer<double>& chia() const noexcept { return chia_; }

 private:
  const op::Operator* eta_;
  exec::DeviceBuffer<double> chi1_;
  exec::DeviceBuffer<double> chia_;
};

/// Z = the factorized double commutator F(eta, gamma) restricted to the
/// topologies selected by `opts` (see the file comment for the expression).
/// Z is overwritten like the output of `Commutator`: reset to zero if it
/// already is a scalar rank-2 operator with gamma's hermiticity on the
/// workspace's model space, otherwise replaced by a new one, so Z always has
/// particle_rank 2 and gamma's hermiticity; its zero-body part stays 0. Z = 0
/// if no topology is selected or if eta or gamma has no two-body part (every
/// topology needs eta_2 and gamma_2; one-body parts do not enter).
/// Preconditions (InvariantError): eta and gamma on the workspace's model
/// space; Z is neither of them.
/// @throws NotImplementedError if eta or gamma is not scalar, number conserving
///         and of particle_rank <= 2, if eta is not anti-Hermitian or gamma is
///         NonHermitian (imsrg++ assumes both, design/reference/02 P-10), or if
///         `opts.two_body_intermediates` is set.
/// Complexity: two ladder-product pairs (four grouped GEMMs of sum over
/// channels N^2 (|Kpp| + |Khh|) flops), a comm221ss-like contraction per
/// pair, a comm121ss-like one-body pass and a comm122ss-like two-body pass;
/// measured at about half of an IMSRG(2) commutator (O16 emax 4).
/// Replicated, not distributed: with a non-serial Workspace communicator every
/// rank must call it with identical inputs (and computes the full result).
void FactorizedDoubleCommutator(const op::Operator& eta,
                                const op::Operator& gamma, op::Operator& Z,
                                const FactorizedOptions& opts, Workspace& ws);

/// The same with eta's intermediates precomputed (`eta.eta()` is eta): saves
/// one ladder-product pair and one contraction per call. Same results to the
/// last bit, same preconditions and exceptions (Z must not alias `eta.eta()`).
void FactorizedDoubleCommutator(const FactorizedEta& eta,
                                const op::Operator& gamma, op::Operator& Z,
                                const FactorizedOptions& opts, Workspace& ws);

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_FACTORIZED_H_
