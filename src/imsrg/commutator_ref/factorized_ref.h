// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_REF_FACTORIZED_REF_H_
#define IMSRG_COMMUTATOR_REF_FACTORIZED_REF_H_

/// @file
/// Slow reference implementation of imsrg++'s factorized double commutator
/// with one-body intermediates only (D32: `SetUse_1b_Intermediates(true)`,
/// `SetUse_2b_Intermediates(false)`), a transcription of
/// `FactorizedDoubleCommutator.cc` `comm223_231_chi1b` and
/// `comm223_232_chi1b` (scalar, channel-diagonal path). It is the independent
/// check of the fast `commutator` implementation (design/30 section 1); tests
/// only, never linked into production targets.
///
/// The result is the factorized part of [eta, [eta, gamma]_3b]_{1b,2b}. Each
/// function ADDS to Z. Conventions: eta anti-Hermitian, gamma Hermitian or
/// anti-Hermitian (the only cases imsrg++ supports; otherwise
/// `NotImplementedError`), Z carries gamma's hermiticity (otherwise
/// `InvariantError`); scalar, number-conserving, particle_rank <= 2 operators
/// on one model space; a missing two-body part of eta or gamma adds nothing.
/// Host-only code (inputs are read from the host mirror, Z is synchronized to
/// the device at the end).

#include "imsrg/operator/operator.h"

namespace imsrg::commutator_ref {

/// imsrg++ `comm223_231_chi1b`: adds the one-body part of the factorized
/// double commutator (diagram I, then diagrams IIIa + IIIb) to Z. Cost
/// O(norb^3 J + nch nk^3).
/// @throws NotImplementedError, InvariantError as described in the file
/// comment.
void Factorized231Chi1b(const op::Operator& eta, const op::Operator& gamma,
                        op::Operator& Z);

/// imsrg++ `comm223_232_chi1b`, scalar path: adds the two-body part of the
/// factorized double commutator (the intermediates CHI_I, CHI_II contracted
/// with gamma and eta on each of the four legs) to Z. Cost O(norb^4 J + nz
/// norb) for nz two-body elements.
/// @throws NotImplementedError, InvariantError as described in the file
/// comment.
void Factorized232Chi1b(const op::Operator& eta, const op::Operator& gamma,
                        op::Operator& Z);

/// `Factorized231Chi1b` followed by `Factorized232Chi1b`: the D32 correction
/// (one-body intermediates on, two-body intermediates off), i.e. imsrg++'s
/// `comm223_231` + `comm223_232` with `SetUse_1b_Intermediates(true)` and
/// `SetUse_2b_Intermediates(false)`.
void FactorizedDoubleCommutator(const op::Operator& eta,
                                const op::Operator& gamma, op::Operator& Z);

}  // namespace imsrg::commutator_ref

#endif  // IMSRG_COMMUTATOR_REF_FACTORIZED_REF_H_
