// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_REF_COMMUTATOR_REF_H_
#define IMSRG_COMMUTATOR_REF_COMMUTATOR_REF_H_

/// @file
/// Slow, transparent reference implementations of the scalar IMSRG(2)
/// commutator terms (design/10 section 2: tests only, never linked into
/// production targets; design/30 section 1). They mirror imsrg++'s
/// `ReferenceImplementations/IMSRG2Commutators.cc`: plain loops over orbits and
/// J with unnormalized matrix elements (`op::TwoBody::get_unnormalized`) and 6j
/// symbols, no GEMMs, no symmetry tricks. Unlike imsrg++'s versions they
/// compute EVERY stored element of Z (both triangles) directly instead of
/// mirroring the upper triangle, so they are valid for any hermiticity of X and
/// Y, including NonHermitian.
///
/// The formulas are those documented in `imsrg/commutator/commutator.h` (same
/// names, same conventions). Every function ADDS its contribution to Z;
/// host-only code (it syncs X, Y and Z to the host before reading and Z to the
/// device after writing).

#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::commutator_ref {

/// Spec of [X, Y]: scalar, particle_rank max of the two, hermiticity
/// AntiHermitian for equal definite hermiticities, Hermitian for opposite ones,
/// NonHermitian otherwise.
/// @throws NotImplementedError for non-scalar specs or particle_rank 3.
op::OperatorSpec CommutatorSpec(const op::OperatorSpec& x,
                                const op::OperatorSpec& y);

/// Z0 += sum_ab (2 j_a + 1) n_a nbar_b (X_ab Y_ba - Y_ab X_ba). O(norb^2).
void Comm110ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z0 += 1/4 sum_abcd sum_J (2J+1) n_a n_b nbar_c nbar_d (X^J_abcd Y^J_cdab -
/// Y^J_abcd X^J_cdab). O(norb^4 J).
void Comm220ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z_ij += sum_a (X_ia Y_aj - Y_ia X_aj). O(norb^3).
void Comm111ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z_ij += sum_ab sum_J (2J+1)/(2 j_i + 1) (n_a - n_b) (X_ab Y^J_biaj - Y_ab
/// X^J_biaj).
void Comm121ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z_ij += 1/2 sum_abc sum_J (2J+1)/(2 j_i + 1) (n_a n_b nbar_c + nbar_a nbar_b
/// n_c)
///         (X^J_ciab Y^J_abcj - Y^J_ciab X^J_abcj).
void Comm221ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z^J_ijkl += sum_a (X_ia Y^J_ajkl + X_ja Y^J_iakl - Y^J_ijal X_ak - Y^J_ijka
/// X_al) - (X <-> Y).
void Comm122ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// Z^J_ijkl += 1/2 sum_ab (nbar_a nbar_b - n_a n_b) (X^J_ijab Y^J_abkl -
/// Y^J_ijab X^J_abkl).
void Comm222PpHhss(const op::Operator& X, const op::Operator& Y,
                   op::Operator& Z);

/// The particle-hole term, see `commutator::terms::Comm222Phss`
/// (Pandya-transformed matrix elements computed explicitly from 6j symbols).
void Comm222Phss(const op::Operator& X, const op::Operator& Y, op::Operator& Z);

/// comm222_pp_hhss followed by comm221ss.
void Comm222PpHh221ss(const op::Operator& X, const op::Operator& Y,
                      op::Operator& Z);

/// [X, Y] with all IMSRG(2) terms (a new operator with spec `CommutatorSpec`);
/// terms whose input parts are missing (particle_rank 1) are skipped.
op::Operator Commutator(const op::Operator& X, const op::Operator& Y);

}  // namespace imsrg::commutator_ref

#endif  // IMSRG_COMMUTATOR_REF_COMMUTATOR_REF_H_
