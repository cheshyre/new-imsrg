// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_IMSRG3_H_
#define IMSRG_COMMUTATOR_IMSRG3_H_

/// @file
/// The seven IMSRG(3)-N7 commutator terms (imsrg++ `imsrg3_n7`,
/// IMSRG3Commutators.cc; design/14 section 4 item 2) on the layout-A
/// three-body storage (`op::ThreeBody`, design/00 D43).
///
/// Notation (m-scheme, sums over all single-particle states): n_a is the
/// reference occupation (possibly fractional), nbar_a = 1 - n_a; X_ab,
/// X_{ab,cd}, X_{abc,def} are the antisymmetrized one-, two- and three-body
/// elements of X = X1 + (1/4) sum X_{ab,cd} {a+ b+ d c} + (1/36) sum
/// X_{abc,def} {a+ b+ c+ f e d}. The J-coupled kernels evaluate the same
/// expressions with imsrg++'s conventions: two-body elements unnormalized
/// (`op::TwoBody::get_unnormalized`, imsrg++ `GetTBME_J`), three-body elements
/// imsrg++'s unnormalized antisymmetrized `GetME_pn(Jab, Jde, 2J, a, b, c, d,
/// e, f)` for any orbit order (recoupled from the canonical kets,
/// detail/n7_views.h), sums over orbit triples as sums over the kept canonical
/// kets with the multiplicity sym = 6, 3, 1 of the triple (distinct, two
/// equal, three equal orbits). Outputs are written in the stored convention:
/// normalized two-body elements (divide by sqrt((1 + d_ij)(1 + d_kl))),
/// canonical three-body elements.
///
/// Each function ADDS its contribution to Z (imsrg++ semantics) and computes
/// every stored element of the part it writes (both triangles of a two-body
/// block, the packed lower triangle or the dense block of a three-body block;
/// no final symmetrization). Operands: scalar, number-conserving X and Y on
/// the context's model space (or one with the same basis, reference and
/// partition); a missing part of X or Y (particle rank below what the term
/// reads) contributes zero. Any hermiticity of X and Y (definite or
/// NonHermitian); Z's hermiticity must be that of [X, Y] (`CommutatorSpec`)
/// and Z must hold the part the term writes. Three-body sums run over the
/// kept three-body space of the context's model space (kets failing E3max,
/// dE3max or OccNat3Cut do not exist, design/00 D34), so the configuration
/// cuts act identically in every term; imsrg++'s per-kernel cut bugs
/// (design/14 section 1) do not exist here.
///
/// Validation: element by element against imsrg-devel on random operators
/// (tests/golden/n7, refharness `comm3`) and against the m-scheme brute force
/// (`commutator_ref::mscheme`) at emax 1, including fractional occupations and
/// non-Hermitian operands.

#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::terms {

/// [X3, Y3] -> Z0 (imsrg++ `comm330ss`, IMSRG3Commutators.cc:72-186):
///   Z0 += 1/36 sum_{abc def} (n_a n_b n_c nbar_d nbar_e nbar_f
///         - nbar_a nbar_b nbar_c n_d n_e n_f) X_{abc,def} Y_{def,abc},
/// i.e. sum over channels (2J + 1) sum_{bra, ket} (sym_bra / 6) (sym_ket / 6)
/// (n_bra nbar_ket - nbar_bra n_ket) X[bra, ket] Y[ket, bra] over the kept
/// canonical rows. Adds nothing unless both have a three-body part. Cost
/// O(sum over channels N^2).
void Comm330ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X3, Y3] -> Z1 (imsrg++ `comm331ss`, IMSRG3Commutators.cc:199-411):
///   Z_ij += 1/12 sum_{ab cde} (n_a n_b nbar_c nbar_d nbar_e + nbar_a nbar_b
///           n_c n_d n_e) (X_{abi,cde} Y_{cde,abj} - Y_{abi,cde} X_{cde,abj}).
void Comm331ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X2, Y3] + [X3, Y2] -> Z1 (imsrg++ `comm231ss`, IMSRG3Commutators.cc:
/// 426-596): with [X2, Y3]_ij = 1/4 sum_{abcd} n_a n_b nbar_c nbar_d
/// (X_{ab,cd} Y_{cdi,abj} - Y_{abi,cdj} X_{cd,ab}),
///   Z_ij += [X2, Y3]_ij - [Y2, X3]_ij.
void Comm231ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X1, Y3] + [X3, Y1] -> Z2 (imsrg++ `comm132ss`, IMSRG3Commutators.cc:
/// 615-811):
///   Z_{ij,kl} += sum_{ab} (n_a - n_b) (X_ab Y_{ijb,kla} - Y_ab X_{ijb,kla}).
void Comm132ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X2, Y3] + [X3, Y2] -> Z2 (imsrg++ `comm232ss`, IMSRG3Commutators.cc:
/// 814-2059): the three-contraction part of the m-scheme Wick expansion
/// (sum over a, b, c with the occupation factor n_a n_b nbar_c + nbar_a
/// nbar_b n_c, antisymmetrized in (ij) and (kl)); see the kernel for the
/// J-coupled form.
void Comm232ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X1, Y3] + [X3, Y1] -> Z3 (imsrg++ `comm133ss`, IMSRG3Commutators.cc:
/// 4517-4989):
///   Z_{ijk,lmn} += sum_a (X_ia Y_{ajk,lmn} + X_ja Y_{iak,lmn} + X_ka
///                  Y_{ija,lmn} - Y_{ijk,amn} X_al - Y_{ijk,lan} X_am
///                  - Y_{ijk,lma} X_an) - (X <-> Y)
/// (no occupation factors: they add up to n + nbar = 1).
void Comm133ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

/// [X2, Y2] -> Z3 (imsrg++ `comm223ss`): adds `commutator::Comm223(X, Y,
/// ctx)` to Z's three-body part (formula in three_body.h).
void Comm223ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx);

}  // namespace imsrg::commutator::terms

#endif  // IMSRG_COMMUTATOR_IMSRG3_H_
