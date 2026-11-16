// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_SUPPORT_THREE_BODY_TEST_SUPPORT_H_
#define TESTS_SUPPORT_THREE_BODY_TEST_SUPPORT_H_

/// @file
/// Test helpers for the IMSRG(3) terms (design/14 sections 3 and 4): model
/// spaces with a three-body space and random particle-rank-3 operators whose
/// three-body part is PHYSICAL, i.e. the canonical matrix elements of an
/// actual antisymmetric three-body operator. Layout A stores every allowed Jab
/// of a triple, so the canonical kets of a triple with equal orbits are
/// linearly dependent; uniformly random canonical values would not belong to
/// any operator, and the J-scheme kernels and the m-scheme brute force would
/// legitimately disagree on them. The three-body part is therefore the
/// [2,2]->3 term (`commutator::Comm223`, validated against imsrg-devel) of two
/// random two-body operators, which is physical by construction.

#include <cstdint>
#include <memory>
#include <string_view>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/operator/operator.h"

namespace imsrg::testing {

/// Model space with hw = 16 MeV, the HO basis e <= emax, the naive filling of
/// nucleus `reference` (valence space = reference, i.e. none) and the
/// three-body space of `cuts`.
std::shared_ptr<const basis::ModelSpace> MakeThreeBodySpace(
    int emax, std::string_view reference, const basis::ThreeBodyCuts& cuts);

/// `ms` (a model space without three-body space, e.g. from
/// `FractionalModelSpace`) with the three-body space of `cuts` added.
std::shared_ptr<const basis::ModelSpace> AddThreeBodySpace(
    const std::shared_ptr<const basis::ModelSpace>& ms,
    const basis::ThreeBodyCuts& cuts);

/// A random scalar particle-rank-3 operator on the context's model space with
/// hermiticity `h`:
/// * zero-body part 0; one- and two-body parts those of
///   `op::RandomOperator(ms, Scalar(2, h), {seed, 0, 1, 1})`;
/// * three-body part `three_body_scale` times Comm223(A, B) of two random
///   two-body operators A, B (seeds 1000 + 2 seed and 1001 + 2 seed, scales 1)
///   whose hermiticities give h: A Hermitian and B Hermitian for an
///   anti-Hermitian h, A anti-Hermitian and B Hermitian for a Hermitian h, A
///   non-Hermitian and B Hermitian for a non-Hermitian h.
/// Deterministic for a given seed up to the summation order of `Comm223`
/// (last-bit differences between thread counts). Synced to the device.
op::Operator RandomThreeBodyOperator(const commutator::ThreeBodyContext& ctx,
                                     Hermiticity h, std::uint64_t seed,
                                     double three_body_scale = 1.0);

}  // namespace imsrg::testing

#endif  // TESTS_SUPPORT_THREE_BODY_TEST_SUPPORT_H_
