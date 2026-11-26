// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_COMMUTATOR_REF_TENSOR_MSCHEME_SUPPORT_H_
#define TESTS_UNIT_COMMUTATOR_REF_TENSOR_MSCHEME_SUPPORT_H_

/// @file
/// Helpers of the tensor m-scheme tests (tests/unit/commutator_ref): random
/// tensor operators and the m-scheme scalar-tensor commutator of one
/// component.

#include <cstdint>
#include <memory>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator_ref/mscheme.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_spec.h"

namespace imsrg::testing {

/// One tensor kind of the m-scheme tests (a spec with two-body part).
struct TensorKind {
  int rank_J;
  Parity parity;
  int delta_Tz;
  Hermiticity hermiticity;
};

/// The spec of a `TensorKind` (particle_rank 2).
op::OperatorSpec SpecOf(const TensorKind& k);

/// A random tensor operator of kind `spec` (splitmix64 seeded with `seed`,
/// values uniform in [-1, 1)): every one-body element allowed by
/// `op::ConnectsOrbits` with its partner from the hermiticity rule O(b,a) = h
/// (-1)^((2j_a-2j_b)/2) O(a,b) (zero diagonal if anti-Hermitian), every stored
/// two-body block random, diagonal blocks h-symmetric (as the tensor tests of
/// the operator module build them). Synced to the device.
op::Operator RandomTensor(std::shared_ptr<const basis::ModelSpace> ms,
                          const op::OperatorSpec& spec, std::uint64_t seed);

/// The component `two_mu` of the commutator [X, Y] (scalar X, tensor Y) in the
/// m-scheme, converted back to reduced J-scheme elements with spec `z_spec`:
/// `Expand` X, `ExpandTensor` Y, the parts [nA, nB] -> nC with nA, nB, nC in
/// 1..2 with the output selection rule of Y's tensor shift (`TensorShift`),
/// `ContractTensor`. Pass `n_c_mask` bit (nC) to select the output ranks (1:
/// one-body, 2: two-body; default both) and `parts_mask` bit (2 (nA - 1) + nB
/// - 1) to select the input pairs ([1,1] 1, [1,2] 2, [2,1] 4, [2,2] 8).
/// Zero-body parts are not computed.
op::Operator MSchemeTensorCommutator(
    const op::Operator& X, const op::Operator& Y,
    const op::OperatorSpec& z_spec, TwoM two_mu,
    const commutator_ref::mscheme::TensorGauge& gauge = {},
    unsigned n_c_mask = 6U, unsigned parts_mask = 15U);

}  // namespace imsrg::testing

#endif  // TESTS_UNIT_COMMUTATOR_REF_TENSOR_MSCHEME_SUPPORT_H_
