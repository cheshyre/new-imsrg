// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_COMMUTATOR_TENSOR_COMM122_REF_H_
#define TESTS_UNIT_COMMUTATOR_TENSOR_COMM122_REF_H_

#include "imsrg/commutator/tensor.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator::test {

/// The element-wise comm122st kernel as it was before the lifted-list rewrite
/// (every matrix element through `TensorTwoBodyView::get_unnormalized`):
/// accumulates [X, Y]_{122st} into `Z` exactly as
/// `tensor_terms::Comm122st`, but with the slow lookups. Preconditions as
/// `Comm122st` (Z allocated with the commutator spec, not aliasing X or Y, all
/// on the workspace's model space). Kept as the reference implementation for
/// `test_tensor_comm122`.
void ReferenceComm122st(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, TensorWorkspace& ws);

}  // namespace imsrg::commutator::test

#endif  // TESTS_UNIT_COMMUTATOR_TENSOR_COMM122_REF_H_
