// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_TRUNCATE_H_
#define IMSRG_OPERATOR_TRUNCATE_H_
/// @file
/// Restriction of an operator to a smaller model space (imsrg++
/// `Operator::Truncate`, Operator.cc:816-876; the emax -> emax_imsrg step of
/// the 27Al pipeline, design/50 C5, reference/06 section 1.5).
#include <memory>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// The operator `O` restricted to the model space `target`: zero-body part
/// copied, one-body part = the submatrix of the orbits of `target`, two-body
/// part = the submatrix of the kets of `target` (normalized elements copied).
/// Orbits are matched by their quantum numbers (n, l, 2j, 2tz), so the result
/// keeps the labels of the source basis (e.g. the natural-orbital labels after
/// a |occupation| re-sorting). No re-normal-ordering: normal-ordered pieces
/// are cut as they are (imsrg++ does the same); the reference of `target` is
/// only attached. Spec and hermiticity are copied.
/// Preconditions: `target` uses the same hw and every orbit of `target` exists
/// in `O`'s basis (InputError otherwise); particle rank <= 2 and, for a
/// two-body part, a scalar operator (NotImplementedError otherwise: tensor
/// storage and 3B truncation come later). Host loops; O(norb_t^2 + number of
/// two-body elements of `target`).
Operator Truncate(const Operator& O,
                  std::shared_ptr<const basis::ModelSpace> target);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_TRUNCATE_H_
