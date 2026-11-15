// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_COMPARE_H_
#define IMSRG_OPERATOR_COMPARE_H_
/// @file
/// Element-wise comparison of operators, for tests and golden comparisons
/// (design/30 section 2: tolerances of the form max|dZ| <= tol * max(1,
/// max|Z|)).
#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// Largest absolute values of the parts of an operator.
struct PartMaxima {
  double zero_body = 0.0;  ///< |O0|.
  double one_body = 0.0;   ///< max |O_ab| over all stored entries.
  double two_body =
      0.0;  ///< max |V| over all stored entries (0 if not allocated).
  double three_body = 0.0;  ///< max over the stored three-body entries (0 if
                            ///< there is no three-body part).
  /// Maximum of the four.
  double max() const noexcept;
};

/// Largest absolute values of the parts of `op` (host computation after syncing
/// to the host).
PartMaxima MaxAbs(const Operator& op);

/// Largest absolute differences, part by part, between `a` and `b`: |a0 - b0|,
/// max |a_ab - b_ab|, max |aV - bV| over the stored elements, and the same over
/// the three-body elements (compared through the full matrices if the
/// hermiticities differ). Requirements (InvariantError): `SameModelSpace(a, b)`
/// and `a.spec().SameSymmetry(b.spec())`; a part allocated in only one operator
/// is compared with zero. Hermiticity flags are not compared.
PartMaxima MaxAbsDifference(const Operator& a, const Operator& b);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_COMPARE_H_
