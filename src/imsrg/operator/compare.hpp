#pragma once
/// @file
/// Element-wise comparison of operators, for tests and golden comparisons (design/30 section 2:
/// tolerances of the form max|dZ| <= tol * max(1, max|Z|)).
#include "imsrg/operator/operator.hpp"

namespace imsrg::op {

/// Largest absolute values of the parts of an operator.
struct PartMaxima {
  double zero_body = 0.0;  ///< |O0|.
  double one_body = 0.0;   ///< max |O_ab| over all stored entries.
  double two_body = 0.0;   ///< max |V| over all stored entries (0 if not allocated).
  /// Maximum of the three.
  double max() const noexcept;
};

/// Largest absolute values of the parts of `op` (host computation after syncing to the host).
PartMaxima max_abs(const Operator& op);

/// Largest absolute differences, part by part, between `a` and `b`: |a0 - b0|, max |a_ab - b_ab|,
/// max |aV - bV| over the stored elements. Requirements (InvariantError): `same_model_space(a, b)`
/// and `a.spec().same_symmetry(b.spec())`; a part allocated in only one operator is compared
/// with zero. Hermiticity flags are not compared.
PartMaxima max_abs_difference(const Operator& a, const Operator& b);

}  // namespace imsrg::op
