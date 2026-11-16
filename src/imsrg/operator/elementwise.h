// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_ELEMENTWISE_H_
#define IMSRG_OPERATOR_ELEMENTWISE_H_
/// @file
/// Element-wise operations on all stored matrix elements of an operator (zero-,
/// one- and two-body parts), as imsrg++ defines them for its adaptive ODE
/// stepper (IMSRGSolver.cc:682-760: `abs`, `operator/`, `double + deque`).
/// They are not algebra of operators (the result is a table of numbers, not an
/// operator of the stated symmetry), so every result is marked non-Hermitian.
/// The elements of the one-body part are those of its dense n x n buffer, as
/// imsrg++'s arma matrix; the norms only read the symmetry-allowed ones.
#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// x_k = |x_k| for every stored element. O(size of x), one kernel per part.
void ElementwiseAbs(Operator& x);

/// num_k /= denom_k for every stored element (IEEE division; 0/0 gives NaN, so
/// callers add a positive offset first). Requirements (InvariantError):
/// `SameModelSpace`, same symmetry and particle_rank, identical layouts.
/// O(size of num).
void ElementwiseDivide(Operator& num, const Operator& denom);

/// x_k += c for every stored element, the zero-body part included (imsrg++
/// `double + Operator` on the ODE state; NOT `operator+=(Operator&, double)`,
/// which only changes E0). O(size of x).
void AddToEveryElement(Operator& x, double c);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_ELEMENTWISE_H_
