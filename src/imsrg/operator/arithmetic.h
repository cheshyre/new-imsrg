// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_ARITHMETIC_H_
#define IMSRG_OPERATOR_ARITHMETIC_H_
/// @file
/// Linear algebra on operators: axpy, scaling and the arithmetic operators
/// (design/10 section 3.3). Whole parts are combined with flat device kernels
/// (the blocks of two operators on the same model space with the same symmetry
/// have identical layouts). Results follow the memory model of `Operator`
/// (device data current; host mirrors stale on GPU backends until
/// `SyncToHost()`).
#include "imsrg/operator/operator.h"

namespace imsrg::op {

/// y += a x, partwise (0B, 1B, 2B, 3B). Requirements, checked with
/// InvariantError: `SameModelSpace(x, y)`, `x.spec().SameSymmetry(y.spec())`,
/// and x.spec().particle_rank <= y.spec().particle_rank (y must hold every part
/// of x). Hermiticity of y afterwards: `CombinedHermiticity(y, x)` (unchanged
/// if a == 0). A three-body part of x is added to that of y (a part missing in
/// x counts as zero); if the hermiticities differ, the three-body storage of
/// both goes to the dense non-Hermitian layout first (a copy of x's part).
/// Complexity O(size of y), one flat kernel per part. Returns y.
Operator& Axpy(double a, const Operator& x, Operator& y);

/// x *= a for all parts. Complexity O(size of x). Returns x.
Operator& Scale(Operator& x, double a);

/// y += x (axpy with a = 1).
Operator& operator+=(Operator& y, const Operator& x);
/// y -= x (axpy with a = -1).
Operator& operator-=(Operator& y, const Operator& x);
/// x *= a.
Operator& operator*=(Operator& x, double a);
/// x /= a (scale by 1 / a). Precondition: a != 0.
Operator& operator/=(Operator& x, double a);
/// x + y. The result has the larger particle_rank of the two (the operand with
/// the larger rank is copied, the other added); requirements as `axpy`.
Operator operator+(const Operator& x, const Operator& y);
/// x - y, as `operator+`.
Operator operator-(const Operator& x, const Operator& y);
/// -x.
Operator operator-(const Operator& x);
/// a x.
Operator operator*(double a, const Operator& x);
/// x a.
Operator operator*(const Operator& x, double a);
/// x / a. Precondition: a != 0.
Operator operator/(const Operator& x, double a);
/// O + c (adds c to the zero-body part; imsrg++ `operator+=(double)`). A
/// nonzero c makes an anti-Hermitian operator non-Hermitian.
Operator& operator+=(Operator& x, double c);
/// O - c (subtracts c from the zero-body part).
Operator& operator-=(Operator& x, double c);

}  // namespace imsrg::op

#endif  // IMSRG_OPERATOR_ARITHMETIC_H_
