#pragma once
/// @file
/// Linear algebra on operators: axpy, scaling and the arithmetic operators (design/10 section
/// 3.3). Whole parts are combined with flat device kernels (the blocks of two operators on the
/// same model space with the same symmetry have identical layouts).
#include "imsrg/operator/operator.hpp"

namespace imsrg::op {

/// y += a x, partwise (0B, 1B, 2B). Requirements, checked with InvariantError:
/// `same_model_space(x, y)`, `x.spec().same_symmetry(y.spec())`, and
/// x.spec().particle_rank <= y.spec().particle_rank (y must hold every part of x).
/// Hermiticity of y afterwards: `combined_hermiticity(y, x)` (unchanged if a == 0).
/// Complexity O(size of y), one flat kernel per part. Returns y.
Operator& axpy(double a, const Operator& x, Operator& y);

/// x *= a for all parts. Complexity O(size of x). Returns x.
Operator& scale(Operator& x, double a);

/// y += x (axpy with a = 1).
Operator& operator+=(Operator& y, const Operator& x);
/// y -= x (axpy with a = -1).
Operator& operator-=(Operator& y, const Operator& x);
/// x *= a.
Operator& operator*=(Operator& x, double a);
/// x /= a (scale by 1 / a). Precondition: a != 0.
Operator& operator/=(Operator& x, double a);
/// x + y. The result has the larger particle_rank of the two (the operand with the larger rank
/// is copied, the other added); requirements as `axpy`.
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
/// O + c (adds c to the zero-body part; imsrg++ `operator+=(double)`).
Operator& operator+=(Operator& x, double c);
/// O - c (subtracts c from the zero-body part).
Operator& operator-=(Operator& x, double c);

}  // namespace imsrg::op
