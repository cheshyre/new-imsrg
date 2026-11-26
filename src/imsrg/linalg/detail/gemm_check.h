// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_LINALG_DETAIL_GEMM_CHECK_H_
#define IMSRG_LINALG_DETAIL_GEMM_CHECK_H_
/// @file
/// Shape validation shared by the host and the device GEMM wrappers.

#include "imsrg/linalg/blas.h"

namespace imsrg::linalg::detail {

/// Validates the shapes of one GEMM problem like `Gemm` (InvariantError on a
/// mismatch, a negative dimension, `ld < rows`, a null pointer with a nonzero
/// size, or a dimension >= 2^31) and returns its flop count 2 m n k.
double CheckGemmProblem(const GemmProblem& p);

}  // namespace imsrg::linalg::detail

#endif  // IMSRG_LINALG_DETAIL_GEMM_CHECK_H_
