// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_LINALG_BLAS_H_
#define IMSRG_LINALG_BLAS_H_
/// @file
/// Dense linear algebra on HOST memory (CPU backends): gemm, grouped gemm,
/// symmetric eigensolver, linear solve. All matrices are column-major (see
/// matrix_view.h). GPU backends will route the same API to vendor BLAS later
/// (design/12 section 3).
///
/// Threading rule (design/12 section 3): the BLAS library may be multithreaded;
/// calls made from inside a parallel region (OpenMP/Kokkos host threads) must
/// run with a single-threaded BLAS (e.g. OPENBLAS_NUM_THREADS=1 /
/// `openblas_set_num_threads(1)`; Accelerate manages this itself), otherwise
/// threads oversubscribe the cores. Only `GroupedGemm` spawns threads, and only
/// with Accelerate (see there); the other wrappers neither spawn nor control
/// threads. Dimension limit: the LP64 BLAS interface restricts every dimension
/// and leading dimension to < 2^31; violations throw InvariantError.

#include <span>
#include <vector>

#include "imsrg/linalg/matrix_view.h"

namespace imsrg::linalg {

/// C = alpha op(A) op(B) + beta C with op(X) = X or X^T (`Op`). Shapes: op(A)
/// is m x k, op(B) is k x n, C is m x n; mismatches, `ld < rows` and dimensions
/// >= 2^31 throw InvariantError. BLAS semantics: beta == 0 means C is not read
/// (NaN/Inf in C do not propagate). C must not overlap A or B. Zero-sized
/// operands are allowed (k == 0 gives C = beta C). Complexity O(m n k).
void Gemm(double alpha, MatrixView<const double> A, Op opA,
          MatrixView<const double> B, Op opB, double beta,
          MatrixView<double> C);

/// Single-precision overload of `gemm`.
void Gemm(float alpha, MatrixView<const float> A, Op opA,
          MatrixView<const float> B, Op opB, float beta, MatrixView<float> C);

/// One gemm of a group: C = alpha op(A) op(B) + beta C (double precision).
struct GemmProblem {
  double alpha = 1.0;
  MatrixView<const double> A;
  Op opA = Op::kNoTrans;
  MatrixView<const double> B;
  Op opB = Op::kNoTrans;
  double beta = 0.0;
  MatrixView<double> C;
};

/// Runs a group of independent gemms (e.g. one per block / channel).
/// Preconditions: the C matrices are pairwise disjoint, and no A or B overlaps
/// another problem's C. All problems are checked first (throws InvariantError
/// before any gemm runs). With Accelerate (`IMSRG_BLAS_ACCELERATE`), if the
/// host execution space has several threads (needs an initialized
/// `exec::Runtime`) and the group has at least ~1 Mflop, the problems then run
/// concurrently, one BLAS call per item, the largest first (design/12 section
/// 3; M12a: at emax 8 the sequential loop left the other threads idle for a
/// third of the commutator); each C is written by exactly one BLAS call, so
/// the result does not depend on the scheduling. Other BLAS libraries run the
/// loop sequentially, each call with the library's own threading, until
/// single-threaded calls inside the parallel region are arranged for them
/// (M12-Q5). Complexity: sum of the individual costs.
void GroupedGemm(std::span<const GemmProblem> problems);

/// Eigen-decomposition of a real symmetric n x n matrix (LAPACK dsyevd). Only
/// the LOWER triangle of `A` is read. `eigenvalues` (size n) receives the
/// eigenvalues in ascending order; `eigenvectors` (n x n, must not alias A)
/// receives the orthonormal eigenvectors as columns, with LAPACK's sign
/// convention. Throws NumericalError if LAPACK does not converge and
/// InvariantError on shape mismatch. Complexity O(n^3).
void Eigh(MatrixView<const double> A, std::span<double> eigenvalues,
          MatrixView<double> eigenvectors);

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.
struct EighResult {
  std::vector<double> eigenvalues;
  Matrix<double> eigenvectors;
};

/// Allocating convenience overload of `eigh`.
EighResult Eigh(MatrixView<const double> A);

/// Solves A X = B for a general square A (LAPACK dgesv, LU with partial
/// pivoting): B <- A^-1 B. A is n x n and is not modified (a copy is
/// factorized); B is n x nrhs. Throws NumericalError if A is singular,
/// InvariantError on shape mismatch. Complexity O(n^3 + n^2 nrhs).
void Solve(MatrixView<const double> A, MatrixView<double> B);

}  // namespace imsrg::linalg

#endif  // IMSRG_LINALG_BLAS_H_
