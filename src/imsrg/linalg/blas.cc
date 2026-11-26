// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/linalg/blas.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/exec/host_parallel.h"
#include "imsrg/linalg/detail/blas.h"
#include "imsrg/linalg/detail/gemm_check.h"

namespace imsrg::linalg {
namespace detail {

int SyevdLower(int n, double* a, int lda, double* w) {
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  double wq = 0.0;
  int iwq = 0;
  const BlasCallGuard guard;  // held for both dsyevd calls
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  char jobz = 'V';
  char uplo = 'L';
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &iwq, &liwork, &info);
  if (info != 0) return info;
  lwork = static_cast<int>(wq);
  liwork = iwq;
  std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));
  std::vector<int> iwork(static_cast<std::size_t>(std::max(liwork, 1)));
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(),
          &liwork, &info);
#else
  const char jobz = 'V';
  const char uplo = 'L';
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &iwq, &liwork, &info);
  if (info != 0) return info;
  lwork = static_cast<int>(wq);
  liwork = iwq;
  std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));
  std::vector<int> iwork(static_cast<std::size_t>(std::max(liwork, 1)));
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(),
          &liwork, &info);
#endif
  return info;
}

}  // namespace detail

namespace {

constexpr std::int64_t kBlasLimit = std::numeric_limits<std::int32_t>::max();

template <class V>
void CheckView(const V& v, const char* name) {
  IMSRG_CHECK(v.rows >= 0 && v.cols >= 0,
              std::string(name) + ": negative dimension");
  IMSRG_CHECK(v.ld >= v.rows, std::string(name) + ": ld < rows");
  IMSRG_CHECK(
      v.rows <= kBlasLimit && v.cols <= kBlasLimit && v.ld <= kBlasLimit,
      std::string(name) + ": dimension exceeds the 2^31 BLAS limit");
  IMSRG_CHECK(v.rows == 0 || v.cols == 0 || v.data != nullptr,
              std::string(name) + ": null data");
}

int Lead(std::int64_t ld) {
  return static_cast<int>(std::max<std::int64_t>(ld, 1));
}

/// Checks the views and shapes of C = alpha op(A) op(B) + beta C (throws
/// InvariantError) and returns the flop count 2 m n k. No BLAS call.
template <class T>
double CheckGemm(MatrixView<const T> A, Op opA, MatrixView<const T> B, Op opB,
                 MatrixView<T> C) {
  CheckView(A, "gemm A");
  CheckView(B, "gemm B");
  CheckView(C, "gemm C");
  const bool ta = opA == Op::kTrans;
  const bool tb = opB == Op::kTrans;
  const std::int64_t m = ta ? A.cols : A.rows;
  const std::int64_t k = ta ? A.rows : A.cols;
  const std::int64_t kb = tb ? B.cols : B.rows;
  const std::int64_t n = tb ? B.rows : B.cols;
  IMSRG_CHECK(k == kb, "gemm: inner dimensions of op(A) and op(B) differ");
  IMSRG_CHECK(C.rows == m && C.cols == n,
              "gemm: C shape does not match op(A) op(B)");
  return 2.0 * static_cast<double>(m) * static_cast<double>(n) *
         static_cast<double>(k);
}

/// The gemm itself on views already checked by `CheckGemm` (does not throw).
template <class T>
void GemmChecked(T alpha, MatrixView<const T> A, Op opA, MatrixView<const T> B,
                 Op opB, T beta, MatrixView<T> C) noexcept {
  const bool ta = opA == Op::kTrans;
  const bool tb = opB == Op::kTrans;
  const std::int64_t m = ta ? A.cols : A.rows;
  const std::int64_t k = ta ? A.rows : A.cols;
  const std::int64_t n = tb ? B.rows : B.cols;
  if (m == 0 || n == 0) return;
  if (k == 0) {
    // Pure scaling of C; beta == 0 must not read C.
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < m; ++i)
        C(i, j) = (beta == T{0}) ? T{0} : beta * C(i, j);
    return;
  }
  detail::GemmRaw(ta, tb, static_cast<int>(m), static_cast<int>(n),
                  static_cast<int>(k), alpha, A.data, Lead(A.ld), B.data,
                  Lead(B.ld), beta, C.data, Lead(C.ld));
}

template <class T>
void GemmImpl(T alpha, MatrixView<const T> A, Op opA, MatrixView<const T> B,
              Op opB, T beta, MatrixView<T> C) {
  CheckGemm<T>(A, opA, B, opB, C);
  GemmChecked<T>(alpha, A, opA, B, opB, beta, C);
}

/// Groups with fewer total flops run sequentially (thread start-up costs
/// more than it saves).
constexpr double kConcurrentGemmMinFlops = 1.0e6;

/// Whether `GroupedGemm` may run its problems concurrently
/// (`IMSRG_BLAS_CONCURRENT`): only with a reentrant BLAS whose calls stay
/// single-threaded inside a parallel region, i.e. Accelerate (default) or the
/// serial cray-libsci `libsci_gnu` (M14). The threaded `libsci_gnu_mp` is not
/// reentrant (FP-Q6); OpenBLAS (pthreads) or MKL would need single-threaded
/// calls per item first (M12-Q5). Never when the calls are serialized anyway
/// (`IMSRG_BLAS_SERIALIZE`).
#if defined(IMSRG_BLAS_CONCURRENT) && IMSRG_BLAS_CONCURRENT && \
    !(defined(IMSRG_BLAS_SERIALIZE) && IMSRG_BLAS_SERIALIZE)
constexpr bool kConcurrentGroupedGemm = true;
#else
constexpr bool kConcurrentGroupedGemm = false;
#endif

}  // namespace

namespace detail {

double CheckGemmProblem(const GemmProblem& p) {
  return CheckGemm<double>(p.A, p.opA, p.B, p.opB, p.C);
}

}  // namespace detail

void Gemm(double alpha, MatrixView<const double> A, Op opA,
          MatrixView<const double> B, Op opB, double beta,
          MatrixView<double> C) {
  GemmImpl<double>(alpha, A, opA, B, opB, beta, C);
}

void Gemm(float alpha, MatrixView<const float> A, Op opA,
          MatrixView<const float> B, Op opB, float beta, MatrixView<float> C) {
  GemmImpl<float>(alpha, A, opA, B, opB, beta, C);
}

void GroupedGemm(std::span<const GemmProblem> problems) {
  // Validate everything first: nothing may throw inside the parallel region.
  std::vector<double> flops(problems.size());
  double total = 0.0;
  for (std::size_t i = 0; i < problems.size(); ++i) {
    const GemmProblem& p = problems[i];
    flops[i] = CheckGemm<double>(p.A, p.opA, p.B, p.opB, p.C);
    total += flops[i];
  }
  if (!kConcurrentGroupedGemm || problems.size() < 2 ||
      total < kConcurrentGemmMinFlops || exec::HostConcurrency() < 2) {
    for (const GemmProblem& p : problems)
      GemmChecked<double>(p.alpha, p.A, p.opA, p.B, p.opB, p.beta, p.C);
    return;
  }
  // One problem per item, the largest first (dynamic scheduling balances the
  // rest). The C matrices are disjoint, and each gemm is one BLAS call made
  // inside the parallel region, so the results do not depend on the order.
  std::vector<std::size_t> order(problems.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::size_t a, std::size_t b) { return flops[a] > flops[b]; });
  exec::HostParallelForDynamic(
      "linalg::grouped_gemm", static_cast<std::int64_t>(order.size()),
      [&](std::int64_t i) {
        const GemmProblem& p = problems[order[static_cast<std::size_t>(i)]];
        GemmChecked<double>(p.alpha, p.A, p.opA, p.B, p.opB, p.beta, p.C);
      });
}

void Eigh(MatrixView<const double> A, std::span<double> eigenvalues,
          MatrixView<double> eigenvectors) {
  CheckView(A, "eigh A");
  CheckView(eigenvectors, "eigh eigenvectors");
  const std::int64_t n = A.rows;
  IMSRG_CHECK(A.cols == n, "eigh: A must be square");
  IMSRG_CHECK(eigenvectors.rows == n && eigenvectors.cols == n,
              "eigh: eigenvectors must be n x n");
  IMSRG_CHECK(static_cast<std::int64_t>(eigenvalues.size()) == n,
              "eigh: eigenvalues size != n");
  if (n == 0) return;
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = j; i < n; ++i) eigenvectors(i, j) = A(i, j);
  const int info =
      detail::SyevdLower(static_cast<int>(n), eigenvectors.data,
                         Lead(eigenvectors.ld), eigenvalues.data());
  if (info != 0)
    throw NumericalError("eigh: LAPACK dsyevd failed to converge (info = " +
                         std::to_string(info) + ")");
}

EighResult Eigh(MatrixView<const double> A) {
  IMSRG_CHECK(A.rows >= 0, "eigh: negative dimension");
  EighResult r{std::vector<double>(static_cast<std::size_t>(A.rows)),
               Matrix<double>(A.rows, A.rows)};
  Eigh(A, std::span<double>(r.eigenvalues), r.eigenvectors.view());
  return r;
}

void Solve(MatrixView<const double> A, MatrixView<double> B) {
  CheckView(A, "solve A");
  CheckView(B, "solve B");
  const std::int64_t n = A.rows;
  IMSRG_CHECK(A.cols == n, "solve: A must be square");
  IMSRG_CHECK(B.rows == n, "solve: B must have n rows");
  if (n == 0 || B.cols == 0) return;
  Matrix<double> lu(n, n);
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < n; ++i) lu(i, j) = A(i, j);
  const int info =
      detail::Gesv(static_cast<int>(n), static_cast<int>(B.cols),
                   lu.view().data, static_cast<int>(n), B.data, Lead(B.ld));
  if (info != 0)
    throw NumericalError("solve: matrix is singular (zero pivot " +
                         std::to_string(info) + ")");
}

}  // namespace imsrg::linalg
