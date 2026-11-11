#include "imsrg/linalg/blas.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/linalg/detail/blas.hpp"

namespace imsrg::linalg {
namespace {

constexpr std::int64_t kBlasLimit = std::numeric_limits<std::int32_t>::max();

template <class V>
void check_view(const V& v, const char* name) {
  IMSRG_CHECK(v.rows >= 0 && v.cols >= 0, std::string(name) + ": negative dimension");
  IMSRG_CHECK(v.ld >= v.rows, std::string(name) + ": ld < rows");
  IMSRG_CHECK(v.rows <= kBlasLimit && v.cols <= kBlasLimit && v.ld <= kBlasLimit,
              std::string(name) + ": dimension exceeds the 2^31 BLAS limit");
  IMSRG_CHECK(v.rows == 0 || v.cols == 0 || v.data != nullptr, std::string(name) + ": null data");
}

int lead(std::int64_t ld) { return static_cast<int>(std::max<std::int64_t>(ld, 1)); }

template <class T>
void gemm_impl(T alpha, MatrixView<const T> A, Op opA, MatrixView<const T> B, Op opB, T beta,
               MatrixView<T> C) {
  check_view(A, "gemm A");
  check_view(B, "gemm B");
  check_view(C, "gemm C");
  const bool ta = opA == Op::Trans;
  const bool tb = opB == Op::Trans;
  const std::int64_t m = ta ? A.cols : A.rows;
  const std::int64_t k = ta ? A.rows : A.cols;
  const std::int64_t kb = tb ? B.cols : B.rows;
  const std::int64_t n = tb ? B.rows : B.cols;
  IMSRG_CHECK(k == kb, "gemm: inner dimensions of op(A) and op(B) differ");
  IMSRG_CHECK(C.rows == m && C.cols == n, "gemm: C shape does not match op(A) op(B)");
  if (m == 0 || n == 0) return;
  if (k == 0) {
    // Pure scaling of C; beta == 0 must not read C.
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < m; ++i) C(i, j) = (beta == T{0}) ? T{0} : beta * C(i, j);
    return;
  }
  detail::gemm_raw(ta, tb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha,
                   A.data, lead(A.ld), B.data, lead(B.ld), beta, C.data, lead(C.ld));
}

}  // namespace

void gemm(double alpha, MatrixView<const double> A, Op opA, MatrixView<const double> B, Op opB,
          double beta, MatrixView<double> C) {
  gemm_impl<double>(alpha, A, opA, B, opB, beta, C);
}

void gemm(float alpha, MatrixView<const float> A, Op opA, MatrixView<const float> B, Op opB,
          float beta, MatrixView<float> C) {
  gemm_impl<float>(alpha, A, opA, B, opB, beta, C);
}

void grouped_gemm(std::span<const GemmProblem> problems) {
  for (const GemmProblem& p : problems) gemm(p.alpha, p.A, p.opA, p.B, p.opB, p.beta, p.C);
}

void eigh(MatrixView<const double> A, std::span<double> eigenvalues,
          MatrixView<double> eigenvectors) {
  check_view(A, "eigh A");
  check_view(eigenvectors, "eigh eigenvectors");
  const std::int64_t n = A.rows;
  IMSRG_CHECK(A.cols == n, "eigh: A must be square");
  IMSRG_CHECK(eigenvectors.rows == n && eigenvectors.cols == n, "eigh: eigenvectors must be n x n");
  IMSRG_CHECK(static_cast<std::int64_t>(eigenvalues.size()) == n, "eigh: eigenvalues size != n");
  if (n == 0) return;
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = j; i < n; ++i) eigenvectors(i, j) = A(i, j);
  const int info = detail::syevd_lower(static_cast<int>(n), eigenvectors.data,
                                       lead(eigenvectors.ld), eigenvalues.data());
  if (info != 0)
    throw NumericalError("eigh: LAPACK dsyevd failed to converge (info = " + std::to_string(info) +
                         ")");
}

EighResult eigh(MatrixView<const double> A) {
  IMSRG_CHECK(A.rows >= 0, "eigh: negative dimension");
  EighResult r{std::vector<double>(static_cast<std::size_t>(A.rows)),
               Matrix<double>(A.rows, A.rows)};
  eigh(A, std::span<double>(r.eigenvalues), r.eigenvectors.view());
  return r;
}

void solve(MatrixView<const double> A, MatrixView<double> B) {
  check_view(A, "solve A");
  check_view(B, "solve B");
  const std::int64_t n = A.rows;
  IMSRG_CHECK(A.cols == n, "solve: A must be square");
  IMSRG_CHECK(B.rows == n, "solve: B must have n rows");
  if (n == 0 || B.cols == 0) return;
  Matrix<double> lu(n, n);
  for (std::int64_t j = 0; j < n; ++j)
    for (std::int64_t i = 0; i < n; ++i) lu(i, j) = A(i, j);
  const int info = detail::gesv(static_cast<int>(n), static_cast<int>(B.cols), lu.view().data,
                                static_cast<int>(n), B.data, lead(B.ld));
  if (info != 0)
    throw NumericalError("solve: matrix is singular (zero pivot " + std::to_string(info) + ")");
}

}  // namespace imsrg::linalg
