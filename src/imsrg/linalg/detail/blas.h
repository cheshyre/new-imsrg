#pragma once
/// @file
/// Private BLAS/LAPACK shim: one thin wrapper per routine, column-major, LP64 (32-bit int)
/// interface. Included only by linalg `.cpp` files. With Apple Accelerate (`IMSRG_BLAS_ACCELERATE`)
/// the CBLAS interface and the new LAPACK headers (`ACCELERATE_NEW_LAPACK`) are used; otherwise
/// the Fortran symbols are declared directly (OpenBLAS, MKL, reference LAPACK).

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
            const double* beta, double* c, const int* ldc);
void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
            const float* beta, float* c, const int* ldc);
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w,
             double* work, const int* lwork, int* iwork, const int* liwork, int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b,
            const int* ldb, int* info);
}
#endif

namespace imsrg::linalg::detail {

/// C = alpha op(A) op(B) + beta C, column-major; `trans_*` true means transposed. Dimensions are
/// already validated by the caller (m, n, k, ld* >= 1 and < 2^31).
inline void gemm_raw(bool trans_a, bool trans_b, int m, int n, int k, double alpha, const double* a,
                     int lda, const double* b, int ldb, double beta, double* c, int ldc) {
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  cblas_dgemm(CblasColMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  const char ta = trans_a ? 'T' : 'N';
  const char tb = trans_b ? 'T' : 'N';
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

/// Single-precision counterpart of `gemm_raw`.
inline void gemm_raw(bool trans_a, bool trans_b, int m, int n, int k, float alpha, const float* a,
                     int lda, const float* b, int ldb, float beta, float* c, int ldc) {
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  cblas_sgemm(CblasColMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  const char ta = trans_a ? 'T' : 'N';
  const char tb = trans_b ? 'T' : 'N';
  sgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

/// dsyevd with JOBZ='V', UPLO='L': on entry `a` (n x n, leading dimension lda) holds the lower
/// triangle; on exit the orthonormal eigenvectors. `w` has n entries. Returns LAPACK's INFO.
/// Performs the workspace query internally.
inline int syevd_lower(int n, double* a, int lda, double* w) {
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  double wq = 0.0;
  int iwq = 0;
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  char jobz = 'V';
  char uplo = 'L';
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &iwq, &liwork, &info);
  if (info != 0) return info;
  lwork = static_cast<int>(wq);
  liwork = iwq;
  std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));
  std::vector<int> iwork(static_cast<std::size_t>(std::max(liwork, 1)));
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);
#else
  const char jobz = 'V';
  const char uplo = 'L';
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wq, &lwork, &iwq, &liwork, &info);
  if (info != 0) return info;
  lwork = static_cast<int>(wq);
  liwork = iwq;
  std::vector<double> work(static_cast<std::size_t>(std::max(lwork, 1)));
  std::vector<int> iwork(static_cast<std::size_t>(std::max(liwork, 1)));
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);
#endif
  return info;
}

/// dgesv: solves A X = B in place (A overwritten by its LU factors, B by X). A is n x n, B is
/// n x nrhs. Returns LAPACK's INFO (> 0: singular).
inline int gesv(int n, int nrhs, double* a, int lda, double* b, int ldb) {
  std::vector<int> ipiv(static_cast<std::size_t>(std::max(n, 1)));
  int info = 0;
  dgesv_(&n, &nrhs, a, &lda, ipiv.data(), b, &ldb, &info);
  return info;
}

}  // namespace imsrg::linalg::detail
