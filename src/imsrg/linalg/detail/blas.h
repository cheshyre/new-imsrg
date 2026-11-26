// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_LINALG_DETAIL_BLAS_H_
#define IMSRG_LINALG_DETAIL_BLAS_H_
/// @file
/// Private BLAS/LAPACK shim: one thin wrapper per routine, column-major, LP64
/// (32-bit int) interface. Included only by linalg `.cc` files. With Apple
/// Accelerate (`IMSRG_BLAS_ACCELERATE`) the CBLAS interface and the new LAPACK
/// headers (`ACCELERATE_NEW_LAPACK`) are used; otherwise the Fortran symbols
/// are declared directly (OpenBLAS, MKL, reference LAPACK).
///
/// With `IMSRG_BLAS_SERIALIZE` (CMake option, ON for every backend but
/// Accelerate) each call holds a process-wide mutex (`BlasCallGuard`): the
/// OpenMP variant of cray-libsci is not reentrant across host threads (two
/// host threads in dgemm at once crash it, FP-Q6).

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
// Fortran BLAS/LAPACK symbols have fixed ABI names (naming check off).
extern "C" {
// NOLINTNEXTLINE(readability-identifier-naming)
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
// NOLINTNEXTLINE(readability-identifier-naming)
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* a, const int* lda,
            const float* b, const int* ldb, const float* beta, float* c,
            const int* ldc);
// NOLINTNEXTLINE(readability-identifier-naming)
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);
// NOLINTNEXTLINE(readability-identifier-naming)
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv,
            double* b, const int* ldb, int* info);
}
#endif

namespace imsrg::linalg::detail {

#if defined(IMSRG_BLAS_SERIALIZE) && IMSRG_BLAS_SERIALIZE
inline constexpr bool kSerializeBlas = true;
#else
inline constexpr bool kSerializeBlas = false;
#endif

/// RAII guard held around every BLAS/LAPACK call of this shim. With
/// `kSerializeBlas` it locks one process-wide mutex, so calls from different
/// host threads (e.g. two Python threads solving at once) run one after
/// another; a threaded BLAS still uses all its OpenMP threads inside each call.
/// Without it the guard does nothing. The mutex is not recursive: no BLAS call
/// may be made while the guard is held. Never call it inside a device kernel.
class [[nodiscard]] BlasCallGuard {
 public:
  BlasCallGuard() {
    if constexpr (kSerializeBlas) Mutex().lock();
  }
  ~BlasCallGuard() {
    if constexpr (kSerializeBlas) Mutex().unlock();
  }
  BlasCallGuard(const BlasCallGuard&) = delete;
  BlasCallGuard& operator=(const BlasCallGuard&) = delete;
  BlasCallGuard(BlasCallGuard&&) = delete;
  BlasCallGuard& operator=(BlasCallGuard&&) = delete;

 private:
  static std::mutex& Mutex() {
    static auto* const kMutex = new std::mutex;  // never destroyed
    return *kMutex;
  }
};

/// C = alpha op(A) op(B) + beta C, column-major; `trans_*` true means
/// transposed. Dimensions are already validated by the caller (m, n, k, ld* >=
/// 1 and < 2^31).
inline void GemmRaw(bool trans_a, bool trans_b, int m, int n, int k,
                    double alpha, const double* a, int lda, const double* b,
                    int ldb, double beta, double* c, int ldc) {
  const BlasCallGuard guard;
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  cblas_dgemm(CblasColMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
              ldb, beta, c, ldc);
#else
  const char ta = trans_a ? 'T' : 'N';
  const char tb = trans_b ? 'T' : 'N';
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

/// Single-precision counterpart of `GemmRaw`.
inline void GemmRaw(bool trans_a, bool trans_b, int m, int n, int k,
                    float alpha, const float* a, int lda, const float* b,
                    int ldb, float beta, float* c, int ldc) {
  const BlasCallGuard guard;
#if defined(IMSRG_BLAS_ACCELERATE) && IMSRG_BLAS_ACCELERATE
  cblas_sgemm(CblasColMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
              ldb, beta, c, ldc);
#else
  const char ta = trans_a ? 'T' : 'N';
  const char tb = trans_b ? 'T' : 'N';
  sgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#endif
}

/// dsyevd with JOBZ='V', UPLO='L': on entry `a` (n x n, leading dimension lda)
/// holds the lower triangle; on exit the orthonormal eigenvectors. `w` has n
/// entries. Returns LAPACK's INFO. Performs the workspace query internally.
int SyevdLower(int n, double* a, int lda, double* w);

/// dgesv: solves A X = B in place (A overwritten by its LU factors, B by X). A
/// is n x n, B is n x nrhs. Returns LAPACK's INFO (> 0: singular).
inline int Gesv(int n, int nrhs, double* a, int lda, double* b, int ldb) {
  std::vector<int> ipiv(static_cast<std::size_t>(std::max(n, 1)));
  int info = 0;
  const BlasCallGuard guard;
  dgesv_(&n, &nrhs, a, &lda, ipiv.data(), b, &ldb, &info);
  return info;
}

}  // namespace imsrg::linalg::detail

#endif  // IMSRG_LINALG_DETAIL_BLAS_H_
