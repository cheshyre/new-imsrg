// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_LINALG_DEVICE_BLAS_H_
#define IMSRG_LINALG_DEVICE_BLAS_H_
/// @file
/// GEMMs on operands in exec DEVICE memory (`exec::DeviceBuffer::device_data()`
/// and pointers into it), design/12 section 3 (M14). The backend is chosen at
/// configure time (`IMSRG_DEVICE_BLAS`):
///
/// * `host` (CPU backends): device memory is host memory; the calls forward to
///   `Gemm` / `GroupedGemm` of blas.h after an `exec::Fence` (pending kernels
///   finish first), and return with the results written.
/// * `rocblas` (HIP): rocBLAS `dgemm` on the stream of the default Kokkos HIP
///   execution space. The calls are ASYNCHRONOUS like a Kokkos kernel: they
///   are ordered after earlier kernels and before later ones on the default
///   execution space; host access to C needs `exec::Fence` (or `SyncToHost`).
///   GroupedGemm spreads the problems over a few extra streams joined by
///   events (the default stream waits for all of them).
/// * `kokkos-kernels`: `KokkosBlas::gemm` on the default execution space
///   (asynchronous in the same way; portable to every Kokkos backend).
///
/// Shapes, BLAS semantics (beta == 0: C is not read) and the dimension checks
/// are those of `linalg::Gemm`; the checks run on the host before anything is
/// launched and throw InvariantError. This header includes neither Kokkos nor
/// HIP (design/10).

#include <span>
#include <string_view>

#include "imsrg/linalg/blas.h"
#include "imsrg/linalg/matrix_view.h"

namespace imsrg::linalg {

/// C = alpha op(A) op(B) + beta C on device memory; see the file comment for
/// the synchronization rule of each backend. Cost 2 m n k flops.
void DeviceGemm(double alpha, MatrixView<const double> A, Op opA,
                MatrixView<const double> B, Op opB, double beta,
                MatrixView<double> C);

/// All `problems` (independent; the C matrices must not overlap each other or
/// any A/B) on device memory. Results equal the sequential loop of
/// `DeviceGemm` (each C is written by exactly one BLAS call). Empty spans are
/// a no-op.
void DeviceGroupedGemm(std::span<const GemmProblem> problems);

/// Name of the configured device BLAS backend: "host", "rocblas" or
/// "kokkos-kernels".
std::string_view DeviceBlasBackend();

}  // namespace imsrg::linalg

#endif  // IMSRG_LINALG_DEVICE_BLAS_H_
