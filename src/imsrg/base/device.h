// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASE_DEVICE_H_
#define IMSRG_BASE_DEVICE_H_
/// @file
/// Device annotations usable in every public header, without including Kokkos
/// (design/20-build-and-tooling.md section 4, design/12-compute-backend.md
/// section 5).
///
/// `IMSRG_FUNCTION` marks a function that may be called from host and device
/// code. It expands to `__host__ __device__` when compiling with nvcc or hipcc
/// and to nothing otherwise, which is exactly what `KOKKOS_FUNCTION` does for
/// the CUDA and HIP backends. `IMSRG_INLINE_FUNCTION` adds `inline` (like
/// `KOKKOS_INLINE_FUNCTION`). `IMSRG_LAMBDA` (device lambdas) is defined only
/// in `imsrg/exec/kokkos_backend.h`, because lambdas are only launched from
/// kernel files.
///
/// Rules for functions with these annotations: no exceptions, no virtual calls,
/// no recursion, no dynamic allocation, no host-only data, and only math
/// functions that exist on the device.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IMSRG_FUNCTION __host__ __device__
#else
#define IMSRG_FUNCTION
#endif

#define IMSRG_INLINE_FUNCTION IMSRG_FUNCTION inline

/// `IMSRG_INLINE_FUNCTION` that the compiler must inline (like
/// `KOKKOS_FORCEINLINE_FUNCTION`): for small helpers called in the inner loops
/// of kernels, where the compiler's heuristics otherwise leave a call.
#if defined(__GNUC__) || defined(__clang__)
#define IMSRG_FORCEINLINE_FUNCTION \
  IMSRG_FUNCTION inline __attribute__((always_inline))
#else
#define IMSRG_FORCEINLINE_FUNCTION IMSRG_INLINE_FUNCTION
#endif

#endif  // IMSRG_BASE_DEVICE_H_
