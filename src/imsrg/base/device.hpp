#pragma once
/// @file
/// Device annotations usable in every public header, without including Kokkos
/// (design/20-build-and-tooling.md section 4, design/12-compute-backend.md section 5).
///
/// `IMSRG_FUNCTION` marks a function that may be called from host and device code. It expands to
/// `__host__ __device__` when compiling with nvcc or hipcc and to nothing otherwise, which is
/// exactly what `KOKKOS_FUNCTION` does for the CUDA and HIP backends. `IMSRG_INLINE_FUNCTION`
/// adds `inline` (like `KOKKOS_INLINE_FUNCTION`). `IMSRG_LAMBDA` (device lambdas) is defined only
/// in `imsrg/exec/kokkos_backend.hpp`, because lambdas are only launched from kernel files.
///
/// Rules for functions with these annotations: no exceptions, no virtual calls, no recursion, no
/// dynamic allocation, no host-only data, and only math functions that exist on the device.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IMSRG_FUNCTION __host__ __device__
#else
#define IMSRG_FUNCTION
#endif

#define IMSRG_INLINE_FUNCTION IMSRG_FUNCTION inline
