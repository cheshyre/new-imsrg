// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_EXEC_HOST_PARALLEL_H_
#define IMSRG_EXEC_HOST_PARALLEL_H_

/// @file
/// Coarse-grained parallel loops on the HOST execution space for ordinary
/// host code (no device lambda), e.g. one BLAS call per item (design/12
/// section 3: BLAS called inside a parallel region runs single-threaded).
/// This header does not include Kokkos.

#include <cstdint>
#include <functional>

namespace imsrg::exec {

/// Runs `f(i)` for i in [0, n) on the host execution space (the OpenMP
/// threads of an OpenMP build, the calling thread otherwise), handing out
/// one item at a time (dynamic scheduling), and returns when all items are
/// done. `f` is plain host code; it must not throw (an exception inside the
/// parallel region terminates the program) and must be safe to run
/// concurrently for different i. Needs an initialized `exec::Runtime`.
void HostParallelForDynamic(const char* name, std::int64_t n,
                            const std::function<void(std::int64_t)>& f);

/// Number of threads of the host execution space (1 for Serial). O(1).
int HostConcurrency();

}  // namespace imsrg::exec

#endif  // IMSRG_EXEC_HOST_PARALLEL_H_
