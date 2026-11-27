// SPDX-License-Identifier: GPL-3.0-only
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

/// Runs `f(i)` for i in [0, n) on the host execution space (its threads with
/// the OpenMP or Threads backend, the calling thread with Serial), handing out
/// one item at a time (dynamic scheduling), and returns when all items are
/// done. `f` is plain host code; it must not throw (an exception inside the
/// parallel region terminates the program) and must be safe to run
/// concurrently for different i. Needs an initialized `exec::Runtime`.
void HostParallelForDynamic(const char* name, std::int64_t n,
                            const std::function<void(std::int64_t)>& f);

/// True inside an active host-parallel region (an OpenMP parallel region,
/// e.g. the body of `HostParallelForDynamic` with the OpenMP host space);
/// always false without OpenMP. O(1). Used by debug guards against
/// allocations and global fences in parallel regions (design/12 section 5).
bool InHostParallelRegion();

/// Number of threads of the host execution space (1 for Serial). O(1).
int HostConcurrency();

}  // namespace imsrg::exec

#endif  // IMSRG_EXEC_HOST_PARALLEL_H_
