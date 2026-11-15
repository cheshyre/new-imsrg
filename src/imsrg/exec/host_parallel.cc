// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/exec/host_parallel.h"

#include <Kokkos_Core.hpp>

#include "imsrg/base/assert.h"

namespace imsrg::exec {

void HostParallelForDynamic(const char* name, std::int64_t n,
                            const std::function<void(std::int64_t)>& f) {
  IMSRG_CHECK(n >= 0, name);
  if (n == 0) return;
  using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace,
                                     Kokkos::IndexType<std::int64_t>,
                                     Kokkos::Schedule<Kokkos::Dynamic>>;
  Policy policy(0, n);
  policy.set_chunk_size(1);
  Kokkos::parallel_for(name, policy, [&f](const std::int64_t i) { f(i); });
  Kokkos::DefaultHostExecutionSpace().fence(name);
}

int HostConcurrency() {
  return Kokkos::DefaultHostExecutionSpace().concurrency();
}

}  // namespace imsrg::exec
