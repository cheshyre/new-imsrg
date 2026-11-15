// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Kernels reading the PairThirdSpace device view. Free functions only
// (design/12 section 5).
#include "tests/unit/operator/pair_third_kernels.h"

#include <cstdint>

#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::op::test {
namespace {

template <class T>
void CopyKernel(const T* src, exec::DeviceBuffer<T>& out) {
  if (out.empty()) return;
  auto o = exec::View(out);
  exec::ParallelFor(
      "test::pair_third_copy", static_cast<exec::Index>(o.extent(0)),
      IMSRG_LAMBDA(const exec::Index i) { o(i) = src[i]; });
  exec::Fence("test::pair_third_copy");
}

}  // namespace

void CopyOnDevice(const std::int32_t* src,
                  exec::DeviceBuffer<std::int32_t>& out) {
  CopyKernel(src, out);
}

void CopyOnDevice(const std::int64_t* src,
                  exec::DeviceBuffer<std::int64_t>& out) {
  CopyKernel(src, out);
}

void CopyOnDevice(const double* src, exec::DeviceBuffer<double>& out) {
  CopyKernel(src, out);
}

}  // namespace imsrg::op::test
