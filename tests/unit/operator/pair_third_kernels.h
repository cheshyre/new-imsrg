// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TESTS_UNIT_OPERATOR_PAIR_THIRD_KERNELS_H_
#define TESTS_UNIT_OPERATOR_PAIR_THIRD_KERNELS_H_
// Test-local kernel entry points (Kokkos-free interface); the kernels are in
// pair_third_kernels.cc.
#include <cstdint>

#include "imsrg/exec/device_buffer.h"

namespace imsrg::op::test {

/// out[i] = src[i] for i < out.size(), executed as a device kernel; `src` is
/// a device pointer (from `PairThirdView` of `device_view()`). No-op for an
/// empty `out`.
void CopyOnDevice(const std::int32_t* src,
                  exec::DeviceBuffer<std::int32_t>& out);
/// 64-bit overload of `CopyOnDevice`.
void CopyOnDevice(const std::int64_t* src,
                  exec::DeviceBuffer<std::int64_t>& out);
/// Double overload of `CopyOnDevice`.
void CopyOnDevice(const double* src, exec::DeviceBuffer<double>& out);

}  // namespace imsrg::op::test

#endif  // TESTS_UNIT_OPERATOR_PAIR_THIRD_KERNELS_H_
