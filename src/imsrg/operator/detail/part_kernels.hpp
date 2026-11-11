#pragma once
/// @file
/// Internal (not a public header of the module): flat device kernels on the buffers of operator
/// parts, defined in `kernels/part_kernels.cpp` (the only translation unit that sees Kokkos).
#include <cstdint>

#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::op::detail {

/// y[i] += a * x[i] for every element. Precondition: x.size() == y.size() (IMSRG_CHECK).
/// Asynchronous on async backends. O(n), embarrassingly parallel.
void axpy(double a, const exec::DeviceBuffer<double>& x, exec::DeviceBuffer<double>& y);

/// x[i] *= a for every element. Asynchronous on async backends. O(n).
void scale(double a, exec::DeviceBuffer<double>& x);

/// Sum of x[i]^2 over i in [offset, offset + count). Precondition: the range lies inside x
/// (IMSRG_CHECK). Blocking (device reduction). O(count).
double sum_of_squares(const exec::DeviceBuffer<double>& x, std::int64_t offset, std::int64_t count);

}  // namespace imsrg::op::detail
