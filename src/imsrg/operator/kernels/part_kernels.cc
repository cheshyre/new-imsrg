// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Flat kernels on operator-part buffers (declared in
// imsrg/operator/detail/part_kernels.h).
#include "imsrg/operator/detail/part_kernels.h"

#include "imsrg/base/assert.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::op::detail {

void Axpy(double a, const exec::DeviceBuffer<double>& x,
          exec::DeviceBuffer<double>& y) {
  IMSRG_CHECK(x.size() == y.size(), "axpy: buffer sizes differ");
  const auto xv = exec::View(x);
  auto yv = exec::View(y);
  exec::ParallelFor(
      "op::axpy", static_cast<exec::Index>(y.size()),
      IMSRG_LAMBDA(const exec::Index i) { yv(i) += a * xv(i); });
}

void Scale(double a, exec::DeviceBuffer<double>& x) {
  auto xv = exec::View(x);
  exec::ParallelFor(
      "op::scale", static_cast<exec::Index>(x.size()),
      IMSRG_LAMBDA(const exec::Index i) { xv(i) *= a; });
}

double SumOfSquares(const exec::DeviceBuffer<double>& x, std::int64_t offset,
                    std::int64_t count) {
  IMSRG_CHECK(offset >= 0 && count >= 0 &&
                  offset + count <= static_cast<std::int64_t>(x.size()),
              "sum_of_squares: range outside the buffer");
  const auto xv = exec::View(x);
  double result = 0.0;
  exec::ParallelReduce(
      "op::sum_of_squares", static_cast<exec::Index>(count),
      IMSRG_LAMBDA(const exec::Index i, double& partial) {
        const double v = xv(offset + i);
        partial += v * v;
      },
      result);
  return result;
}

}  // namespace imsrg::op::detail
