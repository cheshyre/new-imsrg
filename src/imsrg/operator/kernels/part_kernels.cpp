// Flat kernels on operator-part buffers (declared in imsrg/operator/detail/part_kernels.hpp).
#include "imsrg/operator/detail/part_kernels.hpp"

#include "imsrg/base/assert.hpp"
#include "imsrg/exec/kokkos_backend.hpp"

namespace imsrg::op::detail {

void axpy(double a, const exec::DeviceBuffer<double>& x, exec::DeviceBuffer<double>& y) {
  IMSRG_CHECK(x.size() == y.size(), "axpy: buffer sizes differ");
  const auto xv = exec::view(x);
  auto yv = exec::view(y);
  exec::parallel_for(
      "op::axpy", static_cast<exec::Index>(y.size()),
      IMSRG_LAMBDA(const exec::Index i) { yv(i) += a * xv(i); });
}

void scale(double a, exec::DeviceBuffer<double>& x) {
  auto xv = exec::view(x);
  exec::parallel_for(
      "op::scale", static_cast<exec::Index>(x.size()),
      IMSRG_LAMBDA(const exec::Index i) { xv(i) *= a; });
}

double sum_of_squares(const exec::DeviceBuffer<double>& x, std::int64_t offset,
                      std::int64_t count) {
  IMSRG_CHECK(offset >= 0 && count >= 0 && offset + count <= static_cast<std::int64_t>(x.size()),
              "sum_of_squares: range outside the buffer");
  const auto xv = exec::view(x);
  double result = 0.0;
  exec::parallel_reduce(
      "op::sum_of_squares", static_cast<exec::Index>(count),
      IMSRG_LAMBDA(const exec::Index i, double& partial) {
        const double v = xv(offset + i);
        partial += v * v;
      },
      result);
  return result;
}

}  // namespace imsrg::op::detail
