#include "imsrg/operator/norms.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/operator/detail/part_kernels.hpp"

namespace imsrg::op {

double one_body_norm(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "one_body_norm: empty operator");
  const auto& basis = op.model_space().basis();
  const std::int32_t n = basis.num_orbits();
  const std::vector<double> data = op.one_body().buffer().to_vector();
  double sum = 0.0;
  for (std::int32_t p = 0; p < n; ++p) {
    const double w = static_cast<double>(basis.orbit(OrbitIndex{p}).j2.twice + 1);
    for (const OrbitIndex q : basis.same_ljtz(OrbitIndex{p})) {
      const double v = data[static_cast<std::size_t>(p) +
                            static_cast<std::size_t>(q.value()) * static_cast<std::size_t>(n)];
      sum += v * v * w;
    }
  }
  return std::sqrt(sum);
}

double two_body_norm(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "two_body_norm: empty operator");
  const TwoBody& tb = op.two_body();
  if (!tb.allocated()) {
    return 0.0;
  }
  const auto channels = tb.channels().channels();
  const auto& buffer = tb.matrix().buffer();
  double sum = 0.0;
  for (const auto& info : tb.matrix().blocks()) {
    const std::int64_t count = info.rows * info.cols;
    if (count == 0) {
      continue;
    }
    const double s = detail::sum_of_squares(buffer, info.offset, count);
    const double w =
        static_cast<double>(2 * channels[static_cast<std::size_t>(info.row_block.value())].J() + 1);
    sum += w * w * s;
  }
  return std::sqrt(sum);
}

double three_body_norm(const Operator&) { return 0.0; }

double norm(const Operator& op) {
  const double n1 = one_body_norm(op);
  const double n2 = two_body_norm(op);
  const double n3 = three_body_norm(op);
  return std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);
}

}  // namespace imsrg::op
