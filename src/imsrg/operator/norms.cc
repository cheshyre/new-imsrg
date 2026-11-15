// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/operator/norms.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/operator/detail/part_kernels.h"

namespace imsrg::op {

double OneBodyNorm(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "one_body_norm: empty operator");
  const auto& basis = op.model_space().basis();
  const std::int32_t n = basis.num_orbits();
  const std::vector<double> data = op.one_body().buffer().ToVector();
  double sum = 0.0;
  for (std::int32_t p = 0; p < n; ++p) {
    const double w =
        static_cast<double>(basis.orbit(OrbitIndex{p}).j2.twice + 1);
    for (const OrbitIndex q : basis.SameLjtz(OrbitIndex{p})) {
      const double v = data[static_cast<std::size_t>(p) +
                            static_cast<std::size_t>(q.value()) *
                                static_cast<std::size_t>(n)];
      sum += v * v * w;
    }
  }
  return std::sqrt(sum);
}

double TwoBodyNorm(const Operator& op) {
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
    const double s = detail::SumOfSquares(buffer, info.offset, count);
    const double w = static_cast<double>(
        2 * channels[static_cast<std::size_t>(info.row_block.value())].J() + 1);
    sum += w * w * s;
  }
  return std::sqrt(sum);
}

double ThreeBodyNorm(const Operator& op) {
  IMSRG_CHECK(!op.empty(), "three_body_norm: empty operator");
  if (!op.three_body()) return 0.0;
  return ImsrgppNorm(*op.three_body());
}

double Norm(const Operator& op) {
  const double n1 = OneBodyNorm(op);
  const double n2 = TwoBodyNorm(op);
  const double n3 = ThreeBodyNorm(op);
  return std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);
}

}  // namespace imsrg::op
