// SPDX-License-Identifier: GPL-3.0-only
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
  std::vector<std::int64_t> offsets;
  std::vector<double> weights;
  offsets.reserve(tb.matrix().blocks().size() + 1);
  weights.reserve(tb.matrix().blocks().size());
  offsets.push_back(0);
  for (const auto& info : tb.matrix().blocks()) {
    // BlockSparseMatrix stores its blocks contiguously, in block order.
    IMSRG_CHECK(info.offset == offsets.back(),
                "two_body_norm: blocks are not contiguous");
    const double w = static_cast<double>(
        2 * channels[static_cast<std::size_t>(info.row_block.value())].J() + 1);
    weights.push_back(w * w);
    offsets.push_back(info.offset + info.rows * info.cols);
  }
  const double sum =
      detail::WeightedBlockSumOfSquares(tb.matrix().buffer(), offsets, weights);
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
