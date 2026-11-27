// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/norms.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/operator/detail/part_kernels.h"
#include "imsrg/operator/two_body.h"

namespace imsrg::op {

double OneBodyNorm(const Operator& op, TensorOneBodyNorm mode) {
  IMSRG_CHECK(!op.empty(), "one_body_norm: empty operator");
  const auto& basis = op.model_space().basis();
  const std::int32_t n = basis.num_orbits();
  const std::vector<double> data = op.one_body().buffer().ToVector();
  double sum = 0.0;
  if (!op.spec().is_scalar()) {
    // kImsrgppInteger: imsrg++ Operator::OneBodyNorm with its integer
    // degeneracy factor, which mixes 2j with the undoubled rank_J (it drives
    // the BCH thresholds of the fork reference). Default: (2jp+1)(2jq+1)/(2L+1).
    const std::int32_t L = op.spec().rank_J;
    for (std::int32_t p = 0; p < n; ++p) {
      const basis::Orbit& op_ = basis.orbit(OrbitIndex{p});
      const std::int32_t jp = op_.j2.twice;
      for (std::int32_t q = 0; q < n; ++q) {
        const basis::Orbit& oq = basis.orbit(OrbitIndex{q});
        if (!ConnectsOrbits(op.spec(), op_, oq)) continue;
        const std::int32_t jq = oq.j2.twice;
        const double deg =
            mode == TensorOneBodyNorm::kImsrgppInteger
                ? static_cast<double>(
                      (jp + 1) *
                      ((std::min(jq, jp + L) - std::max(-jq, jp - L)) / 2 + 1))
                : static_cast<double>(jp + 1) * static_cast<double>(jq + 1) /
                      static_cast<double>(2 * L + 1);
        const double v =
            data[static_cast<std::size_t>(p) +
                 static_cast<std::size_t>(q) * static_cast<std::size_t>(n)];
        sum += v * v * deg;
      }
    }
    return std::sqrt(sum);
  }
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
    if (tb.is_tensor()) {
      // imsrg++ TwoBodyME::Norm for reduced elements: weight 1/(2 rank_J + 1)
      // per element, and a factor 2 for the implied transposed off-diagonal
      // block.
      const double w = 1.0 / (2.0 * tb.rank_J() + 1.0);
      weights.push_back((info.row_block == info.col_block ? 1.0 : 2.0) * w * w);
    } else {
      const double w = static_cast<double>(
          2 * channels[static_cast<std::size_t>(info.row_block.value())].J() +
          1);
      weights.push_back(w * w);
    }
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

double Norm(const Operator& op, TensorOneBodyNorm mode) {
  const double n1 = OneBodyNorm(op, mode);
  const double n2 = TwoBodyNorm(op);
  const double n3 = ThreeBodyNorm(op);
  return std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);
}

}  // namespace imsrg::op
