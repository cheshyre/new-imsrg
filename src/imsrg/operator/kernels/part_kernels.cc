// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Flat kernels on operator-part buffers (declared in
// imsrg/operator/detail/part_kernels.h).
#include "imsrg/operator/detail/part_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "imsrg/base/assert.h"
#include "imsrg/base/device.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::op::detail {

namespace {

/// Elements per chunk of `WeightedBlockSumOfSquares` (one parallel work item).
/// Device backends use short chunks (M14): with 4096 an emax-8 operator gives
/// ~1700 items, one sequential strided loop each, and the norm took 3.4 ms
/// on an MI250X GCD (a third of the kernel time of a flow). The chunking
/// fixes the summation order, so device and host norms differ in the last
/// bits.
constexpr std::int64_t kNormChunkSize = exec::kDeviceKernels ? 16 : 4096;

/// Last block b in [0, num_blocks) with offsets(b) <= e (so, with empty blocks
/// skipped, the block that contains e). Precondition: offsets(0) <= e <
/// offsets(num_blocks). Binary search, O(log num_blocks).
template <class OffsetsView>
IMSRG_INLINE_FUNCTION int FindNormBlock(const OffsetsView& offsets,
                                        int num_blocks,
                                        std::int64_t e) noexcept {
  int lo = 0;
  int hi = num_blocks - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (offsets(mid) <= e) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

}  // namespace

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

void Abs(exec::DeviceBuffer<double>& x) {
  auto xv = exec::View(x);
  exec::ParallelFor(
      "op::abs", static_cast<exec::Index>(x.size()),
      IMSRG_LAMBDA(const exec::Index i) {
        xv(i) = xv(i) < 0.0 ? -xv(i) : xv(i);
      });
}

void Divide(exec::DeviceBuffer<double>& x,
            const exec::DeviceBuffer<double>& y) {
  IMSRG_CHECK(x.size() == y.size(), "divide: buffer sizes differ");
  auto xv = exec::View(x);
  const auto yv = exec::View(y);
  exec::ParallelFor(
      "op::divide", static_cast<exec::Index>(x.size()),
      IMSRG_LAMBDA(const exec::Index i) { xv(i) /= yv(i); });
}

void AddScalar(double c, exec::DeviceBuffer<double>& x) {
  auto xv = exec::View(x);
  exec::ParallelFor(
      "op::add_scalar", static_cast<exec::Index>(x.size()),
      IMSRG_LAMBDA(const exec::Index i) { xv(i) += c; });
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

double WeightedBlockSumOfSquares(const exec::DeviceBuffer<double>& x,
                                 std::span<const std::int64_t> block_offsets,
                                 std::span<const double> block_weights) {
  if (block_offsets.empty()) {
    IMSRG_CHECK(block_weights.empty(),
                "weighted_block_sum_of_squares: weights without offsets");
    return 0.0;
  }
  IMSRG_CHECK(block_weights.size() + 1 == block_offsets.size(),
              "weighted_block_sum_of_squares: need one weight per block");
  IMSRG_CHECK(block_offsets.front() >= 0 &&
                  std::is_sorted(block_offsets.begin(), block_offsets.end()),
              "weighted_block_sum_of_squares: offsets must be non-negative "
              "and non-decreasing");
  IMSRG_CHECK(block_offsets.back() <= static_cast<std::int64_t>(x.size()),
              "weighted_block_sum_of_squares: blocks outside the buffer");
  const std::int64_t begin = block_offsets.front();
  const std::int64_t end = block_offsets.back();
  if (end == begin) {
    return 0.0;
  }
  const int num_blocks = static_cast<int>(block_weights.size());
  exec::DeviceBuffer<std::int64_t> offsets_dev(block_offsets.size(),
                                               "op::norm_block_offsets");
  exec::DeviceBuffer<double> weights_dev(block_weights.size(),
                                         "op::norm_block_weights");
  offsets_dev.CopyFromHost(block_offsets);
  weights_dev.CopyFromHost(block_weights);
  const auto xv = exec::View(x);
  const auto ov = exec::View(offsets_dev);
  const auto wv = exec::View(weights_dev);
  const std::int64_t num_chunks =
      (end - begin + kNormChunkSize - 1) / kNormChunkSize;
  double result = 0.0;
  exec::ParallelReduce(
      "op::weighted_block_sum_of_squares", static_cast<exec::Index>(num_chunks),
      IMSRG_LAMBDA(const exec::Index chunk, double& partial) {
        const std::int64_t first = begin + chunk * kNormChunkSize;
        const std::int64_t last =
            first + kNormChunkSize < end ? first + kNormChunkSize : end;
        int b = FindNormBlock(ov, num_blocks, first);
        std::int64_t e = first;
        while (e < last) {
          const std::int64_t stop = ov(b + 1) < last ? ov(b + 1) : last;
          double s = 0.0;
          for (; e < stop; ++e) {
            const double v = xv(e);
            s += v * v;
          }
          partial += wv(b) * s;
          if (e == ov(b + 1)) {
            ++b;
          }
        }
      },
      result);
  return result;
}

}  // namespace imsrg::op::detail
