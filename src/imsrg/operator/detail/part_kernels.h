// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPERATOR_DETAIL_PART_KERNELS_H_
#define IMSRG_OPERATOR_DETAIL_PART_KERNELS_H_
/// @file
/// Internal (not a public header of the module): flat device kernels on the
/// buffers of operator parts, defined in `kernels/part_kernels.cc` (the only
/// translation unit that sees Kokkos).
#include <cstdint>
#include <span>

#include "imsrg/exec/device_buffer.h"

namespace imsrg::op::detail {

/// y[i] += a * x[i] for every element. Precondition: x.size() == y.size()
/// (IMSRG_CHECK). Asynchronous on async backends. O(n), embarrassingly
/// parallel.
void Axpy(double a, const exec::DeviceBuffer<double>& x,
          exec::DeviceBuffer<double>& y);

/// x[i] *= a for every element. Asynchronous on async backends. O(n).
void Scale(double a, exec::DeviceBuffer<double>& x);

/// x[i] = |x[i]| for every element. Asynchronous on async backends. O(n).
void Abs(exec::DeviceBuffer<double>& x);

/// x[i] /= y[i] for every element (IEEE division: 0/0 = NaN). Precondition:
/// x.size() == y.size() (IMSRG_CHECK). Asynchronous on async backends. O(n).
void Divide(exec::DeviceBuffer<double>& x, const exec::DeviceBuffer<double>& y);

/// x[i] += c for every element. Asynchronous on async backends. O(n).
void AddScalar(double c, exec::DeviceBuffer<double>& x);

/// Sum of x[i]^2 over i in [offset, offset + count). Precondition: the range
/// lies inside x (IMSRG_CHECK). Blocking (device reduction). O(count).
double SumOfSquares(const exec::DeviceBuffer<double>& x, std::int64_t offset,
                    std::int64_t count);

/// sum over blocks b of block_weights[b] * (sum of x[i]^2 for i in
/// [block_offsets[b], block_offsets[b + 1])), in ONE device reduction over
/// fixed-size chunks (each chunk locates its first block by binary search and
/// then walks its elements). `block_offsets` has num_blocks + 1 non-decreasing
/// entries (empty blocks allowed); elements outside [front, back] of the
/// offsets are ignored. Preconditions (IMSRG_CHECK): block_weights.size() + 1
/// == block_offsets.size() (or both empty), offsets non-negative and
/// non-decreasing, the last offset <= x.size(). Blocking. O(total elements);
/// allocates two small device buffers per call. Differs from the per-block
/// `SumOfSquares` loop only by floating-point reassociation.
double WeightedBlockSumOfSquares(const exec::DeviceBuffer<double>& x,
                                 std::span<const std::int64_t> block_offsets,
                                 std::span<const double> block_weights);

}  // namespace imsrg::op::detail

#endif  // IMSRG_OPERATOR_DETAIL_PART_KERNELS_H_
