// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMM_DETAIL_CHUNKS_H_
#define IMSRG_COMM_DETAIL_CHUNKS_H_
/// @file
/// Host-only helper that splits long messages into chunks that fit MPI's
/// `int` element counts. Pure integer arithmetic; no MPI.

#include <cstdint>
#include <vector>

namespace imsrg::comm::detail {

/// Longest message (in elements) sent by one MPI call of a span collective;
/// longer spans are split into consecutive chunks.
inline constexpr std::int64_t kMaxMessageElements = std::int64_t{1} << 30;

/// A half-open element range [offset, offset + count) of a message.
struct Chunk {
  std::int64_t offset;
  std::int64_t count;
};

/// Splits `n` elements into ceil(n / max_chunk) consecutive, contiguous chunks
/// of at most `max_chunk` elements (all full except possibly the last); empty
/// for `n == 0`. Every rank computes the same chunks from the same `n`.
/// Preconditions: `n >= 0`, `max_chunk >= 1` (IMSRG_CHECK). O(n / max_chunk).
std::vector<Chunk> ChunkRanges(std::int64_t n, std::int64_t max_chunk);

}  // namespace imsrg::comm::detail

#endif  // IMSRG_COMM_DETAIL_CHUNKS_H_
