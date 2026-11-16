// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/comm/detail/chunks.h"

#include <algorithm>
#include <vector>

#include "imsrg/base/assert.h"

namespace imsrg::comm::detail {

std::vector<Chunk> ChunkRanges(std::int64_t n, std::int64_t max_chunk) {
  IMSRG_CHECK(n >= 0, "ChunkRanges: negative length");
  IMSRG_CHECK(max_chunk >= 1, "ChunkRanges: max_chunk must be >= 1");
  std::vector<Chunk> chunks;
  chunks.reserve(static_cast<std::size_t>(n / max_chunk + 1));
  for (std::int64_t offset = 0; offset < n; offset += max_chunk) {
    chunks.push_back({offset, std::min(max_chunk, n - offset)});
  }
  return chunks;
}

}  // namespace imsrg::comm::detail
