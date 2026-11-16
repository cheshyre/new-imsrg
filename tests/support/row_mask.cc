// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "tests/support/row_mask.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "imsrg/base/assert.h"

namespace imsrg::testing {
namespace {

/// Membership flags of the rows of channel `ch` (dimension n) in `set`.
std::vector<char> RowFlags(const basis::RowSet& set, Channel3Index ch,
                           std::int32_t n) {
  std::vector<char> in(static_cast<std::size_t>(n), 0);
  for (const basis::RowRange& r : set.ranges()) {
    if (r.channel != ch) continue;
    IMSRG_CHECK(r.row_end.value() <= n, "row range beyond the channel");
    for (std::int32_t k = r.row_begin.value(); k < r.row_end.value(); ++k) {
      in[static_cast<std::size_t>(k)] = 1;
    }
  }
  return in;
}

/// Calls f(index, i, j) for every stored element of block `ch`; i is the row
/// index m of z(m, k) and j = k the storage row.
template <typename F>
void ForEachStored(const op::ThreeBody& b, Channel3Index ch, F f) {
  const std::int64_t n = b.dim(ch);
  const std::int64_t off =
      b.block_offsets()[static_cast<std::size_t>(ch.value())];
  for (std::int64_t j = 0; j < n; ++j) {
    for (std::int64_t i = b.packed() ? j : 0; i < n; ++i) {
      const std::int64_t idx =
          off + (b.packed() ? op::ThreeBody::PackedIndex(n, i, j) : i + j * n);
      f(idx, i, j);
    }
  }
}

}  // namespace

void MaskOutsideLines(op::ThreeBody& b, const basis::RowSet& lines) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::span<double> data = b.host_span();
  for (std::int32_t c = 0; c < b.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::vector<char> in = RowFlags(lines, ch, b.dim(ch));
    ForEachStored(b, ch, [&](std::int64_t idx, std::int64_t i, std::int64_t j) {
      if (!in[static_cast<std::size_t>(i)] &&
          !in[static_cast<std::size_t>(j)]) {
        data[static_cast<std::size_t>(idx)] = nan;
      }
    });
  }
  b.SyncToDevice();
}

bool OnlyRowsNonzero(const op::ThreeBody& z, const basis::RowSet& rows) {
  z.SyncToHost();
  const std::span<const double> data = z.host_span();
  bool ok = true;
  for (std::int32_t c = 0; c < z.num_channels(); ++c) {
    const Channel3Index ch{c};
    const std::vector<char> in = RowFlags(rows, ch, z.dim(ch));
    ForEachStored(z, ch,
                  [&](std::int64_t idx, std::int64_t /*i*/, std::int64_t j) {
                    if (!in[static_cast<std::size_t>(j)] &&
                        data[static_cast<std::size_t>(idx)] != 0.0) {
                      ok = false;
                    }
                  });
  }
  return ok;
}

}  // namespace imsrg::testing
