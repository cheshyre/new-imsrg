// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The shared pair-third intermediates H and W of the row-tiled IMSRG(3)-N7
// terms (detail/n7_batch.h, design/14 section 6.4). Kernel TU.
//
// Both kernels are flat `exec::ParallelFor`s over the elements of the batch
// buffers: the buffer of H (W) holds the tiles one after the other, so an
// element index i is located by a binary search for the last tile whose
// offset is <= i (a tile of size 0 shares its offset with its successor, so
// the last such tile is the one containing i), then split into (column,
// row s) with s fastest.

#include <cstdint>

#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::exec {
// The one instantiation of the buffer of the tiles (a POD of this module, see
// imsrg/exec/device_buffer.h); needs the impl header, hence this kernel TU.
template class DeviceBuffer<commutator::detail::LineTile>;
}  // namespace imsrg::exec

namespace imsrg::commutator::detail {

namespace {

/// Which of the two buffers of a batch an element index refers to.
enum class Buffer : int { kH = 0, kW = 1 };

/// Index of the tile whose part of buffer `which` contains element `i`
/// (0 <= i < size of the buffer). O(log tiles).
IMSRG_INLINE_FUNCTION int TileOfElement(const LineBatchView& batch,
                                        Buffer which, std::int64_t i) noexcept {
  int lo = 0;
  int hi = batch.num_tiles - 1;
  while (lo < hi) {  // last tile with offset <= i
    const int mid = lo + (hi - lo + 1) / 2;
    const std::int64_t off = which == Buffer::kH ? batch.tiles[mid].h_offset
                                                 : batch.tiles[mid].w_offset;
    if (off <= i) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/// One element of H: row s (sorted, channel-local), line k (channel-local
/// kept row) of channel `ch`.
IMSRG_INLINE_FUNCTION double HElement(const Space3View& sp,
                                      const op::PairThirdView& pt,
                                      const Comm232TablesView& tab,
                                      const ThreeBodyPartView& b,
                                      int transposed, int ch, int s,
                                      int k) noexcept {
  const int f = tab.orig[tab.ch_begin[ch] + s];
  double sum = 0.0;
  for (std::int64_t e = pt.e.ptr[f]; e < pt.e.ptr[f + 1]; ++e) {
    const int r = pt.e.col[e];
    const double bv = transposed != 0 ? ThreeBodyLocal(b, sp, ch, k, r)
                                      : ThreeBodyLocal(b, sp, ch, r, k);
    sum += pt.e.val[e] * bv;
  }
  return sum;
}

/// One element of W: row s (sorted), column t (channel-local state of the
/// tile) of channel `ch`, from the H of the tile at `h`.
IMSRG_INLINE_FUNCTION double WElement(const op::PairThirdView& pt,
                                      const LineTile& tile, const double* h,
                                      int s, int ss, int t) noexcept {
  const std::int64_t f = pt.ch_state_offset[tile.channel] + t;
  double sum = 0.0;
  for (std::int64_t e = pt.e.ptr[f]; e < pt.e.ptr[f + 1]; ++e) {
    const int k = pt.e.col[e];
    sum += h[tile.h_offset + s +
             static_cast<std::int64_t>(k - tile.row_begin) * ss] *
           pt.e.val[e];
  }
  return sum;
}

}  // namespace

void BuildH(const Space3View& sp, const op::PairThirdView& pt,
            const Comm232TablesView& tab, const ThreeBodyPartView& b,
            int transposed, const LineBatchView& batch, double* h) {
  if (batch.num_tiles == 0 || batch.h_doubles == 0) return;
  exec::ParallelFor(
      "commutator::n7_build_h", batch.h_doubles,
      IMSRG_LAMBDA(const exec::Index i) {
        const LineTile& tile = batch.tiles[TileOfElement(batch, Buffer::kH, i)];
        const std::int64_t x = i - tile.h_offset;
        const int ss = pt.ch_num_states[tile.channel];
        const int s = static_cast<int>(x % ss);
        const int k = tile.row_begin + static_cast<int>(x / ss);
        h[i] = HElement(sp, pt, tab, b, transposed, tile.channel, s, k);
      });
}

void BuildW(const op::PairThirdView& pt, const LineBatchView& batch,
            const double* h, double* w) {
  if (batch.num_tiles == 0 || batch.w_doubles == 0) return;
  exec::ParallelFor(
      "commutator::n7_build_w", batch.w_doubles,
      IMSRG_LAMBDA(const exec::Index i) {
        const LineTile& tile = batch.tiles[TileOfElement(batch, Buffer::kW, i)];
        const std::int64_t x = i - tile.w_offset;
        const int ss = pt.ch_num_states[tile.channel];
        const int s = static_cast<int>(x % ss);
        const int t = tile.state_begin + static_cast<int>(x / ss);
        w[i] = WElement(pt, tile, h, s, ss, t);
      });
}

}  // namespace imsrg::commutator::detail
