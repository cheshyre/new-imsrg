// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_N7_BATCH_H_
#define IMSRG_COMMUTATOR_DETAIL_N7_BATCH_H_

/// @file
/// Batches of line tiles and the shared pair-third intermediates H and W of
/// the row-tiled IMSRG(3)-N7 terms (design/14 section 6). Internal to the
/// commutator module and its tests; no Kokkos.
///
/// A *tile* holds whole triples [triple_begin, triple_end) of one channel
/// ch: its kept rows (= lines) [row_begin, row_end) and its pair-third
/// states [state_begin, state_end) (channel-local, `op::PairThirdSpace`
/// order). A *batch* is a list of tiles, at most one per channel, whose
/// intermediates fit a byte budget. For a three-body operand B and a tile,
/// with S the pair-third states of the channel in the SORTED order of
/// `Comm232Tables` (third, Jpq, occupation class, p, q) and E the
/// pair-third -> canonical expansion (`op::PairThirdSpace::e_map`):
///   H[s, k - row_begin] = sum_{(r, e) in E row of sorted state s} e B(r, k),
///   W[s, t - state_begin] = sum_{(k, e) in E row of state t} H[s, k - rb] e
///                         = (E B E^T)[s, t],
/// for EVERY sorted row s in [0, S) and every line k / state t of the tile,
/// stored column-major with leading dimension S at `h_offset` / `w_offset`
/// of per-batch buffers (one block of `h_doubles` / `w_doubles` per operand
/// block). The transposed variant of H uses B(k, r) instead of B(r, k).

#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/device.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Occupation class of a pair with occupations np, nq: 0 if only n_p n_q is
/// nonzero (hh-only), 1 if n_p n_q and nbar_p nbar_q are both nonzero
/// (mixed, fractional occupations), 2 if only nbar_p nbar_q is nonzero
/// (pp-only), 3 if both vanish (inactive, e.g. a particle-hole pair).
/// Host and device use this one function, so the classes agree exactly.
IMSRG_INLINE_FUNCTION int PairClass(double np, double nq) noexcept {
  const bool hh = np != 0.0 && nq != 0.0;
  const bool pp = np != 1.0 && nq != 1.0;
  if (hh) return pp ? 1 : 0;
  return pp ? 2 : 3;
}

/// The position of `key` in the ascending `keys[lo, hi)`, or -1. Binary
/// search, O(log(hi - lo)).
IMSRG_INLINE_FUNCTION int FindKey(const std::int32_t* keys, int lo, int hi,
                                  int key) noexcept {
  int a = lo;
  int b = hi - 1;
  while (a <= b) {
    const int mid = a + (b - a) / 2;
    const int k = keys[mid];
    if (k == key) return mid;
    if (k < key) {
      a = mid + 1;
    } else {
      b = mid - 1;
    }
  }
  return -1;
}

/// One tile of a batch: whole triples of one channel. Indices as in
/// `basis::RowRange` (`triple_*` are global `Triple3Index` values, `row_*`
/// channel-local kept rows) and `op::PairThirdSpace` (`state_*` are
/// channel-local states of the triples). Trivially copyable.
struct LineTile {
  std::int32_t channel = 0;       ///< Three-body channel.
  std::int32_t triple_begin = 0;  ///< First triple.
  std::int32_t triple_end = 0;    ///< One past the last triple.
  std::int32_t row_begin = 0;     ///< First kept row (line), channel-local.
  std::int32_t row_end = 0;       ///< One past the last kept row.
  std::int32_t state_begin = 0;   ///< First pair-third state, channel-local.
  std::int32_t state_end = 0;     ///< One past the last state.
  std::int64_t h_offset = 0;      ///< Start of H_tile (S (re - rb) doubles).
  std::int64_t w_offset = 0;      ///< Start of W_tile (S (se - sb) doubles).
};
static_assert(std::is_trivially_copyable_v<LineTile>);

/// A planned batch (host): tiles in plan order, at most one per channel; the
/// offsets of the tiles are consecutive, so `h_doubles` / `w_doubles` are the
/// sizes of the buffers of ONE operand block.
struct LineBatch {
  std::vector<LineTile> tiles;  ///< The tiles.
  std::int64_t num_lines = 0;   ///< Total lines (kept rows) of the tiles.
  std::int64_t h_doubles = 0;   ///< Doubles of H per operand block.
  std::int64_t w_doubles = 0;   ///< Doubles of W per operand block.
};

/// Device view of a batch (raw device pointers, captured by value).
/// Trivially copyable.
struct LineBatchView {
  const LineTile* tiles = nullptr;  ///< `num_tiles` tiles in plan order.
  /// Tile index of each channel, or -1 (`num_channels` entries).
  const std::int32_t* tile_of_channel = nullptr;
  std::int32_t num_tiles = 0;
  std::int64_t num_lines = 0;  ///< Total lines of the batch.
  std::int64_t h_doubles = 0;  ///< Doubles of H per operand block.
  std::int64_t w_doubles = 0;  ///< Doubles of W per operand block.
};
static_assert(std::is_trivially_copyable_v<LineBatchView>);

/// One batch uploaded to the device. Value type (copies are deep). Needs an
/// initialized `exec::Runtime`.
class LineBatchDevice {
 public:
  /// Uploads `batch` for a space of `num_channels` channels.
  /// Precondition: the channels of the tiles are < `num_channels`
  /// (IMSRG_CHECK). O(tiles + channels).
  LineBatchDevice(const LineBatch& batch, std::int32_t num_channels);

  LineBatchDevice(const LineBatchDevice& other);
  LineBatchDevice& operator=(const LineBatchDevice& other);
  LineBatchDevice(LineBatchDevice&& other) noexcept;
  LineBatchDevice& operator=(LineBatchDevice&& other) noexcept;
  ~LineBatchDevice();

  /// The device view (pointers valid while `*this` lives). O(1).
  LineBatchView view() const noexcept;

 private:
  exec::DeviceBuffer<LineTile> tiles_;
  exec::DeviceBuffer<std::int32_t> tile_of_channel_;
  std::int32_t num_tiles_ = 0;
  std::int64_t num_lines_ = 0;
  std::int64_t h_doubles_ = 0;
  std::int64_t w_doubles_ = 0;
};

/// Plans the batches that cover the rows of `rows` exactly once (pure and
/// deterministic). The ranges of `rows` are walked in order and split into
/// tiles of whole triples; a batch takes at most ONE tile per channel (a
/// second range of the same channel, or the rest of a split range, starts a
/// new batch) and its intermediates, 8 `operand_blocks` (h_doubles +
/// w_doubles) bytes, stay within `budget_bytes`; a batch is closed when the
/// next tile does not fit, and a single triple over the budget is a batch of
/// its own. Tile sizes: H S k doubles and W S m, with S the pair-third states
/// of the channel, k the rows and m the states of the tile's triples.
/// Preconditions: `rows` was built for the space of `im`;
/// `operand_blocks >= 1`, `budget_bytes >= 0` (IMSRG_CHECK). O(triples).
std::vector<LineBatch> PlanBatches(const ThreeBodyContext::Impl& im,
                                   const basis::RowSet& rows,
                                   std::int64_t budget_bytes,
                                   std::int32_t operand_blocks);

/// Offset (into the W buffer of ONE operand block of the batch) of the
/// column of the sorted pair-third state `sorted_state` (flat sorted index,
/// i.e. in [ch_begin[ch], ch_begin[ch + 1]) of `tab`) of channel `ch`: the
/// start of column t - state_begin of the tile of `ch`, i.e. `w_offset + (t -
/// state_begin) S` with t the channel-local `PairThirdSpace` state (`tab.orig`
/// maps sorted to flat), or -1 if the channel has no tile in the batch or the
/// state's triple is not in the tile. O(1).
IMSRG_INLINE_FUNCTION std::int64_t BatchColumn(const LineBatchView& batch,
                                               const Comm232TablesView& tab,
                                               const op::PairThirdView& pt,
                                               int ch,
                                               int sorted_state) noexcept {
  const int ti = batch.tile_of_channel[ch];
  if (ti < 0) return -1;
  const LineTile& tile = batch.tiles[ti];
  const int t = tab.orig[sorted_state] - pt.ch_state_offset[ch];
  if (t < tile.state_begin || t >= tile.state_end) return -1;
  return tile.w_offset +
         static_cast<std::int64_t>(t - tile.state_begin) * pt.ch_num_states[ch];
}

/// Channel-local sorted row of the pair-third state with pair (p, q) (p <= q),
/// pair J `jpq` (J, not 2J) and third `r` in channel `ch`, or -1 if there is
/// none: class of the pair from `occupation`, then `FindKey` in the class
/// range of the block (ch, r, jpq). O(log block size).
IMSRG_INLINE_FUNCTION int FindSortedRow(const Comm232TablesView& tab,
                                        const double* occupation, int ch, int p,
                                        int q, int jpq, int r) noexcept {
  const int cls = PairClass(occupation[p], occupation[q]);
  const std::int64_t b =
      (static_cast<std::int64_t>(ch) * tab.norb + r) * tab.nj + jpq;
  const int pos =
      FindKey(tab.key, tab.blk[kComm232Classes * b + cls],
              tab.blk[kComm232Classes * b + cls + 1], p * tab.norb + q);
  return pos < 0 ? -1 : pos - tab.ch_begin[ch];
}

/// Builds H of every tile of `batch` for the operand part `b` (formula in
/// the file comment) into `h` (`batch.h_doubles` doubles, device memory; every
/// element is written). `transposed` != 0 uses B(k, r) in place of B(r, k).
/// B is read on the lines of the batch only (`ThreeBodyLocal`). One flat
/// work item per H element, s fastest (contiguous writes). Cost: per line
/// k and sorted row s the nnz of the E row of s multiply-adds, i.e. the sum
/// over the triples T of the channel of k_T m_T per line over all S rows.
/// Asynchronous on device backends (the caller fences). The sum order is
/// that of the E row, so the result does not depend on the batching.
void BuildH(const Space3View& sp, const op::PairThirdView& pt,
            const Comm232TablesView& tab, const ThreeBodyPartView& b,
            int transposed, const LineBatchView& batch, double* h);

/// Builds W = (E B E^T) of every tile of `batch` from the H built by `BuildH`
/// into `w` (`batch.w_doubles` doubles; every element is written). One flat
/// work item per W element, s fastest (contiguous reads down a column of
/// H). Cost: per tile column t, S nnz(E row of t) multiply-adds. Same
/// asynchrony and batching invariance as `BuildH`.
void BuildW(const op::PairThirdView& pt, const LineBatchView& batch,
            const double* h, double* w);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_N7_BATCH_H_
