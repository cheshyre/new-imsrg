// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Batch planner and device upload of the line tiles of the row-tiled
// IMSRG(3)-N7 terms (detail/n7_batch.h). Host code.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/three_body_impl.h"

namespace imsrg::commutator::detail {

namespace {

/// Bytes of the intermediates of a tile of a channel with `s` states, `k`
/// rows and `m` states of its triples, for `blocks` operand blocks.
std::int64_t TileBytes(std::int64_t s, std::int64_t k, std::int64_t m,
                       std::int64_t blocks) {
  return 8 * blocks * s * (k + m);
}

/// Host tables the planner reads: the pair-third state ranges and the kept
/// row ranges of the triples.
struct PlanTables {
  op::PairThirdView pt;
  std::vector<std::int32_t> row_begin;  // per triple, channel-local
  std::vector<std::int32_t> row_end;
};

/// The batch under construction.
class BatchBuilder {
 public:
  BatchBuilder(std::int64_t budget, std::int64_t blocks, std::int32_t nch)
      : budget_(budget),
        blocks_(blocks),
        has_tile_(static_cast<std::size_t>(nch), false) {}

  /// True iff the current batch has no tile.
  bool empty() const { return cur_.tiles.empty(); }
  /// True iff the current batch already has a tile of `ch`.
  bool HasChannel(std::int32_t ch) const {
    return has_tile_[static_cast<std::size_t>(ch)];
  }
  /// Bytes still free in the current batch.
  std::int64_t Free() const {
    return budget_ - 8 * blocks_ * (cur_.h_doubles + cur_.w_doubles);
  }

  /// Appends `tile` (its offsets are set here) of a channel with `s` states.
  void Add(LineTile tile, std::int64_t s) {
    const std::int64_t k = tile.row_end - tile.row_begin;
    const std::int64_t m = tile.state_end - tile.state_begin;
    tile.h_offset = cur_.h_doubles;
    tile.w_offset = cur_.w_doubles;
    cur_.h_doubles += s * k;
    cur_.w_doubles += s * m;
    cur_.num_lines += k;
    has_tile_[static_cast<std::size_t>(tile.channel)] = true;
    cur_.tiles.push_back(tile);
  }

  /// Closes the current batch (if nonempty) and starts a new one.
  void Flush() {
    if (cur_.tiles.empty()) return;
    for (const LineTile& t : cur_.tiles) {
      has_tile_[static_cast<std::size_t>(t.channel)] = false;
    }
    done_.push_back(std::move(cur_));
    cur_ = LineBatch();
  }

  /// Closes the last batch and returns all batches.
  std::vector<LineBatch> Finish() {
    Flush();
    return std::move(done_);
  }

 private:
  std::int64_t budget_;
  std::int64_t blocks_;
  std::vector<bool> has_tile_;
  LineBatch cur_;
  std::vector<LineBatch> done_;
};

/// The largest `te` in [tb, end] such that the tile [tb, te) of a channel
/// with `s` states fits `free_bytes` (`tb` if not even one triple fits).
std::int32_t FitTriples(const PlanTables& pl, std::int32_t tb, std::int32_t end,
                        std::int64_t s, std::int64_t blocks,
                        std::int64_t free_bytes) {
  std::int32_t te = tb;
  while (te < end) {
    const auto last = static_cast<std::size_t>(te);
    const auto first = static_cast<std::size_t>(tb);
    const std::int64_t k = pl.row_end[last] - pl.row_begin[first];
    const std::int64_t m = pl.pt.tr_state_end[te] - pl.pt.tr_state_begin[tb];
    if (TileBytes(s, k, m, blocks) > free_bytes) break;
    ++te;
  }
  return te;
}

/// The tile of the triples [tb, te) of channel `ch`.
LineTile MakeTile(const PlanTables& pl, std::int32_t ch, std::int32_t tb,
                  std::int32_t te) {
  LineTile tile;
  tile.channel = ch;
  tile.triple_begin = tb;
  tile.triple_end = te;
  tile.row_begin = pl.row_begin[static_cast<std::size_t>(tb)];
  tile.row_end = pl.row_end[static_cast<std::size_t>(te - 1)];
  tile.state_begin = pl.pt.tr_state_begin[tb];
  tile.state_end = pl.pt.tr_state_end[te - 1];
  return tile;
}

}  // namespace

std::vector<LineBatch> PlanBatches(const ThreeBodyContext::Impl& im,
                                   const basis::RowSet& rows,
                                   std::int64_t budget_bytes,
                                   std::int32_t operand_blocks) {
  IMSRG_CHECK(operand_blocks >= 1, "PlanBatches: operand_blocks < 1");
  IMSRG_CHECK(budget_bytes >= 0, "PlanBatches: negative budget");
  PlanTables pl;
  pl.pt = im.pair_third.host_view();
  pl.row_begin = im.tr_row_begin.ToVector();
  pl.row_end = im.tr_row_end.ToVector();
  BatchBuilder bb(budget_bytes, operand_blocks, pl.pt.num_channels);
  for (const basis::RowRange& range : rows.ranges()) {
    const std::int32_t ch = range.channel.value();
    const std::int64_t s = pl.pt.ch_num_states[ch];
    std::int32_t tb = range.triple_begin.value();
    const std::int32_t end = range.triple_end.value();
    while (tb < end) {
      if (bb.HasChannel(ch)) bb.Flush();
      std::int32_t te = FitTriples(pl, tb, end, s, operand_blocks, bb.Free());
      if (te == tb) {
        if (!bb.empty()) {
          bb.Flush();
          continue;
        }
        te = tb + 1;  // a single triple over the budget: a batch of its own
      }
      bb.Add(MakeTile(pl, ch, tb, te), s);
      tb = te;
    }
  }
  return bb.Finish();
}

LineBatchDevice::LineBatchDevice(const LineBatch& batch,
                                 std::int32_t num_channels)
    : num_tiles_(static_cast<std::int32_t>(batch.tiles.size())),
      num_lines_(batch.num_lines),
      h_doubles_(batch.h_doubles),
      w_doubles_(batch.w_doubles) {
  std::vector<std::int32_t> of_channel(static_cast<std::size_t>(num_channels),
                                       -1);
  for (std::size_t i = 0; i < batch.tiles.size(); ++i) {
    const std::int32_t ch = batch.tiles[i].channel;
    IMSRG_CHECK(ch >= 0 && ch < num_channels,
                "LineBatchDevice: channel out of range");
    of_channel[static_cast<std::size_t>(ch)] = static_cast<std::int32_t>(i);
  }
  tiles_ = exec::DeviceBuffer<LineTile>(batch.tiles.size(),
                                        "commutator::n7_batch_tiles");
  if (!batch.tiles.empty()) tiles_.CopyFromHost(batch.tiles);
  tile_of_channel_ = exec::DeviceBuffer<std::int32_t>(
      of_channel.size(), "commutator::n7_batch_tile_of_channel");
  if (!of_channel.empty()) tile_of_channel_.CopyFromHost(of_channel);
}

LineBatchDevice::LineBatchDevice(const LineBatchDevice& other) = default;
LineBatchDevice& LineBatchDevice::operator=(const LineBatchDevice& other) =
    default;
LineBatchDevice::LineBatchDevice(LineBatchDevice&& other) noexcept = default;
LineBatchDevice& LineBatchDevice::operator=(LineBatchDevice&& other) noexcept =
    default;
LineBatchDevice::~LineBatchDevice() = default;

LineBatchView LineBatchDevice::view() const noexcept {
  LineBatchView v;
  v.tiles = tiles_.device_data();
  v.tile_of_channel = tile_of_channel_.device_data();
  v.num_tiles = num_tiles_;
  v.num_lines = num_lines_;
  v.h_doubles = h_doubles_;
  v.w_doubles = w_doubles_;
  return v;
}

}  // namespace imsrg::commutator::detail
