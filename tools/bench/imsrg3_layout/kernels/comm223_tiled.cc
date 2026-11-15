// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/kernels/tile_engine.h"

namespace imsrg::layout3 {

namespace detail {

/// K1 body of one bra triple: all tiles (t, t' <= t) of its channel, with the
/// layout epilogue (`write_z`) or only W (the checksum of W goes to `acc`).
struct Comm223Body {
  double* z = nullptr;
  bool write_z = true;

  IMSRG_FUNCTION void operator()(const TileViews& tv, const TileScratch& sc,
                                 std::int32_t t, double& acc) const {
    const std::int32_t c = tv.cv.tr_channel[t];
    for (std::int32_t tp = tv.cv.ch_triple_begin[c]; tp <= t; ++tp) {
      const TileDims d = ComputeW(tv, sc, t, tp);
      if (write_z) {
        StoreTile(tv, sc, t, tp, d, z);
      } else {
        double sum = 0.0;
        for (std::int32_t i = 0; i < d.m * d.mp; ++i) sum += sc.wt[i];
        acc += sum;
      }
    }
  }
};
static_assert(std::is_trivially_copyable_v<Comm223Body>);

}  // namespace detail

/// K1 launch: see kernel_context.h. The layout tables are read on the host
/// (OpenMP/Serial backends, where device data is host data) to size the
/// scratch and to check `z`.
void Comm223Tiled(const KernelContext& ctx, const LayoutView& layout,
                  const op::Operator& X, const op::Operator& Y,
                  std::span<double> z, TiledMode mode) {
  const std::int64_t total = layout.ch_block_offset[layout.num_channels];
  IMSRG_CHECK(z.size() >= static_cast<std::size_t>(total),
              "Comm223Tiled: z is smaller than the layout blocks");
  const detail::TileSizes sz = detail::MakeTileSizes(ctx, layout);
  detail::GatherTables tables;
  const detail::TileViews tv =
      detail::MakeTileViews(ctx, layout, X, Y, sz, &tables);
  detail::Comm223Body body;
  body.z = z.data();
  body.write_z = mode == TiledMode::kFull;
  const double checksum =
      detail::RunTileLeague("imsrg3_layout::comm223_tiled", tv, sz,
                            ctx.WorkOrder(), ctx.Space().NumTriples(), body);
  (void)checksum;  // Only keeps the W-only kernel from being optimized away.
}

}  // namespace imsrg::layout3
