// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/exec/device_buffer.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"
#include "tools/bench/imsrg3_layout/kernels/tile_engine.h"

namespace imsrg::layout3 {

namespace detail {

/// Triples body of one bra triple t: the tiles (t, t' < t) with a nonzero
/// occupation factor, E_t = (2J + 1) sum f sum (w w Z)^2.
struct TriplesBody {
  const double* epsilon = nullptr;

  IMSRG_FUNCTION void operator()(const TileViews& tv, const TileScratch& sc,
                                 std::int32_t t, double& acc) const {
    constexpr double kSkip = 1.0e-8;
    const double occ_t = Occ(tv, t);
    const double unocc_t = Unocc(tv, t);
    if (occ_t < kSkip && unocc_t < kSkip) return;
    const std::int32_t c = tv.cv.tr_channel[t];
    const double two_j_plus_1 = static_cast<double>(tv.cv.ch_two_j[c] + 1);
    const double e_t = Energy(tv, t);
    for (std::int32_t tp = tv.cv.ch_triple_begin[c]; tp < t; ++tp) {
      const double num = occ_t * Unocc(tv, tp) - Occ(tv, tp) * unocc_t;
      if (Abs(num) < kSkip) continue;
      const double f = num / (e_t - Energy(tv, tp));
      const TileDims d = ComputeW(tv, sc, t, tp);
      acc += two_j_plus_1 * f * TileEnergy(tv, sc, t, tp, d);
    }
  }

 private:
  IMSRG_INLINE_FUNCTION static double Abs(double x) { return x < 0 ? -x : x; }
  IMSRG_INLINE_FUNCTION static double Occ(const TileViews& tv, std::int32_t t) {
    return tv.ob.occupation[tv.cv.tr_a[t]] * tv.ob.occupation[tv.cv.tr_b[t]] *
           tv.ob.occupation[tv.cv.tr_c[t]];
  }
  IMSRG_INLINE_FUNCTION static double Unocc(const TileViews& tv,
                                            std::int32_t t) {
    return (1.0 - tv.ob.occupation[tv.cv.tr_a[t]]) *
           (1.0 - tv.ob.occupation[tv.cv.tr_b[t]]) *
           (1.0 - tv.ob.occupation[tv.cv.tr_c[t]]);
  }
  IMSRG_INLINE_FUNCTION double Energy(const TileViews& tv,
                                      std::int32_t t) const {
    return epsilon[tv.cv.tr_a[t]] + epsilon[tv.cv.tr_b[t]] +
           epsilon[tv.cv.tr_c[t]];
  }
};
static_assert(std::is_trivially_copyable_v<TriplesBody>);

}  // namespace detail

double TriplesTiled(const KernelContext& ctx, const LayoutView& layout,
                    const op::Operator& X, const op::Operator& Y,
                    std::span<const double> epsilon) {
  IMSRG_CHECK(
      static_cast<std::int32_t>(epsilon.size()) == ctx.Orbits().num_orbits,
      "TriplesTiled: one epsilon per orbit expected");
  const detail::TileSizes sz = detail::MakeTileSizes(ctx, layout);
  detail::GatherTables tables;
  const detail::TileViews tv =
      detail::MakeTileViews(ctx, layout, X, Y, sz, &tables);
  exec::DeviceBuffer<double> eps(epsilon.size(), "imsrg3_layout::epsilon");
  eps.CopyFromHost(epsilon);
  detail::TriplesBody body;
  body.epsilon = eps.device_data();
  return detail::RunTileLeague("imsrg3_layout::triples_tiled", tv, sz,
                               ctx.WorkOrder(), ctx.Space().NumTriples(), body);
}

}  // namespace imsrg::layout3
