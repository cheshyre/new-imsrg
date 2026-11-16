// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The N7 term comm330ss, [X3, Y3] -> Z0 (imsrg3.h). Kernel TU.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/three_body_rows.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/commutator/detail/n7_batch.h"
#include "imsrg/commutator/detail/n7_common.h"
#include "imsrg/commutator/detail/n7_terms.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_impl.h"
#include "imsrg/commutator/imsrg3.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"

namespace imsrg::commutator {

namespace {

/// Occupation products n_abc, nbar_abc and the multiplicity sym/6 of the
/// triple of flat row `r`.
struct RowOcc {
  double n = 0.0;
  double nbar = 0.0;
  double sym6 = 0.0;
};
static_assert(std::is_trivially_copyable_v<RowOcc>);

IMSRG_INLINE_FUNCTION RowOcc RowOccupation(const detail::N7Views& v,
                                           std::int64_t r) noexcept {
  const int t = v.row_triple[r];
  const double na = v.ob.occupation[v.sp.tr_a[t]];
  const double nb = v.ob.occupation[v.sp.tr_b[t]];
  const double nc = v.ob.occupation[v.sp.tr_c[t]];
  RowOcc o;
  o.n = na * nb * nc;
  o.nbar = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
  o.sym6 = v.sp.tr_sym[t] / 6.0;
  return o;
}

}  // namespace

namespace detail {

void Comm330ssReference(const op::Operator& X, const op::Operator& Y,
                        op::Operator& Z, const ThreeBodyContext& ctx) {
  const N7Views v = PrepareN7("Comm330ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr || v.y3.local == nullptr) return;
  // The scalar part of an anti-Hermitian Z vanishes (the sum below cancels
  // pairwise); skip it to keep Z0 exactly zero.
  if (Z.hermiticity() == Hermiticity::kAntiHermitian) return;
  const exec::Index nrows = static_cast<exec::Index>(
      ctx.model_space_ptr()->three_body_space()->NumRows());
  double sum = 0.0;
  exec::ParallelReduce(
      "commutator::comm330ss_reference", nrows,
      IMSRG_LAMBDA(const exec::Index r, double& partial) {
        const int t = v.row_triple[r];
        const int ch = v.sp.tr_channel[t];
        const std::int64_t base = v.sp.ch_row_offset[ch];
        const int i = static_cast<int>(r - base);
        const int n = v.sp.ch_dim[ch];
        const RowOcc oi = RowOccupation(v, r);
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
          if (j == i) continue;
          const RowOcc oj = RowOccupation(v, base + j);
          const double occ = oi.n * oj.nbar - oi.nbar * oj.n;
          if (occ == 0.0) continue;
          acc += oj.sym6 * occ * ThreeBodyLocal(v.x3, v.sp, ch, i, j) *
                 ThreeBodyLocal(v.y3, v.sp, ch, j, i);
        }
        partial += (v.sp.ch_two_j[ch] + 1) * oi.sym6 * acc;
      },
      sum);
  Z.zero_body() += sum;
}

double Comm330Batch(const N7BatchInputs& in) {
  const N7Views v = in.v;
  if (v.x3.local == nullptr || v.y3.local == nullptr) return 0.0;
  IMSRG_CHECK(in.plan != nullptr, "Comm330Batch: the batch inputs lack a plan");
  const std::vector<LineTile>& tiles = in.plan->tiles;
  // Line offsets of the tiles (host), so that a flat line index finds its
  // tile by binary search.
  std::vector<std::int64_t> off(tiles.size() + 1, 0);
  std::vector<std::int32_t> tile_ch(tiles.size());
  std::vector<std::int32_t> tile_row0(tiles.size());
  for (std::size_t t = 0; t < tiles.size(); ++t) {
    off[t + 1] = off[t] + (tiles[t].row_end - tiles[t].row_begin);
    tile_ch[t] = tiles[t].channel;
    tile_row0[t] = tiles[t].row_begin;
  }
  if (off.back() == 0) return 0.0;
  exec::DeviceBuffer<std::int64_t> d_off(off.size(), "commutator::comm330_off");
  d_off.CopyFromHost(std::span<const std::int64_t>(off));
  exec::DeviceBuffer<std::int32_t> d_ch(tile_ch.size(),
                                        "commutator::comm330_tile_ch");
  d_ch.CopyFromHost(std::span<const std::int32_t>(tile_ch));
  exec::DeviceBuffer<std::int32_t> d_row0(tile_row0.size(),
                                          "commutator::comm330_tile_row0");
  d_row0.CopyFromHost(std::span<const std::int32_t>(tile_row0));
  const std::int64_t* line_off = d_off.device_data();
  const std::int32_t* ch_of = d_ch.device_data();
  const std::int32_t* row0_of = d_row0.device_data();
  const int num_tiles = static_cast<int>(tiles.size());
  double sum = 0.0;
  exec::ParallelReduce(
      "commutator::comm330ss_batch", static_cast<exec::Index>(off.back()),
      IMSRG_LAMBDA(const exec::Index l, double& partial) {
        const int ti = FindBlock(line_off, num_tiles, l);
        const int ch = ch_of[ti];
        const std::int64_t base = v.sp.ch_row_offset[ch];
        const int k = row0_of[ti] + static_cast<int>(l - line_off[ti]);
        const int n = v.sp.ch_dim[ch];
        const RowOcc ok = RowOccupation(v, base + k);
        double acc = 0.0;
        for (int m = 0; m < n; ++m) {
          if (m == k) continue;
          const RowOcc om = RowOccupation(v, base + m);
          const double occ = om.n * ok.nbar - om.nbar * ok.n;
          if (occ == 0.0) continue;
          acc += om.sym6 * occ * ThreeBodyLocal(v.x3, v.sp, ch, m, k) *
                 ThreeBodyLocal(v.y3, v.sp, ch, k, m);
        }
        partial += (v.sp.ch_two_j[ch] + 1) * ok.sym6 * acc;
      },
      sum);
  return sum;
}

void Comm330Rows(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
                 const ThreeBodyContext& ctx, const basis::RowSet& rows,
                 std::int64_t budget_bytes) {
  const N7Views v = PrepareN7("Comm330ss", X, Y, Z, ctx);
  if (v.x3.local == nullptr || v.y3.local == nullptr) return;
  // The scalar part of an anti-Hermitian Z vanishes (the sum cancels
  // pairwise); skip it to keep Z0 exactly zero.
  if (Z.hermiticity() == Hermiticity::kAntiHermitian) return;
  double total = 0.0;
  ForEachN7Batch(v, ctx, rows, budget_bytes, N7Needs{false, false},
                 [&total](const N7BatchInputs& in) {
                   total += Comm330Batch(in);
                 });
  Z.zero_body() += total;
}

}  // namespace detail

namespace terms {

void Comm330ss(const op::Operator& X, const op::Operator& Y, op::Operator& Z,
               const ThreeBodyContext& ctx) {
  detail::Comm330Rows(
      X, Y, Z, ctx,
      basis::RowSet::Whole(*ctx.model_space_ptr()->three_body_space()),
      detail::kN7BatchBytes);
}

}  // namespace terms

}  // namespace imsrg::commutator
