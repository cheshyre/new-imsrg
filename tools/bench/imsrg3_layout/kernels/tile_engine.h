// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_KERNELS_TILE_ENGINE_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_KERNELS_TILE_ENGINE_H_
/// @file
/// Tile code shared by the tiled [2,2]->3 kernel K1 (comm223_tiled.cc) and the
/// perturbative-triples energy (triples.cc); internal to the layout
/// benchmark, includes Kokkos (design/12 section 5 device rules: raw-pointer
/// views captured by value, no allocation or exceptions in device code,
/// scratch from the team scratch).
///
/// A tile is the pair of bra triple t and ket triple t' <= t of one channel.
/// Its intermediate W (m x m', m = pair-third states of t) is
///   W[r, s] = Hat(J1') Hat(J2') sum_a six (X_{P,(w a)} Y~_{(z a),Q}
///                                          - Y_{P,(w a)} X~_{(z a),Q})
/// for the row r = (P, z, J1') of t and the column s = (Q, w, J2') of t'.
///
/// Gather first: no two-body element is read inside the r x s x a loop. The
/// vectors over the internal orbit a of the elements are gathered ONCE per
/// kernel launch for every (two-body ket P, J, third orbit w) into the
/// gather tables (`GatherTables`, a parallel pre-pass; its cost is
/// O(kets J orbits S) get_normalized calls, a small fraction of the tile
/// work), a = a_k the k-th orbit of the orbit group of (P, w):
///   table_x[((slot(P, J)) n_orb + w) S + k] =
///       X_{P,(w a_k)} (w == a_k ? sqrt 2 : 1),
/// and table_y for Y (S = the largest group). The tile reads both legs from
/// these rows: the ket leg X~_{(z a),Q} equals h_X X_{Q,(z a)}, because a
/// scalar operator of definite hermiticity h_X has symmetric (h_X = +1) or
/// antisymmetric (-1) two-body blocks, so the same table serves the bra and
/// the ket side and
///   W[r, s] = Hat Hat sum_k six_k (h_Y x_{P,w}[k] y_{Q,z}[k]
///                                  - h_X y_{P,w}[k] x_{Q,z}[k]),
/// with one short dot product per (r, s). The 6j weights come from the dense
/// `SixJ3View` data through precomputed strides.
/// The team scratch holds W^T (`wt[s m + r]`, column-major in W), the
/// half-transformed U^T (`ut[k' m + r]`, k' the layout rows of t') and the
/// small per-state index tables of the tile.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/operator/operator_spec.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"

namespace imsrg::layout3::detail {

/// Maximum number of distinct third orbits of a triple.
inline constexpr std::int32_t kMaxThirds = 3;

/// Everything a tile kernel reads (trivially copyable, captured by value).
struct TileViews {
  CanonicalView cv;
  PairThirdView pv;
  SixJ3View sj;
  basis::OrbitTableView ob;
  OrbitGroupView groups;
  commutator::detail::TwoBodyView xv;
  commutator::detail::TwoBodyView yv;
  LayoutView lv;
  const double* table_x = nullptr;  ///< Gather tables (see the file comment).
  const double* table_y = nullptr;
  /// Strides of the dense 6j table (see SixJ3Table): the element
  /// {j1 j2 J12; j3 J J23} is at c1 i1 + c2 i2 + c3 J12 + c4 i3 + c5 iJ + J23
  /// with i = (2j - 1) / 2, iJ = (2J - 1) / 2, J12 and J23 integer.
  std::int64_t c1 = 0;
  std::int64_t c2 = 0;
  std::int64_t c3 = 0;
  std::int64_t c4 = 0;
  std::int64_t c5 = 0;
  std::int32_t stride = 0;  ///< S, the largest orbit group.
  double hx = 1.0;          ///< Hermiticity signs of X and Y.
  double hy = 1.0;
  double hz = 1.0;  ///< h_Z of Z = [X, Y].
};
static_assert(std::is_trivially_copyable_v<TileViews>);

/// Largest extents of a tile, which size the team scratch.
struct TileSizes {
  std::int32_t m_max = 0;  ///< Pair-third states of the largest triple.
  std::int32_t k_max = 0;  ///< Layout rows of the largest triple.
  std::int32_t s_max = 0;  ///< Largest orbit group.
};
static_assert(std::is_trivially_copyable_v<TileSizes>);

/// Number of doubles of the team scratch for `sz`.
inline std::size_t ScratchDoubles(const TileSizes& sz) {
  const auto m = static_cast<std::size_t>(sz.m_max);
  const auto k = static_cast<std::size_t>(sz.k_max);
  const auto s = static_cast<std::size_t>(sz.s_max);
  const std::size_t ints = 12 * m + 4 * s + 4;
  return m * m + k * m + 2 * m + (ints + 1) / 2 + 1;
}

/// Pointers into the team scratch (carved by `CarveScratch`).
struct TileScratch {
  double* wt = nullptr;
  double* ut = nullptr;
  double* row_hat = nullptr;           ///< Hat(J1') of row r.
  double* col_hat = nullptr;           ///< Hat(J2') of column s.
  std::int32_t* row_slot = nullptr;    ///< Gather slot of (P, J1') of row r.
  std::int32_t* row_z = nullptr;       ///< Third orbit z of row r.
  std::int32_t* row_j = nullptr;       ///< Jpq of row r (integer).
  std::int32_t* row_jz = nullptr;      ///< (2 j_z - 1) / 2 of row r.
  std::int32_t* row_group = nullptr;   ///< Orbit group of (row, w slot).
  std::int32_t* col_slot = nullptr;    ///< Gather slot of (Q, J2').
  std::int32_t* col_w = nullptr;       ///< Third orbit w of column s.
  std::int32_t* col_j = nullptr;       ///< Jpq of column s (integer).
  std::int32_t* col_jw = nullptr;      ///< (2 j_w - 1) / 2 of column s.
  std::int32_t* col_slot_w = nullptr;  ///< Slot of w among thirds(t').
  std::int32_t* ja_off = nullptr;      ///< 4 S: c2 (2 j_a - 1) / 2 of a.
  std::int32_t* group_size = nullptr;  ///< 4 entries.
};
static_assert(std::is_trivially_copyable_v<TileScratch>);

/// The distinct third orbits of a triple.
struct Thirds {
  std::int32_t n = 0;
  std::int32_t orbit[kMaxThirds] = {-1, -1, -1};
};

/// The distinct orbits among (a, b, c), in this order.
IMSRG_INLINE_FUNCTION Thirds DistinctOrbits(std::int32_t a, std::int32_t b,
                                            std::int32_t c) noexcept {
  Thirds t;
  t.orbit[0] = a;
  t.n = 1;
  if (b != a) t.orbit[t.n++] = b;
  if (c != a && c != b) t.orbit[t.n++] = c;
  return t;
}

/// Index of `orbit` in `t` (0 if absent; never absent for the thirds of the
/// pair-third states of the triple).
IMSRG_INLINE_FUNCTION std::int32_t SlotOf(const Thirds& t,
                                          std::int32_t orbit) noexcept {
  for (std::int32_t i = 0; i < t.n; ++i) {
    if (t.orbit[i] == orbit) return i;
  }
  return 0;
}

/// Orbit group of parity `parity` and 2tz sum `tz2`, or -1 if |tz2| != 1.
IMSRG_INLINE_FUNCTION std::int32_t GroupOrEmpty(std::int32_t parity,
                                                std::int32_t tz2) noexcept {
  return (tz2 == 1 || tz2 == -1) ? OrbitGroupView::Group(parity, tz2) : -1;
}

/// Gather slot of the two-body state (ket (p, q), `two_j`): the position of
/// the (ket, J) pair in the basis table (`ket_J_offsets`); `p <= q`, J in the
/// range of the pair. O(1).
IMSRG_INLINE_FUNCTION std::int32_t GatherSlot(const basis::TwoBodyTableView& tb,
                                              std::int32_t p, std::int32_t q,
                                              std::int32_t two_j) noexcept {
  const std::int32_t jp = tb.orbit_j2[p];
  const std::int32_t jq = tb.orbit_j2[q];
  const std::int32_t jmin = (jp > jq ? jp - jq : jq - jp) / 2;
  return tb.ket_J_offsets[tb.ket_index(OrbitIndex{p}, OrbitIndex{q}).value()] +
         (two_j / 2 - jmin);
}

/// Parity of l_p + l_q + l_w and 2tz sum (tz_p + tz_q - tz_w) select the group
/// of the internal orbit; this returns it (or -1 if empty). O(1).
IMSRG_INLINE_FUNCTION std::int32_t GatherGroup(const basis::OrbitTableView& ob,
                                               std::int32_t p, std::int32_t q,
                                               std::int32_t w) noexcept {
  return GroupOrEmpty((ob.l[p] + ob.l[q] + ob.l[w]) % 2,
                      ob.tz2[p] + ob.tz2[q] - ob.tz2[w]);
}

/// The two gather tables (device data).
struct GatherTables {
  exec::DeviceBuffer<double> x;
  exec::DeviceBuffer<double> y;
};

/// Views of the pre-pass `FillGatherTables`.
struct GatherViews {
  commutator::detail::TwoBodyView xv;
  commutator::detail::TwoBodyView yv;
  basis::OrbitTableView ob;
  OrbitGroupView groups;
  double* tx = nullptr;
  double* ty = nullptr;
  std::int32_t stride = 0;
};
static_assert(std::is_trivially_copyable_v<GatherViews>);

/// Fills the gather table entries of one two-body ket: for every (J, third
/// w) with a nonempty group, the vectors over a of X (and Y) elements times
/// sqrt 2 for w == a. Entries not written stay zero. Cost O(sum over the kets
/// of (J range) n_orb S) `get_normalized` calls; no allocation.
IMSRG_INLINE_FUNCTION void FillGatherKet(const GatherViews& gv,
                                         std::int32_t ket) noexcept {
  const basis::TwoBodyTableView& tb = gv.xv.tb;
  const std::int32_t p = tb.ket_p[ket];
  const std::int32_t q = tb.ket_q[ket];
  const std::int32_t jp = tb.orbit_j2[p];
  const std::int32_t jq = tb.orbit_j2[q];
  const std::int32_t jmin = (jp > jq ? jp - jq : jq - jp) / 2;
  const std::int32_t jmax = (jp + jq) / 2;
  const std::int32_t norb = tb.num_orbits;
  for (std::int32_t J = jmin; J <= jmax; ++J) {
    if (tb.local_index(KetIndex{ket}, TwoJ{2 * J}) < 0) continue;
    const std::int64_t slot = tb.ket_J_offsets[ket] + (J - jmin);
    for (std::int32_t w = 0; w < norb; ++w) {
      const std::int32_t g = GatherGroup(gv.ob, p, q, w);
      if (g < 0) continue;
      const std::int32_t lo = gv.groups.offset[g];
      const std::int32_t n = gv.groups.offset[g + 1] - lo;
      double* const ox = gv.tx + (slot * norb + w) * gv.stride;
      double* const oy = gv.ty + (slot * norb + w) * gv.stride;
      for (std::int32_t k = 0; k < n; ++k) {
        const std::int32_t a = gv.groups.orbits[lo + k];
        const double fac = w == a ? constants::kSqrt2 : 1.0;
        ox[k] = fac * gv.xv.get_normalized(p, q, w, a, 2 * J);
        oy[k] = fac * gv.yv.get_normalized(p, q, w, a, 2 * J);
      }
    }
  }
}

/// Launches the pre-pass: one task per two-body ket, dynamic schedule. A free
/// function (design/12 section 5). Asynchronous on async backends.
inline void FillGatherTables(const GatherViews& gv) {
  using Policy = Kokkos::RangePolicy<exec::DefaultExecSpace,
                                     Kokkos::Schedule<Kokkos::Dynamic>,
                                     Kokkos::IndexType<std::int32_t>>;
  const std::int32_t num_kets = gv.xv.tb.num_kets;
  Kokkos::parallel_for(
      "imsrg3_layout::gather_tables", Policy(0, num_kets),
      IMSRG_LAMBDA(const std::int32_t ket) { FillGatherKet(gv, ket); });
}

/// Carves the scratch (`base` with `ScratchDoubles(sz)` doubles) and fills the
/// tile-independent tables (6j offsets and sizes of the orbit groups). Cost
/// O(4 S).
IMSRG_INLINE_FUNCTION TileScratch CarveScratch(double* base,
                                               const TileViews& tv,
                                               const TileSizes& sz) noexcept {
  const auto m = static_cast<std::size_t>(sz.m_max);
  const auto k = static_cast<std::size_t>(sz.k_max);
  const auto s = static_cast<std::size_t>(sz.s_max);
  TileScratch sc;
  double* p = base;
  sc.wt = p;
  p += m * m;
  sc.ut = p;
  p += k * m;
  sc.row_hat = p;
  p += m;
  sc.col_hat = p;
  p += m;
  auto* ip = reinterpret_cast<std::int32_t*>(p);
  sc.row_slot = ip;
  ip += m;
  sc.row_z = ip;
  ip += m;
  sc.row_j = ip;
  ip += m;
  sc.row_jz = ip;
  ip += m;
  sc.row_group = ip;
  ip += kMaxThirds * m;
  sc.col_slot = ip;
  ip += m;
  sc.col_w = ip;
  ip += m;
  sc.col_j = ip;
  ip += m;
  sc.col_jw = ip;
  ip += m;
  sc.col_slot_w = ip;
  ip += m;
  sc.ja_off = ip;
  ip += 4 * s;
  sc.group_size = ip;
  for (std::int32_t g = 0; g < 4; ++g) {
    const std::int32_t lo = tv.groups.offset[g];
    const std::int32_t n = tv.groups.offset[g + 1] - lo;
    sc.group_size[g] = n;
    for (std::int32_t i = 0; i < n; ++i) {
      const std::int32_t a = tv.groups.orbits[lo + i];
      sc.ja_off[g * sz.s_max + i] =
          static_cast<std::int32_t>(tv.c2 * ((tv.ob.j2[a] - 1) / 2));
    }
  }
  return sc;
}

/// Extents of a prepared tile.
struct TileDims {
  std::int32_t m = 0;   ///< Pair-third states of t (rows of W).
  std::int32_t mp = 0;  ///< Pair-third states of t' (columns of W).
};

/// Computes W[t, t'] into `sc.wt` (W^T column-major, `wt[s m + r]`) from the
/// gather tables and the 6j table (formula in the file comment).
/// Preconditions: t' in the channel of t, `sc` carved from scratch of at least
/// the extents of both triples. Cost O(m m' n_a) multiply-adds plus O(m + m')
/// index set-up; no allocation.
IMSRG_INLINE_FUNCTION TileDims ComputeW(const TileViews& tv,
                                        const TileScratch& sc, std::int32_t t,
                                        std::int32_t tp) noexcept {
  const std::int32_t c = tv.cv.tr_channel[t];
  const std::int32_t two_J = tv.cv.ch_two_j[c];
  const std::int32_t pt_base = tv.pv.ch_pt_offset[c];
  const std::int32_t f0 = pt_base + tv.pv.tr_pt_begin[t];
  const std::int32_t f1 = pt_base + tv.pv.tr_pt_begin[tp];
  TileDims d;
  d.m = tv.pv.tr_pt_end[t] - tv.pv.tr_pt_begin[t];
  d.mp = tv.pv.tr_pt_end[tp] - tv.pv.tr_pt_begin[tp];
  const std::int32_t S = tv.stride;
  const std::int64_t norb = tv.ob.num_orbits;
  const basis::TwoBodyTableView& tb = tv.xv.tb;
  const Thirds ws =
      DistinctOrbits(tv.cv.tr_a[tp], tv.cv.tr_b[tp], tv.cv.tr_c[tp]);

  // Index set-up of the rows (P, z, J1') of t and the columns (Q, w, J2') of
  // t'.
  for (std::int32_t r = 0; r < d.m; ++r) {
    const std::int32_t f = f0 + r;
    const std::int32_t p = tv.pv.pt_p[f];
    const std::int32_t q = tv.pv.pt_q[f];
    const std::int32_t z = tv.pv.pt_r[f];
    const std::int32_t tj = tv.pv.pt_two_jpq[f];
    sc.row_slot[r] = GatherSlot(tb, p, q, tj);
    sc.row_z[r] = z;
    sc.row_j[r] = tj / 2;
    sc.row_jz[r] = (tv.ob.j2[z] - 1) / 2;
    sc.row_hat[r] = tv.sj.Hat(tj / 2);
    for (std::int32_t iw = 0; iw < ws.n; ++iw) {
      sc.row_group[r * kMaxThirds + iw] =
          GatherGroup(tv.ob, p, q, ws.orbit[iw]);
    }
  }
  for (std::int32_t s = 0; s < d.mp; ++s) {
    const std::int32_t f = f1 + s;
    const std::int32_t p = tv.pv.pt_p[f];
    const std::int32_t q = tv.pv.pt_q[f];
    const std::int32_t w = tv.pv.pt_r[f];
    const std::int32_t tj = tv.pv.pt_two_jpq[f];
    sc.col_slot[s] = GatherSlot(tb, p, q, tj);
    sc.col_w[s] = w;
    sc.col_j[s] = tj / 2;
    sc.col_jw[s] = (tv.ob.j2[w] - 1) / 2;
    sc.col_hat[s] = tv.sj.Hat(tj / 2);
    sc.col_slot_w[s] = SlotOf(ws, w);
  }

  // The tile: short dot products with the 6j weights.
  const double* const six = tv.sj.data;
  const std::int64_t j_base = tv.c5 * ((two_J - 1) / 2);
  for (std::int32_t s = 0; s < d.mp; ++s) {
    const std::int32_t iw = sc.col_slot_w[s];
    const std::int64_t w = sc.col_w[s];
    const std::int64_t col_base =
        j_base + tv.c1 * sc.col_jw[s] + sc.col_j[s];  // + c3 J1' + c4 jz + c2 a
    const std::int64_t qrow = static_cast<std::int64_t>(sc.col_slot[s]) * norb;
    const double hat_s = sc.col_hat[s];
    double* const out = sc.wt + static_cast<std::int64_t>(s) * d.m;
    for (std::int32_t r = 0; r < d.m; ++r) {
      const std::int32_t g = sc.row_group[r * kMaxThirds + iw];
      if (g < 0) {
        out[r] = 0.0;
        continue;
      }
      const std::int32_t n = sc.group_size[g];
      const std::int64_t base =
          col_base + tv.c3 * sc.row_j[r] + tv.c4 * sc.row_jz[r];
      const std::int64_t prow =
          (static_cast<std::int64_t>(sc.row_slot[r]) * norb + w) * S;
      const std::int64_t crow = (qrow + sc.row_z[r]) * S;
      const double* const px = tv.table_x + prow;
      const double* const py = tv.table_y + prow;
      const double* const qx = tv.table_x + crow;
      const double* const qy = tv.table_y + crow;
      const std::int32_t* const jo = sc.ja_off + g * S;
      double a1 = 0.0;
      double a2 = 0.0;
      for (std::int32_t k = 0; k < n; ++k) {
        const double w6 = six[base + jo[k]];
        a1 += w6 * px[k] * qy[k];
        a2 += w6 * py[k] * qx[k];
      }
      out[r] = sc.row_hat[r] * hat_s * (tv.hy * a1 - tv.hx * a2);
    }
  }
  return d;
}

/// First half of the epilogue: U^T[k', r] = sum_s W[r, s] L_t'[k', s] into
/// `sc.ut` (`ut[k' m + r]`) for the layout rows k' of t'. Cost O(nnz(L_t') m).
IMSRG_INLINE_FUNCTION void OuterRight(const TileViews& tv,
                                      const TileScratch& sc, std::int32_t t,
                                      std::int32_t tp,
                                      const TileDims& d) noexcept {
  const std::int32_t c = tv.cv.tr_channel[t];
  const std::int64_t row0 = static_cast<std::int64_t>(tv.lv.ch_row_offset[c]) +
                            tv.lv.tr_row_begin[tp];
  const std::int32_t kp = tv.lv.tr_row_end[tp] - tv.lv.tr_row_begin[tp];
  const std::int32_t pt0 = tv.pv.tr_pt_begin[tp];
  for (std::int32_t k = 0; k < kp; ++k) {
    double* const u = sc.ut + static_cast<std::int64_t>(k) * d.m;
    for (std::int32_t r = 0; r < d.m; ++r) u[r] = 0.0;
    for (std::int64_t e = tv.lv.outer.ptr[row0 + k];
         e < tv.lv.outer.ptr[row0 + k + 1]; ++e) {
      const double v = tv.lv.outer.val[e];
      const double* const w =
          sc.wt + static_cast<std::int64_t>(tv.lv.outer.col[e] - pt0) * d.m;
      for (std::int32_t r = 0; r < d.m; ++r) u[r] += v * w[r];
    }
  }
}

/// Entry Z[k, k'] = sum_j L_t[k, j] U[j, k'] (second half of the epilogue):
/// `ut_col` is the column U^T[k', .] of `OuterRight`, `row` the flat layout row
/// of k, `pt0` the channel-local first pair-third state of t. Cost O(nnz of
/// the row).
IMSRG_INLINE_FUNCTION double OuterLeft(const TileViews& tv,
                                       const double* ut_col, std::int64_t row,
                                       std::int32_t pt0) noexcept {
  double s = 0.0;
  for (std::int64_t e = tv.lv.outer.ptr[row]; e < tv.lv.outer.ptr[row + 1];
       ++e) {
    s += tv.lv.outer.val[e] * ut_col[tv.lv.outer.col[e] - pt0];
  }
  return s;
}

/// Full epilogue of a computed tile: writes Z[t, t'] and, for t' != t,
/// h_Z Z[t, t']^T to the layout block of the channel, `z` the base pointer of
/// the layout blocks (column-major). Cost O(nnz(L_t') m + nnz(L_t) k').
IMSRG_INLINE_FUNCTION void StoreTile(const TileViews& tv, const TileScratch& sc,
                                     std::int32_t t, std::int32_t tp,
                                     const TileDims& d, double* z) noexcept {
  OuterRight(tv, sc, t, tp, d);
  const std::int32_t c = tv.cv.tr_channel[t];
  const std::int64_t n_dim = tv.lv.ch_dim[c];
  double* const blk = z + tv.lv.ch_block_offset[c];
  const std::int32_t rb = tv.lv.tr_row_begin[t];
  const std::int32_t cb = tv.lv.tr_row_begin[tp];
  const std::int32_t kt = tv.lv.tr_row_end[t] - rb;
  const std::int32_t kp = tv.lv.tr_row_end[tp] - cb;
  const std::int64_t row0 =
      static_cast<std::int64_t>(tv.lv.ch_row_offset[c]) + rb;
  const std::int32_t pt0 = tv.pv.tr_pt_begin[t];
  const bool mirror = t != tp;
  for (std::int32_t kc = 0; kc < kp; ++kc) {
    const double* const u = sc.ut + static_cast<std::int64_t>(kc) * d.m;
    double* const col = blk + (cb + kc) * n_dim + rb;
    for (std::int32_t k = 0; k < kt; ++k) {
      const double v = OuterLeft(tv, u, row0 + k, pt0);
      col[k] = v;
      if (mirror) blk[(cb + kc) + (rb + k) * n_dim] = tv.hz * v;
    }
  }
}

/// Sum over the tile of (weight_k weight_k' Z[k, k'])^2 with the layout's
/// energy weights (flat layout rows), without storing Z. Cost as `StoreTile`.
IMSRG_INLINE_FUNCTION double TileEnergy(const TileViews& tv,
                                        const TileScratch& sc, std::int32_t t,
                                        std::int32_t tp,
                                        const TileDims& d) noexcept {
  OuterRight(tv, sc, t, tp, d);
  const std::int32_t c = tv.cv.tr_channel[t];
  const std::int32_t rb = tv.lv.tr_row_begin[t];
  const std::int32_t cb = tv.lv.tr_row_begin[tp];
  const std::int32_t kt = tv.lv.tr_row_end[t] - rb;
  const std::int32_t kp = tv.lv.tr_row_end[tp] - cb;
  const std::int64_t row0 =
      static_cast<std::int64_t>(tv.lv.ch_row_offset[c]) + rb;
  const std::int64_t col0 =
      static_cast<std::int64_t>(tv.lv.ch_row_offset[c]) + cb;
  const std::int32_t pt0 = tv.pv.tr_pt_begin[t];
  double sum = 0.0;
  for (std::int32_t kc = 0; kc < kp; ++kc) {
    const double wc = tv.lv.energy_weight[col0 + kc];
    if (wc == 0.0) continue;
    const double* const u = sc.ut + static_cast<std::int64_t>(kc) * d.m;
    for (std::int32_t k = 0; k < kt; ++k) {
      const double wr = tv.lv.energy_weight[row0 + k];
      const double v = wr * wc * OuterLeft(tv, u, row0 + k, pt0);
      sum += v * v;
    }
  }
  return sum;
}

/// Largest tile extents over all triples. Reads the host-accessible tables of
/// the layout and the orbit groups (OpenMP and Serial backends: the device
/// memory is the host memory). O(num_triples).
inline TileSizes MakeTileSizes(const KernelContext& ctx,
                               const LayoutView& layout) {
  const PairThirdView pv = ctx.PairThird().host_view();
  const OrbitGroupView g = ctx.Groups();
  const std::int32_t num_triples = ctx.Space().NumTriples();
  TileSizes sz;
  for (std::int32_t t = 0; t < num_triples; ++t) {
    sz.m_max = std::max(sz.m_max, pv.tr_pt_end[t] - pv.tr_pt_begin[t]);
    sz.k_max =
        std::max(sz.k_max, layout.tr_row_end[t] - layout.tr_row_begin[t]);
  }
  for (std::int32_t i = 0; i < 4; ++i) {
    sz.s_max = std::max(sz.s_max, g.offset[i + 1] - g.offset[i]);
  }
  return sz;
}

/// Builds the tile views of a kernel launch (context tables, operands,
/// layout) and fills the gather tables `tables` (blocking). `X` and `Y` must
/// be scalar with definite hermiticity (IMSRG_CHECK in
/// `CommutatorHermiticity`). Cost: the pre-pass of the file comment.
inline TileViews MakeTileViews(const KernelContext& ctx,
                               const LayoutView& layout, const op::Operator& X,
                               const op::Operator& Y, const TileSizes& sz,
                               GatherTables* tables) {
  TileViews tv;
  tv.cv = ctx.Space().device_view();
  tv.pv = ctx.PairThird().device_view();
  tv.sj = ctx.SixJ().device_view();
  tv.ob = ctx.Orbits();
  tv.groups = ctx.Groups();
  tv.xv = ctx.TwoBody(X);
  tv.yv = ctx.TwoBody(Y);
  tv.lv = layout;
  const std::int64_t nj = tv.sj.emax + 1;
  const std::int64_t nJ = 2 * static_cast<std::int64_t>(tv.sj.emax) + 2;
  const std::int64_t nJ3 = 3 * static_cast<std::int64_t>(tv.sj.emax) + 2;
  tv.c5 = nJ;
  tv.c4 = nJ3 * nJ;
  tv.c3 = nj * tv.c4;
  tv.c2 = nJ * tv.c3;
  tv.c1 = nj * tv.c2;
  tv.stride = sz.s_max;
  tv.hx = op::HermiticitySign(X.spec().hermiticity);
  tv.hy = op::HermiticitySign(Y.spec().hermiticity);
  tv.hz = CommutatorHermiticity(X, Y);
  if (tables != nullptr) {
    const std::int64_t slots =
        tv.xv.tb.ket_J_offsets[tv.xv.tb.num_kets];  // host-accessible
    const auto n = static_cast<std::size_t>(slots) *
                   static_cast<std::size_t>(tv.ob.num_orbits) *
                   static_cast<std::size_t>(sz.s_max);
    tables->x = exec::DeviceBuffer<double>(n, "imsrg3_layout::gather_x");
    tables->y = exec::DeviceBuffer<double>(n, "imsrg3_layout::gather_y");
    GatherViews gv;
    gv.xv = tv.xv;
    gv.yv = tv.yv;
    gv.ob = tv.ob;
    gv.groups = tv.groups;
    gv.tx = tables->x.device_data();
    gv.ty = tables->y.device_data();
    gv.stride = sz.s_max;
    FillGatherTables(gv);
    exec::Fence("imsrg3_layout::gather_tables");
    tv.table_x = tables->x.device_data();
    tv.table_y = tables->y.device_data();
  }
  return tv;
}

/// Launches `body(tv, scratch, t, acc)` once per bra triple t = order[i], one
/// team (size 1, dynamic schedule) per league index i, with a `TileScratch`
/// carved from team scratch (level 0 if it fits `scratch_size_max(0)`, else
/// level 1), and returns the sum of the `acc` contributions. `body` is a
/// trivially copyable functor `IMSRG_FUNCTION void operator()(const
/// TileViews&, const TileScratch&, std::int32_t t, double& acc) const`.
/// Blocking.
template <class Body>
double RunTileLeague(const char* name, const TileViews& tv, const TileSizes& sz,
                     const std::int32_t* order, std::int32_t num_triples,
                     const Body& body) {
  using Policy = Kokkos::TeamPolicy<exec::DefaultExecSpace,
                                    Kokkos::Schedule<Kokkos::Dynamic>>;
  using Member = Policy::member_type;
  using Scratch =
      Kokkos::View<double*, exec::DefaultExecSpace::scratch_memory_space,
                   Kokkos::MemoryUnmanaged>;
  const std::size_t doubles = ScratchDoubles(sz);
  const std::size_t bytes = Scratch::shmem_size(doubles);
  const int level =
      bytes <= static_cast<std::size_t>(Policy::scratch_size_max(0)) ? 0 : 1;
  IMSRG_CHECK(bytes <= static_cast<std::size_t>(Policy::scratch_size_max(1)),
              "RunTileLeague: the team scratch does not fit");
  Policy policy(num_triples, 1);
  policy.set_scratch_size(level, Kokkos::PerTeam(bytes));
  double result = 0.0;
  Kokkos::parallel_reduce(
      name, policy,
      IMSRG_LAMBDA(const Member& member, double& acc) {
        Scratch buf(member.team_scratch(level), doubles);
        const TileScratch sc = CarveScratch(buf.data(), tv, sz);
        body(tv, sc, order[member.league_rank()], acc);
      },
      result);
  return result;
}

}  // namespace imsrg::layout3::detail

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_KERNELS_TILE_ENGINE_H_
