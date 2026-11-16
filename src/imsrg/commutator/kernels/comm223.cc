// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The [2,2]->3 pair-third tiles (comm223ss and the perturbative triples;
// formula in three_body.h, decomposition in detail/three_body_tiles.h).
// Kernel TU: Kokkos team policies with team scratch. Nested team-level
// lambdas run inside the kernel and capture by reference (Kokkos idiom); the
// launched functor is trivially copyable and holds only views.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "imsrg/base/assert.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/exec/kokkos_backend.h"
#include "imsrg/operator/three_body.h"

namespace imsrg::commutator::detail {

namespace {

using Policy = Kokkos::TeamPolicy<exec::DefaultExecSpace,
                                  Kokkos::Schedule<Kokkos::Dynamic>>;
using Member = Policy::member_type;
using ScratchSpace = exec::DefaultExecSpace::scratch_memory_space;
template <class T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;

/// The distinct orbits of a triple, in the order a, b, c.
struct Thirds {
  std::int32_t n = 0;
  std::int32_t orbit[kMaxThirds] = {-1, -1, -1};
};
static_assert(std::is_trivially_copyable_v<Thirds>);

/// The distinct orbits among (a, b, c). O(1).
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
/// pair-third states of the triple). O(1).
IMSRG_INLINE_FUNCTION std::int32_t SlotOf(const Thirds& t,
                                          std::int32_t orbit) noexcept {
  for (std::int32_t i = 0; i < t.n; ++i) {
    if (t.orbit[i] == orbit) return i;
  }
  return 0;
}

IMSRG_INLINE_FUNCTION std::int32_t AbsDiff(std::int32_t a,
                                           std::int32_t b) noexcept {
  return a > b ? a - b : b - a;
}

/// Pointers into the team scratch of one team (layout: `ScratchCounts`).
struct TileScratch {
  double* bra = nullptr;  ///< [slot][row][X: S | Y: S], kBraSlots x m_max.
  double* ket = nullptr;  ///< [iz][column][X: S | Y: S], kMaxThirds x m_max.
  double* w = nullptr;    ///< W[r, s] at r + s m (column-major, tile m).
  double* u = nullptr;    ///< U[i, k'] = (W R_T'^T)[i, k'] at i + k' m.
  double* row_hat = nullptr;         ///< Hat(J1') of row r.
  double* col_hat = nullptr;         ///< Hat(J2') of column s.
  std::int32_t* row_J = nullptr;     ///< J1' of row r (integer).
  std::int32_t* row_zi = nullptr;    ///< Index of z among the thirds of T.
  std::int32_t* row_six = nullptr;   ///< c3 J1' + c4 (2 j_z - 1) / 2 of row r.
  std::int32_t* col_J = nullptr;     ///< J2' of column s (integer).
  std::int32_t* col_wi = nullptr;    ///< Index of w among the thirds of T'.
  std::int32_t* col_slot = nullptr;  ///< Bra-cache slot of w.
  std::int32_t* col_jw = nullptr;    ///< 2 j_w of column s.
};
static_assert(std::is_trivially_copyable_v<TileScratch>);

/// Carves the team scratch (level `level`) into the tables of `TileScratch`.
/// The typed views are taken in the order of `TeamScratchBytes`.
IMSRG_INLINE_FUNCTION TileScratch CarveScratch(const Member& team, int level,
                                               const TileExtents& ext) {
  const TileScratchCounts cnt = ScratchCounts(ext);
  ScratchView<double> d(team.team_scratch(level),
                        static_cast<std::size_t>(cnt.doubles));
  ScratchView<std::int32_t> i(team.team_scratch(level),
                              static_cast<std::size_t>(cnt.int32s));
  const std::int64_t m = ext.m_max;
  const std::int64_t leg = 2 * static_cast<std::int64_t>(ext.s_max);
  TileScratch sc;
  double* p = d.data();
  sc.bra = p;
  p += kBraSlots * m * leg;
  sc.ket = p;
  p += kMaxThirds * m * leg;
  sc.w = p;
  p += m * m;
  sc.u = p;
  p += m * ext.k_max;
  sc.row_hat = p;
  p += m;
  sc.col_hat = p;
  std::int32_t* q = i.data();
  sc.row_J = q;
  sc.row_zi = q + m;
  sc.row_six = q + 2 * m;
  sc.col_J = q + 3 * m;
  sc.col_wi = q + 4 * m;
  sc.col_slot = q + 5 * m;
  sc.col_jw = q + 6 * m;
  return sc;
}

/// Occupation product n_a n_b n_c and vacancy product of triple `t`.
IMSRG_INLINE_FUNCTION void TripleOccupation(const TileViews& v, std::int32_t t,
                                            double& occ,
                                            double& unocc) noexcept {
  const double na = v.ob.occupation[v.sp.tr_a[t]];
  const double nb = v.ob.occupation[v.sp.tr_b[t]];
  const double nc = v.ob.occupation[v.sp.tr_c[t]];
  occ = na * nb * nc;
  unocc = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
}

/// Sum of the single-particle energies of triple `t`.
IMSRG_INLINE_FUNCTION double TripleEnergy(const TileViews& v,
                                          std::int32_t t) noexcept {
  return v.epsilon[v.sp.tr_a[t]] + v.epsilon[v.sp.tr_b[t]] +
         v.epsilon[v.sp.tr_c[t]];
}

/// Gathers one leg: for the positions k of group `g` whose orbit a has 2 j_a
/// in [|2J - 2 j_o|, 2J + 2 j_o] (`J` the pair J of the leg, `o` its third
/// orbit; the range comes from `leg_range`), writes out_x[k] = f X[base + li
/// stride], out_y[k] = f Y[...] with the local index li of the ket (min(o,
/// a), max(o, a)) at J and its factor f from the leg tables (f = 0 for an
/// absent ket). `base` and `stride` come from the state tables: the row of
/// the pair P (stride: channel dimension) for a bra leg X_{P,(o a)}, its
/// column (stride 1) for a ket leg X_{(o a),P}. Branch-free and
/// vector-parallel over k. O(n_a).
IMSRG_INLINE_FUNCTION void GatherLeg(const Member& team, const TileViews& v,
                                     std::int32_t g, std::int32_t o,
                                     std::int32_t J, std::int64_t base,
                                     std::int64_t stride, double* out_x,
                                     double* out_y) {
  const std::int64_t key = static_cast<std::int64_t>(o) * v.legs.num_j + J;
  const std::int32_t* range = v.legs.leg_range + (key * 4 + g) * 2;
  const std::int64_t e0 = key * v.legs.num_orbits + v.groups.offset[g];
  const std::int32_t* li = v.legs.pair_local + e0;
  const double* f = v.legs.pair_factor + e0;
  const double* x2 = v.x2;
  const double* y2 = v.y2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, range[0], range[1]),
                       [&](const std::int32_t k) {
                         const std::int64_t idx = base + li[k] * stride;
                         out_x[k] = f[k] * x2[idx];
                         out_y[k] = f[k] * y2[idx];
                       });
}

/// The sum over a of W[r, s] / (Hat(J1') Hat(J2')) from the gathered legs:
/// the j blocks ij_lo..ij_hi (ij = (2 j_a - 1) / 2, the range allowed by both
/// triangles) of the orbit group whose block positions are `jp`, one 6j per
/// nonempty block from `six` (six[c2 ij] = {j_w j_a J1'; j_z J J2'}).
/// `xb`/`yb` are the bra legs of (r, w), `xk`/`yk` the ket legs of (s, z).
/// Sequential (one vector lane per (r, s)). O(n_a).
IMSRG_INLINE_FUNCTION double TileSum(const std::int32_t* jp, std::int32_t ij_lo,
                                     std::int32_t ij_hi, const double* six,
                                     std::int64_t c2, const double* xb,
                                     const double* yb, const double* xk,
                                     const double* yk) noexcept {
  double acc = 0.0;
  if (ij_lo > ij_hi) return acc;
  std::int32_t k = jp[ij_lo];
  for (std::int32_t ij = ij_lo; ij <= ij_hi; ++ij) {
    const std::int32_t k1 = jp[ij + 1];
    if (k1 == k) continue;
    double d = 0.0;
    for (; k < k1; ++k) d += xb[k] * yk[k] - yb[k] * xk[k];
    acc += six[c2 * ij] * d;
  }
  return acc;
}

/// The work of one team: bra triple `T` against its ket triples (by `mode`),
/// adding the triples energy of the team to `acc` (kTriples) or writing Z.
/// Barriers separate the phases of a tile (design/12 section 5), so any team
/// size is correct; with team size 1 they are no-ops.
IMSRG_INLINE_FUNCTION void TeamTiles(const Member& team, const TileViews& v,
                                     TileMode mode, std::int32_t T, int level,
                                     double& acc) {
  const TileScratch sc = CarveScratch(team, level, v.ext);
  const std::int32_t S = v.ext.s_max;
  const std::int64_t leg = 2 * static_cast<std::int64_t>(S);
  const std::int64_t m_max = v.ext.m_max;
  const std::int32_t c = v.sp.tr_channel[T];
  const std::int32_t two_J = v.sp.ch_two_j[c];
  const std::int32_t iJ = (two_J - 1) / 2;
  const std::int32_t st0 = v.pt.ch_state_offset[c];  // flat state of local 0
  const std::int32_t rs0 = v.pt.tr_state_begin[T];
  const std::int32_t m = v.pt.tr_state_end[T] - rs0;
  const std::int32_t ta = v.sp.tr_a[T];
  const std::int32_t tb = v.sp.tr_b[T];
  const std::int32_t tc = v.sp.tr_c[T];
  const Thirds zs = DistinctOrbits(ta, tb, tc);
  const std::int32_t ch_parity = v.ob.l[ta] + v.ob.l[tb] + v.ob.l[tc];
  const std::int32_t ch_tz2 = v.ob.tz2[ta] + v.ob.tz2[tb] + v.ob.tz2[tc];
  std::int32_t jz_of[kMaxThirds] = {0, 0, 0};  // 2 j_z of the thirds of T
  for (std::int32_t iz = 0; iz < zs.n; ++iz) jz_of[iz] = v.ob.j2[zs.orbit[iz]];

  double occ_t = 0.0;
  double unocc_t = 0.0;
  double e_t = 0.0;
  if (mode == TileMode::kTriples) {
    TripleOccupation(v, T, occ_t, unocc_t);
    if (occ_t < kTriplesSkip && unocc_t < kTriplesSkip) return;
    e_t = TripleEnergy(v, T);
  }

  // Rows r = (P, z, J1') of T.
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, m),
                       [&](const std::int32_t r) {
                         const std::int32_t f = st0 + rs0 + r;
                         const std::int32_t z = v.pt.st_r[f];
                         const std::int32_t J1 = v.pt.st_two_jpq[f] / 2;
                         sc.row_J[r] = J1;
                         sc.row_zi[r] = SlotOf(zs, z);
                         sc.row_six[r] = static_cast<std::int32_t>(
                             v.c3 * J1 + v.c4 * ((v.ob.j2[z] - 1) / 2));
                         sc.row_hat[r] = v.sj.Hat(J1);
                       });
  team.team_barrier();

  // Bra-leg cache: slot -> third orbit w and the tick of its last use. Every
  // thread of the team keeps an identical copy (deterministic updates).
  std::int32_t slot_orbit[kBraSlots];
  std::int32_t slot_tick[kBraSlots];
  for (std::int32_t k = 0; k < kBraSlots; ++k) {
    slot_orbit[k] = -1;
    slot_tick[k] = 0;
  }
  std::int32_t tick = 0;

  const std::int32_t tp_begin = v.sp.ch_triple_begin[c];
  const std::int32_t tp_end = mode == TileMode::kFullDense
                                  ? v.sp.ch_triple_end[c]
                                  : (mode == TileMode::kTriples ? T : T + 1);
  const std::int64_t r_row_t = v.sp.ch_row_offset[c] + v.sp.tr_row_begin[T];
  const std::int32_t kt = v.sp.tr_row_end[T] - v.sp.tr_row_begin[T];
  for (std::int32_t tp = tp_begin; tp < tp_end; ++tp) {
    double factor = 0.0;
    if (mode == TileMode::kTriples) {
      double occ_p = 0.0;
      double unocc_p = 0.0;
      TripleOccupation(v, tp, occ_p, unocc_p);
      const double num = occ_t * unocc_p - occ_p * unocc_t;
      if (num < kTriplesSkip && num > -kTriplesSkip) continue;
      factor = (two_J + 1) * (num / (e_t - TripleEnergy(v, tp))) *
               (v.sp.tr_sym[T] / 6.0) * (v.sp.tr_sym[tp] / 6.0);
    }
    const Thirds ws =
        DistinctOrbits(v.sp.tr_a[tp], v.sp.tr_b[tp], v.sp.tr_c[tp]);
    // Orbit group of a for each (z of T, w of T'): l_a = l_P + l_w = pi -
    // l_z + l_w (mod 2), tz_a = tz_P - tz_w = 2Tz - tz_z - tz_w.
    std::int32_t gzw[kMaxThirds][kMaxThirds] = {
        {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}};
    for (std::int32_t iz = 0; iz < zs.n; ++iz) {
      for (std::int32_t iw = 0; iw < ws.n; ++iw) {
        const std::int32_t z = zs.orbit[iz];
        const std::int32_t w = ws.orbit[iw];
        gzw[iz][iw] =
            OrbitGroupsView::GroupOf(ch_parity + v.ob.l[z] + v.ob.l[w],
                                     ch_tz2 - v.ob.tz2[z] - v.ob.tz2[w]);
      }
    }
    // Cache slots of the thirds of T': hits first, then the least recently
    // used free slots for the misses (which are gathered below).
    ++tick;
    std::int32_t wslot[kMaxThirds] = {-1, -1, -1};
    std::int32_t miss[kMaxThirds] = {0, 0, 0};
    std::int32_t n_miss = 0;
    for (std::int32_t iw = 0; iw < ws.n; ++iw) {
      for (std::int32_t k = 0; k < kBraSlots; ++k) {
        if (slot_orbit[k] == ws.orbit[iw]) {
          wslot[iw] = k;
          slot_tick[k] = tick;
        }
      }
    }
    for (std::int32_t iw = 0; iw < ws.n; ++iw) {
      if (wslot[iw] >= 0) continue;
      std::int32_t best = -1;
      for (std::int32_t k = 0; k < kBraSlots; ++k) {
        if (slot_tick[k] == tick) continue;
        if (best < 0 || slot_tick[k] < slot_tick[best]) best = k;
      }
      wslot[iw] = best;
      slot_orbit[best] = ws.orbit[iw];
      slot_tick[best] = tick;
      miss[n_miss++] = iw;
    }

    // Columns s = (Q, w, J2') of T'.
    const std::int32_t cs0 = v.pt.tr_state_begin[tp];
    const std::int32_t mp = v.pt.tr_state_end[tp] - cs0;
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, mp),
                         [&](const std::int32_t s) {
                           const std::int32_t f = st0 + cs0 + s;
                           const std::int32_t w = v.pt.st_r[f];
                           const std::int32_t J2 = v.pt.st_two_jpq[f] / 2;
                           const std::int32_t iw = SlotOf(ws, w);
                           sc.col_J[s] = J2;
                           sc.col_wi[s] = iw;
                           sc.col_slot[s] = wslot[iw];
                           sc.col_jw[s] = v.ob.j2[w];
                           sc.col_hat[s] = v.sj.Hat(J2);
                         });
    team.team_barrier();

    // Gathers: the bra legs of the missing thirds (all rows of T) and the ket
    // legs of the thirds of T (all columns of T'); independent outputs, one
    // barrier.
    for (std::int32_t im = 0; im < n_miss; ++im) {
      const std::int32_t iw = miss[im];
      const std::int32_t w = ws.orbit[iw];
      double* bra_slot = sc.bra + wslot[iw] * m_max * leg;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, m), [&](const std::int32_t r) {
            const std::int32_t g = gzw[sc.row_zi[r]][iw];
            if (g < 0) return;
            const std::int32_t f = st0 + rs0 + r;
            double* xb = bra_slot + r * leg;
            GatherLeg(team, v, g, w, sc.row_J[r], v.legs.state_bra_base[f],
                      v.legs.state_dim[f], xb, xb + S);
          });
    }
    for (std::int32_t iz = 0; iz < zs.n; ++iz) {
      const std::int32_t z = zs.orbit[iz];
      double* ket_z = sc.ket + iz * m_max * leg;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, mp), [&](const std::int32_t s) {
            const std::int32_t g = gzw[iz][sc.col_wi[s]];
            if (g < 0) return;
            double* xk = ket_z + s * leg;
            GatherLeg(team, v, g, z, sc.col_J[s],
                      v.legs.state_ket_base[st0 + cs0 + s], 1, xk, xk + S);
          });
    }
    team.team_barrier();

    // W[r, s]: a team thread per column s, a vector lane per row r. The
    // allowed 2 j_a range of (r, s) is the intersection of the bra side
    // [|2J1' - 2j_w|, 2J1' + 2j_w] and the ket side [|2J2' - 2j_z|, 2J2' +
    // 2j_z] (per third z of T, hoisted out of the row loop).
    const std::int32_t nj = v.groups.nj;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, mp), [&](const std::int32_t s) {
          const std::int32_t J2 = sc.col_J[s];
          const std::int32_t jw = sc.col_jw[s];
          const std::int32_t iw = sc.col_wi[s];
          const double hat_s = sc.col_hat[s];
          const double* xb_slot = sc.bra + sc.col_slot[s] * m_max * leg;
          const double* six_s =
              v.sj.data() + v.c1 * ((jw - 1) / 2) + v.c5 * iJ + J2;
          std::int32_t lo_z[kMaxThirds] = {0, 0, 0};
          std::int32_t hi_z[kMaxThirds] = {0, 0, 0};
          const std::int32_t* jp_z[kMaxThirds] = {nullptr, nullptr, nullptr};
          for (std::int32_t iz = 0; iz < zs.n; ++iz) {
            lo_z[iz] = AbsDiff(2 * J2, jz_of[iz]);
            hi_z[iz] = 2 * J2 + jz_of[iz];
            const std::int32_t g = gzw[iz][iw];
            jp_z[iz] = g >= 0 ? v.groups.j_pos + g * (nj + 1) : nullptr;
          }
          double* wcol = sc.w + static_cast<std::int64_t>(s) * m;
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, m), [&](const std::int32_t r) {
                const std::int32_t iz = sc.row_zi[r];
                double val = 0.0;
                if (jp_z[iz] != nullptr) {
                  const std::int32_t J1 = sc.row_J[r];
                  const std::int32_t lo1 = AbsDiff(2 * J1, jw);
                  const std::int32_t hi1 = 2 * J1 + jw;
                  const std::int32_t lo = lo1 > lo_z[iz] ? lo1 : lo_z[iz];
                  const std::int32_t hi = hi1 < hi_z[iz] ? hi1 : hi_z[iz];
                  const std::int32_t ij_hi = (hi - 1) / 2;
                  const double* xb = xb_slot + r * leg;
                  const double* xk = sc.ket + (iz * m_max + s) * leg;
                  val = sc.row_hat[r] * hat_s *
                        TileSum(jp_z[iz], (lo - 1) / 2,
                                ij_hi < nj - 1 ? ij_hi : nj - 1,
                                six_s + sc.row_six[r], v.c2, xb, xb + S, xk,
                                xk + S);
                }
                wcol[r] = val;
              });
        });
    team.team_barrier();

    // U = W R_T'^T (m x k'): a team thread per column k', vector over rows.
    const std::int64_t r_row_p = v.sp.ch_row_offset[c] + v.sp.tr_row_begin[tp];
    const std::int32_t kp = v.sp.tr_row_end[tp] - v.sp.tr_row_begin[tp];
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, kp), [&](const std::int32_t kc) {
          const std::int64_t row = r_row_p + kc;
          double* uc = sc.u + static_cast<std::int64_t>(kc) * m;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, m),
                               [&](const std::int32_t i) { uc[i] = 0.0; });
          for (std::int64_t e = v.pt.r.ptr[row]; e < v.pt.r.ptr[row + 1]; ++e) {
            const double rv = v.pt.r.val[e];
            const double* wc =
                sc.w + static_cast<std::int64_t>(v.pt.r.col[e] - cs0) * m;
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, m),
                [&](const std::int32_t i) { uc[i] += rv * wc[i]; });
          }
        });
    team.team_barrier();

    // Z[k, k'] = (R_T U)[k, k']: a team thread per column k', vector over k.
    if (mode == TileMode::kTriples) {
      double tile = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, kp),
          [&](const std::int32_t kc, double& t) {
            const double* uc = sc.u + static_cast<std::int64_t>(kc) * m;
            double tk = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, kt),
                [&](const std::int32_t k, double& tt) {
                  const std::int64_t row = r_row_t + k;
                  double zv = 0.0;
                  for (std::int64_t e = v.pt.r.ptr[row];
                       e < v.pt.r.ptr[row + 1]; ++e) {
                    zv += v.pt.r.val[e] * uc[v.pt.r.col[e] - rs0];
                  }
                  tt += zv * zv;
                },
                tk);
            t += tk;
          },
          tile);
      Kokkos::single(Kokkos::PerTeam(team), [&]() { acc += factor * tile; });
    } else {
      const std::int64_t rb = v.sp.tr_row_begin[T];
      const std::int64_t cb = v.sp.tr_row_begin[tp];
      const bool diagonal = tp == T;
      const bool packed = mode == TileMode::kLowerPacked;
      const std::int64_t f = v.sp.ch_row_offset[c];
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, kp), [&](const std::int32_t kc) {
            const double* uc = sc.u + static_cast<std::int64_t>(kc) * m;
            const std::int64_t j = cb + kc;
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, packed && diagonal ? kc : 0,
                                          kt),
                [&](const std::int32_t k) {
                  const std::int64_t row = r_row_t + k;
                  double zv = 0.0;
                  for (std::int64_t e = v.pt.r.ptr[row];
                       e < v.pt.r.ptr[row + 1]; ++e) {
                    zv += v.pt.r.val[e] * uc[v.pt.r.col[e] - rs0];
                  }
                  if (v.zero_diagonal != 0 && diagonal && k == kc) zv = 0.0;
                  const std::int64_t i = rb + k;
                  // Each stored element belongs to exactly one tile entry, so
                  // the accumulation is race-free.
                  v.z[v.z_row_off[f + j] + i] += zv;
                });
          });
    }
    team.team_barrier();
  }
}

/// The launched functor: one team per entry of `order` (trivially copyable).
struct TileFunctor {
  TileViews v;
  const std::int32_t* order = nullptr;
  TileMode mode = TileMode::kLowerPacked;
  int level = 0;

  IMSRG_FUNCTION void operator()(const Member& team) const {
    double unused = 0.0;
    TeamTiles(team, v, mode, order[team.league_rank()], level, unused);
  }
  IMSRG_FUNCTION void operator()(const Member& team, double& acc) const {
    TeamTiles(team, v, mode, order[team.league_rank()], level, acc);
  }
};
static_assert(std::is_trivially_copyable_v<TileFunctor>);

/// The team policy: `league` teams, AUTO team size and vector length (team
/// size 1 on OpenMP), chunk 1, `bytes` of team scratch at `level`.
Policy MakePolicy(std::int32_t league, std::int64_t bytes, int level) {
  Policy policy(league, Kokkos::AUTO, Kokkos::AUTO);
  policy.set_chunk_size(1);
  policy.set_scratch_size(level,
                          Kokkos::PerTeam(static_cast<std::size_t>(bytes)));
  return policy;
}

/// Scratch level for `bytes` per team: 0 if it fits, else 1 (IMSRG_CHECK
/// that level 1 holds it). Also checks that the 6j row offsets (`row_six`,
/// below c2) fit the int32 scratch table.
int ScratchLevel(const TileViews& v, std::int64_t bytes) {
  IMSRG_CHECK(v.c2 <= INT32_MAX, "comm223: the 6j table is too large");
  if (bytes <= static_cast<std::int64_t>(Policy::scratch_size_max(0))) return 0;
  IMSRG_CHECK(bytes <= static_cast<std::int64_t>(Policy::scratch_size_max(1)),
              "comm223: the team scratch of the largest tile does not fit");
  return 1;
}

}  // namespace

std::int64_t TeamScratchBytes(const TileExtents& ext) {
  const TileScratchCounts c = ScratchCounts(ext);
  return static_cast<std::int64_t>(
      ScratchView<double>::shmem_size(static_cast<std::size_t>(c.doubles)) +
      ScratchView<std::int32_t>::shmem_size(
          static_cast<std::size_t>(c.int32s)));
}

void RunComm223Tiles(const TileViews& v, TileMode mode,
                     const std::int32_t* order, std::int32_t league) {
  IMSRG_CHECK(mode != TileMode::kTriples,
              "RunComm223Tiles: use RunTriplesTiles for the triples");
  if (league <= 0) return;
  const std::int64_t bytes = TeamScratchBytes(v.ext);
  const int level = ScratchLevel(v, bytes);
  TileFunctor f;
  f.v = v;
  f.order = order;
  f.mode = mode;
  f.level = level;
  Kokkos::parallel_for("commutator::comm223_tiles",
                       MakePolicy(league, bytes, level), f);
  exec::Fence("commutator::comm223_tiles");
}

double RunTriplesTiles(const TileViews& v, const std::int32_t* order,
                       std::int32_t league) {
  if (league <= 0) return 0.0;
  const std::int64_t bytes = TeamScratchBytes(v.ext);
  const int level = ScratchLevel(v, bytes);
  TileFunctor f;
  f.v = v;
  f.order = order;
  f.mode = TileMode::kTriples;
  f.level = level;
  double energy = 0.0;
  Kokkos::parallel_reduce("commutator::triples_tiles",
                          MakePolicy(league, bytes, level), f,
                          Kokkos::Sum<double>(energy));
  return energy;
}

}  // namespace imsrg::commutator::detail
