// SPDX-License-Identifier: GPL-3.0-or-later
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
#include "imsrg/base/constants.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/two_body_channels.h"
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
  std::int64_t* row_base = nullptr;  ///< Block offset + local index of P.
  std::int64_t* col_base = nullptr;  ///< Block offset + n * local index of Q.
  std::int32_t* row_z = nullptr;     ///< Third z of row r.
  std::int32_t* row_J = nullptr;     ///< J1' of row r (integer).
  std::int32_t* row_zi = nullptr;    ///< Index of z among the thirds of T.
  std::int32_t* row_n = nullptr;     ///< Dimension of P's two-body channel.
  std::int32_t* row_jz = nullptr;    ///< 2 j_z of row r.
  std::int32_t* col_w = nullptr;     ///< Third w of column s.
  std::int32_t* col_J = nullptr;     ///< J2' of column s (integer).
  std::int32_t* col_wi = nullptr;    ///< Index of w among the thirds of T'.
  std::int32_t* col_slot = nullptr;  ///< Bra-cache slot of w.
  std::int32_t* col_jw = nullptr;    ///< 2 j_w of column s.
};
static_assert(std::is_trivially_copyable_v<TileScratch>);

/// Carves the team scratch (level `level`) into the tables of `TileScratch`.
/// The three typed views are taken in the order of `TeamScratchBytes`.
IMSRG_INLINE_FUNCTION TileScratch CarveScratch(const Member& team, int level,
                                               const TileExtents& ext) {
  const TileScratchCounts cnt = ScratchCounts(ext);
  ScratchView<double> d(team.team_scratch(level),
                        static_cast<std::size_t>(cnt.doubles));
  ScratchView<std::int64_t> l(team.team_scratch(level),
                              static_cast<std::size_t>(cnt.int64s));
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
  sc.row_base = l.data();
  sc.col_base = l.data() + m;
  std::int32_t* q = i.data();
  sc.row_z = q;
  sc.row_J = q + m;
  sc.row_zi = q + 2 * m;
  sc.row_n = q + 3 * m;
  sc.row_jz = q + 4 * m;
  sc.col_w = q + 5 * m;
  sc.col_J = q + 6 * m;
  sc.col_wi = q + 7 * m;
  sc.col_slot = q + 8 * m;
  sc.col_jw = q + 9 * m;
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

/// Two-body channel and local index of the pair (p <= q) at 2J = `two_j`
/// (an existing ket: pair-third states satisfy Pauli and the triangle).
struct PairSlot {
  std::int32_t channel;
  std::int32_t local;
};

IMSRG_INLINE_FUNCTION PairSlot PairOf(const TileViews& v, std::int32_t p,
                                      std::int32_t q,
                                      std::int32_t two_j) noexcept {
  const KetIndex ket = v.tb.ket_index(OrbitIndex{p}, OrbitIndex{q});
  PairSlot s;
  s.local = v.tb.local_index(ket, TwoJ{two_j});
  s.channel =
      v.tb.channel_of(OrbitIndex{p}, OrbitIndex{q}, TwoJ{two_j}).value();
  return s;
}

/// Gathers one leg: for the positions k of group `g` whose orbit a has
/// 2 j_a in [|2J - j2_o|, 2J + j2_o] (`two_J` = 2J of the leg's pair, `o` the
/// leg's third orbit with 2j = `j2_o`), writes
///   out_x[k] = f X[base + li(o, a) stride], out_y[k] = f Y[...],
/// li(o, a) the local index of the ket (min(o, a), max(o, a)) at J (zero if
/// it does not exist, i.e. o == a with odd J), f = ExchangePhase(j_o, j_a, J)
/// for o > a, sqrt 2 for o == a, else 1. `stride` is the channel dimension
/// for a bra leg (element (P, (o a)): row P fixed) and 1 for a ket leg
/// (element ((o a), Q): column Q fixed). Vector-parallel over k. O(n_a).
IMSRG_INLINE_FUNCTION void GatherLeg(const Member& team, const TileViews& v,
                                     std::int32_t g, std::int32_t o,
                                     std::int32_t two_J, std::int64_t base,
                                     std::int64_t stride, double* out_x,
                                     double* out_y) {
  const std::int32_t nj = v.groups.nj;
  const std::int32_t j2o = v.ob.j2[o];
  const std::int32_t ij_lo = (AbsDiff(two_J, j2o) - 1) / 2;
  std::int32_t ij_hi = (two_J + j2o - 1) / 2;
  if (ij_hi > nj - 1) ij_hi = nj - 1;
  if (ij_lo > ij_hi) return;
  const std::int32_t* jp = v.groups.j_pos + g * (nj + 1);
  const std::int32_t* orbs = v.groups.orbits + v.groups.offset[g];
  const std::int32_t J = two_J / 2;
  const std::int32_t norb = v.tb.num_orbits;
  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(team, jp[ij_lo], jp[ij_hi + 1]),
      [&](const std::int32_t k) {
        const std::int32_t a = orbs[k];
        const std::int32_t j2a = v.ob.j2[a];
        double f = 1.0;
        std::int32_t lo = o;
        std::int32_t hi = a;
        if (o > a) {
          lo = a;
          hi = o;
          f = basis::ExchangePhase(TwoJ{j2o}, TwoJ{j2a}, TwoJ{two_J});
        } else if (o == a) {
          f = constants::kSqrt2;
        }
        const std::int32_t ket =
            basis::ImsrgppKetIndex(OrbitIndex{lo}, OrbitIndex{hi}, norb)
                .value();
        const std::int32_t li =
            v.tb.local_index_table[v.tb.ket_J_offsets[ket] + J -
                                   AbsDiff(j2o, j2a) / 2];
        if (li < 0) {
          out_x[k] = 0.0;
          out_y[k] = 0.0;
          return;
        }
        const std::int64_t idx = base + li * stride;
        out_x[k] = f * v.x2[idx];
        out_y[k] = f * v.y2[idx];
      });
}

/// W[r, s] of the tile (formula in three_body.h) from the gathered legs: the
/// sum over the j blocks of the orbit group `g` allowed by both triangles,
/// one 6j per block. `xb`/`yb` are the bra legs of (r, w_s), `xk`/`yk` the
/// ket legs of (s, z_r). O(n_a).
IMSRG_INLINE_FUNCTION double TileElement(const Member& team, const TileViews& v,
                                         std::int32_t g, std::int32_t iJ,
                                         std::int32_t J1, std::int32_t jz,
                                         std::int32_t J2, std::int32_t jw,
                                         const double* xb, const double* yb,
                                         const double* xk, const double* yk) {
  const std::int32_t nj = v.groups.nj;
  const std::int32_t lo1 = AbsDiff(2 * J1, jw);
  const std::int32_t lo2 = AbsDiff(2 * J2, jz);
  const std::int32_t hi1 = 2 * J1 + jw;
  const std::int32_t hi2 = 2 * J2 + jz;
  const std::int32_t ij_lo = ((lo1 > lo2 ? lo1 : lo2) - 1) / 2;
  std::int32_t ij_hi = ((hi1 < hi2 ? hi1 : hi2) - 1) / 2;
  if (ij_hi > nj - 1) ij_hi = nj - 1;
  const std::int32_t* jp = v.groups.j_pos + g * (nj + 1);
  // {j_w j_a J1'; j_z J J2'} = six[c2 ia] with ia = (2 j_a - 1) / 2.
  const double* six = v.sj.data + v.c1 * ((jw - 1) / 2) + v.c3 * J1 +
                      v.c4 * ((jz - 1) / 2) + v.c5 * iJ + J2;
  double acc = 0.0;
  for (std::int32_t ij = ij_lo; ij <= ij_hi; ++ij) {
    const std::int32_t k0 = jp[ij];
    const std::int32_t k1 = jp[ij + 1];
    if (k0 == k1) continue;
    double d = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(team, k0, k1),
        [&](const std::int32_t k, double& t) {
          t += xb[k] * yk[k] - yb[k] * xk[k];
        },
        d);
    acc += six[v.c2 * ij] * d;
  }
  return acc;
}

/// The work of one team: bra triple `T` against its ket triples (by `mode`),
/// adding the triples energy of the team to `acc` (kTriples) or writing Z.
/// Barriers separate the phases of a tile (design/12 section 5; with team
/// size 1, as on OpenMP, they are no-ops).
IMSRG_INLINE_FUNCTION void TeamTiles(const Member& team, const TileViews& v,
                                     TileMode mode, std::int32_t T, int level,
                                     double& acc) {
  const TileScratch sc = CarveScratch(team, level, v.ext);
  const std::int32_t S = v.ext.s_max;
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

  double occ_t = 0.0;
  double unocc_t = 0.0;
  double e_t = 0.0;
  if (mode == TileMode::kTriples) {
    TripleOccupation(v, T, occ_t, unocc_t);
    if (occ_t < kTriplesSkip && unocc_t < kTriplesSkip) return;
    e_t = TripleEnergy(v, T);
  }

  // Rows r = (P, z, J1') of T.
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, m), [&](const std::int32_t r) {
        const std::int32_t f = st0 + rs0 + r;
        const std::int32_t z = v.pt.st_r[f];
        const std::int32_t tj = v.pt.st_two_jpq[f];
        const PairSlot ps = PairOf(v, v.pt.st_p[f], v.pt.st_q[f], tj);
        sc.row_z[r] = z;
        sc.row_jz[r] = v.ob.j2[z];
        sc.row_J[r] = tj / 2;
        sc.row_zi[r] = SlotOf(zs, z);
        sc.row_hat[r] = v.sj.Hat(tj / 2);
        sc.row_n[r] = v.tb.channel_num_kets[ps.channel];
        sc.row_base[r] = v.two_body_offset[ps.channel] + ps.local;
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
    std::int32_t gzw[kMaxThirds][kMaxThirds];
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
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, mp), [&](const std::int32_t s) {
          const std::int32_t f = st0 + cs0 + s;
          const std::int32_t w = v.pt.st_r[f];
          const std::int32_t tj = v.pt.st_two_jpq[f];
          const PairSlot ps = PairOf(v, v.pt.st_p[f], v.pt.st_q[f], tj);
          const std::int32_t iw = SlotOf(ws, w);
          sc.col_w[s] = w;
          sc.col_jw[s] = v.ob.j2[w];
          sc.col_J[s] = tj / 2;
          sc.col_wi[s] = iw;
          sc.col_slot[s] = wslot[iw];
          sc.col_hat[s] = v.sj.Hat(tj / 2);
          sc.col_base[s] = v.two_body_offset[ps.channel] +
                           static_cast<std::int64_t>(ps.local) *
                               v.tb.channel_num_kets[ps.channel];
        });
    team.team_barrier();

    // Gathers: the bra legs of the missing thirds (all rows of T) and the ket
    // legs of the thirds of T (all columns of T').
    const std::int32_t n_bra = m * n_miss;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, n_bra + mp * zs.n),
        [&](const std::int32_t item) {
          if (item < n_bra) {
            const std::int32_t r = item % m;
            const std::int32_t iw = miss[item / m];
            const std::int32_t g = gzw[sc.row_zi[r]][iw];
            if (g < 0) return;
            double* xb = sc.bra + (wslot[iw] * m_max + r) * 2 * S;
            GatherLeg(team, v, g, ws.orbit[iw], 2 * sc.row_J[r], sc.row_base[r],
                      sc.row_n[r], xb, xb + S);
          } else {
            const std::int32_t s = (item - n_bra) % mp;
            const std::int32_t iz = (item - n_bra) / mp;
            const std::int32_t g = gzw[iz][sc.col_wi[s]];
            if (g < 0) return;
            double* xk = sc.ket + (iz * m_max + s) * 2 * S;
            GatherLeg(team, v, g, zs.orbit[iz], 2 * sc.col_J[s], sc.col_base[s],
                      1, xk, xk + S);
          }
        });
    team.team_barrier();

    // W[r, s].
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, m * mp), [&](const std::int32_t idx) {
          const std::int32_t r = idx % m;
          const std::int32_t s = idx / m;
          const std::int32_t iz = sc.row_zi[r];
          const std::int32_t g = gzw[iz][sc.col_wi[s]];
          double val = 0.0;
          if (g >= 0) {
            const double* xb = sc.bra + (sc.col_slot[s] * m_max + r) * 2 * S;
            const double* xk = sc.ket + (iz * m_max + s) * 2 * S;
            val =
                sc.row_hat[r] * sc.col_hat[s] *
                TileElement(team, v, g, iJ, sc.row_J[r], sc.row_jz[r],
                            sc.col_J[s], sc.col_jw[s], xb, xb + S, xk, xk + S);
          }
          Kokkos::single(Kokkos::PerThread(team), [&]() { sc.w[idx] = val; });
        });
    team.team_barrier();

    // U = W R_T'^T (m x k').
    const std::int64_t r_row_p = v.sp.ch_row_offset[c] + v.sp.tr_row_begin[tp];
    const std::int32_t kp = v.sp.tr_row_end[tp] - v.sp.tr_row_begin[tp];
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, m * kp), [&](const std::int32_t idx) {
          const std::int32_t i = idx % m;
          const std::int64_t row = r_row_p + idx / m;
          double sum = 0.0;
          for (std::int64_t e = v.pt.r.ptr[row]; e < v.pt.r.ptr[row + 1]; ++e) {
            sum += v.pt.r.val[e] * sc.w[i + (v.pt.r.col[e] - cs0) * m];
          }
          Kokkos::single(Kokkos::PerThread(team), [&]() { sc.u[idx] = sum; });
        });
    team.team_barrier();

    // Z[k, k'] = (R_T U)[k, k'].
    if (mode == TileMode::kTriples) {
      double tile = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, kt * kp),
          [&](const std::int32_t idx, double& t) {
            const std::int64_t row = r_row_t + idx % kt;
            const double* uc = sc.u + (idx / kt) * m;
            double zv = 0.0;
            for (std::int64_t e = v.pt.r.ptr[row]; e < v.pt.r.ptr[row + 1];
                 ++e) {
              zv += v.pt.r.val[e] * uc[v.pt.r.col[e] - rs0];
            }
            t += zv * zv;
          },
          tile);
      Kokkos::single(Kokkos::PerTeam(team), [&]() { acc += factor * tile; });
    } else {
      const std::int64_t n_dim = v.sp.ch_dim[c];
      const std::int64_t rb = v.sp.tr_row_begin[T];
      const std::int64_t cb = v.sp.tr_row_begin[tp];
      const bool diagonal = tp == T;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, kt * kp), [&](const std::int32_t idx) {
            const std::int32_t k = idx % kt;
            const std::int32_t kc = idx / kt;
            if (mode == TileMode::kLowerPacked && diagonal && k < kc) return;
            const std::int64_t row = r_row_t + k;
            const double* uc = sc.u + kc * m;
            double zv = 0.0;
            for (std::int64_t e = v.pt.r.ptr[row]; e < v.pt.r.ptr[row + 1];
                 ++e) {
              zv += v.pt.r.val[e] * uc[v.pt.r.col[e] - rs0];
            }
            if (v.zero_diagonal != 0 && diagonal && k == kc) zv = 0.0;
            const std::int64_t i = rb + k;
            const std::int64_t j = cb + kc;
            const std::int64_t dst =
                mode == TileMode::kLowerPacked
                    ? v.sp.ch_packed_offset[c] +
                          op::ThreeBody::PackedIndex(n_dim, i, j)
                    : v.sp.ch_dense_offset[c] + i + j * n_dim;
            Kokkos::single(Kokkos::PerThread(team), [&]() { v.z[dst] = zv; });
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
/// that level 1 holds it).
int ScratchLevel(std::int64_t bytes) {
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
      ScratchView<std::int64_t>::shmem_size(
          static_cast<std::size_t>(c.int64s)) +
      ScratchView<std::int32_t>::shmem_size(
          static_cast<std::size_t>(c.int32s)));
}

void RunComm223Tiles(const TileViews& v, TileMode mode,
                     const std::int32_t* order, std::int32_t league) {
  IMSRG_CHECK(mode != TileMode::kTriples,
              "RunComm223Tiles: use RunTriplesTiles for the triples");
  if (league <= 0) return;
  const std::int64_t bytes = TeamScratchBytes(v.ext);
  const int level = ScratchLevel(bytes);
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
  const int level = ScratchLevel(bytes);
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
