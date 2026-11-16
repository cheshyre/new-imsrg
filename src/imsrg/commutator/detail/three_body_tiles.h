// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_THREE_BODY_TILES_H_
#define IMSRG_COMMUTATOR_DETAIL_THREE_BODY_TILES_H_

/// @file
/// Device views and launchers of the [2,2]->3 pair-third tiles (internal to
/// the commutator module; formula in three_body.h). No Kokkos: three_body.cc
/// builds the views from a `ThreeBodyContext` and the operands,
/// kernels/comm223.cc runs the tiles (design/12 section 5 device rules: the
/// views are trivially copyable and captured by value).
///
/// Work decomposition: one team per bra triple T (league in a precomputed
/// work order, largest estimated work first, dynamic schedule). The team
/// visits the ket triples T' of the channel (T' <= T, all T', or the
/// triples-active T' < T, by `TileMode`) and per tile
///  1. gathers the bra legs X/Y_{P_r,(w a)} for the rows r of T and the
///     distinct thirds w of T' into a cache of `kBraSlots` slots keyed by w
///     (consecutive T' share thirds, so most legs are reused), and the ket
///     legs X/Y_{(z a),Q_s} for the columns s of T' and the distinct thirds z
///     of T; only the internal orbits a whose j lies in the triangle range of
///     the leg are gathered, through the index tables of `LegTablesView`
///     (O(norb^2 emax) pair entries and O(states) base indices, not gathered
///     values);
///  2. computes W[r, s] over the 6j-allowed j blocks of the orbit group of a
///     (one 6j lookup per j block);
///  3. applies the epilogue R_T W R_T'^T and writes the packed lower
///     triangle, the dense block, or accumulates the triples energy.
/// All tile data live in team scratch (level 0 if it fits, else level 1).

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/base/device.h"
#include "imsrg/basis/device_tables.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator::detail {

/// Most distinct orbits of a triple.
inline constexpr std::int32_t kMaxThirds = 3;
/// Slots of the per-team bra-leg cache (>= kMaxThirds).
inline constexpr std::int32_t kBraSlots = 4;
/// imsrg++'s occupation thresholds of the perturbative triples
/// (IMSRG3Commutators.cc:5065, 5160).
inline constexpr double kTriplesSkip = 1.0e-8;

static_assert(kBraSlots >= kMaxThirds);

/// Device tables of the kept canonical space (`basis::ThreeBodySpace`), struct
/// of arrays. Channels are those of the space (possibly of dimension 0);
/// triples are grouped by channel; rows of a triple are contiguous.
struct Space3View {
  const std::int32_t* ch_two_j = nullptr;         ///< 2J per channel.
  const std::int32_t* ch_dim = nullptr;           ///< Kept rows N per channel.
  const std::int64_t* ch_row_offset = nullptr;    ///< Flat row of local row 0.
  const std::int32_t* ch_triple_begin = nullptr;  ///< First triple.
  const std::int32_t* ch_triple_end = nullptr;    ///< One past the last.
  /// Offsets of the packed lower-triangle blocks (num_channels + 1).
  const std::int64_t* ch_packed_offset = nullptr;
  /// Offsets of the dense column-major blocks (num_channels + 1).
  const std::int64_t* ch_dense_offset = nullptr;
  const std::int32_t* tr_a = nullptr;  ///< Orbits a <= b <= c per triple.
  const std::int32_t* tr_b = nullptr;
  const std::int32_t* tr_c = nullptr;
  const std::int32_t* tr_channel = nullptr;    ///< Channel per triple.
  const std::int32_t* tr_row_begin = nullptr;  ///< Channel-local first row.
  const std::int32_t* tr_row_end = nullptr;    ///< One past the last row.
  const std::int32_t* tr_sym = nullptr;        ///< 6, 3 or 1.
  std::int32_t num_channels = 0;
  std::int32_t num_triples = 0;
};
static_assert(std::is_trivially_copyable_v<Space3View>);

/// All orbits of the basis grouped by (parity, 2tz): group g = 2 parity + (tz2
/// + 1) / 2 holds `orbits[offset[g] .. offset[g + 1])`, sorted by (2j, index)
/// (the range of the internal orbit a). `j_pos[g * (nj + 1) + ij]` is the
/// first position in group g (relative to offset[g]) whose orbit has (2j - 1)
/// / 2 >= ij, for ij in [0, nj]; `j_pos[g * (nj + 1) + nj]` is the group
/// size. The orbits with (2j - 1) / 2 == ij are the positions [j_pos[ij],
/// j_pos[ij + 1]) (a "j block").
struct OrbitGroupsView {
  const std::int32_t* offset = nullptr;  ///< 5 entries.
  const std::int32_t* orbits = nullptr;  ///< num_orbits entries.
  const std::int32_t* j_pos = nullptr;   ///< 4 (nj + 1) entries.
  std::int32_t nj = 0;                   ///< emax + 1 (number of 2j values).

  /// Group of parity `parity` (0, 1) and 2tz `tz2` (-1, +1); -1 for |tz2| != 1
  /// (no orbit). O(1).
  static IMSRG_INLINE_FUNCTION constexpr std::int32_t GroupOf(
      std::int32_t parity, std::int32_t tz2) noexcept {
    return (tz2 == 1 || tz2 == -1) ? 2 * (parity & 1) + (tz2 + 1) / 2 : -1;
  }
};
static_assert(std::is_trivially_copyable_v<OrbitGroupsView>);

/// Largest tile extents over all triples (they size the team scratch).
struct TileExtents {
  std::int32_t m_max = 0;  ///< Most pair-third states of a triple.
  std::int32_t k_max = 0;  ///< Most kept rows of a triple.
  std::int32_t s_max = 0;  ///< Largest orbit group.
};
static_assert(std::is_trivially_copyable_v<TileExtents>);

/// Element counts of the team scratch of one team, by type.
struct TileScratchCounts {
  std::int64_t doubles = 0;  ///< Bra cache, ket legs, W, U, hats.
  std::int64_t int32s = 0;   ///< Per-state index tables.
};
static_assert(std::is_trivially_copyable_v<TileScratchCounts>);

/// Number of per-state int32 tables in the team scratch (rows: J1', index of
/// z, 6j offset c3 J1' + c4 (2 j_z - 1) / 2; columns: J2', index of w, cache
/// slot, 2 j_w).
inline constexpr std::int64_t kStateTables = 7;

/// The scratch counts for `ext`: bra cache kBraSlots m S 2, ket legs
/// kMaxThirds m S 2, W m^2, U m k, two hat tables of m (m = m_max, S = s_max,
/// k = k_max) doubles; kStateTables m int32. O(1).
IMSRG_INLINE_FUNCTION constexpr TileScratchCounts ScratchCounts(
    const TileExtents& ext) noexcept {
  const std::int64_t m = ext.m_max;
  const std::int64_t s = ext.s_max;
  const std::int64_t k = ext.k_max;
  TileScratchCounts c;
  c.doubles = (kBraSlots + kMaxThirds) * m * 2 * s + m * m + m * k + 2 * m;
  c.int32s = kStateTables * m;
  return c;
}

/// What a tile launch computes.
enum class TileMode : std::int8_t {
  kLowerPacked,  ///< Tiles T' <= T, packed lower triangle (Z of definite
                 ///< hermiticity; the diagonal tile only k >= k').
  kFullDense,    ///< All tiles of the channel, dense blocks (non-Hermitian Z).
  kTriples,      ///< Tiles T' < T with a nonzero occupation factor; energy.
};

/// Gather tables of the two-body legs (built once per model space; they
/// replace the full `get_normalized` per gathered element).
///
/// Per orbit pair: for a leg with third orbit o (w of a bra leg, z of a ket
/// leg), pair J and internal orbit a at grouped position gpos (index into
/// `OrbitGroupsView::orbits`), entry e = (o num_j + J) num_orbits + gpos holds
/// * `pair_local[e]`: the local index of the two-body ket (min(o, a),
///   max(o, a)) at J in its channel, or 0 if that ket does not exist;
/// * `pair_factor[e]`: 1 for o < a, `basis::ExchangePhase(j_o, j_a, J)` for
///   o > a, sqrt 2 for o == a (the unnormalized (o a) pair), and 0 if the ket
///   does not exist (triangle, Pauli), so a gather is branch-free.
/// Per leg key (o, J) and orbit group g: `leg_range[((o num_j + J) 4 + g) 2]`
/// and `[... + 1]` are the first and one-past-last positions in group g
/// (relative to its offset) of the orbits a with 2 j_a in [|2J - 2 j_o|, 2J
/// + 2 j_o] (the triangle range of the leg; empty if none).
/// Per flat pair-third state f with pair (p, q) and Jpq (always an existing
/// ket): `state_bra_base[f]` = block offset + local index of (p, q), the
/// element (P, k) at `state_bra_base[f] + k state_dim[f]`;
/// `state_ket_base[f]` = block offset + state_dim[f] local index, the element
/// (k, P) at `state_ket_base[f] + k`; `state_dim[f]` the channel dimension.
struct LegTablesView {
  const std::int32_t* pair_local = nullptr;
  const double* pair_factor = nullptr;
  const std::int32_t* leg_range = nullptr;
  const std::int64_t* state_bra_base = nullptr;
  const std::int64_t* state_ket_base = nullptr;
  const std::int32_t* state_dim = nullptr;
  std::int32_t num_j = 0;       ///< Two-body J values, 2 emax + 2.
  std::int32_t num_orbits = 0;  ///< Orbits of the basis.
};
static_assert(std::is_trivially_copyable_v<LegTablesView>);

/// Everything a tile kernel reads or writes (raw device pointers, captured
/// by value).
struct TileViews {
  Space3View sp;               ///< Kept canonical space.
  op::PairThirdView pt;        ///< Pair-third states and R.
  angmom::SixJ3View sj;        ///< 3B 6j table.
  basis::OrbitTableView ob;    ///< Orbit quantum numbers and occupations.
  OrbitGroupsView groups;      ///< Range of the internal orbit.
  LegTablesView legs;          ///< Gather tables of the two-body legs.
  const double* x2 = nullptr;  ///< Two-body data of X (op::TwoBody layout).
  const double* y2 = nullptr;  ///< Two-body data of Y.
  double* z = nullptr;         ///< Output blocks (packed or dense modes).
  const double* epsilon = nullptr;  ///< One energy per orbit (triples mode).
  /// Strides of the dense 6j table: {j1 j2 J12; j3 J J23} is at c1 i1 + c2 i2
  /// + c3 J12 + c4 i3 + c5 iJ + J23 with i = (2j - 1) / 2, iJ = (2J - 1) / 2.
  std::int64_t c1 = 0;
  std::int64_t c2 = 0;
  std::int64_t c3 = 0;
  std::int64_t c4 = 0;
  std::int64_t c5 = 0;
  /// 1 for anti-Hermitian Z in kLowerPacked mode (diagonal written as 0).
  std::int32_t zero_diagonal = 0;
  TileExtents ext;  ///< Scratch extents.
};
static_assert(std::is_trivially_copyable_v<TileViews>);

/// Runs the tiles of mode kLowerPacked or kFullDense: one team per bra triple
/// `order[i]`, i in [0, league), ADDING to every stored element of the layout
/// in `v.z` (each element is written by one work item; nothing else).
/// Blocking. Defined in kernels/comm223.cc. Cost: see `Comm223` in
/// three_body.h.
void RunComm223Tiles(const TileViews& v, TileMode mode,
                     const std::int32_t* order, std::int32_t league);

/// Runs the tiles of mode kTriples for the bra triples `order[i]`, i in [0,
/// league), and returns the energy sum (Kokkos sum reducer). Blocking.
/// Defined in kernels/comm223.cc.
double RunTriplesTiles(const TileViews& v, const std::int32_t* order,
                       std::int32_t league);

/// Bytes the kernels request per team for `ext` (Kokkos scratch sizes,
/// alignment padding included). Defined in kernels/comm223.cc.
std::int64_t TeamScratchBytes(const TileExtents& ext);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_THREE_BODY_TILES_H_
