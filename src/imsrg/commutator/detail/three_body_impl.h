// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_THREE_BODY_IMPL_H_
#define IMSRG_COMMUTATOR_DETAIL_THREE_BODY_IMPL_H_

/// @file
/// Internal tables of `commutator::ThreeBodyContext` (no Kokkos), shared by
/// three_body.cc and the tests of the tile kernels. Not a public API.

#include <cstdint>
#include <memory>
#include <vector>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/basis/device_tables.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/commutator/detail/comm232.h"
#include "imsrg/commutator/detail/n7_views.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/commutator/three_body.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator {

/// The device tables of a `ThreeBodyContext` (built once by its constructor;
/// read-only afterwards). Host copies are kept where host code needs them.
struct ThreeBodyContext::Impl {
  /// Builds every table for `ms3` (non-null, with a three-body space). Needs
  /// an initialized `exec::Runtime`. Cost: see the `ThreeBodyContext`
  /// constructor.
  explicit Impl(const std::shared_ptr<const basis::ModelSpace>& ms3);

  basis::DeviceTables tables;     ///< Orbit and two-body tables.
  angmom::SixJ3Table sixj;        ///< {j1 j2 J12; j3 J J23} up to the emax.
  op::PairThirdSpace pair_third;  ///< Pair-third states, R and E.

  /// Offsets of the two-body channel blocks of `op::TwoBody` (one n x n block
  /// per channel, in channel order; num_channels + 1 entries).
  std::vector<std::int64_t> two_body_offset_host;

  /// Kept canonical space (see `detail::Space3View`).
  exec::DeviceBuffer<std::int32_t> ch_two_j, ch_dim, ch_triple_begin,
      ch_triple_end;
  exec::DeviceBuffer<std::int64_t> ch_row_offset, ch_packed_offset,
      ch_dense_offset;
  std::vector<std::int64_t> packed_offset_host;  ///< `op::ThreeBody` packed.
  std::vector<std::int64_t> dense_offset_host;   ///< `op::ThreeBody` dense.
  exec::DeviceBuffer<std::int32_t> tr_a, tr_b, tr_c, tr_channel, tr_row_begin,
      tr_row_end, tr_sym;

  /// Orbit groups (see `detail::OrbitGroupsView`).
  exec::DeviceBuffer<std::int32_t> group_offset, group_orbits, group_j_pos;
  std::int32_t nj = 0;  ///< emax + 1.

  /// Gather tables of the legs (see `detail::LegTablesView`).
  exec::DeviceBuffer<std::int32_t> pair_local;
  exec::DeviceBuffer<double> pair_factor;
  exec::DeviceBuffer<std::int32_t> leg_range;
  exec::DeviceBuffer<std::int64_t> state_bra_base, state_ket_base;
  exec::DeviceBuffer<std::int32_t> state_dim;
  std::int32_t num_j = 0;  ///< 2 emax + 2.

  /// Team work orders (bra triples, largest estimated work first): all
  /// triples for the tiles T' <= T (`order_lower`) and for all tiles
  /// (`order_full`); the first `triples_league` entries of `order_triples`
  /// are the bra triples with at least one triples tile.
  exec::DeviceBuffer<std::int32_t> order_lower, order_full, order_triples;
  std::int32_t num_triples = 0;     ///< Triples of the kept space.
  std::int32_t triples_league = 0;  ///< Bra triples with triples work.

  detail::TileExtents ext;         ///< Largest tile extents.
  std::int64_t table_bytes = 0;    ///< `ThreeBodyContext::TableBytes`.
  std::int64_t scratch_bytes = 0;  ///< `ThreeBodyContext::ScratchBytesPerTeam`.

  /// Tables of the N7 terms (see `detail::N7Views`): 2 Jab and triple per
  /// flat kept row, the two-body block offsets and the one-body channel CSR
  /// on the device.
  exec::DeviceBuffer<std::int32_t> row_two_jab, row_triple;
  exec::DeviceBuffer<std::int64_t> two_body_offset;
  exec::DeviceBuffer<std::int32_t> ob_offsets, ob_orbits;

  /// The sorted pair-third tables of comm232ss (detail/comm232.h).
  detail::Comm232Tables comm232;

  /// The kernel views of the tables (operands, output and epsilon unset).
  /// O(1).
  detail::TileViews Views() const;

  /// The N7 kernel views with the parts of X and Y (either may be null: its
  /// parts read as zero; a missing one-, two- or three-body part also reads
  /// as zero). The operands must be laid out as the context's model space
  /// (checked by the callers). O(1).
  detail::N7Views N7(const op::Operator* X, const op::Operator* Y) const;
};

namespace detail {

/// Adds the [2,2]->3 term of [X, Y] (`Comm223`, three_body.h) to `z` in
/// place (no temporary three-body part). Preconditions as `Comm223`, and `z`
/// laid out as the context's kept space with the hermiticity of [X, Y]
/// (InvariantError otherwise).
void AddComm223(const op::Operator& X, const op::Operator& Y, op::ThreeBody& z,
                const ThreeBodyContext& ctx);

}  // namespace detail

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_DETAIL_THREE_BODY_IMPL_H_
