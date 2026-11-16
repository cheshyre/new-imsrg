// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_WORK_ITEMS_H_
#define IMSRG_COMMUTATOR_DETAIL_WORK_ITEMS_H_
/// @file
/// Work items, costs and the per-rank assignment of the expensive IMSRG(2)
/// commutator terms (the pp/hh ladder, comm122ss and the particle-hole term;
/// design/12 section 6, M12b phase 1). Host code only (no Kokkos, no MPI): the
/// sizes come from the model space, every rank builds the same items and the
/// same `comm::OwnershipMap`s. `AssignWork` stores the assignment of one part
/// (rank) in the `Workspace`; with one part everything is owned and the serial
/// kernels are unchanged.

#include <cstdint>
#include <vector>

#include "imsrg/basis/model_space.h"
#include "imsrg/comm/ownership_map.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/device_buffer.h"

namespace imsrg::commutator::detail {

/// Rows [row_begin, row_end) of the block of two-body channel `channel`.
struct RowBlock {
  std::int32_t channel = 0;    ///< Two-body channel index.
  std::int32_t row_begin = 0;  ///< First row (inclusive).
  std::int32_t row_end = 0;    ///< One past the last row.
};

/// Equality of row blocks (all members).
inline bool operator==(const RowBlock& a, const RowBlock& b) noexcept {
  return a.channel == b.channel && a.row_begin == b.row_begin &&
         a.row_end == b.row_end;
}

/// The sizes the work costs are built from (passive data, host only). Arrays
/// of the two-body channels have `num_channels` entries, those of the
/// cross-coupled (Pandya) channels `num_cc`.
struct WorkSizes {
  std::vector<std::int32_t> num_kets;  ///< N_c: kets of each channel.
  /// Kets with non-zero pp weight (1 - n_p)(1 - n_q) (`LadderData` rule).
  std::vector<std::int32_t> k_pp;
  /// Kets with non-zero hh weight n_p n_q (`LadderData` rule).
  std::vector<std::int32_t> k_hh;
  /// Lifted one-body entries of all rows of the channel (`LiftedData` rule).
  std::vector<std::int32_t> lifted;
  std::vector<std::int32_t> ncc;  ///< Ncc: kets of each cc channel.
  std::vector<std::int32_t> nk;   ///< |K| of each cc channel (`PandyaData`).
};

/// Computes the `WorkSizes` of `ms` with the counting rules of
/// `BuildLadderData`, `BuildLiftedData` and `BuildPandyaData` (the unit tests
/// compare them with the Workspace's own data). Pure host code, no device
/// buffers. Complexity O(kets x orbits per one-body channel).
WorkSizes ComputeWorkSizes(const basis::ModelSpace& ms);

/// Ladder work items (row blocks) with their costs.
struct LadderItems {
  std::vector<RowBlock> blocks;  ///< Ascending (channel, row_begin).
  std::vector<double> cost;      ///< Cost of each block.
};

/// Whole-channel work items (comm122 or particle-hole) with their costs.
struct ChannelItems {
  std::vector<std::int32_t> channels;  ///< Ascending channel indices.
  std::vector<double> cost;            ///< Cost of each item.
};

/// Ladder items: one per channel with N_c > 0 and k_pp + k_hh > 0, cost 2 rows
/// N_c (k_pp + k_hh). Only if `num_parts > 1`, a channel whose cost exceeds
/// total / (4 num_parts) is split into nb row blocks of near-equal rows (block
/// b is [b N / nb, (b + 1) N / nb)), nb = min(N_c, ceil(cost / (total / (4
/// num_parts)))), raised until the largest block fits the threshold (or is one
/// row). With `num_parts == 1` nothing is split. Precondition: `num_parts >=
/// 1`. Complexity O(channels + items).
LadderItems LadderWorkItems(const WorkSizes& sizes, std::int32_t num_parts);

/// comm122 items: one per channel with N_c > 0 and lifted_c > 0, cost N_c
/// lifted_c + N_c^2 / 2. O(channels).
ChannelItems Comm122WorkItems(const WorkSizes& sizes);

/// Particle-hole items: one per cc channel with Ncc > 0 and nk > 0, cost 4
/// Ncc^2 nk. O(cc channels).
ChannelItems PhWorkItems(const WorkSizes& sizes);

/// The work of one part (rank) of `num_parts`: three `OwnershipMap::Greedy`
/// maps (one per term, balanced per term) and the owned items. Value type.
class WorkAssignment {
 public:
  /// The assignment of a single part (owns nothing: no items).
  WorkAssignment();

  WorkAssignment(const WorkAssignment&) = default;
  WorkAssignment& operator=(const WorkAssignment&) = default;
  WorkAssignment(WorkAssignment&&) = default;
  WorkAssignment& operator=(WorkAssignment&&) = default;

  /// Builds the assignment of `part` of `num_parts`. Deterministic: every part
  /// builds the same maps. Precondition: 0 <= part < num_parts (IMSRG_CHECK).
  /// Complexity O(items log items).
  static WorkAssignment Build(const WorkSizes& sizes, std::int32_t part,
                              std::int32_t num_parts);

  /// This part, in [0, num_parts()).
  std::int32_t part() const noexcept { return part_; }
  /// Number of parts (>= 1).
  std::int32_t num_parts() const noexcept { return num_parts_; }
  /// True if the work is split (`num_parts() > 1`).
  bool distributed() const noexcept { return num_parts_ > 1; }
  /// The ladder map; item i is `all ladder items[i]` of `LadderWorkItems`.
  const comm::OwnershipMap& ladder_map() const noexcept { return ladder_map_; }
  /// The comm122 map (items = `Comm122WorkItems` channels in order).
  const comm::OwnershipMap& comm122_map() const noexcept {
    return comm122_map_;
  }
  /// The particle-hole map (items = `PhWorkItems` channels in order).
  const comm::OwnershipMap& ph_map() const noexcept { return ph_map_; }
  /// Row blocks owned by this part, ascending (channel, row_begin).
  const std::vector<RowBlock>& ladder_blocks() const noexcept {
    return ladder_blocks_;
  }
  /// Channels holding at least one owned ladder block, ascending.
  const std::vector<std::int32_t>& ladder_channels() const noexcept {
    return ladder_channels_;
  }
  /// comm122 channels owned by this part, ascending.
  const std::vector<std::int32_t>& comm122_channels() const noexcept {
    return comm122_channels_;
  }
  /// Cross-coupled channels owned by this part, ascending.
  const std::vector<std::int32_t>& ph_channels() const noexcept {
    return ph_channels_;
  }

 private:
  std::int32_t part_ = 0;
  std::int32_t num_parts_ = 1;
  comm::OwnershipMap ladder_map_, comm122_map_, ph_map_;
  std::vector<RowBlock> ladder_blocks_;
  std::vector<std::int32_t> ladder_channels_, comm122_channels_, ph_channels_;
};

/// Device copies of the owned-work lists that the kernels read. Allocated by
/// `AssignWork`, never per commutator call.
struct WorkDeviceData {
  /// Per two-body channel: 1 if it holds an owned ladder block, else 0.
  exec::DeviceBuffer<std::int32_t> ladder_active;
  /// Global columns e (indices into `Workspace::Impl::entry_channel`) of all
  /// columns of the active ladder channels, ascending.
  exec::DeviceBuffer<std::int32_t> ladder_columns;
  /// The same for the owned comm122 channels.
  exec::DeviceBuffer<std::int32_t> comm122_columns;
  /// Per cc channel: 1 if owned by this part, else 0.
  exec::DeviceBuffer<std::int32_t> ph_owned;
};

/// Builds the `WorkAssignment` of `part` of `num_parts` for the model space of
/// `ws` and its device lists (`Workspace::Impl::work`, `work_device`),
/// replacing the previous ones. The `Workspace` constructor calls it with
/// `(comm.rank(), comm.size())`; tests call it to give one process a fake part
/// of a split. Precondition: 0 <= part < num_parts (IMSRG_CHECK). Needs an
/// initialized `exec::Runtime`. Complexity O(kets + items log items).
void AssignWork(Workspace& ws, std::int32_t part, std::int32_t num_parts);

}  // namespace imsrg::commutator::detail

#endif  // IMSRG_COMMUTATOR_DETAIL_WORK_ITEMS_H_
