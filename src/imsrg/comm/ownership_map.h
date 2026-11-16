// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMM_OWNERSHIP_MAP_H_
#define IMSRG_COMM_OWNERSHIP_MAP_H_
/// @file
/// `imsrg::comm::OwnershipMap`: a deterministic assignment of weighted work
/// items to ranks. Pure host code, no MPI: every rank builds the same map from
/// the same costs.

#include <cstdint>
#include <span>
#include <vector>

namespace imsrg::comm {

/// Immutable assignment of `num_items()` work items to `num_ranks()` ranks
/// with the per-rank loads (summed costs). Value type.
class OwnershipMap {
 public:
  /// An empty map: no items, one rank.
  OwnershipMap();

  OwnershipMap(const OwnershipMap&) = default;
  OwnershipMap& operator=(const OwnershipMap&) = default;
  OwnershipMap(OwnershipMap&&) = default;
  OwnershipMap& operator=(OwnershipMap&&) = default;

  /// Longest-processing-time greedy assignment: items in order of decreasing
  /// cost (ties: lower index first), each given to the currently least-loaded
  /// rank (ties: lower rank). Deterministic.
  /// Throws `InputError` if `num_ranks < 1`, a cost is negative or not
  /// finite, or there are more than INT32_MAX items.
  /// Complexity O(n log n + n log P).
  static OwnershipMap Greedy(std::span<const double> cost,
                             std::int32_t num_ranks);

  /// Assignment given explicitly: item i goes to rank `owner[i]` with cost
  /// `cost[i]`. Same layout and invariants as `Greedy` (items of a rank in
  /// ascending order, per-rank loads summed in item order, `Imbalance`).
  /// Throws `InputError` if `num_ranks < 1`, `owner.size() != cost.size()`,
  /// an owner is outside [0, num_ranks), a cost is negative or not finite,
  /// or there are more than INT32_MAX items. Complexity O(n + P).
  static OwnershipMap FromOwners(std::span<const std::int32_t> owner,
                                 std::span<const double> cost,
                                 std::int32_t num_ranks);

  /// Number of work items.
  std::int32_t num_items() const noexcept {
    return static_cast<std::int32_t>(owner_.size());
  }
  /// Number of ranks (>= 1).
  std::int32_t num_ranks() const noexcept {
    return static_cast<std::int32_t>(load_.size());
  }
  /// Rank owning `item`, in [0, num_ranks()). Precondition: `item` in
  /// [0, num_items()) (IMSRG_ASSERT).
  std::int32_t owner(std::int32_t item) const;
  /// Items owned by `rank`, in ascending item order. Precondition: `rank` in
  /// [0, num_ranks()) (IMSRG_ASSERT).
  std::span<const std::int32_t> items_of(std::int32_t rank) const;
  /// Summed cost of the items of `rank`. Precondition: `rank` in
  /// [0, num_ranks()) (IMSRG_ASSERT).
  double load(std::int32_t rank) const;
  /// Load imbalance: max load / mean load - 1, the mean taken over all ranks;
  /// 0 if the total cost is 0.
  double Imbalance() const noexcept;

 private:
  // Fills items_ and item_begin_ (CSR by rank) from owner_ and the rank count
  // implied by load_.
  void BuildCsr();

  std::vector<std::int32_t> owner_;       // item -> rank
  std::vector<std::int32_t> items_;       // items grouped by rank (CSR)
  std::vector<std::int64_t> item_begin_;  // num_ranks + 1 offsets into items_
  std::vector<double> load_;              // per rank
};

}  // namespace imsrg::comm

#endif  // IMSRG_COMM_OWNERSHIP_MAP_H_
