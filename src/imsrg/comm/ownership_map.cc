// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/comm/ownership_map.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"

namespace imsrg::comm {

OwnershipMap::OwnershipMap() : item_begin_(2, 0), load_(1, 0.0) {}

namespace {

void ValidateGreedyInput(std::span<const double> cost, std::int32_t num_ranks) {
  if (num_ranks < 1) {
    throw InputError("OwnershipMap: num_ranks must be >= 1");
  }
  if (cost.size() > static_cast<std::size_t>(INT32_MAX)) {
    throw InputError("OwnershipMap: more than INT32_MAX items");
  }
  for (const double c : cost) {
    if (!std::isfinite(c) || c < 0.0) {
      throw InputError("OwnershipMap: cost must be finite and >= 0");
    }
  }
}

}  // namespace

OwnershipMap OwnershipMap::Greedy(std::span<const double> cost,
                                  std::int32_t num_ranks) {
  ValidateGreedyInput(cost, num_ranks);
  std::vector<std::int32_t> order(cost.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::int32_t a, std::int32_t b) {
                     return cost[static_cast<std::size_t>(a)] >
                            cost[static_cast<std::size_t>(b)];
                   });
  using Entry = std::pair<double, std::int32_t>;  // (load, rank)
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
  for (std::int32_t r = 0; r < num_ranks; ++r) heap.push({0.0, r});

  OwnershipMap map;
  map.owner_.assign(cost.size(), 0);
  map.load_.assign(static_cast<std::size_t>(num_ranks), 0.0);
  for (const std::int32_t item : order) {
    const auto [load, rank] = heap.top();
    heap.pop();
    const double new_load = load + cost[static_cast<std::size_t>(item)];
    map.owner_[static_cast<std::size_t>(item)] = rank;
    map.load_[static_cast<std::size_t>(rank)] = new_load;
    heap.push({new_load, rank});
  }

  map.BuildCsr();
  return map;
}

void OwnershipMap::BuildCsr() {
  const auto n = static_cast<std::int32_t>(owner_.size());
  const std::size_t num_ranks = load_.size();
  item_begin_.assign(num_ranks + 1, 0);
  for (const std::int32_t r : owner_) {
    ++item_begin_[static_cast<std::size_t>(r) + 1];
  }
  std::partial_sum(item_begin_.begin(), item_begin_.end(), item_begin_.begin());
  items_.resize(owner_.size());
  std::vector<std::int64_t> fill(item_begin_.begin(), item_begin_.end() - 1);
  for (std::int32_t item = 0; item < n; ++item) {  // ascending per rank
    const auto r =
        static_cast<std::size_t>(owner_[static_cast<std::size_t>(item)]);
    items_[static_cast<std::size_t>(fill[r]++)] = item;
  }
}

OwnershipMap OwnershipMap::FromOwners(std::span<const std::int32_t> owner,
                                      std::span<const double> cost,
                                      std::int32_t num_ranks) {
  ValidateGreedyInput(cost, num_ranks);
  if (owner.size() != cost.size()) {
    throw InputError("OwnershipMap::FromOwners: owner and cost sizes differ");
  }
  OwnershipMap map;
  map.owner_.assign(owner.begin(), owner.end());
  map.load_.assign(static_cast<std::size_t>(num_ranks), 0.0);
  for (std::size_t i = 0; i < owner.size(); ++i) {
    if (owner[i] < 0 || owner[i] >= num_ranks) {
      throw InputError("OwnershipMap::FromOwners: owner out of range");
    }
    map.load_[static_cast<std::size_t>(owner[i])] += cost[i];
  }
  map.BuildCsr();
  return map;
}

std::int32_t OwnershipMap::owner(std::int32_t item) const {
  IMSRG_ASSERT(item >= 0 && item < num_items(),
               "OwnershipMap::owner: item out of range");
  return owner_[static_cast<std::size_t>(item)];
}

std::span<const std::int32_t> OwnershipMap::items_of(std::int32_t rank) const {
  IMSRG_ASSERT(rank >= 0 && rank < num_ranks(),
               "OwnershipMap::items_of: rank out of range");
  const auto r = static_cast<std::size_t>(rank);
  return std::span<const std::int32_t>(items_).subspan(
      static_cast<std::size_t>(item_begin_[r]),
      static_cast<std::size_t>(item_begin_[r + 1] - item_begin_[r]));
}

double OwnershipMap::load(std::int32_t rank) const {
  IMSRG_ASSERT(rank >= 0 && rank < num_ranks(),
               "OwnershipMap::load: rank out of range");
  return load_[static_cast<std::size_t>(rank)];
}

double OwnershipMap::Imbalance() const noexcept {
  const double total = std::accumulate(load_.begin(), load_.end(), 0.0);
  if (!(total > 0.0)) return 0.0;
  const double mean = total / static_cast<double>(load_.size());
  return *std::max_element(load_.begin(), load_.end()) / mean - 1.0;
}

}  // namespace imsrg::comm
