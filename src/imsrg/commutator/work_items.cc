// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Work items, costs and per-part assignment of the expensive IMSRG(2) terms
// (design/12 section 6). Host-side only; no Kokkos.

#include "imsrg/commutator/detail/work_items.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/basis/particle_hole_channels.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/workspace_impl.h"

namespace imsrg::commutator::detail {

namespace {

/// Counts the kets of `ch` with non-zero pp and hh weights (the rule of
/// `FillProjector` in ladder_data.cc, same expressions).
void CountProjectors(const basis::TwoBodyChannels& tb,
                     std::span<const double> occ, std::int32_t ch,
                     std::int32_t& k_pp, std::int32_t& k_hh) {
  k_pp = 0;
  k_hh = 0;
  for (const KetIndex k : tb.channel_kets(ChannelIndex{ch})) {
    const auto& ket = tb.ket(k);
    const double np = occ[static_cast<std::size_t>(ket.p.value())];
    const double nq = occ[static_cast<std::size_t>(ket.q.value())];
    if ((1.0 - np) * (1.0 - nq) != 0.0) ++k_pp;
    if (np * nq != 0.0) ++k_hh;
  }
}

/// Counts the lifted one-body entries of the rows of `ch` (the rule of
/// `BuildLiftedData` in lifted_data.cc: an entry exists if the lifted ket
/// exists for this J).
std::int32_t CountLifted(const basis::ModelSpace& ms, std::int32_t ch) {
  const auto& tb = ms.two_body();
  const auto& sp = ms.basis();
  const ChannelIndex cidx{ch};
  const TwoJ two_j = tb.channel(cidx).two_j;
  auto exists = [&](int a, int b) {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    return tb.local_index(tb.ket_index(OrbitIndex{lo}, OrbitIndex{hi}), two_j)
        .has_value();
  };
  std::int32_t count = 0;
  for (const KetIndex kidx : tb.channel_kets(cidx)) {
    const auto& ket = tb.ket(kidx);
    const int i = ket.p.value();
    const int j = ket.q.value();
    for (const OrbitIndex c : sp.SameLjtz(OrbitIndex{i}))
      count += exists(c.value(), j) ? 1 : 0;
    for (const OrbitIndex c : sp.SameLjtz(OrbitIndex{j}))
      count += exists(i, c.value()) ? 1 : 0;
  }
  return count;
}

/// Counts |K| of a cc channel (the rule of `BuildChannelTables` in
/// pandya_data.cc: two entries per ket with n_p != n_q).
std::int32_t CountPandyaK(const basis::TwoBodyChannels& tb,
                          const basis::ParticleHoleChannels& ph,
                          std::span<const double> occ, std::int32_t c) {
  std::int32_t nk = 0;
  for (const KetIndex k : ph.channel_kets(ChannelIndex{c})) {
    const auto& ket = tb.ket(k);
    const double np = occ[static_cast<std::size_t>(ket.p.value())];
    const double nq = occ[static_cast<std::size_t>(ket.q.value())];
    if (np != nq) nk += 2;
  }
  return nk;
}

/// Maps the item indices owned by `part` to `items[...]`.
template <class T>
std::vector<T> Owned(const comm::OwnershipMap& map, std::int32_t part,
                     const std::vector<T>& items) {
  std::vector<T> out;
  for (const std::int32_t i : map.items_of(part))
    out.push_back(items[static_cast<std::size_t>(i)]);
  return out;
}

/// Uploads `host` to a new device buffer named `label` (empty: no copy).
exec::DeviceBuffer<std::int32_t> Upload(const std::vector<std::int32_t>& host,
                                        const char* label) {
  exec::DeviceBuffer<std::int32_t> b(host.size(), label);
  if (!host.empty()) b.CopyFromHost(std::span<const std::int32_t>(host));
  return b;
}

/// The global columns (indices into `entry_channel`) of `channels`, ascending;
/// `base` holds the first column of every channel (num_channels + 1 entries).
std::vector<std::int32_t> ColumnsOf(const std::vector<std::int32_t>& channels,
                                    const std::vector<std::int32_t>& base) {
  std::vector<std::int32_t> cols;
  for (const std::int32_t c : channels) {
    const auto cs = static_cast<std::size_t>(c);
    for (std::int32_t e = base[cs]; e < base[cs + 1]; ++e) cols.push_back(e);
  }
  return cols;
}

/// Flags (1) the entries of `list` in a vector of `size` zeros.
std::vector<std::int32_t> Flags(const std::vector<std::int32_t>& list,
                                std::size_t size) {
  std::vector<std::int32_t> flags(size, 0);
  for (const std::int32_t c : list) flags[static_cast<std::size_t>(c)] = 1;
  return flags;
}

}  // namespace

WorkSizes ComputeWorkSizes(const basis::ModelSpace& ms) {
  const auto& tb = ms.two_body();
  const auto& ph = ms.particle_hole();
  const auto occ = ms.reference().occupations();
  const auto nch = static_cast<std::size_t>(tb.num_channels());
  const auto ncc = static_cast<std::size_t>(ph.num_channels());
  WorkSizes s;
  s.num_kets.assign(nch, 0);
  s.k_pp.assign(nch, 0);
  s.k_hh.assign(nch, 0);
  s.lifted.assign(nch, 0);
  for (std::size_t c = 0; c < nch; ++c) {
    const auto ci = static_cast<std::int32_t>(c);
    s.num_kets[c] =
        static_cast<std::int32_t>(tb.channel_kets(ChannelIndex{ci}).size());
    CountProjectors(tb, occ, ci, s.k_pp[c], s.k_hh[c]);
    s.lifted[c] = CountLifted(ms, ci);
  }
  s.ncc.assign(ncc, 0);
  s.nk.assign(ncc, 0);
  for (std::size_t c = 0; c < ncc; ++c) {
    const auto ci = static_cast<std::int32_t>(c);
    s.ncc[c] =
        static_cast<std::int32_t>(ph.channel_kets(ChannelIndex{ci}).size());
    s.nk[c] = CountPandyaK(tb, ph, occ, ci);
  }
  return s;
}

// Calibration of the term weights (ladder units): one rank, 16 OpenMP threads
// (OMP_NUM_THREADS=16 OMP_PROC_BIND=false), release-openmp build, Apple
// silicon laptop, 2026-11-17, O16 emax 8, `commutator_bench --emax 8 --reps 5`
// (best of 3 runs of the per-term minimum times: 10.59 ms, 4.76 ms, 15.52 ms;
// the machine was shared, load average 12-50, so the weights are good to about
// 15 %; the same bench at emax 6 gives w_ph 3.0, w_122 44, the balance is not
// sensitive to them): w_t = (time_t / units_t) /
// (time_ladder / units_ladder) with time_ladder = comm222_pp_hhss, time_122 =
// comm122ss, time_ph = comm222_phss and units_t the totals of
// `LadderWorkItems` (5.841e9), `Comm122WorkItems` (4.452e7) and `PhWorkItems`
// (2.913e9). See design/benchmarks/mpi-scaling.md, "Joint cut".
const double kComm122WeightInLadderUnits = 59.0;
const double kPhWeightInLadderUnits = 2.9;

LadderItems LadderWorkItems(const WorkSizes& sizes, std::int32_t num_parts) {
  IMSRG_CHECK(num_parts >= 1, "LadderWorkItems: num_parts must be >= 1");
  return LadderWorkItems(
      sizes, std::vector<double>(static_cast<std::size_t>(num_parts), 1.0));
}

LadderItems LadderWorkItems(const WorkSizes& sizes,
                            const std::vector<double>& capacity) {
  IMSRG_CHECK(!capacity.empty(), "LadderWorkItems: need at least one part");
  const auto num_parts = static_cast<std::int32_t>(capacity.size());
  double cap_total = 0.0;
  for (const double c : capacity) {
    IMSRG_CHECK(std::isfinite(c) && c >= 0.0,
                "LadderWorkItems: capacities must be finite and >= 0");
    cap_total += c;
  }
  const bool equal = !(cap_total > 0.0);
  std::vector<double> cum(capacity.size() + 1, 0.0);  // C_r (or r if equal)
  for (std::size_t r = 0; r < capacity.size(); ++r)
    cum[r + 1] = cum[r] + (equal ? 1.0 : capacity[r]);
  const std::size_t nch = sizes.num_kets.size();
  auto unit_of = [&](std::size_t c) {  // cost of one row
    return 2.0 * sizes.num_kets[c] * (sizes.k_pp[c] + sizes.k_hh[c]);
  };
  auto eligible = [&](std::size_t c) {
    return sizes.num_kets[c] > 0 && sizes.k_pp[c] + sizes.k_hh[c] > 0;
  };
  // Prefix sums S_i of the row costs over the global row sequence (exact:
  // integers below 2^53), and the first global row g_c of each channel.
  std::vector<double> prefix(1, 0.0);
  std::vector<std::int64_t> first_row(nch, 0);
  for (std::size_t c = 0; c < nch; ++c) {
    first_row[c] = static_cast<std::int64_t>(prefix.size()) - 1;
    if (!eligible(c)) continue;
    const double unit = unit_of(c);
    for (std::int32_t j = 0; j < sizes.num_kets[c]; ++j)
      prefix.push_back(prefix.back() + unit);
  }
  const auto num_rows = static_cast<std::int64_t>(prefix.size()) - 1;
  const double total = prefix.back();
  // cut[r] = the boundary whose S_i is nearest to T C_r / C_P (ties: lower i).
  std::vector<std::int64_t> cut(static_cast<std::size_t>(num_parts) + 1, 0);
  cut.back() = num_rows;
  for (std::int32_t r = 1; r < num_parts; ++r) {
    const double target = total * cum[static_cast<std::size_t>(r)] / cum.back();
    auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
    auto i = static_cast<std::int64_t>(it - prefix.begin());  // S_i >= target
    if (i > 0 && target - prefix[static_cast<std::size_t>(i - 1)] <=
                     prefix[static_cast<std::size_t>(i)] - target)
      --i;
    cut[static_cast<std::size_t>(r)] =
        std::max(i, cut[static_cast<std::size_t>(r - 1)]);
  }
  LadderItems items;
  std::int32_t rank = 0;
  for (std::size_t c = 0; c < nch; ++c) {
    if (!eligible(c)) continue;
    const double unit = unit_of(c);
    const std::int64_t g0 = first_row[c];
    const std::int64_t g1 = g0 + sizes.num_kets[c];
    for (std::int64_t pos = g0; pos < g1;) {
      while (cut[static_cast<std::size_t>(rank) + 1] <= pos) ++rank;
      const std::int64_t hi =
          std::min(g1, cut[static_cast<std::size_t>(rank) + 1]);
      items.blocks.push_back({static_cast<std::int32_t>(c),
                              static_cast<std::int32_t>(pos - g0),
                              static_cast<std::int32_t>(hi - g0)});
      items.cost.push_back(unit * static_cast<double>(hi - pos));
      items.owner.push_back(rank);
      pos = hi;
    }
  }
  return items;
}

ChannelItems Comm122WorkItems(const WorkSizes& sizes) {
  ChannelItems items;
  for (std::size_t c = 0; c < sizes.num_kets.size(); ++c) {
    const double n = sizes.num_kets[c];
    if (sizes.num_kets[c] <= 0 || sizes.lifted[c] <= 0) continue;
    items.channels.push_back(static_cast<std::int32_t>(c));
    items.cost.push_back(n * sizes.lifted[c] + 0.5 * n * n);
  }
  return items;
}

ChannelItems PhWorkItems(const WorkSizes& sizes) {
  ChannelItems items;
  for (std::size_t c = 0; c < sizes.ncc.size(); ++c) {
    const double n = sizes.ncc[c];
    if (sizes.ncc[c] <= 0 || sizes.nk[c] <= 0) continue;
    items.channels.push_back(static_cast<std::int32_t>(c));
    items.cost.push_back(4.0 * n * n * sizes.nk[c]);
  }
  return items;
}

LadderItems JointLadderItems(const WorkSizes& sizes,
                             const comm::OwnershipMap& comm122_map,
                             const comm::OwnershipMap& ph_map) {
  const std::int32_t num_parts = comm122_map.num_ranks();
  IMSRG_CHECK(ph_map.num_ranks() == num_parts,
              "JointLadderItems: maps must have the same number of ranks");
  if (num_parts == 1) return LadderWorkItems(sizes, 1);
  const auto np = static_cast<std::size_t>(num_parts);
  std::vector<double> other(np, 0.0);
  double other_sum = 0.0;
  for (std::int32_t r = 0; r < num_parts; ++r) {
    other[static_cast<std::size_t>(r)] =
        kComm122WeightInLadderUnits * comm122_map.load(r) +
        kPhWeightInLadderUnits * ph_map.load(r);
    other_sum += other[static_cast<std::size_t>(r)];
  }
  double ladder_total = 0.0;
  for (const double c : LadderWorkItems(sizes, 1).cost) ladder_total += c;
  const double target = (ladder_total + other_sum) / num_parts;
  std::vector<double> capacity(np, 0.0);
  double cap_sum = 0.0;
  for (std::size_t r = 0; r < np; ++r) {
    capacity[r] = std::max(0.0, target - other[r]);
    cap_sum += capacity[r];
  }
  if (cap_sum > 0.0)
    for (double& c : capacity) c *= ladder_total / cap_sum;
  return LadderWorkItems(sizes, capacity);
}

WorkAssignment::WorkAssignment() = default;

WorkAssignment WorkAssignment::Build(const WorkSizes& sizes, std::int32_t part,
                                     std::int32_t num_parts) {
  IMSRG_CHECK(num_parts >= 1 && part >= 0 && part < num_parts,
              "WorkAssignment: need 0 <= part < num_parts");
  WorkAssignment a;
  a.part_ = part;
  a.num_parts_ = num_parts;
  const ChannelItems c122 = Comm122WorkItems(sizes);
  const ChannelItems ph = PhWorkItems(sizes);
  a.comm122_map_ = comm::OwnershipMap::Greedy(c122.cost, num_parts);
  a.ph_map_ = comm::OwnershipMap::Greedy(ph.cost, num_parts);
  const LadderItems lad = JointLadderItems(sizes, a.comm122_map_, a.ph_map_);
  a.ladder_map_ =
      comm::OwnershipMap::FromOwners(lad.owner, lad.cost, num_parts);
  a.ladder_blocks_ = Owned(a.ladder_map_, part, lad.blocks);
  for (const RowBlock& b : a.ladder_blocks_)
    if (a.ladder_channels_.empty() || a.ladder_channels_.back() != b.channel)
      a.ladder_channels_.push_back(b.channel);
  a.comm122_channels_ = Owned(a.comm122_map_, part, c122.channels);
  a.ph_channels_ = Owned(a.ph_map_, part, ph.channels);
  return a;
}

void AssignWork(Workspace& ws, std::int32_t part, std::int32_t num_parts) {
  const WorkSizes sizes = ComputeWorkSizes(ws.model_space());
  WorkAssignment a = WorkAssignment::Build(sizes, part, num_parts);
  const std::size_t nch = sizes.num_kets.size();
  std::vector<std::int32_t> base(nch + 1, 0);
  for (std::size_t c = 0; c < nch; ++c)
    base[c + 1] = base[c] + sizes.num_kets[c];
  WorkDeviceData d;
  d.ladder_active =
      Upload(Flags(a.ladder_channels(), nch), "commutator::work_ladder_active");
  d.ladder_columns = Upload(ColumnsOf(a.ladder_channels(), base),
                            "commutator::work_ladder_columns");
  d.comm122_columns = Upload(ColumnsOf(a.comm122_channels(), base),
                             "commutator::work_comm122_columns");
  d.ph_owned =
      Upload(Flags(a.ph_channels(), sizes.ncc.size()), "commutator::work_ph");
  Workspace::Impl& impl = ws.impl();
  impl.work = std::move(a);
  impl.work_device = std::move(d);
  AttachLadderWork(impl);
  AttachPandyaWork(impl.pandya, impl.work.ph_channels(),
                   impl.work.distributed());
}

}  // namespace imsrg::commutator::detail
