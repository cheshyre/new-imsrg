// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/three_body_rows.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"

namespace imsrg::basis {
namespace {

using Size = std::size_t;

Size Sz(std::int64_t v) { return static_cast<Size>(v); }

/// Stored doubles of the rows [k0, k1) of an n x n block.
std::int64_t RowsStored(std::int64_t n, std::int64_t k0, std::int64_t k1,
                        bool packed) {
  const std::int64_t rows = k1 - k0;
  if (!packed) return rows * n;
  return rows * n - (k1 * (k1 - 1) - k0 * (k0 - 1)) / 2;
}

[[noreturn]] void Fail(const std::string& msg) {
  throw InputError("three-body rows: " + msg);
}

void CheckRange(const ThreeBodySpace& s, const RowRange& r) {
  if (r.channel.value() < 0 || r.channel.value() >= s.num_channels()) {
    Fail("range with an invalid channel");
  }
  const KeptChannel3& kc = s.channel(r.channel);
  if (r.triple_begin.value() >= r.triple_end.value()) {
    Fail("empty range in channel " + std::to_string(r.channel.value()));
  }
  if (r.triple_begin.value() < kc.triple_begin.value() ||
      r.triple_end.value() > kc.triple_end.value()) {
    Fail("range triples outside channel " + std::to_string(r.channel.value()));
  }
  const OrbitTriple& first = s.triple(r.triple_begin);
  const OrbitTriple& last = s.triple(Triple3Index{r.triple_end.value() - 1});
  if (r.row_begin != first.row_begin || r.row_end != last.row_end) {
    Fail("range rows not aligned to its triples in channel " +
         std::to_string(r.channel.value()));
  }
}

bool RangeBefore(const RowRange& a, const RowRange& b) {
  if (a.channel != b.channel) return a.channel.value() < b.channel.value();
  return a.row_begin.value() < b.row_begin.value();
}

void CheckWeights(const ThreeBodySpace& s, const LineWeights& w,
                  std::int32_t parts) {
  if (parts < 1) Fail("number of parts must be >= 1");
  const Size nt = Sz(s.num_triples());
  if (w.compute.size() != nt || w.memory.size() != nt) {
    Fail("weights need one entry per triple");
  }
  for (Size t = 0; t < nt; ++t) {
    const bool ok = std::isfinite(w.compute[t]) && w.compute[t] >= 0.0 &&
                    std::isfinite(w.memory[t]) && w.memory[t] >= 0.0;
    if (!ok) Fail("weights must be finite and non-negative");
  }
}

double Ratio(double part, double total) {
  return total > 0.0 ? part / total : 0.0;
}

double SumRange(const std::vector<double>& v, std::int32_t b, std::int32_t e) {
  double sum = 0.0;
  for (std::int32_t t = b; t < e; ++t) sum += v[Sz(t)];
  return sum;
}

/// Zipper order of the triples of channel `ch` (see RowPartition::Balanced).
std::vector<std::int32_t> ZipperOrder(const KeptChannel3& kc,
                                      const LineWeights& w) {
  const std::int32_t tb = kc.triple_begin.value();
  const std::int32_t te = kc.triple_end.value();
  const double total_c = SumRange(w.compute, tb, te);
  const double total_m = SumRange(w.memory, tb, te);
  std::vector<std::int32_t> order;
  order.reserve(Sz(te - tb));
  std::int32_t front = tb;
  std::int32_t back = te - 1;
  double diff = 0.0;  // taken compute ratio - taken memory ratio
  auto step = [&](std::int32_t t) {
    return Ratio(w.compute[Sz(t)], total_c) - Ratio(w.memory[Sz(t)], total_m);
  };
  while (front <= back) {
    const bool take_front = front == back || std::abs(diff + step(front)) <=
                                                 std::abs(diff + step(back));
    const std::int32_t t = take_front ? front++ : back--;
    diff += step(t);
    order.push_back(t);
  }
  return order;
}

/// Channels with triples by decreasing dimension (ties: lower index).
std::vector<std::int32_t> ChannelsByDim(const ThreeBodySpace& s) {
  std::vector<std::int32_t> chans;
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    if (s.dim(Channel3Index{c}) > 0) chans.push_back(c);
  }
  std::stable_sort(chans.begin(), chans.end(),
                   [&](std::int32_t a, std::int32_t b) {
                     return s.dim(Channel3Index{a}) > s.dim(Channel3Index{b});
                   });
  return chans;
}

/// Turns a per-triple part assignment into one RowSet per part.
std::vector<RowSet> BuildParts(const ThreeBodySpace& s,
                               const std::vector<std::int32_t>& owner,
                               std::int32_t parts) {
  std::vector<std::vector<RowRange>> ranges(Sz(parts));
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const KeptChannel3& kc = s.channel(Channel3Index{c});
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      std::vector<RowRange>& mine = ranges[Sz(owner[Sz(t)])];
      const OrbitTriple& tr = s.triple(Triple3Index{t});
      if (!mine.empty() && mine.back().channel.value() == c &&
          mine.back().triple_end.value() == t) {
        mine.back().triple_end = Triple3Index{t + 1};
        mine.back().row_end = tr.row_end;
      } else {
        mine.push_back(RowRange{Channel3Index{c}, Triple3Index{t},
                                Triple3Index{t + 1}, tr.row_begin, tr.row_end});
      }
    }
  }
  std::vector<RowSet> sets;
  sets.reserve(Sz(parts));
  for (auto& r : ranges) sets.push_back(RowSet::FromRanges(s, std::move(r)));
  return sets;
}

}  // namespace

RowSet RowSet::Whole(const ThreeBodySpace& s) {
  std::vector<RowRange> r;
  for (std::int32_t c = 0; c < s.num_channels(); ++c) {
    const KeptChannel3& kc = s.channel(Channel3Index{c});
    if (kc.dim == 0) continue;
    r.push_back(RowRange{Channel3Index{c}, kc.triple_begin, kc.triple_end,
                         KeptRow3Index{0}, KeptRow3Index{kc.dim}});
  }
  return FromRanges(s, std::move(r));
}

RowSet RowSet::FromRanges(const ThreeBodySpace& s, std::vector<RowRange> r) {
  for (const RowRange& x : r) CheckRange(s, x);
  std::sort(r.begin(), r.end(), RangeBefore);
  for (Size i = 1; i < r.size(); ++i) {
    if (r[i - 1].channel == r[i].channel &&
        r[i - 1].row_end.value() > r[i].row_begin.value()) {
      Fail("overlapping ranges in channel " +
           std::to_string(r[i].channel.value()));
    }
  }
  RowSet set;
  set.ranges_ = std::move(r);
  set.packed_prefix_.assign(1, 0);
  set.dense_prefix_.assign(1, 0);
  for (const RowRange& x : set.ranges_) {
    const std::int32_t n = s.dim(x.channel);
    const std::int64_t k0 = x.row_begin.value();
    const std::int64_t k1 = x.row_end.value();
    set.dims_.push_back(n);
    set.num_rows_ += k1 - k0;
    set.packed_prefix_.push_back(set.packed_prefix_.back() +
                                 RowsStored(n, k0, k1, true));
    set.dense_prefix_.push_back(set.dense_prefix_.back() +
                                RowsStored(n, k0, k1, false));
  }
  return set;
}

std::int64_t RowSet::StoredSize(bool packed) const noexcept {
  return packed ? packed_prefix_.back() : dense_prefix_.back();
}

std::int64_t RowSet::FindRange(Channel3Index ch, KeptRow3Index k) const {
  // First range that starts after (ch, k); the candidate is its predecessor.
  const RowRange probe{ch, Triple3Index{0}, Triple3Index{0}, k, k};
  auto it =
      std::upper_bound(ranges_.begin(), ranges_.end(), probe, RangeBefore);
  if (it == ranges_.begin()) return -1;
  --it;
  if (it->channel != ch || k.value() >= it->row_end.value()) return -1;
  return it - ranges_.begin();
}

bool RowSet::Contains(Channel3Index ch, KeptRow3Index k) const {
  return FindRange(ch, k) >= 0;
}

std::int64_t RowSet::LocalRowOffset(Channel3Index ch, KeptRow3Index k,
                                    bool packed) const {
  const std::int64_t i = FindRange(ch, k);
  if (i < 0) return -1;
  const RowRange& r = ranges_[Sz(i)];
  const std::vector<std::int64_t>& prefix =
      packed ? packed_prefix_ : dense_prefix_;
  return prefix[Sz(i)] +
         RowsStored(dims_[Sz(i)], r.row_begin.value(), k.value(), packed);
}

RowPartition::RowPartition(std::vector<RowSet> parts)
    : parts_(std::move(parts)) {
  for (Size p = 0; p < parts_.size(); ++p) {
    for (const RowRange& r : parts_[p].ranges()) {
      index_.push_back(Entry{r.channel, r.row_begin, r.row_end,
                             static_cast<std::int32_t>(p)});
    }
  }
  std::sort(index_.begin(), index_.end(), [](const Entry& a, const Entry& b) {
    if (a.channel != b.channel) return a.channel.value() < b.channel.value();
    return a.row_begin.value() < b.row_begin.value();
  });
}

RowPartition RowPartition::Whole(const ThreeBodySpace& s) {
  return RowPartition(std::vector<RowSet>{RowSet::Whole(s)});
}

RowPartition RowPartition::Balanced(const ThreeBodySpace& s,
                                    const LineWeights& w, std::int32_t parts) {
  CheckWeights(s, w, parts);
  const double total_c = SumRange(w.compute, 0, s.num_triples());
  const double total_m = SumRange(w.memory, 0, s.num_triples());
  std::vector<std::int32_t> owner(Sz(s.num_triples()), 0);
  double cum = 0.0;
  for (const std::int32_t c : ChannelsByDim(s)) {
    for (const std::int32_t t : ZipperOrder(s.channel(Channel3Index{c}), w)) {
      const double wt = 0.5 * Ratio(w.compute[Sz(t)], total_c) +
                        0.5 * Ratio(w.memory[Sz(t)], total_m);
      const double mid = cum + 0.5 * wt;
      const auto p = static_cast<std::int32_t>(
          std::floor(mid * static_cast<double>(parts)));
      owner[Sz(t)] = std::clamp(p, 0, parts - 1);
      cum += wt;
    }
  }
  return RowPartition(BuildParts(s, owner, parts));
}

RowPartition RowPartition::WholeChannels(const ThreeBodySpace& s,
                                         const LineWeights& w,
                                         std::int32_t parts) {
  CheckWeights(s, w, parts);
  std::vector<std::int32_t> chans = ChannelsByDim(s);
  std::vector<double> chan_compute(Sz(s.num_channels()), 0.0);
  for (const std::int32_t c : chans) {
    const KeptChannel3& kc = s.channel(Channel3Index{c});
    chan_compute[Sz(c)] =
        SumRange(w.compute, kc.triple_begin.value(), kc.triple_end.value());
  }
  std::stable_sort(chans.begin(), chans.end(),
                   [&](std::int32_t a, std::int32_t b) {
                     if (chan_compute[Sz(a)] != chan_compute[Sz(b)]) {
                       return chan_compute[Sz(a)] > chan_compute[Sz(b)];
                     }
                     return a < b;
                   });
  std::vector<double> load(Sz(parts), 0.0);
  std::vector<std::int32_t> owner(Sz(s.num_triples()), 0);
  for (const std::int32_t c : chans) {
    const auto it = std::min_element(load.begin(), load.end());
    const auto p = static_cast<std::int32_t>(it - load.begin());
    *it += chan_compute[Sz(c)];
    const KeptChannel3& kc = s.channel(Channel3Index{c});
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      owner[Sz(t)] = p;
    }
  }
  return RowPartition(BuildParts(s, owner, parts));
}

const RowSet& RowPartition::part(std::int32_t p) const {
  IMSRG_ASSERT(p >= 0 && p < num_parts());
  return parts_[Sz(p)];
}

std::int32_t RowPartition::Owner(Channel3Index ch, KeptRow3Index k) const {
  auto it = std::upper_bound(index_.begin(), index_.end(), Entry{ch, k, k, 0},
                             [](const Entry& a, const Entry& b) {
                               if (a.channel != b.channel) {
                                 return a.channel.value() < b.channel.value();
                               }
                               return a.row_begin.value() < b.row_begin.value();
                             });
  if (it == index_.begin()) return -1;
  --it;
  if (it->channel != ch || k.value() >= it->row_end.value()) return -1;
  return it->part;
}

PartitionImbalanceResult PartitionImbalance(const RowPartition& p,
                                            const LineWeights& w) {
  std::vector<double> compute(Sz(p.num_parts()), 0.0);
  std::vector<double> memory(Sz(p.num_parts()), 0.0);
  for (std::int32_t q = 0; q < p.num_parts(); ++q) {
    for (const RowRange& r : p.part(q).ranges()) {
      IMSRG_CHECK(Sz(r.triple_end.value()) <= w.compute.size() &&
                      Sz(r.triple_end.value()) <= w.memory.size(),
                  "weights too short for the partition");
      compute[Sz(q)] +=
          SumRange(w.compute, r.triple_begin.value(), r.triple_end.value());
      memory[Sz(q)] +=
          SumRange(w.memory, r.triple_begin.value(), r.triple_end.value());
    }
  }
  auto imbalance = [](const std::vector<double>& v) {
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    if (sum <= 0.0) return 0.0;
    const double mean = sum / static_cast<double>(v.size());
    return *std::max_element(v.begin(), v.end()) / mean - 1.0;
  };
  return PartitionImbalanceResult{imbalance(compute), imbalance(memory)};
}

}  // namespace imsrg::basis
