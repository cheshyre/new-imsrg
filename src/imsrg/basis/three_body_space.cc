// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/basis/three_body_space.h"

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "imsrg/base/assert.h"

namespace imsrg::basis {

ThreeBodySpace::ThreeBodySpace(const ThreeBodyChannels& channels,
                               const SingleParticleBasis& basis,
                               const Reference& ref,
                               const ThreeBodyCuts& cuts) {
  channels_.reserve(static_cast<std::size_t>(channels.num_channels()));
  std::int64_t num_rows = 0;
  for (std::int32_t bc = 0; bc < channels.num_channels(); ++bc) {
    const Channel3Index ch{bc};
    const ThreeBodyChannel& src = channels.channel(ch);
    KeptChannel3 kc;
    kc.two_j = src.two_j;
    kc.parity = src.parity;
    kc.two_tz = src.two_tz;
    kc.row_offset = num_rows;
    const std::size_t first_triple = triples_.size();
    kc.triple_begin = Triple3Index{static_cast<std::int32_t>(first_triple)};
    std::int32_t dim = 0;
    for (const Ket3Index k : channels.channel_kets(ch)) {
      if (!channels.PassesConfigurationCut(k, basis, ref, cuts.de3max,
                                           cuts.occnat3_cut,
                                           cuts.natural_occupations)) {
        continue;
      }
      const Ket3& ket = channels.ket(k);
      const bool new_triple =
          triples_.size() == first_triple || triples_.back().a != ket.p ||
          triples_.back().b != ket.q || triples_.back().c != ket.r;
      if (new_triple) {
        if (triples_.size() > first_triple) {
          triples_.back().row_end = KeptRow3Index{dim};
        }
        OrbitTriple t;
        t.a = ket.p;
        t.b = ket.q;
        t.c = ket.r;
        t.channel = ch;
        t.row_begin = KeptRow3Index{dim};
        t.row_end = KeptRow3Index{dim};
        t.sym = (ket.p == ket.q && ket.q == ket.r)   ? 1
                : (ket.p == ket.q || ket.q == ket.r) ? 3
                                                     : 6;
        triples_.push_back(t);
      }
      row_two_jab_.push_back(ket.two_jpq.twice);
      row_triple_.push_back(static_cast<std::int32_t>(triples_.size()) - 1);
      row_ket_.push_back(k);
      ++dim;
    }
    if (dim > 0) triples_.back().row_end = KeptRow3Index{dim};
    kc.dim = dim;
    kc.triple_end = Triple3Index{static_cast<std::int32_t>(triples_.size())};
    num_rows += dim;
    channels_.push_back(kc);
  }
}

const KeptChannel3& ThreeBodySpace::channel(Channel3Index ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels(),
               "three-body channel index out of range");
  return channels_[ch.ToSize()];
}

const OrbitTriple& ThreeBodySpace::triple(Triple3Index t) const {
  IMSRG_ASSERT(t.value() >= 0 && t.value() < num_triples(),
               "triple index out of range");
  return triples_[t.ToSize()];
}

TwoJ ThreeBodySpace::row_two_jab(std::int64_t row) const {
  IMSRG_ASSERT(row >= 0 && row < NumRows(), "flat row out of range");
  return TwoJ{row_two_jab_[static_cast<std::size_t>(row)]};
}

Triple3Index ThreeBodySpace::row_triple(std::int64_t row) const {
  IMSRG_ASSERT(row >= 0 && row < NumRows(), "flat row out of range");
  return Triple3Index{row_triple_[static_cast<std::size_t>(row)]};
}

Ket3Index ThreeBodySpace::row_ket(std::int64_t row) const {
  IMSRG_ASSERT(row >= 0 && row < NumRows(), "flat row out of range");
  return row_ket_[static_cast<std::size_t>(row)];
}

ChannelRow3 ThreeBodySpace::ChannelRowOf(std::int64_t row) const {
  IMSRG_ASSERT(row >= 0 && row < NumRows(), "flat row out of range");
  // The first channel with row_offset + dim > row (empty channels have
  // row_offset + dim == row_offset <= row of a later channel, so they are
  // skipped automatically).
  const auto it = std::partition_point(
      channels_.begin(), channels_.end(),
      [row](const KeptChannel3& c) { return c.row_offset + c.dim <= row; });
  IMSRG_ASSERT(it != channels_.end(), "flat row without a channel");
  return ChannelRow3{
      Channel3Index{static_cast<std::int32_t>(it - channels_.begin())},
      KeptRow3Index{static_cast<std::int32_t>(row - it->row_offset)}};
}

std::int64_t ThreeBodySpace::FlatRow(Channel3Index ch,
                                     KeptRow3Index row) const {
  const KeptChannel3& kc = channel(ch);
  IMSRG_ASSERT(row.value() >= 0 && row.value() < kc.dim,
               "local row out of range");
  return kc.row_offset + row.value();
}

std::optional<KeptRow3Index> ThreeBodySpace::find_row(Channel3Index ch,
                                                      OrbitIndex a,
                                                      OrbitIndex b,
                                                      OrbitIndex c,
                                                      TwoJ two_jab) const {
  if (ch.value() < 0 || ch.value() >= num_channels()) return std::nullopt;
  const KeptChannel3& kc = channels_[ch.ToSize()];
  const auto first = triples_.begin() + kc.triple_begin.value();
  const auto last = triples_.begin() + kc.triple_end.value();
  using Key = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
  const auto it = std::lower_bound(
      first, last, Key{a.value(), b.value(), c.value()},
      [](const OrbitTriple& t, const Key& key) {
        return Key{t.a.value(), t.b.value(), t.c.value()} < key;
      });
  if (it == last || it->a != a || it->b != b || it->c != c) {
    return std::nullopt;
  }
  const std::int64_t base = kc.row_offset;
  const auto rows_first = row_two_jab_.begin() + (base + it->row_begin.value());
  const auto rows_last = row_two_jab_.begin() + (base + it->row_end.value());
  const auto r = std::lower_bound(rows_first, rows_last, two_jab.twice);
  if (r == rows_last || *r != two_jab.twice) return std::nullopt;
  return KeptRow3Index{
      static_cast<std::int32_t>((r - row_two_jab_.begin()) - base)};
}

ThreeBodySpaceCounts ThreeBodySpace::Counts() const {
  ThreeBodySpaceCounts s;
  for (const KeptChannel3& c : channels_) {
    const std::int64_t n = c.dim;
    s.sum_dim += n;
    s.sum_dim2 += n * n;
    s.sum_packed += n * (n + 1) / 2;
    s.max_dim = std::max(s.max_dim, c.dim);
    if (n > 0) ++s.num_nonempty_channels;
  }
  s.num_triples = num_triples();
  return s;
}

}  // namespace imsrg::basis
