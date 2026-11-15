// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASIS_THREE_BODY_SPACE_H_
#define IMSRG_BASIS_THREE_BODY_SPACE_H_
/// @file
/// The kept canonical three-body space (design/14 section 2 and decision D43,
/// layout A): the imsrg++ kets |(ab) Jab, c; J> with a <= b <= c of
/// `ThreeBodyChannels` that pass the configuration cut (dE3max, OccNat3Cut) of
/// `ThreeBodyCuts` for a given reference. Kets failing the cut are dropped at
/// enumeration (owner decision D34), so the cut defines the space identically
/// for every term. Host-only (no Kokkos).
///
/// Index conventions (the contract of the operator and kernel code):
/// * channels: exactly those of `ThreeBodyChannels`, same `Channel3Index`; a
///   channel may have kept dimension 0;
/// * local row `KeptRow3Index` in [0, dim(ch)): the position of a kept ket in
///   its channel; the order is that of `ThreeBodyChannels::channel_kets`
///   (ascending global ket, i.e. lexicographic in (a, b, c, Jab)) restricted
///   to the kept kets;
/// * flat row (std::int64_t) in [0, NumRows()): channel `ch` owns the flat
///   rows [row_offset(ch), row_offset(ch) + dim(ch)), in channel order, so
///   flat row = row_offset(ch) + local row;
/// * triple `Triple3Index` t in [0, num_triples()): the orbit triples a <= b
///   <= c that have at least one kept row, grouped by channel (channel `ch`
///   owns [triple_begin, triple_end)) and ordered lexicographically by (a, b,
///   c) inside it; the rows of one triple are contiguous and ordered by
///   ascending Jab, and a triple stores its channel-local row range
///   [row_begin, row_end).

#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/reference.h"
#include "imsrg/basis/single_particle_basis.h"
#include "imsrg/basis/three_body_channels.h"

namespace imsrg::basis {

/// Channel-local row of a kept three-body ket (see the file comment).
using KeptRow3Index = StrongIndex<struct KeptRow3Tag>;
/// Flat index of an orbit triple (a, b, c) with a <= b <= c (see the file
/// comment).
using Triple3Index = StrongIndex<struct Triple3Tag>;

/// One three-body channel of the kept space.
struct KeptChannel3 {
  TwoJ two_j{1};                  ///< 2J (odd).
  Parity parity = Parity::kEven;  ///< (-1)^(l_a + l_b + l_c).
  TwoTz two_tz{-3};               ///< 2Tz in {-3, -1, 1, 3}.
  std::int32_t dim = 0;           ///< Number of kept rows.
  std::int64_t row_offset = 0;    ///< Flat index of local row 0.
  Triple3Index triple_begin;      ///< First triple of the channel.
  Triple3Index triple_end;        ///< One past the last triple.
};
static_assert(std::is_trivially_copyable_v<KeptChannel3>);

/// An orbit triple (a, b, c), a <= b <= c, inside one channel.
struct OrbitTriple {
  OrbitIndex a;             ///< First orbit.
  OrbitIndex b;             ///< Second orbit (b >= a).
  OrbitIndex c;             ///< Third orbit (c >= b).
  Channel3Index channel;    ///< Channel of the triple.
  KeptRow3Index row_begin;  ///< First channel-local row of the triple.
  KeptRow3Index row_end;    ///< One past the last channel-local row.
  std::int32_t sym = 6;     ///< 6 (a, b, c distinct), 3 (two equal), 1.
};
static_assert(std::is_trivially_copyable_v<OrbitTriple>);

/// A flat row resolved to its channel and channel-local row.
struct ChannelRow3 {
  Channel3Index channel;  ///< The channel.
  KeptRow3Index row;      ///< The channel-local row.
};

/// Size summary of a `ThreeBodySpace`.
struct ThreeBodySpaceCounts {
  std::int64_t sum_dim = 0;     ///< Sum over channels of the dimension N.
  std::int64_t sum_dim2 = 0;    ///< Sum of N^2 (dense doubles per operator).
  std::int64_t sum_packed = 0;  ///< Sum of N (N + 1) / 2 (packed triangle).
  std::int32_t max_dim = 0;     ///< Largest N.
  std::int32_t num_nonempty_channels = 0;  ///< Channels with N > 0.
  std::int64_t num_triples = 0;            ///< Number of orbit triples.
};

/// Immutable kept canonical three-body space. Value type (copies are deep).
class ThreeBodySpace {
 public:
  /// Builds the space: for every channel of `channels` (same indices), the
  /// kets for which `channels.PassesConfigurationCut(k, basis, ref,
  /// cuts.de3max, cuts.occnat3_cut, cuts.natural_occupations)` holds, in the
  /// channel's ket order. Preconditions: `ref` belongs to `basis`
  /// (IMSRG_ASSERT in the cut). Cost O(SumChannelKets()).
  ThreeBodySpace(const ThreeBodyChannels& channels,
                 const SingleParticleBasis& basis, const Reference& ref,
                 const ThreeBodyCuts& cuts);

  ThreeBodySpace(const ThreeBodySpace&) = default;
  ThreeBodySpace& operator=(const ThreeBodySpace&) = default;
  ThreeBodySpace(ThreeBodySpace&&) = default;
  ThreeBodySpace& operator=(ThreeBodySpace&&) = default;

  /// Number of channels (equal to `ThreeBodyChannels::num_channels()`).
  std::int32_t num_channels() const noexcept {
    return static_cast<std::int32_t>(channels_.size());
  }
  /// All channels in `Channel3Index` order.
  std::span<const KeptChannel3> channels() const noexcept { return channels_; }
  /// Channel `ch`. Precondition: valid index (IMSRG_ASSERT).
  const KeptChannel3& channel(Channel3Index ch) const;
  /// Kept dimension of channel `ch` (possibly 0). Precondition: valid index.
  std::int32_t dim(Channel3Index ch) const { return channel(ch).dim; }

  /// Number of kept rows (sum of the channel dimensions).
  std::int64_t NumRows() const noexcept {
    return static_cast<std::int64_t>(row_two_jab_.size());
  }
  /// Number of orbit triples.
  std::int32_t num_triples() const noexcept {
    return static_cast<std::int32_t>(triples_.size());
  }
  /// All triples in `Triple3Index` order.
  std::span<const OrbitTriple> triples() const noexcept { return triples_; }
  /// Triple `t`. Precondition: valid index (IMSRG_ASSERT).
  const OrbitTriple& triple(Triple3Index t) const;

  /// 2 Jab of flat row `row`. Precondition: 0 <= row < NumRows().
  TwoJ row_two_jab(std::int64_t row) const;
  /// Triple of flat row `row`. Precondition: 0 <= row < NumRows().
  Triple3Index row_triple(std::int64_t row) const;
  /// Global ket (index into `ThreeBodyChannels::kets()`) of flat row `row`.
  /// Precondition: 0 <= row < NumRows().
  Ket3Index row_ket(std::int64_t row) const;
  /// Channel and channel-local row of flat row `row`. O(log num_channels()).
  /// Precondition: 0 <= row < NumRows().
  ChannelRow3 ChannelRowOf(std::int64_t row) const;
  /// Flat row of (`ch`, `row`): row_offset(ch) + row. Precondition: `row <
  /// dim(ch)` (IMSRG_ASSERT).
  std::int64_t FlatRow(Channel3Index ch, KeptRow3Index row) const;

  /// Channel-local row of the kept ket |(a b) Jab, c; ch> or nullopt if the
  /// ket is dropped (cut, not a member of the channel, or not canonical, i.e.
  /// not a <= b <= c). O(log triples + log Jab values).
  std::optional<KeptRow3Index> find_row(Channel3Index ch, OrbitIndex a,
                                        OrbitIndex b, OrbitIndex c,
                                        TwoJ two_jab) const;

  /// Sums over the channels.
  ThreeBodySpaceCounts Counts() const;

 private:
  std::vector<KeptChannel3> channels_;
  std::vector<OrbitTriple> triples_;
  std::vector<std::int32_t> row_two_jab_;  // per flat row
  std::vector<std::int32_t> row_triple_;   // per flat row
  std::vector<Ket3Index> row_ket_;         // per flat row
};

}  // namespace imsrg::basis

#endif  // IMSRG_BASIS_THREE_BODY_SPACE_H_
