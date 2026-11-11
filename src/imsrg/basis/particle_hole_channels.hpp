#pragma once
/// @file
/// Cross-coupled (particle-hole, Pandya) two-body channels (design/reference/01 sections 2.2-2.3,
/// imsrg++ `TwoBodyChannel_CC`). Only the channel and ket structure lives here; the 6j-weighted
/// Pandya maps are built by the commutator module.
///
/// Conventions (identical to imsrg++, design/00 D11):
/// * the channel index is the normal one, `imsrgpp_channel_index` = 6 J + 2 (Tz + 1) + parity,
///   with 6 (J_max + 1) channels; the struct `TwoBodyChannel` is reused, with `two_tz` = 2 Tz =
///   |tz2_p - tz2_q| of the kets, i.e. Tz = 0 holds pp and nn kets, Tz = +1 holds pn kets; the
///   Tz = -1 channels exist (as slots) but are always empty;
/// * the ket |pq> (p <= q, global `KetIndex` of the normal channels) is in channel (J, P, Tz) if
///   parity (-1)^(l_p + l_q) = P, |tz2_p - tz2_q| = 2 Tz and |j_p - j_q| <= J <= j_p + j_q
///   (`TwoBodyChannel_CC::CheckChannel_ket`); there is no Pauli rule, so p == q kets appear for
///   every J of their range;
/// * inside a channel kets are in ascending global order (the local index).

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/two_body_channels.hpp"

namespace imsrg::basis {

/// Immutable cross-coupled channel structure of a two-body ket set. Value type.
class ParticleHoleChannels {
 public:
  /// Builds the channels from the orbits and kets of `two_body`. Cost O(sum over kets of
  /// (j_p + j_q - |j_p - j_q| + 1)).
  explicit ParticleHoleChannels(const TwoBodyChannels& two_body);

  /// Number of orbits.
  std::int32_t num_orbits() const noexcept { return num_orbits_; }
  /// Number of global kets (the same as the normal channels).
  std::int32_t num_kets() const noexcept { return num_kets_; }
  /// Largest J of any channel.
  std::int32_t max_J() const noexcept { return max_J_; }
  /// Number of channels, 6 (max_J() + 1), including empty ones.
  std::int32_t num_channels() const noexcept { return static_cast<std::int32_t>(channels_.size()); }
  /// All channels in index order.
  std::span<const TwoBodyChannel> channels() const noexcept { return channels_; }
  /// Channel `ch`. Precondition: 0 <= ch < num_channels() (IMSRG_ASSERT).
  const TwoBodyChannel& channel(ChannelIndex ch) const;

  /// Index of the channel (2J, parity, 2Tz) with 2Tz in {-2, 0, 2}, or nullopt if J is negative or
  /// above max_J(), or 2J or 2Tz is odd or out of range. O(1).
  std::optional<ChannelIndex> find_channel(TwoJ two_j, Parity parity, TwoTz two_tz) const noexcept;
  /// Channel containing the pair (p, q) (either order) coupled to 2J = `two_j`; the pair need
  /// not be allowed in it (use `local_index`). Precondition: valid orbits and 0 <= J <= max_J()
  /// (IMSRG_ASSERT). O(1).
  ChannelIndex channel_of(OrbitIndex p, OrbitIndex q, TwoJ two_j) const;

  /// Global kets (p <= q) of channel `ch` in local order (ascending).
  std::span<const KetIndex> channel_kets(ChannelIndex ch) const;
  /// Local index of global ket `k` in channel `ch`, or nullopt if the ket does not belong to the
  /// channel (parity, |Tz difference| or triangle rule fails). O(1).
  std::optional<LocalKetIndex> local_index(ChannelIndex ch, KetIndex k) const;
  /// Local index of global ket `k` in the channel with 2J = `two_j` that its orbits imply (parity
  /// and Tz are fixed by the ket), or nullopt if J is outside |j_p - j_q|..j_p + j_q. O(1).
  std::optional<LocalKetIndex> local_index(KetIndex k, TwoJ two_j) const;

  /// Index of the ORDERED pair (p, q) in the 2 N-dimensional cross-coupled matrix of channel
  /// `ch` (N = channel(ch).num_kets): local(|pq>) if p <= q, local(|qp>) + N if p > q; nullopt
  /// if the (sorted) ket is absent from the channel. Deviation from imsrg++ (design/30 section
  /// 6): `GetLocalIndex` returns NumberKets - 1 for an absent swapped ket instead of reporting
  /// absence. Precondition: valid orbit indices and `ch` (IMSRG_ASSERT). O(1).
  std::optional<std::int32_t> pair_index(ChannelIndex ch, OrbitIndex p, OrbitIndex q) const;

  /// Non-empty channels by descending number of kets, ties by ascending index.
  std::span<const ChannelIndex> channels_by_size() const noexcept { return channels_by_size_; }

  /// Flat table: kets of all channels concatenated in channel order (channel `ch` occupies
  /// [ket_offset, ket_offset + num_kets)). For device mirrors.
  std::span<const KetIndex> channel_ket_table() const noexcept { return channel_ket_table_; }
  /// Flat table: for ket k, entries [ket_J_offsets()[k], ket_J_offsets()[k+1]) of
  /// `local_index_table()` belong to J = |j_p - j_q|, ..., j_p + j_q. Size num_kets() + 1.
  std::span<const std::int32_t> ket_J_offsets() const noexcept { return ket_J_offsets_; }
  /// Flat table: local index of (ket, J) in its channel (never negative here). For device mirrors.
  std::span<const std::int32_t> local_index_table() const noexcept { return local_index_table_; }

 private:
  struct KetInfo {
    std::int32_t jmin, jmax;  // integer J range
    Parity parity;
    std::int32_t tz_diff;  // |tz2_p - tz2_q| / 2 in {0, 1}
  };
  std::int32_t num_orbits_ = 0;
  std::int32_t num_kets_ = 0;
  std::int32_t max_J_ = 0;
  std::vector<KetInfo> info_;
  std::vector<TwoBodyChannel> channels_;
  std::vector<KetIndex> channel_ket_table_;
  std::vector<std::int32_t> ket_J_offsets_;
  std::vector<std::int32_t> local_index_table_;
  std::vector<ChannelIndex> channels_by_size_;
  std::vector<std::int32_t> orbit_tz2_;
  std::vector<std::int32_t> orbit_parity_;
  std::vector<std::int32_t> orbit_j2_;
};

}  // namespace imsrg::basis
