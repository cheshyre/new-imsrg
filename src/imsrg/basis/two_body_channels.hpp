#pragma once
/// @file
/// Two-body kets, J-coupled two-body channels (J, parity, Tz) and their per-channel ket subsets
/// (design/10 section 3.2, design/13 section 1, design/reference/01 sections 2.1-2.4).
///
/// Conventions (identical to imsrg++, design/00 D11):
/// * a ket |pq> has p <= q (orbit indices); its global index is imsrg++'s `Index2(p, q) =
///   p (2N - 1 - p) / 2 + q` (N = number of orbits), i.e. lexicographic order, p == q included;
///   every pair p <= q is a ket (no E2max cut);
/// * a channel has J (integer), parity (-1)^(l_p + l_q) and Tz = (tz2_p + tz2_q) / 2 in
///   {-1, 0, +1} (pp, pn, nn); its index is `6 J + 2 (Tz + 1) + parity`, and all
///   6 (J_max + 1) channels exist, including empty ones (J_max = the basis' max_two_body_J());
/// * |pq> belongs to channel (J, P, Tz) if parity and Tz match, |j_p - j_q| <= J <= j_p + j_q,
///   and J is even when p == q (Pauli); inside a channel the kets are in ascending global order
///   (the local index);
/// * the exchange phase is |qp; J> = (-1)^(j_p + j_q - J + 1) |pq; J> (`exchange_phase`).

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis {

/// A two-body ket |pq> with p <= q.
struct Ket {
  OrbitIndex p;  ///< First orbit (p <= q).
  OrbitIndex q;  ///< Second orbit.

  /// Field-wise equality.
  friend IMSRG_INLINE_FUNCTION constexpr bool operator==(const Ket& a, const Ket& b) noexcept {
    return a.p == b.p && a.q == b.q;
  }
};
static_assert(std::is_trivially_copyable_v<Ket>);

/// imsrg++ `ModelSpace::Index2` (ModelSpace.cc:1166-1176): global index of the ket |pq> in a basis
/// of `num_orbits` orbits, p (2N - 1 - p) / 2 + q. Precondition: 0 <= p <= q < num_orbits
/// (not checked; p > q gives a wrong index). O(1).
IMSRG_INLINE_FUNCTION constexpr KetIndex imsrgpp_ket_index(OrbitIndex p, OrbitIndex q,
                                                           std::int32_t num_orbits) noexcept {
  return KetIndex{p.value() * (2 * num_orbits - 1 - p.value()) / 2 + q.value()};
}

/// imsrg++ two-body channel index (`GetTwoBodyChannelIndex`, ModelSpace.cc:1104-1112):
/// 6 J + 2 (Tz + 1) + parity, for `two_j` = 2J and `two_tz` = 2Tz = tz2_p + tz2_q. O(1).
IMSRG_INLINE_FUNCTION constexpr ChannelIndex imsrgpp_channel_index(TwoJ two_j, Parity parity,
                                                                   TwoTz two_tz) noexcept {
  return ChannelIndex{3 * two_j.twice + two_tz.twice + 2 + static_cast<std::int32_t>(parity)};
}

/// Exchange phase of imsrg++ (`Ket::Phase`, ModelSpace.hh:77-80):
/// |qp; J> = phase |pq; J> with phase = (-1)^((2j_p + 2j_q)/2 + 1 + J) = -(-1)^(j_p + j_q - J).
/// `two_j` is 2J of the pair. O(1).
IMSRG_INLINE_FUNCTION constexpr std::int32_t exchange_phase(TwoJ jp, TwoJ jq, TwoJ two_j) noexcept {
  return (((jp.twice + jq.twice) / 2 + 1 + two_j.twice / 2) % 2 == 0) ? 1 : -1;
}

/// Quantum numbers and extent of one two-body channel. Trivially copyable.
struct TwoBodyChannel {
  TwoJ two_j{0};                 ///< 2J (J is an integer).
  Parity parity = Parity::Even;  ///< (-1)^(l_p + l_q).
  TwoTz two_tz{0};               ///< 2Tz = tz2_p + tz2_q: -2 (pp), 0 (pn), +2 (nn).
  std::int32_t num_kets = 0;     ///< Number of kets (the block dimension).
  std::int32_t ket_offset =
      0;  ///< Position of local ket 0 in `TwoBodyChannels::channel_ket_table()`.

  /// J as an integer.
  IMSRG_INLINE_FUNCTION constexpr std::int32_t J() const noexcept { return two_j.twice / 2; }
  /// Tz as an integer (-1, 0, +1).
  IMSRG_INLINE_FUNCTION constexpr std::int32_t Tz() const noexcept { return two_tz.twice / 2; }
};
static_assert(std::is_trivially_copyable_v<TwoBodyChannel>);

/// Immutable two-body ket and channel structure of a single-particle basis. It depends only on
/// the orbits (not on occupations or the partition), so model spaces that differ only in their
/// reference share it. Build cost O(sum over kets of (j_p + j_q - |j_p - j_q| + 1)).
class TwoBodyChannels {
 public:
  /// Builds the kets and channels of `basis` (imsrg++ `SetupKets` and `TwoBodyChannel::
  /// Initialize`, ModelSpace.cc:1178-1267, TwoBodyChannel.cc:19-152).
  explicit TwoBodyChannels(const SingleParticleBasis& basis);

  /// Number of orbits of the basis.
  std::int32_t num_orbits() const noexcept { return static_cast<std::int32_t>(orbits_.size()); }
  /// The orbits of the basis (copied, so the structure is self-contained).
  std::span<const Orbit> orbits() const noexcept { return orbits_; }

  /// Number of kets, N (N + 1) / 2.
  std::int32_t num_kets() const noexcept { return static_cast<std::int32_t>(kets_.size()); }
  /// All kets in global index order.
  std::span<const Ket> kets() const noexcept { return kets_; }
  /// The ket with global index `k`. Precondition: 0 <= k < num_kets() (IMSRG_ASSERT).
  const Ket& ket(KetIndex k) const;
  /// Global index of |pq>. Precondition: 0 <= p <= q < num_orbits() (IMSRG_ASSERT).
  KetIndex ket_index(OrbitIndex p, OrbitIndex q) const;

  /// Largest J of any channel (the basis' max_two_body_J()).
  std::int32_t max_J() const noexcept { return max_J_; }
  /// Number of channels, 6 (max_J() + 1), including empty channels.
  std::int32_t num_channels() const noexcept { return static_cast<std::int32_t>(channels_.size()); }
  /// All channels in index order.
  std::span<const TwoBodyChannel> channels() const noexcept { return channels_; }
  /// Channel `ch`. Precondition: 0 <= ch < num_channels() (IMSRG_ASSERT).
  const TwoBodyChannel& channel(ChannelIndex ch) const;
  /// Index of the channel (2J, parity, 2Tz), or nullopt if J > max_J(), J < 0, or 2J or 2Tz is
  /// odd or out of range.
  std::optional<ChannelIndex> find_channel(TwoJ two_j, Parity parity, TwoTz two_tz) const noexcept;
  /// Channel of the pair (p, q) (any order) coupled to 2J = `two_j`; the pair need not be allowed
  /// in it (use `local_index` for that). Precondition: valid orbits and 0 <= J <= max_J().
  ChannelIndex channel_of(OrbitIndex p, OrbitIndex q, TwoJ two_j) const;

  /// Global indices of the kets of channel `ch` in local order (ascending).
  std::span<const KetIndex> channel_kets(ChannelIndex ch) const;
  /// Global index of local ket `i` of channel `ch`. Precondition: i < channel(ch).num_kets.
  KetIndex global_ket(ChannelIndex ch, LocalKetIndex i) const;
  /// Local index of global ket `k` in channel `ch`, or nullopt if the ket is not in the channel.
  /// O(1).
  std::optional<LocalKetIndex> local_index(ChannelIndex ch, KetIndex k) const;
  /// Local index of global ket `k` in the channel with 2J = `two_j` that its orbits imply, or
  /// nullopt if J is outside |j_p - j_q|..j_p + j_q or Pauli-forbidden (p == q, odd J). O(1).
  std::optional<LocalKetIndex> local_index(KetIndex k, TwoJ two_j) const;

  /// Non-empty channels by descending number of kets, ties by ascending index (imsrg++
  /// `SortedTwoBodyChannels`, which it uses for load balancing; its tie order is unspecified).
  std::span<const ChannelIndex> channels_by_size() const noexcept { return channels_by_size_; }

  /// Flat table: the kets of all channels concatenated in channel order (channel `ch` occupies
  /// [ket_offset, ket_offset + num_kets)). For device mirrors.
  std::span<const KetIndex> channel_ket_table() const noexcept { return channel_ket_table_; }
  /// Flat table: for ket k, the entries [ket_J_offsets()[k], ket_J_offsets()[k+1]) of
  /// `local_index_table()` belong to J = |j_p - j_q|, ..., j_p + j_q. Size num_kets() + 1.
  std::span<const std::int32_t> ket_J_offsets() const noexcept { return ket_J_offsets_; }
  /// Flat table: local index of (ket, J) in its channel, -1 if Pauli-forbidden. For device
  /// mirrors.
  std::span<const std::int32_t> local_index_table() const noexcept { return local_index_table_; }

  /// True if built from bases with identical orbits.
  friend bool operator==(const TwoBodyChannels& a, const TwoBodyChannels& b) noexcept {
    return a.orbits_ == b.orbits_;
  }

 private:
  std::vector<Orbit> orbits_;
  std::vector<Ket> kets_;
  std::int32_t max_J_ = 0;
  std::vector<TwoBodyChannel> channels_;
  std::vector<KetIndex> channel_ket_table_;
  std::vector<std::int32_t> ket_J_offsets_;
  std::vector<std::int32_t> local_index_table_;
  std::vector<ChannelIndex> channels_by_size_;
};

/// Classes of two-body kets used to select blocks in the commutators and generators: by
/// occupation (pp, hh, ph) and by partition class (cc, vc, qc, vv, qv, qq). The enumerator order
/// is the order of the golden `channel_subset` records.
enum class KetClass : std::uint8_t { PP = 0, HH, PH, CC, VC, QC, VV, QV, QQ };

/// Number of KetClass values.
inline constexpr std::int32_t kNumKetClasses = 9;

/// All KetClass values in enumerator order.
inline constexpr std::array<KetClass, kNumKetClasses> kAllKetClasses = {
    KetClass::PP, KetClass::HH, KetClass::PH, KetClass::CC, KetClass::VC,
    KetClass::QC, KetClass::VV, KetClass::QV, KetClass::QQ};

/// Lower-case name of a class: "pp", "hh", "ph", "cc", "vc", "qc", "vv", "qv", "qq".
std::string_view ket_class_name(KetClass c) noexcept;

/// Immutable per-channel ket subsets for given occupations and partition (imsrg++
/// `ModelSpace::SetupKets` and `TwoBodyChannel::Initialize`, ModelSpace.cc:1178-1234,
/// TwoBodyChannel.cc:19-66). A ket |pq> is in
/// * pp if both orbits are particles (n < kOccupationCut), hh if both are holes
///   (n >= kOccupationCut), ph if exactly one is a hole;
/// * cc, vc, qc, vv, qv, qq by the cvq values c_p, c_q (0 core, 1 valence, 2 q-space): cc if
///   c_p + c_q = 0, vc if = 1, qc if |c_p - c_q| = 2, vv if c_p c_q = 1, qv if c_p + c_q = 3,
///   qq if c_p + c_q = 4.
/// Each subset lists local ket indices of the channel in ascending order. For hh and ph the
/// products n_p n_q and (1 - n_p)(1 - n_q) are available aligned with the list (they carry
/// fractional occupations into the commutators).
///
/// Deviation from imsrg++ (design/30 section 6): imsrg++ tests hh/ph with n > OCC_CUT but
/// holes with n >= OCC_CUT, so an orbit with n == 1e-6 exactly is in no subset; here holes are
/// n >= kOccupationCut throughout.
class TwoBodyKetSubsets {
 public:
  /// Subsets of `channels` for per-orbit `occupations` and `classes` (each of size
  /// channels.num_orbits()).
  /// @throws InputError if a size differs from channels.num_orbits().
  TwoBodyKetSubsets(const TwoBodyChannels& channels, std::span<const double> occupations,
                    std::span<const OrbitClass> classes);

  /// Subsets for the occupations of `reference` and the classes of `partition`.
  TwoBodyKetSubsets(const TwoBodyChannels& channels, const Reference& reference,
                    const Partition& partition)
      : TwoBodyKetSubsets(channels, reference.occupations(), partition.classes()) {}

  /// Number of channels.
  std::int32_t num_channels() const noexcept { return num_channels_; }
  /// Local indices (ascending) of the kets of class `c` in channel `ch`.
  std::span<const LocalKetIndex> kets(ChannelIndex ch, KetClass c) const;
  /// n_p n_q for each ket of `kets(ch, c)`. Precondition: c is KetClass::HH or KetClass::PH
  /// (IMSRG_CHECK).
  std::span<const double> occupation_products(ChannelIndex ch, KetClass c) const;
  /// (1 - n_p)(1 - n_q) for each ket of `kets(ch, c)`. Precondition: c is HH or PH.
  std::span<const double> vacancy_products(ChannelIndex ch, KetClass c) const;

  /// Flat table: all subsets concatenated, subset (ch, c) at
  /// [subset_offsets()[ch * kNumKetClasses + c], subset_offsets()[ch * kNumKetClasses + c + 1]).
  /// For device mirrors.
  std::span<const LocalKetIndex> subset_table() const noexcept { return subset_table_; }
  /// Offsets into `subset_table()`, size num_channels() * kNumKetClasses + 1.
  std::span<const std::int32_t> subset_offsets() const noexcept { return subset_offsets_; }

 private:
  std::int32_t num_channels_ = 0;
  std::vector<LocalKetIndex> subset_table_;
  std::vector<std::int32_t> subset_offsets_;
  // Aligned with subset_table_ (only the HH and PH ranges are meaningful; others hold 0).
  std::vector<double> occupation_products_;
  std::vector<double> vacancy_products_;
};

}  // namespace imsrg::basis
