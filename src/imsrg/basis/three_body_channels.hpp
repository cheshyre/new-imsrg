#pragma once
/// @file
/// Three-body kets and channels (design/reference/01 section 3, imsrg++ `ModelSpace::Setup3bKets`
/// and `ThreeBodyChannel`). Only the structure lives here; the storage of three-body matrix
/// elements is built on top of it by the operator module.
///
/// Conventions (identical to imsrg++, design/00 D11):
/// * global kets |pqr; Jpq> have orbit indices p <= q <= r (all with shell e <= emax_3body and
///   2 (n_p + n_q + n_r) + l_p + l_q + l_r <= e3max); Jpq runs from |j_p - j_q| to j_p + j_q, with
///   p == q and odd Jpq dropped and p == q == r with 2j = 1 dropped. The order is the loop order
///   (p, q, r, Jpq), i.e. lexicographic in (p, q, r, Jpq);
/// * channels (2J, parity, 2Tz) cover 2J = 1, 3, ..., 6 emax_3body + 1, parity 0, 1 and
///   2Tz = -3, -1, 1, 3, in `std::set` order, so the index is
///   ((2J - 1) / 2) * 8 + parity * 4 + (2Tz + 3) / 2;
/// * a ket belongs to a channel if it passes `ThreeBodyChannel::CheckChannel_ket` (parity of
///   l_p + l_q + l_r, tz2 sum, the triangle rule between Jpq and j_r, and the stretched-state
///   rules, including those imsrg++ marks "TODO put these back in", which are redundant but are
///   kept so that the membership is identical); members are in ascending global ket order.

#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis {

/// Index of a three-body ket inside one three-body channel.
using LocalKet3Index = StrongIndex<struct LocalKet3Tag>;

/// Truncation of the three-body space.
struct ThreeBodyCuts {
  std::int32_t e3max = 12;       ///< Largest 2 (n_p + n_q + n_r) + l_p + l_q + l_r of a ket.
  std::int32_t emax_3body = -1;  ///< Largest single-particle shell e; -1 means the basis emax.
};

/// A three-body ket |pqr; Jpq> with p <= q <= r (orbit indices); Jpq couples p and q.
struct Ket3 {
  OrbitIndex p;  ///< First orbit.
  OrbitIndex q;  ///< Second orbit (q >= p).
  OrbitIndex r;  ///< Third orbit (r >= q).
  TwoJ two_jpq;  ///< 2 Jpq (Jpq is an integer, so this is even).

  /// Field-wise equality.
  friend IMSRG_INLINE_FUNCTION constexpr bool operator==(const Ket3& a, const Ket3& b) noexcept {
    return a.p == b.p && a.q == b.q && a.r == b.r && a.two_jpq == b.two_jpq;
  }
};
static_assert(std::is_trivially_copyable_v<Ket3>);

/// A three-body channel (2J, parity, 2Tz).
struct ThreeBodyChannel {
  TwoJ two_j{1};                 ///< 2J (odd).
  Parity parity = Parity::Even;  ///< (-1)^(l_p + l_q + l_r).
  TwoTz two_tz{-3};              ///< 2Tz = tz2_p + tz2_q + tz2_r in {-3, -1, 1, 3}.
  std::int64_t num_kets = 0;     ///< Number of kets (the block dimension).
  std::int64_t ket_offset = 0;   ///< Position of local ket 0 in `channel_ket_table()`.
};
static_assert(std::is_trivially_copyable_v<ThreeBodyChannel>);

/// Immutable three-body ket and channel structure of a single-particle basis. Value type.
class ThreeBodyChannels {
 public:
  /// Builds the kets and channels. `cuts.emax_3body` < 0 selects `basis.emax()`; it must not
  /// exceed `basis.emax()` (IMSRG_CHECK) and `cuts.e3max` must be >= 0. Cost O(N_orbits^3 +
  /// sum over kets of (2 j_r + 1)).
  explicit ThreeBodyChannels(const SingleParticleBasis& basis, const ThreeBodyCuts& cuts = {});

  /// The cuts used, with `emax_3body` resolved (never negative).
  const ThreeBodyCuts& cuts() const noexcept { return cuts_; }

  /// Number of global kets.
  std::int64_t num_kets() const noexcept { return static_cast<std::int64_t>(kets_.size()); }
  /// All global kets in imsrg++ order.
  std::span<const Ket3> kets() const noexcept { return kets_; }
  /// Global ket `k`. Precondition: 0 <= k < num_kets() (IMSRG_ASSERT).
  const Ket3& ket(Ket3Index k) const;
  /// Index of the ket |pqr; Jpq>, or nullopt if it is not in the set (also if the orbits are
  /// not ordered p <= q <= r). O(log num_kets()).
  std::optional<Ket3Index> find_ket(OrbitIndex p, OrbitIndex q, OrbitIndex r,
                                    TwoJ two_jpq) const noexcept;

  /// Number of channels, 8 * (3 emax_3body + 1).
  std::int32_t num_channels() const noexcept { return static_cast<std::int32_t>(channels_.size()); }
  /// All channels in index order.
  std::span<const ThreeBodyChannel> channels() const noexcept { return channels_; }
  /// Channel `ch`. Precondition: valid index (IMSRG_ASSERT).
  const ThreeBodyChannel& channel(Channel3Index ch) const;

  /// Channel index of (2J, parity, 2Tz) by the imsrg++ formula, without range checks. O(1).
  static IMSRG_INLINE_FUNCTION constexpr Channel3Index channel_index(TwoJ two_j, Parity parity,
                                                                     TwoTz two_tz) noexcept {
    return Channel3Index{((two_j.twice - 1) / 2) * 8 + static_cast<std::int32_t>(parity) * 4 +
                         (two_tz.twice + 3) / 2};
  }
  /// Index of the channel (2J, parity, 2Tz) or nullopt if 2J is not an odd number in
  /// 1..6 emax_3body + 1 or 2Tz is not in {-3, -1, 1, 3}. O(1).
  std::optional<Channel3Index> find_channel(TwoJ two_j, Parity parity, TwoTz two_tz) const noexcept;

  /// Global kets of channel `ch` in ascending order (the local order).
  std::span<const Ket3Index> channel_kets(Channel3Index ch) const;
  /// Local index of global ket `k` in channel `ch`, or nullopt if the ket is not a member. O(1)
  /// from a compact per-ket table (the J range of a ket has at most 2 j_r + 1 values).
  std::optional<LocalKet3Index> local_index(Channel3Index ch, Ket3Index k) const;
  /// Sum over channels of the number of kets (a ket belongs to up to j_r + 1 channels).
  std::int64_t sum_channel_kets() const noexcept {
    return static_cast<std::int64_t>(channel_ket_table_.size());
  }

  /// Configuration cut of imsrg++ (`dE3max`, `occnat3cut`) for ket `k`: true if
  /// sum_i |e_i - e_fermi(tz_i)| <= `de3max` and prod_i n_i (1 - n_i) >= `occnat3_cut`, over the
  /// three orbits. n_i is `natural_occupations[i]` if that span is non-empty (one value per
  /// orbit, index order), else the occupation of `ref`. Orbit shells are e = 2n + l and the
  /// Fermi levels are `ref.e_fermi`. Precondition: `ref` belongs to `basis` (same orbit
  /// count; IMSRG_ASSERT). O(1).
  bool passes_configuration_cut(Ket3Index k, const SingleParticleBasis& basis, const Reference& ref,
                                double de3max, double occnat3_cut,
                                std::span<const double> natural_occupations = {}) const;

 private:
  ThreeBodyCuts cuts_;
  std::vector<Ket3> kets_;
  std::vector<ThreeBodyChannel> channels_;
  std::vector<Ket3Index> channel_ket_table_;
  // For ket k, entries [ket_twoJ_offsets_[k], ket_twoJ_offsets_[k+1]) of local_index_table_ belong
  // to 2J = |2 Jpq - j2_r|, ..., 2 Jpq + j2_r (step 2); -1 marks "not a member".
  std::vector<std::int64_t> ket_twoJ_offsets_;
  std::vector<std::int32_t> local_index_table_;
  std::vector<std::int32_t> ket_twoJ_min_;
  std::vector<std::int32_t> ket_channel_base_;  // parity*4 + (2Tz+3)/2 of each ket
};

}  // namespace imsrg::basis
