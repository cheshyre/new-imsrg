#pragma once
/// @file
/// Device-copyable index tables of the basis for kernels (design/12 sections 2 and 5): trivially
/// copyable views with raw device pointers and `IMSRG_INLINE_FUNCTION` accessors, plus
/// `DeviceTables`, which owns the device buffers. No Kokkos include: kernel translation units
/// capture the views by value in `IMSRG_LAMBDA`s.
///
/// All accessors are O(1), allocation-free and have no checks (preconditions are documented);
/// they mirror the host semantics of `TwoBodyChannels` and `TwoBodyKetSubsets`.

#include <cstdint>
#include <type_traits>

#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/basis/two_body_channels.hpp"
#include "imsrg/exec/device_buffer.hpp"

namespace imsrg::basis {

/// Per-orbit quantum numbers, occupations and partition classes (struct of arrays), indexed by
/// the orbit index.
struct OrbitTableView {
  const std::int32_t* n = nullptr;     ///< Radial quantum number.
  const std::int32_t* l = nullptr;     ///< Orbital angular momentum.
  const std::int32_t* j2 = nullptr;    ///< 2j.
  const std::int32_t* tz2 = nullptr;   ///< 2tz (-1 proton, +1 neutron).
  const double* occupation = nullptr;  ///< Reference occupation.
  const std::int32_t* cvq = nullptr;   ///< imsrg++ cvq: 0 core, 1 valence, 2 q-space.
  std::int32_t num_orbits = 0;         ///< Number of orbits.
};
static_assert(std::is_trivially_copyable_v<OrbitTableView>);

/// Two-body kets, channels, and the compact (ket, J) -> local-index map.
struct TwoBodyTableView {
  const std::int32_t* ket_p = nullptr;  ///< First orbit of each ket (p <= q).
  const std::int32_t* ket_q = nullptr;  ///< Second orbit of each ket.
  std::int32_t num_kets = 0;            ///< Number of kets.
  std::int32_t num_orbits = 0;          ///< Number of orbits.

  const std::int32_t* channel_two_j = nullptr;       ///< 2J per channel.
  const std::int32_t* channel_parity = nullptr;      ///< 0 even, 1 odd, per channel.
  const std::int32_t* channel_two_tz = nullptr;      ///< 2Tz per channel.
  const std::int32_t* channel_num_kets = nullptr;    ///< Number of kets per channel.
  const std::int32_t* channel_ket_offset = nullptr;  ///< Offset into `channel_kets`.
  std::int32_t num_channels = 0;                     ///< Number of channels.
  const std::int32_t* channel_kets = nullptr;        ///< Flat global ket indices of all channels.

  const std::int32_t* ket_J_offsets = nullptr;      ///< Size num_kets + 1.
  const std::int32_t* local_index_table = nullptr;  ///< Local index per (ket, J), -1 if forbidden.

  const std::int32_t* orbit_j2 = nullptr;   ///< 2j per orbit.
  const std::int32_t* orbit_l = nullptr;    ///< l per orbit.
  const std::int32_t* orbit_tz2 = nullptr;  ///< 2tz per orbit.

  /// Global ket index of |pq> (imsrg++ Index2). Precondition: 0 <= p <= q < num_orbits.
  IMSRG_INLINE_FUNCTION KetIndex ket_index(OrbitIndex p, OrbitIndex q) const noexcept {
    return imsrgpp_ket_index(p, q, num_orbits);
  }

  /// Local index of global ket `k` at 2J = `two_j` in the channel its orbits imply; -1 if J is
  /// negative/odd/outside |j_p - j_q|..j_p + j_q or the ket is Pauli-forbidden. Same semantics as
  /// `TwoBodyChannels::local_index(KetIndex, TwoJ)`. Precondition: 0 <= k < num_kets.
  IMSRG_INLINE_FUNCTION std::int32_t local_index(KetIndex k, TwoJ two_j) const noexcept {
    if (two_j.twice < 0 || two_j.twice % 2 != 0) return -1;
    const std::int32_t jp = orbit_j2[ket_p[k.value()]];
    const std::int32_t jq = orbit_j2[ket_q[k.value()]];
    const std::int32_t d = jp > jq ? jp - jq : jq - jp;
    const std::int32_t jmin = d / 2;
    const std::int32_t jmax = (jp + jq) / 2;
    const std::int32_t J = two_j.twice / 2;
    if (J < jmin || J > jmax) return -1;
    return local_index_table[ket_J_offsets[k.value()] + (J - jmin)];
  }

  /// Channel of the pair (p, q) (any order) coupled to `two_j`. Preconditions: valid orbits,
  /// even 0 <= two_j <= 2 max_J.
  IMSRG_INLINE_FUNCTION ChannelIndex channel_of(OrbitIndex p, OrbitIndex q,
                                                TwoJ two_j) const noexcept {
    const std::int32_t parity = ((orbit_l[p.value()] + orbit_l[q.value()]) % 2 == 0) ? 0 : 1;
    return ChannelIndex{3 * two_j.twice + orbit_tz2[p.value()] + orbit_tz2[q.value()] + 2 + parity};
  }

  /// Global ket of local ket `i` of channel `ch`. Precondition: i < channel_num_kets[ch].
  IMSRG_INLINE_FUNCTION KetIndex global_ket(ChannelIndex ch, LocalKetIndex i) const noexcept {
    return KetIndex{channel_kets[channel_ket_offset[ch.value()] + i.value()]};
  }
};
static_assert(std::is_trivially_copyable_v<TwoBodyTableView>);

/// Per-channel ket subsets (flat tables of `TwoBodyKetSubsets`).
struct KetSubsetTableView {
  const std::int32_t* offsets = nullptr;        ///< Size num_channels * kNumKetClasses + 1.
  const std::int32_t* local_kets = nullptr;     ///< Local ket indices of all subsets.
  const double* occupation_products = nullptr;  ///< n_p n_q aligned with `local_kets` (HH, PH).
  const double* vacancy_products = nullptr;     ///< (1-n_p)(1-n_q) aligned with `local_kets`.
  std::int32_t num_channels = 0;                ///< Number of channels.

  /// Begin of subset (ch, c) in `local_kets` (and the product tables).
  IMSRG_INLINE_FUNCTION std::int32_t begin(ChannelIndex ch, KetClass c) const noexcept {
    return offsets[ch.value() * kNumKetClasses + static_cast<std::int32_t>(c)];
  }
  /// End (one past the last) of subset (ch, c).
  IMSRG_INLINE_FUNCTION std::int32_t end(ChannelIndex ch, KetClass c) const noexcept {
    return offsets[ch.value() * kNumKetClasses + static_cast<std::int32_t>(c) + 1];
  }
  /// Number of kets in subset (ch, c).
  IMSRG_INLINE_FUNCTION std::int32_t size(ChannelIndex ch, KetClass c) const noexcept {
    return end(ch, c) - begin(ch, c);
  }
};
static_assert(std::is_trivially_copyable_v<KetSubsetTableView>);

/// Owner of the device copies of the basis tables. Requires an initialized `exec::Runtime`
/// (DeviceBuffer enforces it). Movable, not copyable.
class DeviceTables {
 public:
  /// Copies the tables of the given host objects to the device (`copy_from_host`).
  /// Cost O(orbits + kets * J range + subset sizes). `channels` and `subsets` must have been
  /// built from `basis` and `reference`/`partition` (same orbit count).
  DeviceTables(const SingleParticleBasis& basis, const Reference& reference,
               const Partition& partition, const TwoBodyChannels& channels,
               const TwoBodyKetSubsets& subsets);

  DeviceTables(const DeviceTables&) = delete;
  DeviceTables& operator=(const DeviceTables&) = delete;
  DeviceTables(DeviceTables&&) noexcept = default;
  DeviceTables& operator=(DeviceTables&&) noexcept = default;
  ~DeviceTables() = default;

  /// Orbit table view; valid while this object lives (device pointers survive moves).
  const OrbitTableView& orbits() const noexcept { return orbits_view_; }
  /// Two-body table view; same validity as `orbits()`.
  const TwoBodyTableView& two_body() const noexcept { return two_body_view_; }
  /// Subset table view; same validity as `orbits()`.
  const KetSubsetTableView& subsets() const noexcept { return subsets_view_; }

 private:
  exec::DeviceBuffer<std::int32_t> orb_n_, orb_l_, orb_j2_, orb_tz2_, orb_cvq_;
  exec::DeviceBuffer<double> orb_occ_;
  exec::DeviceBuffer<std::int32_t> ket_p_, ket_q_;
  exec::DeviceBuffer<std::int32_t> ch_two_j_, ch_parity_, ch_two_tz_, ch_num_kets_, ch_offset_;
  exec::DeviceBuffer<std::int32_t> ch_kets_, ket_j_offsets_, local_index_;
  exec::DeviceBuffer<std::int32_t> sub_offsets_, sub_kets_;
  exec::DeviceBuffer<double> sub_occ_, sub_vac_;
  OrbitTableView orbits_view_;
  TwoBodyTableView two_body_view_;
  KetSubsetTableView subsets_view_;
};

}  // namespace imsrg::basis
