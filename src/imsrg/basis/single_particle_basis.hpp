#pragma once
/// @file
/// The harmonic-oscillator single-particle basis: the orbit list in imsrg++ order and its
/// truncations (design/10 section 3.2, design/13 section 1, design/reference/01 sections 1.2-1.3).

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/orbit.hpp"

namespace imsrg::basis {

/// Truncation of the harmonic-oscillator basis, with the meaning of the imsrg++ parameters of the
/// same names. There is no E2max cut on two-body kets (E2max only matters to file readers).
struct BasisTruncation {
  /// Keep orbits with shell e = 2n + l <= emax (emax >= 0).
  std::int32_t emax = 0;
  /// Keep orbits with l <= lmax; a negative value means no cut (imsrg++ default lmax = 99999).
  std::int32_t lmax = -1;
  /// Unoccupied-shell cut (imsrg++ `emax_unocc`): when 0 <= emax_unocc < emax, an orbit with
  /// e > emax_unocc is dropped unless l <= lmax_unocc. A negative value means no cut.
  std::int32_t emax_unocc = -1;
  /// The "hole-like" l range kept above emax_unocc. imsrg++ (`ModelSpace::SetEmaxUnocc`,
  /// ModelSpace.cc:1420-1459) uses min(emax, (largest l of the reference holes) + 2); the caller
  /// computes it. Required (>= 0) when the emax_unocc cut is active.
  std::int32_t lmax_unocc = -1;

  /// Field-wise equality.
  friend bool operator==(const BasisTruncation&, const BasisTruncation&) = default;
};

/// Immutable list of harmonic-oscillator orbits in imsrg++ order (design/00 D11).
///
/// The orbits are generated as in `ModelSpace::Init` (ModelSpace.cc:381-408): shell
/// N = 0..emax; l = N, N-2, ..., 0; 2j = 2l+1, then 2l-1; tz2 = -1 (proton), then +1 (neutron);
/// truncated orbits are skipped, and the orbit index is the position in this sequence. Without
/// truncations other than emax the index equals `imsrgpp_orbit_hash(orbit)`. Protons and
/// neutrons are always truncated alike, so proton indices are even and the neutron partner of
/// proton `a` is `a + 1` in every basis. Single-species ("atomic") bases are not supported.
///
/// Value type: copies are independent and compare equal.
class SingleParticleBasis {
 public:
  /// The harmonic-oscillator basis with truncation `trunc` and oscillator frequency `hw_MeV`
  /// (hbar*omega in MeV; it does not change the orbit list). O(number of orbits).
  /// @throws InputError if emax < 0, hw_MeV <= 0 (or not finite), or the emax_unocc cut is
  ///         active without lmax_unocc >= 0.
  static SingleParticleBasis harmonic_oscillator(const BasisTruncation& trunc, double hw_MeV);

  /// All orbits, index order.
  std::span<const Orbit> orbits() const noexcept { return orbits_; }
  /// Number of orbits.
  std::int32_t num_orbits() const noexcept { return static_cast<std::int32_t>(orbits_.size()); }
  /// The orbit with index `i`. Precondition: 0 <= i < num_orbits() (IMSRG_ASSERT).
  const Orbit& orbit(OrbitIndex i) const;

  /// Index of the orbit with the quantum numbers of `o`, or nullopt if it is not in the basis
  /// (truncated away or invalid). O(1).
  std::optional<OrbitIndex> find(const Orbit& o) const noexcept;
  /// Index of the orbit with the quantum numbers of `o`.
  /// @throws InputError (with the orbit label) if it is not in the basis.
  OrbitIndex index_of(const Orbit& o) const;
  /// True if the orbit `o` is in the basis.
  bool contains(const Orbit& o) const noexcept { return find(o).has_value(); }

  /// The truncation this basis was built with (as given).
  const BasisTruncation& truncation() const noexcept { return truncation_; }
  /// Largest shell e of the basis (the emax parameter).
  std::int32_t emax() const noexcept { return truncation_.emax; }
  /// Effective l cut, min(lmax, emax) (emax when there is no l cut).
  std::int32_t lmax() const noexcept;
  /// Oscillator frequency hbar*omega in MeV.
  double hbar_omega() const noexcept { return hbar_omega_; }

  /// Largest 2j of any orbit (imsrg++ `OneBodyJmax`); TwoJ{0} for an empty basis.
  TwoJ max_j2() const noexcept { return max_j2_; }
  /// Largest two-body J (an integer, J = (2j_max + 2j_max) / 2 = max_j2().twice), which sets the
  /// number of two-body channels 6 (J_max + 1) (imsrg++ `TwoBodyJmax`).
  std::int32_t max_two_body_J() const noexcept { return max_j2_.twice; }

  /// Indices of the orbits with the same (l, 2j, tz2) as orbit `i`, ascending (imsrg++
  /// `OneBodyChannels`): the orbits a scalar, isospin-conserving one-body operator connects to
  /// `i`. O(num_orbits()).
  std::vector<OrbitIndex> same_ljtz(OrbitIndex i) const;

  /// Equality of emax, hbar*omega and the orbit list (the truncation parameters may differ when
  /// they produce the same orbits).
  friend bool operator==(const SingleParticleBasis& a, const SingleParticleBasis& b) noexcept;

 private:
  SingleParticleBasis() = default;

  BasisTruncation truncation_;
  double hbar_omega_ = 0.0;
  std::vector<Orbit> orbits_;
  std::vector<std::int32_t> index_by_hash_;  // imsrgpp_orbit_hash -> index, -1 if absent
  TwoJ max_j2_{0};
};

}  // namespace imsrg::basis
