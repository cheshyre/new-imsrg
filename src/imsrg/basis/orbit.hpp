#pragma once
/// @file
/// Single-particle orbit quantum numbers and the imsrg++ orbit-ordering formula (design/00 D11,
/// design/reference/01 sections 1.1-1.2).
///
/// Conventions (identical to imsrg++, so that results compare element by element):
/// * an orbit is |n l j tz> of the spherical harmonic oscillator; e = 2n + l is its shell;
/// * `tz2 = -1` is a proton, `tz2 = +1` a neutron;
/// * orbit order: shell e ascending, then 2j descending (which is l descending, then j
///   descending), then proton before neutron. In an untruncated basis the index of an orbit is
///   `imsrgpp_orbit_hash(o)`: protons have even indices and the neutron partner of proton `a` is
///   `a + 1`;
/// * labels are written as imsrg++ writes them, e.g. "p0d5" for the proton 0d5/2 orbit, with the
///   l letters s p d f g h i j k l m n o for l = 0..12 (note that imsrg++ uses `j` for l = 7,
///   unlike standard spectroscopic notation, e.g. "n0j15").

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"

namespace imsrg::basis {

/// Occupation threshold of imsrg++ (`ModelSpace::OCC_CUT`): an orbit with occupation
/// n >= kOccupationCut is a hole, otherwise a particle.
inline constexpr double kOccupationCut = 1e-6;

/// Quantum numbers of one spherical harmonic-oscillator orbit |n l j tz>. Trivially copyable,
/// usable in device code. A default-constructed orbit is the proton 0s1/2.
struct Orbit {
  std::int32_t n = 0;  ///< Radial quantum number (number of nodes), n >= 0.
  std::int32_t l = 0;  ///< Orbital angular momentum, l >= 0.
  TwoJ j2{1};          ///< Twice the total angular momentum, 2j = 2l + 1 or 2l - 1 (> 0).
  TwoTz tz2{-1};       ///< Twice the isospin projection: -1 proton, +1 neutron.

  /// Harmonic-oscillator shell e = 2n + l.
  IMSRG_INLINE_FUNCTION constexpr std::int32_t e() const noexcept { return 2 * n + l; }
  /// Parity (-1)^l.
  IMSRG_INLINE_FUNCTION constexpr Parity parity() const noexcept { return parity_of_l(l); }
  /// True for a proton orbit (tz2 < 0).
  IMSRG_INLINE_FUNCTION constexpr bool is_proton() const noexcept { return tz2.twice < 0; }
  /// True for a neutron orbit (tz2 > 0).
  IMSRG_INLINE_FUNCTION constexpr bool is_neutron() const noexcept { return tz2.twice > 0; }

  /// True if all four quantum numbers agree.
  friend IMSRG_INLINE_FUNCTION constexpr bool operator==(const Orbit& a, const Orbit& b) noexcept {
    return a.n == b.n && a.l == b.l && a.j2 == b.j2 && a.tz2 == b.tz2;
  }
  /// Negation of `==`.
  friend IMSRG_INLINE_FUNCTION constexpr bool operator!=(const Orbit& a, const Orbit& b) noexcept {
    return !(a == b);
  }
};

static_assert(std::is_trivially_copyable_v<Orbit> && std::is_standard_layout_v<Orbit>);
static_assert(sizeof(Orbit) == 4 * sizeof(std::int32_t));

/// True if `o` has valid quantum numbers: n >= 0, l >= 0, 2j = 2l +- 1 with 2j > 0, tz2 = +-1.
IMSRG_INLINE_FUNCTION constexpr bool is_valid(const Orbit& o) noexcept {
  return o.n >= 0 && o.l >= 0 && (o.j2.twice == 2 * o.l + 1 || o.j2.twice == 2 * o.l - 1) &&
         o.j2.twice > 0 && (o.tz2.twice == -1 || o.tz2.twice == 1);
}

/// imsrg++ orbit hash (`ModelSpace::Index1_hash`, ModelSpace.cc:1154-1164):
/// `e (e + 3) + 1 - 2j + (tz2 + 1) / 2` with e = 2n + l. It orders orbits by shell, then 2j
/// descending, then proton before neutron, and equals the orbit index in a basis without l or
/// unoccupied-shell truncations (so the index of an orbit does not depend on emax).
/// Precondition: `is_valid(o)`. O(1).
IMSRG_INLINE_FUNCTION constexpr std::int32_t imsrgpp_orbit_hash(const Orbit& o) noexcept {
  const std::int32_t e = o.e();
  return e * (e + 3) + 1 - o.j2.twice + (o.tz2.twice + 1) / 2;
}

/// Number of orbits with shell e <= emax in an untruncated two-species basis:
/// (emax + 1)(emax + 2). Precondition: emax >= -1.
IMSRG_INLINE_FUNCTION constexpr std::int32_t num_orbits_untruncated(std::int32_t emax) noexcept {
  return (emax + 1) * (emax + 2);
}

/// The imsrg++ label of `o`, e.g. "p0d5" (proton 0d5/2) or "n1s1" (neutron 1s1/2): 'p' or 'n',
/// n, the l letter (s p d f g h i j k l m n o for l = 0..12), then 2j.
/// @throws InputError if `o` is not valid or l > 12.
std::string orbit_label(const Orbit& o);

/// Parses an imsrg++ orbit label ("p0d5", "n1s1", "n0j15"): 'p' or 'n', the decimal n (several
/// digits are accepted; imsrg++ reads only one), an l letter from "spdfghijklmno" (j is l = 7, as
/// in imsrg++), then the decimal 2j.
/// @throws InputError (naming the label) if the label is malformed or the quantum numbers are not
///         valid (`is_valid`).
Orbit parse_orbit_label(std::string_view label);

}  // namespace imsrg::basis
