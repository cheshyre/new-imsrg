#pragma once
/// @file
/// Nucleus names ("O16", "vacuum", "mix_A4.5_Z2.5") and the naive shell-model filling of
/// imsrg++ (design/13 section 1, design/reference/01 section 1.5).

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/basis/orbit.hpp"

namespace imsrg::basis {

/// A nucleus, or an ensemble of nuclei, given by its mass number A and proton number Z. Both are
/// doubles so that the fractional mixtures of imsrg++ ("mix_A4.5_Z2.5") are representable.
struct Nucleus {
  double A = 0.0;  ///< Mass number (protons + neutrons), A >= Z >= 0.
  double Z = 0.0;  ///< Proton number.

  /// Neutron number A - Z.
  double N() const noexcept { return A - Z; }

  /// Parses an imsrg++ nucleus name (`ModelSpace::GetAZfromString`, ModelSpace.cc:568-605):
  /// an element symbol followed by A ("O16", "He4", "Ca48"; "n" is the neutron, Z = 0);
  /// "vacuum" (A = Z = 0); or "mix_A<A>_Z<Z>" with decimal A and Z ("mix_A4.5_Z2.5").
  /// @throws InputError (naming the input) for an unknown element, a missing or malformed A,
  ///         trailing characters, or A < Z.
  static Nucleus parse(std::string_view name);

  /// A canonical name: "vacuum" for A = Z = 0, element symbol + A for integer A and Z
  /// (e.g. "O16"), otherwise "mix_A<A>_Z<Z>" (shortest round-trip decimal form).
  std::string name() const;

  /// Field-wise equality.
  friend bool operator==(const Nucleus&, const Nucleus&) = default;
};

/// Element symbol of proton number Z, from "n" (Z = 0, the neutron) to "Og" (Z = 118), with the
/// spelling of the imsrg++ periodic table (ModelSpace.cc:570-576).
/// @throws InputError if Z is outside [0, 118].
std::string_view element_symbol(std::int32_t Z);

/// Proton number of an element symbol ("He" -> 2; case sensitive, "n" -> 0).
/// @throws InputError if the symbol is unknown.
std::int32_t element_proton_number(std::string_view symbol);

/// An orbit and its occupation, e.g. one entry of a filling or of an occupation file.
struct OrbitOccupation {
  Orbit orbit;            ///< Quantum numbers.
  double occupation = 0;  ///< Occupation n in (0, 1].

  /// Field-wise equality.
  friend bool operator==(const OrbitOccupation&, const OrbitOccupation&) = default;
};

/// The naive shell-model filling of imsrg++ (`ModelSpace::GetOrbitsAZ`, ModelSpace.cc:609-650).
///
/// Within each major shell N = 0, 1, ..., emax the orbits are filled in the order
/// g = 2N+1, 2N-3, ... (>= -2N), with 2j = |g| and l = (2j-1)/2 for g > 0, l = (2j+1)/2 for
/// g < 0 (N = 2: d5/2, s1/2, d3/2; N = 3: f7/2, p3/2, p1/2, f5/2). Protons and neutrons fill
/// independently; each orbit takes min(remaining nucleons, 2j+1) nucleons, so the last orbit of a
/// species gets the fractional occupation (remaining)/(2j+1) (equal filling of an open shell).
/// Orbits listed in `skip` are passed over (imsrg++ does this for the core of comma-separated
/// valence spaces). The arithmetic is the same as in imsrg++, so occupations agree bitwise.
///
/// Returns the occupied orbits in filling order (per orbit step: proton, then neutron).
/// @throws InputError if the nucleons do not fit into the shells 0..emax (imsrg++ prints
///         "Trouble!" and continues with a partial filling; we refuse).
std::vector<OrbitOccupation> naive_filling(const Nucleus& nucleus, std::int32_t emax,
                                           std::span<const Orbit> skip = {});

}  // namespace imsrg::basis
