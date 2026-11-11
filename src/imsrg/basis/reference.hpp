#pragma once
/// @file
/// The reference state: occupations of the orbits of a single-particle basis (design/10
/// section 3.2, design/13 section 1, design/reference/01 sections 1.4-1.5).

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis {

/// Immutable occupation numbers n_a in [0, 1] of every orbit of a `SingleParticleBasis` (the
/// orbit with index a has occupation `occupations()[a]`). Fractional occupations describe
/// ensemble (equal-filling) references of open-shell nuclei.
///
/// Derived quantities follow imsrg++ exactly:
/// * holes: n_a >= kOccupationCut (1e-6); particles: n_a < kOccupationCut (only completely empty
///   orbits are particles; a partially filled orbit is a hole);
/// * A = sum over holes of (2j+1) n_a, Z the same over proton holes (`GetAref`/`GetZref`);
///   `target_mass()` and `target_Z()` truncate them to int (imsrg++ `SetTargetMass(Aref)`);
/// * the Fermi level `e_fermi(species)` of `ModelSpace::FindEFermi` (ModelSpace.cc:1052-1087).
///
/// A Reference stores no pointer to its basis; `ModelSpace` checks that the sizes agree. Value
/// type: copies are independent.
class Reference {
 public:
  /// The naive shell-model filling of `nucleus` (`naive_filling(nucleus, basis.emax())`), as
  /// imsrg++ builds the reference from a nucleus name (ModelSpace.cc:213-221). `name()` is
  /// `nucleus.name()`.
  /// @throws InputError if the nucleus does not fit into shells e <= emax, or an occupied orbit
  ///         is not in the basis (e.g. removed by an lmax cut; imsrg++ exits in `SetLmax`).
  static Reference from_nucleus(const SingleParticleBasis& basis, const Nucleus& nucleus);

  /// Explicit occupations, one per orbit of `basis` in index order.
  /// @throws InputError if the size differs from basis.num_orbits() or a value is outside
  ///         [0, 1] (or not finite).
  static Reference from_occupations(const SingleParticleBasis& basis,
                                    std::vector<double> occupations, std::string name = "custom");

  /// Occupations of listed orbits, all others empty: the semantics of an imsrg++ occupation file
  /// (`Init_occ_from_file`, ModelSpace.cc:319-355; lines "n l 2j 2tz occ"). Entries with
  /// |occupation| < 1e-8 are ignored, as in imsrg++; a later entry for the same orbit overrides
  /// an earlier one.
  /// @throws InputError if a listed orbit is not in the basis or an occupation is outside [0, 1].
  static Reference from_orbit_occupations(const SingleParticleBasis& basis,
                                          std::span<const OrbitOccupation> entries,
                                          std::string name = "custom");

  /// Number of orbits (equals the basis size).
  std::int32_t num_orbits() const noexcept { return static_cast<std::int32_t>(occ_.size()); }
  /// Occupation of every orbit, index order.
  std::span<const double> occupations() const noexcept { return occ_; }
  /// Occupation of orbit `a`. Precondition: 0 <= a < num_orbits() (IMSRG_ASSERT).
  double occupation(OrbitIndex a) const;
  /// True if orbit `a` is a hole (occupation >= kOccupationCut).
  bool is_hole(OrbitIndex a) const { return occupation(a) >= kOccupationCut; }

  /// Hole orbits (occupation >= kOccupationCut), ascending (imsrg++ `holes`).
  std::span<const OrbitIndex> holes() const noexcept { return holes_; }
  /// Particle orbits (occupation < kOccupationCut), ascending (imsrg++ `particles`).
  std::span<const OrbitIndex> particles() const noexcept { return particles_; }

  /// Mass number of the reference, sum over holes of (2j+1) n (imsrg++ `GetAref`).
  double A() const noexcept { return A_; }
  /// Proton number of the reference (imsrg++ `GetZref`).
  double Z() const noexcept { return Z_; }
  /// Neutron number A - Z, summed over neutron holes.
  double N() const noexcept { return N_; }
  /// `static_cast<int>(A())` (imsrg++ sets `target_mass` from Aref by truncation).
  std::int32_t target_mass() const noexcept { return static_cast<std::int32_t>(A_); }
  /// `static_cast<int>(Z())`.
  std::int32_t target_Z() const noexcept { return static_cast<std::int32_t>(Z_); }

  /// Fermi level of a species in oscillator-shell units (imsrg++ `e_fermi[tz2]`): the largest
  /// shell e = 2n + l among the holes of that species (-1 if there are none); if the first hole
  /// orbit (lowest index) with that e has occupation > 0.99, the level is moved halfway to the
  /// lowest particle shell of the species (to itself if there are no particles). Used by the
  /// dE3max cut on three-body configurations.
  /// @throws InputError if `species` is not TwoTz{-1} or TwoTz{+1}.
  double e_fermi(TwoTz species) const;

  /// Informational name ("O16", "custom", ...).
  const std::string& name() const noexcept { return name_; }

  /// True if the occupations are identical (names are ignored).
  friend bool operator==(const Reference& a, const Reference& b) noexcept {
    return a.occ_ == b.occ_;
  }

 private:
  Reference() = default;

  std::vector<double> occ_;
  std::vector<OrbitIndex> holes_;
  std::vector<OrbitIndex> particles_;
  double A_ = 0.0;
  double Z_ = 0.0;
  double N_ = 0.0;
  double e_fermi_proton_ = -1.0;
  double e_fermi_neutron_ = -1.0;
  std::string name_;
};

}  // namespace imsrg::basis
