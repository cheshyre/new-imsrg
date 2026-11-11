#pragma once
/// @file
/// Partition of the orbits into core, valence and q-space (imsrg++ `cvq`; design/10 section 3.2,
/// design/reference/01 sections 1.4-1.5).

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "imsrg/base/types.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/single_particle_basis.hpp"
#include "imsrg/basis/valence_space.hpp"

namespace imsrg::basis {

/// Class of an orbit in the decoupling partition. The numeric values are imsrg++'s `cvq`
/// (0 core, 1 valence, 2 q-space), which the two-body subset rules use arithmetically.
enum class OrbitClass : std::uint8_t { Core = 0, Valence = 1, QSpace = 2 };

/// imsrg++ `cvq` value of a class (0, 1 or 2).
constexpr std::int32_t cvq(OrbitClass c) noexcept { return static_cast<std::int32_t>(c); }

/// Name of a class: "core", "valence" or "qspace".
std::string_view orbit_class_name(OrbitClass c) noexcept;

/// Immutable assignment of every orbit of a basis to the core, the valence space or the q-space.
/// A single-reference calculation has no valence orbits. Stores no pointer to the basis.
class Partition {
 public:
  /// Explicit classes, one per orbit of `basis` in index order.
  /// @throws InputError if the size differs from basis.num_orbits().
  static Partition from_classes(const SingleParticleBasis& basis, std::vector<OrbitClass> classes);

  /// The partition of `def` on `basis`, as `ModelSpace::Init` (version 0, ModelSpace.cc:358-422)
  /// assigns `cvq`: core if the orbit is in `def.core`, else valence if in `def.valence`, else
  /// q-space (an orbit in both lists is core; imsrg++ prints a warning).
  /// @throws InputError if a core orbit is not in the basis (imsrg++ exits in `SetLmax`). Valence
  ///         orbits outside the basis are dropped, as imsrg++ does (e.g. under an lmax cut), with a
  ///         warning in the log.
  static Partition from_definition(const SingleParticleBasis& basis,
                                   const ValenceSpaceDefinition& def);

  /// The rule of imsrg++ `ModelSpace::SetReference(map)` (ModelSpace.cc:807-826), which the HF
  /// re-filling uses when the occupations change: the core becomes the holes of `ref` that are not
  /// valence orbits of `previous`; the valence orbits stay; every other orbit is q-space.
  /// @throws InputError if the sizes of `previous` and `ref` differ.
  static Partition with_holes_as_core(const Partition& previous, const Reference& ref,
                                      const SingleParticleBasis& basis);

  /// Number of orbits.
  std::int32_t num_orbits() const noexcept { return static_cast<std::int32_t>(classes_.size()); }
  /// Class of every orbit, index order.
  std::span<const OrbitClass> classes() const noexcept { return classes_; }
  /// Class of orbit `a`. Precondition: 0 <= a < num_orbits() (IMSRG_ASSERT).
  OrbitClass orbit_class(OrbitIndex a) const;

  /// Core orbits, ascending (imsrg++ `core`).
  std::span<const OrbitIndex> core() const noexcept { return core_; }
  /// Valence orbits, ascending (imsrg++ `valence`).
  std::span<const OrbitIndex> valence() const noexcept { return valence_; }
  /// Q-space orbits, ascending (imsrg++ `qspace`).
  std::span<const OrbitIndex> qspace() const noexcept { return qspace_; }

  /// Number of nucleons in the full core shells, sum over core orbits of (2j+1) (imsrg++
  /// `GetAcore`; no occupation weights).
  double A_core() const noexcept { return A_core_; }
  /// Number of protons in the full core shells (imsrg++ `GetZcore`).
  double Z_core() const noexcept { return Z_core_; }

  /// True if there are no valence orbits.
  bool is_single_reference() const noexcept { return valence_.empty(); }

  /// True if the classes are identical.
  friend bool operator==(const Partition& a, const Partition& b) noexcept {
    return a.classes_ == b.classes_;
  }

 private:
  Partition() = default;

  std::vector<OrbitClass> classes_;
  std::vector<OrbitIndex> core_;
  std::vector<OrbitIndex> valence_;
  std::vector<OrbitIndex> qspace_;
  double A_core_ = 0.0;
  double Z_core_ = 0.0;
};

}  // namespace imsrg::basis
