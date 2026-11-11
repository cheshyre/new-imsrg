#pragma once
/// @file
/// Valence-space definitions: the named valence spaces of imsrg++ and the interpretation of its
/// `valence_space` strings (design/13 section 1, design/reference/01 section 1.5).

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis {

/// One entry of the imsrg++ table of named valence spaces (`ModelSpace::ValenceSpaces`,
/// ModelSpace.cc:23-70): the core nucleus and the valence orbit labels, in table order.
struct NamedValenceSpace {
  std::string_view name;                     ///< e.g. "sd-shell"
  std::string_view core;                     ///< core nucleus name, e.g. "O16"
  std::span<const std::string_view> orbits;  ///< valence orbit labels, e.g. "p0d5", "n0d5", ...
};

/// All named valence spaces, copied exactly from imsrg++ and sorted by name (std::map order, as
/// in imsrg++).
std::span<const NamedValenceSpace> named_valence_spaces() noexcept;

/// The named valence space `name`, or nullptr if there is none.
const NamedValenceSpace* find_named_valence_space(std::string_view name) noexcept;

/// A decoupling partition given by orbit quantum numbers: the core orbits and the valence
/// orbits; all other orbits of a basis are q-space (imsrg++ cvq 0, 1, 2).
struct ValenceSpaceDefinition {
  std::vector<Orbit> core;     ///< Core orbits (may overlap `valence`; core wins, see Partition).
  std::vector<Orbit> valence;  ///< Valence orbits (empty for a single-reference calculation).
  std::string core_name;       ///< Name of the core nucleus ("O16"); empty for "FCI".

  /// True if there are no valence orbits (single-reference decoupling).
  bool is_single_reference() const noexcept { return valence.empty(); }
};

/// Interprets an imsrg++ `valence_space` string as `ModelSpace::Init` (version 2,
/// ModelSpace.cc:224-290) does, for the basis `basis` and the reference nucleus `reference`:
/// * "0hw-shell": one major oscillator shell per species above the largest closed HO shell of
///   the reference (`Get0hwSpace`, ModelSpace.cc:659-694; uses int(A), int(Z) of `reference`);
/// * "jj-shell": the jj-shell analogue with the largest-j intruder of the next shell
///   (`GetjjSpace`, ModelSpace.cc:696-747);
/// * a string containing ',': "Core,orb,orb,..." (`ParseCommaSeparatedValenceSpace`,
///   ModelSpace.cc:753-770): the listed valence orbits, and the core = naive filling of the core
///   nucleus that SKIPS the valence orbits;
/// * a string containing "FCI": every orbit of `basis` is valence, there is no core;
/// * a name from `named_valence_spaces()`: its valence orbits, and the core = naive filling of
///   its core nucleus WITHOUT skipping valence orbits;
/// * anything else is a nucleus name: single reference, core = its naive filling, no valence.
///
/// Core fillings use `naive_filling(core, basis.emax())`; every orbit with nonzero filling is a
/// core orbit. The orbits are returned as quantum numbers in the order of their construction;
/// whether they exist in `basis` is checked by `Partition`.
/// @throws InputError for a malformed orbit label or nucleus name, or a core that does not fit
///         into shells e <= emax.
ValenceSpaceDefinition define_valence_space(std::string_view spec, const SingleParticleBasis& basis,
                                            const Nucleus& reference);

/// The nucleus imsrg++ uses as the reference when `reference=default` (imsrg++.cc:332 and the
/// two-argument ModelSpace constructor, ModelSpace.cc:201-211): the core of a named valence
/// space, the core of a comma-separated space ("O16,p0d5,..." gives O16), otherwise the string
/// parsed as a nucleus name.
/// @throws InputError for "0hw-shell", "jj-shell" and "FCI" spaces (they need an explicit
///         reference) and for unparsable names.
Nucleus default_reference_nucleus(std::string_view spec);

}  // namespace imsrg::basis
