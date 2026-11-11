#include "imsrg/basis/partition.hpp"

#include <utility>

#include <fmt/format.h>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"
#include "imsrg/base/log.hpp"

namespace imsrg::basis {

std::string_view orbit_class_name(OrbitClass c) noexcept {
  switch (c) {
    case OrbitClass::Core:
      return "core";
    case OrbitClass::Valence:
      return "valence";
    case OrbitClass::QSpace:
      return "qspace";
  }
  return "unknown";
}

Partition Partition::from_classes(const SingleParticleBasis& basis,
                                  std::vector<OrbitClass> classes) {
  if (static_cast<std::int32_t>(classes.size()) != basis.num_orbits()) {
    throw InputError(fmt::format("Partition: {} classes given for a basis of {} orbits",
                                 classes.size(), basis.num_orbits()));
  }
  Partition p;
  p.classes_ = std::move(classes);
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
    const OrbitIndex ia{a};
    switch (p.classes_[static_cast<std::size_t>(a)]) {
      case OrbitClass::Core: {
        p.core_.push_back(ia);
        const Orbit& o = basis.orbit(ia);
        p.A_core_ += o.j2.twice + 1;
        if (o.is_proton()) p.Z_core_ += o.j2.twice + 1;
        break;
      }
      case OrbitClass::Valence:
        p.valence_.push_back(ia);
        break;
      case OrbitClass::QSpace:
        p.qspace_.push_back(ia);
        break;
    }
  }
  return p;
}

Partition Partition::from_definition(const SingleParticleBasis& basis,
                                     const ValenceSpaceDefinition& def) {
  std::vector<OrbitClass> classes(static_cast<std::size_t>(basis.num_orbits()), OrbitClass::QSpace);
  for (const Orbit& o : def.valence) {
    if (const auto idx = basis.find(o)) {
      classes[idx->to_size()] = OrbitClass::Valence;
    } else {
      log::warn("valence orbit {} is not in the basis and is dropped", orbit_label(o));
    }
  }
  for (const Orbit& o : def.core) {
    const auto idx = basis.find(o);
    if (!idx) {
      throw InputError(fmt::format("Partition: core orbit {} is not in the basis", orbit_label(o)));
    }
    if (classes[idx->to_size()] == OrbitClass::Valence) {
      log::warn("orbit {} is in both the core and the valence space; it is core", orbit_label(o));
    }
    classes[idx->to_size()] = OrbitClass::Core;
  }
  return from_classes(basis, std::move(classes));
}

Partition Partition::with_holes_as_core(const Partition& previous, const Reference& ref,
                                        const SingleParticleBasis& basis) {
  if (previous.num_orbits() != ref.num_orbits() || ref.num_orbits() != basis.num_orbits()) {
    throw InputError(fmt::format(
        "Partition::with_holes_as_core: sizes differ (partition {}, reference {}, basis {})",
        previous.num_orbits(), ref.num_orbits(), basis.num_orbits()));
  }
  std::vector<OrbitClass> classes(static_cast<std::size_t>(basis.num_orbits()), OrbitClass::QSpace);
  for (const OrbitIndex h : ref.holes()) {
    classes[h.to_size()] = OrbitClass::Core;
  }
  for (const OrbitIndex v : previous.valence()) {
    classes[v.to_size()] = OrbitClass::Valence;  // valence is never core
  }
  return from_classes(basis, std::move(classes));
}

OrbitClass Partition::orbit_class(OrbitIndex a) const {
  IMSRG_ASSERT(a.value() >= 0 && a.value() < num_orbits(), "orbit index out of range");
  return classes_[a.to_size()];
}

}  // namespace imsrg::basis
