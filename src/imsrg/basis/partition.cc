// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/basis/partition.h"

#include <utility>
#include <vector>

#include <fmt/format.h>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/base/log.h"

namespace imsrg::basis {

std::string_view OrbitClassName(OrbitClass c) noexcept {
  switch (c) {
    case OrbitClass::kCore:
      return "core";
    case OrbitClass::kValence:
      return "valence";
    case OrbitClass::kQSpace:
      return "qspace";
  }
  return "unknown";
}

Partition Partition::FromClasses(const SingleParticleBasis& basis,
                                 std::vector<OrbitClass> classes) {
  if (static_cast<std::int32_t>(classes.size()) != basis.num_orbits()) {
    throw InputError(
        fmt::format("Partition: {} classes given for a basis of {} orbits",
                    classes.size(), basis.num_orbits()));
  }
  Partition p;
  p.classes_ = std::move(classes);
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
    const OrbitIndex ia{a};
    switch (p.classes_[static_cast<std::size_t>(a)]) {
      case OrbitClass::kCore: {
        p.core_.push_back(ia);
        const Orbit& o = basis.orbit(ia);
        p.A_core_ += o.j2.twice + 1;
        if (o.is_proton()) p.Z_core_ += o.j2.twice + 1;
        break;
      }
      case OrbitClass::kValence:
        p.valence_.push_back(ia);
        break;
      case OrbitClass::kQSpace:
        p.qspace_.push_back(ia);
        break;
    }
  }
  return p;
}

Partition Partition::FromDefinition(const SingleParticleBasis& basis,
                                    const ValenceSpaceDefinition& def) {
  std::vector<OrbitClass> classes(static_cast<std::size_t>(basis.num_orbits()),
                                  OrbitClass::kQSpace);
  for (const Orbit& o : def.valence) {
    if (const auto idx = basis.find(o)) {
      classes[idx->ToSize()] = OrbitClass::kValence;
    } else {
      log::Warn("valence orbit {} is not in the basis and is dropped",
                OrbitLabel(o));
    }
  }
  for (const Orbit& o : def.core) {
    const auto idx = basis.find(o);
    if (!idx) {
      throw InputError(fmt::format(
          "Partition: core orbit {} is not in the basis", OrbitLabel(o)));
    }
    if (classes[idx->ToSize()] == OrbitClass::kValence) {
      log::Warn(
          "orbit {} is in both the core and the valence space; it is core",
          OrbitLabel(o));
    }
    classes[idx->ToSize()] = OrbitClass::kCore;
  }
  return FromClasses(basis, std::move(classes));
}

Partition Partition::WithHolesAsCore(const Partition& previous,
                                     const Reference& ref,
                                     const SingleParticleBasis& basis) {
  if (previous.num_orbits() != ref.num_orbits() ||
      ref.num_orbits() != basis.num_orbits()) {
    throw InputError(fmt::format(
        "Partition::with_holes_as_core: sizes differ (partition {}, reference "
        "{}, basis {})",
        previous.num_orbits(), ref.num_orbits(), basis.num_orbits()));
  }
  std::vector<OrbitClass> classes(static_cast<std::size_t>(basis.num_orbits()),
                                  OrbitClass::kQSpace);
  for (const OrbitIndex h : ref.holes()) {
    classes[h.ToSize()] = OrbitClass::kCore;
  }
  for (const OrbitIndex v : previous.valence()) {
    classes[v.ToSize()] = OrbitClass::kValence;  // valence is never core
  }
  return FromClasses(basis, std::move(classes));
}

OrbitClass Partition::orbit_class(OrbitIndex a) const {
  IMSRG_ASSERT(a.value() >= 0 && a.value() < num_orbits(),
               "orbit index out of range");
  return classes_[a.ToSize()];
}

}  // namespace imsrg::basis
