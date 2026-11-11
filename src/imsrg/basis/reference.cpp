#include "imsrg/basis/reference.hpp"

#include <cmath>
#include <utility>

#include <fmt/format.h>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::basis {

Reference Reference::from_occupations(const SingleParticleBasis& basis,
                                      std::vector<double> occupations, std::string name) {
  if (static_cast<std::int32_t>(occupations.size()) != basis.num_orbits()) {
    throw InputError(fmt::format("Reference: {} occupations given for a basis of {} orbits",
                                 occupations.size(), basis.num_orbits()));
  }
  for (std::size_t a = 0; a < occupations.size(); ++a) {
    const double n = occupations[a];
    if (!std::isfinite(n) || n < 0.0 || n > 1.0) {
      throw InputError(fmt::format("Reference: occupation {} of orbit {} ({}) is outside [0, 1]", n,
                                   a, orbit_label(basis.orbits()[a])));
    }
  }

  Reference ref;
  ref.occ_ = std::move(occupations);
  ref.name_ = std::move(name);

  // Same operation order as imsrg++ (CountInSet over holes ascending, FindEFermi).
  double e_fermi_p = -1.0;
  double e_fermi_n = -1.0;
  double occmax_p = 0.0;
  double occmax_n = 0.0;
  for (std::int32_t a = 0; a < basis.num_orbits(); ++a) {
    const OrbitIndex ia{a};
    const double n = ref.occ_[static_cast<std::size_t>(a)];
    if (n >= kOccupationCut) {
      ref.holes_.push_back(ia);
      const Orbit& o = basis.orbit(ia);
      const double weight = (o.j2.twice + 1) * n;
      ref.A_ += weight;
      double& e_f = o.is_proton() ? e_fermi_p : e_fermi_n;
      double& occmax = o.is_proton() ? occmax_p : occmax_n;
      if (o.is_proton()) {
        ref.Z_ += weight;
      } else {
        ref.N_ += weight;
      }
      if (static_cast<double>(o.e()) > e_f) {
        e_f = static_cast<double>(o.e());
        occmax = n;
      }
    } else {
      ref.particles_.push_back(ia);
    }
  }
  double pmin_p = 1e6;
  double pmin_n = 1e6;
  for (const OrbitIndex ia : ref.particles_) {
    const Orbit& o = basis.orbit(ia);
    double& pmin = o.is_proton() ? pmin_p : pmin_n;
    if (static_cast<double>(o.e()) < pmin) {
      pmin = static_cast<double>(o.e());
    }
  }
  if (pmin_p == 1e6) pmin_p = e_fermi_p;
  if (pmin_n == 1e6) pmin_n = e_fermi_n;
  if (occmax_p > 0.99) e_fermi_p = 0.5 * (e_fermi_p + pmin_p);
  if (occmax_n > 0.99) e_fermi_n = 0.5 * (e_fermi_n + pmin_n);
  ref.e_fermi_proton_ = e_fermi_p;
  ref.e_fermi_neutron_ = e_fermi_n;
  return ref;
}

Reference Reference::from_nucleus(const SingleParticleBasis& basis, const Nucleus& nucleus) {
  std::vector<double> occ(static_cast<std::size_t>(basis.num_orbits()), 0.0);
  for (const OrbitOccupation& oo : naive_filling(nucleus, basis.emax())) {
    const auto idx = basis.find(oo.orbit);
    if (!idx) {
      throw InputError(fmt::format("Reference for {}: occupied orbit {} is not in the basis",
                                   nucleus.name(), orbit_label(oo.orbit)));
    }
    occ[idx->to_size()] = oo.occupation;
  }
  return from_occupations(basis, std::move(occ), nucleus.name());
}

Reference Reference::from_orbit_occupations(const SingleParticleBasis& basis,
                                            std::span<const OrbitOccupation> entries,
                                            std::string name) {
  std::vector<double> occ(static_cast<std::size_t>(basis.num_orbits()), 0.0);
  for (const OrbitOccupation& oo : entries) {
    if (std::abs(oo.occupation) < 1e-8) {
      continue;
    }
    if (!is_valid(oo.orbit)) {
      throw InputError("Reference: invalid orbit in the occupation list");
    }
    const auto idx = basis.find(oo.orbit);
    if (!idx) {
      throw InputError(fmt::format("Reference: orbit {} of the occupation list is not in the basis",
                                   orbit_label(oo.orbit)));
    }
    occ[idx->to_size()] = oo.occupation;
  }
  return from_occupations(basis, std::move(occ), std::move(name));
}

double Reference::occupation(OrbitIndex a) const {
  IMSRG_ASSERT(a.value() >= 0 && a.value() < num_orbits(), "orbit index out of range");
  return occ_[a.to_size()];
}

double Reference::e_fermi(TwoTz species) const {
  if (species == TwoTz{-1}) return e_fermi_proton_;
  if (species == TwoTz{+1}) return e_fermi_neutron_;
  throw InputError(
      fmt::format("Reference::e_fermi: species 2tz = {} is neither -1 nor +1", species.twice));
}

}  // namespace imsrg::basis
