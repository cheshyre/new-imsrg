#include "imsrg/basis/single_particle_basis.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::basis {

SingleParticleBasis SingleParticleBasis::harmonic_oscillator(const BasisTruncation& trunc,
                                                             double hw_MeV) {
  if (trunc.emax < 0) {
    throw InputError("SingleParticleBasis: emax must be >= 0 (got " + std::to_string(trunc.emax) +
                     ")");
  }
  if (!(hw_MeV > 0.0) || !std::isfinite(hw_MeV)) {
    throw InputError("SingleParticleBasis: hbar*omega must be a positive number of MeV");
  }
  const bool unocc_cut = trunc.emax_unocc >= 0 && trunc.emax_unocc < trunc.emax;
  if (unocc_cut && trunc.lmax_unocc < 0) {
    throw InputError("SingleParticleBasis: emax_unocc = " + std::to_string(trunc.emax_unocc) +
                     " needs lmax_unocc >= 0 (the hole-like l range kept above emax_unocc)");
  }

  SingleParticleBasis b;
  b.truncation_ = trunc;
  b.hbar_omega_ = hw_MeV;
  b.index_by_hash_.assign(static_cast<std::size_t>(num_orbits_untruncated(trunc.emax)), -1);

  // ModelSpace::Init, ModelSpace.cc:381-408.
  for (std::int32_t N = 0; N <= trunc.emax; ++N) {
    for (std::int32_t l = N; l >= 0; l -= 2) {
      if (trunc.lmax >= 0 && l > trunc.lmax) continue;
      const std::int32_t n = (N - l) / 2;
      for (std::int32_t j2 = 2 * l + 1; j2 >= 2 * l - 1 && j2 > 0; j2 -= 2) {
        for (const TwoTz tz2 : {TwoTz::proton(), TwoTz::neutron()}) {
          if (unocc_cut && N > trunc.emax_unocc && l > trunc.lmax_unocc) continue;
          const Orbit o{n, l, TwoJ{j2}, tz2};
          b.index_by_hash_[static_cast<std::size_t>(imsrgpp_orbit_hash(o))] =
              static_cast<std::int32_t>(b.orbits_.size());
          b.orbits_.push_back(o);
          b.max_j2_ = std::max(b.max_j2_, o.j2);
        }
      }
    }
  }
  return b;
}

const Orbit& SingleParticleBasis::orbit(OrbitIndex i) const {
  IMSRG_ASSERT(i.value() >= 0 && i.value() < num_orbits(), "orbit index out of range");
  return orbits_[i.to_size()];
}

std::optional<OrbitIndex> SingleParticleBasis::find(const Orbit& o) const noexcept {
  if (!is_valid(o) || o.e() > truncation_.emax) return std::nullopt;
  const std::int32_t idx = index_by_hash_[static_cast<std::size_t>(imsrgpp_orbit_hash(o))];
  if (idx < 0) return std::nullopt;
  return OrbitIndex{idx};
}

OrbitIndex SingleParticleBasis::index_of(const Orbit& o) const {
  if (const auto i = find(o)) return *i;
  std::string label;
  try {
    label = orbit_label(o);
  } catch (const InputError&) {
    label = "(invalid orbit)";
  }
  throw InputError("orbit " + label + " is not in the basis (emax=" +
                   std::to_string(truncation_.emax) + ", lmax=" + std::to_string(lmax()) + ")");
}

std::int32_t SingleParticleBasis::lmax() const noexcept {
  return truncation_.lmax < 0 ? truncation_.emax : std::min(truncation_.lmax, truncation_.emax);
}

std::vector<OrbitIndex> SingleParticleBasis::same_ljtz(OrbitIndex i) const {
  const Orbit& oi = orbit(i);
  std::vector<OrbitIndex> result;
  for (std::int32_t k = 0; k < num_orbits(); ++k) {
    const Orbit& ok = orbits_[static_cast<std::size_t>(k)];
    if (ok.l == oi.l && ok.j2 == oi.j2 && ok.tz2 == oi.tz2) result.emplace_back(k);
  }
  return result;
}

bool operator==(const SingleParticleBasis& a, const SingleParticleBasis& b) noexcept {
  return a.truncation_.emax == b.truncation_.emax && a.hbar_omega_ == b.hbar_omega_ &&
         a.orbits_ == b.orbits_;
}

}  // namespace imsrg::basis
