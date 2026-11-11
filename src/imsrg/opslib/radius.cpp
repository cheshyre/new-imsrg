#include "imsrg/opslib/radius.hpp"

#include <stdexcept>

#include "imsrg/base/constants.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/opslib/center_of_mass.hpp"
#include "imsrg/opslib/detail/pair_engine.hpp"

namespace imsrg::opslib {

namespace {

double b_squared(const basis::ModelSpace& ms) {
  return constants::hbarc * constants::hbarc / constants::m_nucleon / ms.basis().hbar_omega();
}

bool in_species(const basis::Orbit& o, Species s) {
  switch (s) {
    case Species::Proton:
      return o.tz2.twice < 0;
    case Species::Neutron:
      return o.tz2.twice > 0;
    case Species::Matter:
      return true;
  }
  return false;
}

op::Operator zero_operator(const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator z(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  z.sync_to_device();
  return z;
}

}  // namespace

Species species_from_string(const std::string& option) {
  if (option == "proton") {
    return Species::Proton;
  }
  if (option == "neutron") {
    return Species::Neutron;
  }
  if (option == "matter") {
    return Species::Matter;
  }
  throw std::invalid_argument("species_from_string: unknown option '" + option + "'");
}

op::Operator r2_one_body(const std::shared_ptr<const basis::ModelSpace>& ms, Species species) {
  op::Operator r(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  detail::add_oscillator_one_body(
      r, b_squared(*ms), -1.0, [species](const basis::Orbit& o) { return in_species(o, species); });
  r.sync_to_device();
  return r;
}

op::Operator r2_two_body(const std::shared_ptr<const basis::ModelSpace>& ms, Species species) {
  op::Operator r(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const detail::PairEngine engine(detail::max_orbit_shell(*ms), ms->basis().lmax(),
                                  ms->basis().hbar_omega());
  // Channel filter and pn prefactor are applied inside the value callback (zero skips the entry).
  detail::add_two_body(
      r, 0.0,
      [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a, const basis::Orbit& b,
          const basis::Orbit& c, const basis::Orbit& d) {
        if ((species == Species::Proton && ch.Tz() > 0) ||
            (species == Species::Neutron && ch.Tz() < 0)) {
          return 0.0;
        }
        const double prefactor = (ch.Tz() == 0 && species != Species::Matter) ? 0.5 : 1.0;
        return prefactor * engine.element(detail::PairKind::Position, a, b, c, d, ch.two_j);
      });
  r.sync_to_device();
  return r;
}

op::Operator rp_spin_orbit_correction(const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator r(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const double m2 =
      constants::m_nucleon * constants::m_nucleon / (constants::hbarc * constants::hbarc);
  const auto& basis = ms->basis();
  for (std::int32_t i = 0; i < basis.num_orbits(); ++i) {
    const basis::Orbit& o = basis.orbit(OrbitIndex{i});
    const double mu = (o.tz2.twice < 0) ? 2.793 - 0.5 : -1.913;
    const double kappa = (o.j2.twice < 2 * o.l) ? o.l : -(o.l + 1);
    r.one_body().add(OrbitIndex{i}, OrbitIndex{i}, -mu / m2 * (kappa + 1.0));
  }
  r.sync_to_device();
  return r;
}

op::Operator rp2(const std::shared_ptr<const basis::ModelSpace>& ms, int A, int Z,
                 bool include_spin_orbit) {
  if (Z == 0) {
    return zero_operator(ms);
  }
  const double dA = A;
  const double dZ = Z;
  op::Operator r = r2_cm(ms, A);
  op::axpy((dA - 2.0) / (dA * dZ), r2_one_body(ms, Species::Proton), r);
  op::axpy(-4.0 / (dA * dZ), r2_two_body(ms, Species::Proton), r);
  if (include_spin_orbit) {
    op::axpy(1.0 / dZ, rp_spin_orbit_correction(ms), r);
  }
  return r;
}

op::Operator rn2(const std::shared_ptr<const basis::ModelSpace>& ms, int A, int Z) {
  if (Z == A) {
    return zero_operator(ms);
  }
  const double dA = A;
  const double dN = A - Z;
  op::Operator r = r2_cm(ms, A);
  op::axpy((dA - 2.0) / (dA * dN), r2_one_body(ms, Species::Neutron), r);
  op::axpy(-4.0 / (dA * dN), r2_two_body(ms, Species::Neutron), r);
  return r;
}

op::Operator rm2(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator r = r2_one_body(ms, Species::Matter);
  r *= 1.0 / A;
  r -= r2_cm(ms, A);
  return r;
}

}  // namespace imsrg::opslib
