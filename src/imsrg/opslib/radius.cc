// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/opslib/radius.h"

#include <memory>
#include <string>

#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/opslib/center_of_mass.h"
#include "imsrg/opslib/detail/pair_engine.h"

namespace imsrg::opslib {

namespace {

double BSquared(const basis::ModelSpace& ms) {
  return constants::kHbarc * constants::kHbarc / constants::kMNucleon /
         ms.basis().hbar_omega();
}

bool InSpecies(const basis::Orbit& o, Species s) {
  switch (s) {
    case Species::kProton:
      return o.tz2.twice < 0;
    case Species::kNeutron:
      return o.tz2.twice > 0;
    case Species::kMatter:
      return true;
  }
  return false;
}

op::Operator ZeroOperator(const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator z(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  z.SyncToDevice();
  return z;
}

}  // namespace

Species SpeciesFromString(const std::string& option) {
  if (option == "proton") {
    return Species::kProton;
  }
  if (option == "neutron") {
    return Species::kNeutron;
  }
  if (option == "matter") {
    return Species::kMatter;
  }
  throw InputError("species_from_string: unknown option '" + option + "'");
}

op::Operator R2OneBody(const std::shared_ptr<const basis::ModelSpace>& ms,
                       Species species) {
  op::Operator r(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  detail::AddOscillatorOneBody(
      r, BSquared(*ms), -1.0,
      [species](const basis::Orbit& o) { return InSpecies(o, species); });
  r.SyncToDevice();
  return r;
}

op::Operator R2TwoBody(const std::shared_ptr<const basis::ModelSpace>& ms,
                       Species species) {
  op::Operator r(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  const detail::PairEngine engine(detail::MaxOrbitShell(*ms),
                                  ms->basis().lmax(), ms->basis().hbar_omega());
  // Channel filter and pn prefactor are applied inside the value callback (zero
  // skips the entry).
  detail::AddTwoBody(
      r, 0.0,
      [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a,
          const basis::Orbit& b, const basis::Orbit& c, const basis::Orbit& d) {
        if ((species == Species::kProton && ch.Tz() > 0) ||
            (species == Species::kNeutron && ch.Tz() < 0)) {
          return 0.0;
        }
        const double prefactor =
            (ch.Tz() == 0 && species != Species::kMatter) ? 0.5 : 1.0;
        return prefactor * engine.Element(detail::PairKind::kPosition, a, b, c,
                                          d, ch.two_j);
      });
  r.SyncToDevice();
  return r;
}

op::Operator RpSpinOrbitCorrection(
    const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator r(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  const double m2 = constants::kMNucleon * constants::kMNucleon /
                    (constants::kHbarc * constants::kHbarc);
  const auto& basis = ms->basis();
  for (std::int32_t i = 0; i < basis.num_orbits(); ++i) {
    const basis::Orbit& o = basis.orbit(OrbitIndex{i});
    const double mu = (o.tz2.twice < 0) ? 2.793 - 0.5 : -1.913;
    const double kappa = (o.j2.twice < 2 * o.l) ? o.l : -(o.l + 1);
    r.one_body().Add(OrbitIndex{i}, OrbitIndex{i}, -mu / m2 * (kappa + 1.0));
  }
  r.SyncToDevice();
  return r;
}

op::Operator Rp2(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                 int Z) {
  if (Z == 0) {
    return ZeroOperator(ms);
  }
  const double dA = A;
  const double dZ = Z;
  op::Operator r = R2Cm(ms, A);
  op::Axpy((dA - 2.0) / (dA * dZ), R2OneBody(ms, Species::kProton), r);
  op::Axpy(-4.0 / (dA * dZ), R2TwoBody(ms, Species::kProton), r);
  return r;
}

op::Operator Rn2(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                 int Z) {
  if (Z == A) {
    return ZeroOperator(ms);
  }
  const double dA = A;
  const double dN = A - Z;
  op::Operator r = R2Cm(ms, A);
  op::Axpy((dA - 2.0) / (dA * dN), R2OneBody(ms, Species::kNeutron), r);
  op::Axpy(-4.0 / (dA * dN), R2TwoBody(ms, Species::kNeutron), r);
  return r;
}

op::Operator Rm2(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator r = R2OneBody(ms, Species::kMatter);
  r *= 1.0 / A;
  r -= R2Cm(ms, A);
  return r;
}

}  // namespace imsrg::opslib
