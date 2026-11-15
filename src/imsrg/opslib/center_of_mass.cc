// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/opslib/center_of_mass.h"

#include "imsrg/base/constants.h"
#include "imsrg/operator/arithmetic.h"
#include "imsrg/opslib/detail/pair_engine.h"
#include "imsrg/opslib/kinetic.h"

namespace imsrg::opslib {

namespace {

constexpr double kTcmThreshold =
    1.0e-7;  // imsrg++ TCM_Op drops |p1p2/A| <= 1e-7

detail::PairEngine MakeEngine(const basis::ModelSpace& ms) {
  return detail::PairEngine(detail::MaxOrbitShell(ms), ms.basis().lmax(),
                            ms.basis().hbar_omega());
}

}  // namespace

op::Operator TCm(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator t(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  detail::AddOscillatorOneBody(t, 0.5 * ms->basis().hbar_omega() / A, 1.0,
                               [](const basis::Orbit&) { return true; });
  const detail::PairEngine engine = MakeEngine(*ms);
  detail::AddTwoBody(
      t, kTcmThreshold,
      [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a,
          const basis::Orbit& b, const basis::Orbit& c, const basis::Orbit& d) {
        return engine.Element(detail::PairKind::kMomentum, a, b, c, d,
                              ch.two_j) /
               A;
      });
  t.SyncToDevice();
  return t;
}

op::Operator TRel(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator t = KineticEnergy(ms);
  t -= TCm(ms, A);
  t.SyncToHost();
  return t;
}

op::Operator R2Cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator r(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  const double b2 = constants::kHbarc * constants::kHbarc /
                    constants::kMNucleon / ms->basis().hbar_omega();
  detail::AddOscillatorOneBody(r, b2 / (double(A) * A), -1.0,
                               [](const basis::Orbit&) { return true; });
  const detail::PairEngine engine = MakeEngine(*ms);
  detail::AddTwoBody(
      r, 0.0,
      [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a,
          const basis::Orbit& b, const basis::Orbit& c, const basis::Orbit& d) {
        return 2.0 *
               engine.Element(detail::PairKind::kPosition, a, b, c, d,
                              ch.two_j) /
               (double(A) * A);
      });
  r.SyncToDevice();
  return r;
}

op::Operator HCm(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                 std::optional<double> hbar_omega_trap) {
  const double hw_trap = hbar_omega_trap.value_or(ms->basis().hbar_omega());
  op::Operator h = TCm(ms, A);
  const double k = 0.5 * A * constants::kMNucleon * hw_trap * hw_trap /
                   (constants::kHbarc * constants::kHbarc);
  op::Axpy(k, R2Cm(ms, A), h);
  h.SyncToHost();
  return h;
}

}  // namespace imsrg::opslib
