#include "imsrg/opslib/center_of_mass.hpp"

#include "imsrg/base/constants.hpp"
#include "imsrg/operator/arithmetic.hpp"
#include "imsrg/opslib/detail/pair_engine.hpp"
#include "imsrg/opslib/kinetic.hpp"

namespace imsrg::opslib {

namespace {

constexpr double kTcmThreshold = 1e-7;  // imsrg++ TCM_Op drops |p1p2/A| <= 1e-7

detail::PairEngine make_engine(const basis::ModelSpace& ms) {
  return detail::PairEngine(detail::max_orbit_shell(ms), ms.basis().lmax(),
                            ms.basis().hbar_omega());
}

}  // namespace

op::Operator t_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator t(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  detail::add_oscillator_one_body(t, 0.5 * ms->basis().hbar_omega() / A, 1.0,
                                  [](const basis::Orbit&) { return true; });
  const detail::PairEngine engine = make_engine(*ms);
  detail::add_two_body(t, kTcmThreshold,
                       [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a,
                           const basis::Orbit& b, const basis::Orbit& c, const basis::Orbit& d) {
                         return engine.element(detail::PairKind::Momentum, a, b, c, d, ch.two_j) /
                                A;
                       });
  t.sync_to_device();
  return t;
}

op::Operator t_rel(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator t = kinetic_energy(ms);
  t -= t_cm(ms, A);
  t.sync_to_host();
  return t;
}

op::Operator r2_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A) {
  op::Operator r(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  const double b2 =
      constants::hbarc * constants::hbarc / constants::m_nucleon / ms->basis().hbar_omega();
  detail::add_oscillator_one_body(r, b2 / (double(A) * A), -1.0,
                                  [](const basis::Orbit&) { return true; });
  const detail::PairEngine engine = make_engine(*ms);
  detail::add_two_body(r, 0.0,
                       [&](const basis::TwoBodyChannel& ch, const basis::Orbit& a,
                           const basis::Orbit& b, const basis::Orbit& c, const basis::Orbit& d) {
                         return 2.0 *
                                engine.element(detail::PairKind::Position, a, b, c, d, ch.two_j) /
                                (double(A) * A);
                       });
  r.sync_to_device();
  return r;
}

op::Operator h_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                  std::optional<double> hbar_omega_trap) {
  const double hw_trap = hbar_omega_trap.value_or(ms->basis().hbar_omega());
  op::Operator h = t_cm(ms, A);
  const double k =
      0.5 * A * constants::m_nucleon * hw_trap * hw_trap / (constants::hbarc * constants::hbarc);
  op::axpy(k, r2_cm(ms, A), h);
  h.sync_to_host();
  return h;
}

}  // namespace imsrg::opslib
