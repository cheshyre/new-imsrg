#include "imsrg/opslib/kinetic.hpp"

#include "imsrg/opslib/detail/pair_engine.hpp"

namespace imsrg::opslib {

op::Operator kinetic_energy(const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator t(ms, op::OperatorSpec::scalar(2, Hermiticity::Hermitian));
  detail::add_oscillator_one_body(t, 0.5 * ms->basis().hbar_omega(), 1.0,
                                  [](const basis::Orbit&) { return true; });
  t.sync_to_device();
  return t;
}

}  // namespace imsrg::opslib
