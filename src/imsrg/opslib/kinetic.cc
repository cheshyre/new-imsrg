// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/opslib/kinetic.h"

#include <memory>

#include "imsrg/opslib/detail/pair_engine.h"

namespace imsrg::opslib {

op::Operator KineticEnergy(const std::shared_ptr<const basis::ModelSpace>& ms) {
  op::Operator t(ms, op::OperatorSpec::Scalar(2, Hermiticity::kHermitian));
  detail::AddOscillatorOneBody(t, 0.5 * ms->basis().hbar_omega(), 1.0,
                               [](const basis::Orbit&) { return true; });
  t.SyncToDevice();
  return t;
}

}  // namespace imsrg::opslib
