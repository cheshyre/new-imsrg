// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// The N7 term comm232ss (imsrg3.h). Kernel TU. STUB: not implemented yet.

#include "imsrg/base/error.h"
#include "imsrg/commutator/imsrg3.h"

namespace imsrg::commutator::terms {

void Comm232ss(const op::Operator& /*X*/, const op::Operator& /*Y*/,
               op::Operator& /*Z*/, const ThreeBodyContext& /*ctx*/) {
  throw NotImplementedError("commutator: comm232ss is not implemented yet");
}

}  // namespace imsrg::commutator::terms
