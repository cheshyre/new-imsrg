// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPSLIB_KINETIC_H_
#define IMSRG_OPSLIB_KINETIC_H_
/// @file
/// Kinetic energy in the harmonic-oscillator basis (imsrg++
/// `imsrg_util::KineticEnergy_Op`).

#include <memory>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::opslib {

/// Laboratory-frame kinetic energy T = sum_i p_i^2 / (2 m) in MeV, as a
/// Hermitian scalar operator with particle rank 2 whose two-body part is zero.
///
/// One-body matrix elements in the oscillator basis with hbar_omega =
/// `ms->basis().hbar_omega()`: <n l j tz| T |n l j tz> = (hw/2)(2n + l + 3/2)
/// and <n+1 l j tz| T |n l j tz> = (hw/2) sqrt((n+1)(n + l + 3/2)); all other
/// elements vanish (T conserves l, j, tz). Reproduces imsrg++
/// `KineticEnergy_Op`. Complexity O(num_orbits^2) host operations; the result
/// is synchronised to the device before returning.
/// @pre `ms` is not null.
op::Operator KineticEnergy(const std::shared_ptr<const basis::ModelSpace>& ms);

}  // namespace imsrg::opslib

#endif  // IMSRG_OPSLIB_KINETIC_H_
