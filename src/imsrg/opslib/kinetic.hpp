#pragma once
/// @file
/// Kinetic energy in the harmonic-oscillator basis (imsrg++ `imsrg_util::KineticEnergy_Op`).

#include <memory>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::opslib {

/// Laboratory-frame kinetic energy T = sum_i p_i^2 / (2 m) in MeV, as a Hermitian scalar operator
/// with particle rank 2 whose two-body part is zero.
///
/// One-body matrix elements in the oscillator basis with hbar_omega = `ms->basis().hbar_omega()`:
/// <n l j tz| T |n l j tz> = (hw/2)(2n + l + 3/2) and
/// <n+1 l j tz| T |n l j tz> = (hw/2) sqrt((n+1)(n + l + 3/2)); all other elements vanish
/// (T conserves l, j, tz). Reproduces imsrg++ `KineticEnergy_Op`. Complexity O(num_orbits^2) host
/// operations; the result is synchronised to the device before returning.
/// @pre `ms` is not null.
op::Operator kinetic_energy(const std::shared_ptr<const basis::ModelSpace>& ms);

}  // namespace imsrg::opslib
