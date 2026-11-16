// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPSLIB_CENTER_OF_MASS_H_
#define IMSRG_OPSLIB_CENTER_OF_MASS_H_
/// @file
/// Center-of-mass operators in the harmonic-oscillator basis: T_cm, T_rel,
/// R^2_cm and H_cm (imsrg++ `imsrg_util::TCM_Op`, `Trel_Op`, `R2CM_Op`,
/// `HCM_Op`).
///
/// All operators are Hermitian scalars of particle rank 2 with normalized
/// antisymmetrized J-coupled two-body storage (kets p <= q, both triangles
/// filled), in MeV (T, H) or fm^2 (R^2). The oscillator frequency hbar_omega is
/// that of `ms->basis()`; the mass number A is an explicit argument (imsrg++
/// takes it from the model space's target mass). Construction is host-only
/// (two-body elements by Moshinsky transformation) and ends with
/// `SyncToDevice()`.
///
/// imsrg++ skips two-body kets with 2(n_a+n_b)+l_a+l_b > E2max in TCM_Op; the
/// model space of this code has no E2max cut (its default, 2 emax, never cuts),
/// so no such skip is implemented.

#include <memory>
#include <optional>

#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"

namespace imsrg::opslib {

/// Center-of-mass kinetic energy T_cm = (sum_i p_i)^2 / (2 m A) = T/A +
/// sum_{i<j} p_i.p_j/(m A), MeV. One body part: T/A (see `KineticEnergy`).
/// Two-body part: the normalized antisymmetrized element of p1.p2/(m A),
/// obtained from the (T_cm - T_rel) oscillator sums of the Moshinsky
/// transformation times hbar_omega/2 (see `Calculate_p1p2`). Every two-body
/// entry is kept (D35): imsrg++ `TCM_Op` drops elements with |v| <= 1e-7, which
/// only ever removes round-off (<= 2.4e-15 MeV at emax 4); the goldens come
/// from an imsrg++ build without that threshold. Complexity: O(sum over
/// channels of num_kets^2 x E^3) with O(1) table lookups; about a second at
/// emax 6.
/// @pre `ms` not null, `A` >= 1.
op::Operator TCm(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Intrinsic kinetic energy T_rel = T - T_cm in MeV, computed as the exact
/// operator difference `KineticEnergy(ms) - TCm(ms, A)` (imsrg++ `Trel_Op`).
/// Same complexity as `TCm`.
/// @pre `ms` not null, `A` >= 1.
op::Operator TRel(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Center-of-mass radius squared R_cm^2 = (1/A^2) (sum_i r_i)^2 in fm^2:
/// (1/A^2) [ sum_i r_i^2 + 2 sum_{i<j} r_i.r_j ], with the oscillator length
/// b^2 = hbar^2 c^2 / (m_nucleon hbar_omega). The one-body part is r^2/A^2 with
/// <n l|r^2|n l> = b^2 (2n + l + 3/2) and off-diagonal elements -b^2 sqrt(n_big
/// (n_big + l + 1/2)); the two-body part is 2 r1.r2/A^2 (every element stored,
/// no threshold). Reproduces imsrg++ `R2CM_Op`. Complexity as `TCm`.
/// @pre `ms` not null, `A` >= 1.
op::Operator R2Cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Center-of-mass Hamiltonian H_cm = T_cm + (1/2) m A omega_trap^2 R_cm^2, MeV,
/// with (1/2) m A omega_trap^2 = (1/2) A m_nucleon (hbar_omega_trap)^2 / (hbar
/// c)^2 [MeV/fm^2] multiplying `R2Cm` (imsrg++ `HCM_Op` / `HCM_<hw>`: only
/// that prefactor uses the trap frequency; T_cm and R_cm^2 are evaluated at the
/// basis frequency).
///
/// Lawson usage: the imsrg++ driver adds beta_cm * H_cm to the Hamiltonian
/// (with the trap frequency defaulting to the basis hbar_omega) to push
/// center-of-mass excitations up, then subtracts it after the evolution; its
/// expectation value in an oscillator Slater determinant at the basis frequency
/// is a measure of spuriousness. `hbar_omega_trap` is in MeV; the default
/// (empty) is the basis frequency. Complexity as `TCm`.
/// @pre `ms` not null, `A` >= 1, `hbar_omega_trap` (if given) > 0.
op::Operator HCm(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                 std::optional<double> hbar_omega_trap = std::nullopt);

}  // namespace imsrg::opslib

#endif  // IMSRG_OPSLIB_CENTER_OF_MASS_H_
