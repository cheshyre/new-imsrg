#pragma once
/// @file
/// Center-of-mass operators in the harmonic-oscillator basis: T_cm, T_rel, R^2_cm and H_cm
/// (imsrg++ `imsrg_util::TCM_Op`, `Trel_Op`, `R2CM_Op`, `HCM_Op`).
///
/// All operators are Hermitian scalars of particle rank 2 with normalized antisymmetrized
/// J-coupled two-body storage (kets p <= q, both triangles filled), in MeV (T, H) or fm^2 (R^2).
/// The oscillator frequency hbar_omega is that of `ms->basis()`; the mass number A is an explicit
/// argument (imsrg++ takes it from the model space's target mass). Construction is host-only
/// (two-body elements by Moshinsky transformation) and ends with `sync_to_device()`.
///
/// imsrg++ skips two-body kets with 2(n_a+n_b)+l_a+l_b > E2max in TCM_Op; the model space of this
/// code has no E2max cut (its default, 2 emax, never cuts), so no such skip is implemented.

#include <memory>
#include <optional>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::opslib {

/// Center-of-mass kinetic energy T_cm = (sum_i p_i)^2 / (2 m A) = T/A + sum_{i<j} p_i.p_j/(m A),
/// MeV. One body part: T/A (see `kinetic_energy`). Two-body part: the normalized antisymmetrized
/// element of p1.p2/(m A), obtained from the (T_cm - T_rel) oscillator sums of the Moshinsky
/// transformation times hbar_omega/2 (see `Calculate_p1p2`); elements with |v| <= 1e-7 are not
/// stored, as in imsrg++ `TCM_Op` (an imsrg++ quirk kept on purpose). Reproduces `TCM_Op`.
/// Complexity: O(sum over channels of num_kets^2 x E^3) with O(1) table lookups; about a second at
/// emax 6.
/// @pre `ms` not null, `A` >= 1.
op::Operator t_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Intrinsic kinetic energy T_rel = T - T_cm in MeV, computed as the exact operator difference
/// `kinetic_energy(ms) - t_cm(ms, A)` (imsrg++ `Trel_Op`). Same complexity as `t_cm`.
/// @pre `ms` not null, `A` >= 1.
op::Operator t_rel(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Center-of-mass radius squared R_cm^2 = (1/A^2) (sum_i r_i)^2 in fm^2: (1/A^2) [ sum_i r_i^2 +
/// 2 sum_{i<j} r_i.r_j ], with the oscillator length b^2 = hbar^2 c^2 / (m_nucleon hbar_omega).
/// The one-body part is r^2/A^2 with <n l|r^2|n l> = b^2 (2n + l + 3/2) and off-diagonal elements
/// -b^2 sqrt(n_big (n_big + l + 1/2)); the two-body part is 2 r1.r2/A^2 (every element stored, no
/// threshold). Reproduces imsrg++ `R2CM_Op`. Complexity as `t_cm`.
/// @pre `ms` not null, `A` >= 1.
op::Operator r2_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

/// Center-of-mass Hamiltonian H_cm = T_cm + (1/2) m A omega_trap^2 R_cm^2, MeV, with
/// (1/2) m A omega_trap^2 = (1/2) A m_nucleon (hbar_omega_trap)^2 / (hbar c)^2 [MeV/fm^2]
/// multiplying `r2_cm` (imsrg++ `HCM_Op` / `HCM_<hw>`: only that prefactor uses the trap
/// frequency; T_cm and R_cm^2 are evaluated at the basis frequency).
///
/// Lawson usage: the imsrg++ driver adds beta_cm * H_cm to the Hamiltonian (with the trap
/// frequency defaulting to the basis hbar_omega) to push center-of-mass excitations up, then
/// subtracts it after the evolution; its expectation value in an oscillator Slater determinant
/// at the basis frequency is a measure of spuriousness. `hbar_omega_trap` is in MeV; the default
/// (empty) is the basis frequency. Complexity as `t_cm`.
/// @pre `ms` not null, `A` >= 1, `hbar_omega_trap` (if given) > 0.
op::Operator h_cm(const std::shared_ptr<const basis::ModelSpace>& ms, int A,
                  std::optional<double> hbar_omega_trap = std::nullopt);

}  // namespace imsrg::opslib
