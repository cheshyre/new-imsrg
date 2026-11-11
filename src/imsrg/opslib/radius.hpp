#pragma once
/// @file
/// Intrinsic point-nucleon radius operators in the harmonic-oscillator basis (imsrg++
/// `imsrg_util::Rp2_corrected_Op`, `Rn2_corrected_Op`, `Rm2_corrected_Op`, `R2_1body_Op`,
/// `R2_2body_Op`, `RpSpinOrbitCorrection`).
///
/// All operators are Hermitian scalars of particle rank 2 with normalized antisymmetrized
/// J-coupled two-body storage, in fm^2. hbar_omega comes from `ms->basis()`; A and Z are explicit
/// arguments. Convention: protons have tz2 = -1, neutrons tz2 = +1.

#include <memory>
#include <string>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::opslib {

/// Nucleon species selecting the orbits of `r2_one_body` / `r2_two_body`; replaces the imsrg++
/// option strings "proton", "neutron" and "matter".
enum class Species { Proton, Neutron, Matter };

/// Maps an imsrg++ option string ("proton", "neutron", "matter") to a `Species`.
/// @throws std::invalid_argument for any other string.
Species species_from_string(const std::string& option);

/// One-body r^2 restricted to one species (imsrg++ `R2_1body_Op(modelspace, option)`):
/// <n l|r^2|n l> = b^2 (2n + l + 3/2), <n+1 l|r^2|n l> = -b^2 sqrt((n+1)(n + l + 3/2)) for the
/// orbits of the species (all orbits for Matter, which equals imsrg++ `RSquaredOp`), with
/// b^2 = hbar^2 c^2 / (m_nucleon hbar_omega). No two-body part. Complexity O(num_orbits^2).
/// @pre `ms` not null.
op::Operator r2_one_body(const std::shared_ptr<const basis::ModelSpace>& ms, Species species);

/// Two-body r1.r2 for one species (imsrg++ `R2_2body_Op(modelspace, option)`): the normalized
/// antisymmetrized element of r1.r2 (see `r2_cm`, without the factor 2/A^2) in channels that can
/// hold the species: Proton drops Tz > 0 channels, Neutron drops Tz < 0 channels, Matter keeps
/// all. In the Tz = 0 (pn) channels of Proton/Neutron the element is multiplied by 1/2 (imsrg++
/// convention: each pn pair contributes to the radius of one species only). Zero one-body part.
/// Complexity O(sum over channels of num_kets^2 x E^3) table-driven; seconds at emax 6.
/// @pre `ms` not null.
op::Operator r2_two_body(const std::shared_ptr<const basis::ModelSpace>& ms, Species species);

/// Spin-orbit correction to the proton radius (imsrg++ `RpSpinOrbitCorrection`): a diagonal
/// one-body operator, for every orbit, of -mu_i (kappa + 1) / M^2 with M^2 = (m_nucleon / hbar c)^2
/// [fm^-2], mu_i = 2.793 - 1/2 for protons and -1.913 for neutrons (anomalous plus Dirac moment
/// as in imsrg++) and kappa = l for j = l - 1/2, -(l + 1) for j = l + 1/2. fm^2 units.
/// @pre `ms` not null.
op::Operator rp_spin_orbit_correction(const std::shared_ptr<const basis::ModelSpace>& ms);

/// Intrinsic proton radius squared (imsrg++ `Rp2_corrected_Op(ms, A, Z)`), fm^2:
/// R_cm^2 + (A-2)/(A Z) R2_1b(p) - 4/(A Z) R2_2b(p) [+ (1/Z) R_so], that is the point-proton
/// mean square radius about the center of mass; the bracketed spin-orbit term is added when
/// `include_spin_orbit` is true (the imsrg++ default and what the golden dumps contain). Returns
/// the zero operator for Z == 0 (as imsrg++). The nucleon-size corrections are not included
/// (imsrg++ applies them to the expectation value). Cost: that of `r2_cm` plus `r2_two_body`.
/// @pre `ms` not null, A >= 1, 0 <= Z <= A.
op::Operator rp2(const std::shared_ptr<const basis::ModelSpace>& ms, int A, int Z,
                 bool include_spin_orbit = true);

/// Intrinsic neutron radius squared (imsrg++ `Rn2_corrected_Op`), fm^2:
/// R_cm^2 + (A-2)/(A N) R2_1b(n) - 4/(A N) R2_2b(n), N = A - Z; no spin-orbit term. Zero operator
/// for Z == A. Cost as `rp2`.
/// @pre `ms` not null, A >= 1, 0 <= Z <= A.
op::Operator rn2(const std::shared_ptr<const basis::ModelSpace>& ms, int A, int Z);

/// Intrinsic matter radius squared (imsrg++ `Rm2_corrected_Op`), fm^2:
/// (1/A) sum_i r_i^2 - R_cm^2 (Z is not needed). Cost: that of `r2_cm`.
/// @pre `ms` not null, A >= 1.
op::Operator rm2(const std::shared_ptr<const basis::ModelSpace>& ms, int A);

}  // namespace imsrg::opslib
