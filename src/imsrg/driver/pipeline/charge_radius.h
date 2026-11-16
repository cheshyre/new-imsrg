// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_PIPELINE_CHARGE_RADIUS_H_
#define IMSRG_DRIVER_PIPELINE_CHARGE_RADIUS_H_
/// @file
/// The point-proton to charge-radius conversion of the output stage (owner
/// decision D35, design/13 section 8, design/16 section 2.10; imsrg++.cc:845
/// and :1367 with the constants of PhysicalConstants.hh:29-33).
#include <optional>

namespace imsrg::driver::pipeline {

/// Mean square charge radius in fm^2 from the zero-body expectation values of
/// the intrinsic point-proton radius operator `Rp2` and the spin-orbit
/// correction `Rso` (catalog operators of the same names, both fm^2):
///
///   Rch2 = Rp2 + Rso / Z + r_p^2 + (N / Z) r_n^2 + Darwin-Foldy,
///
/// with N = A - Z, r_p^2 = 0.707 fm^2, r_n^2 = -0.1161 fm^2 (PDG 2020) and the
/// Darwin-Foldy term 0.033172 fm^2 (`imsrg::constants`, copied exactly from
/// imsrg++). imsrg++'s Rp2 contained the Rso/Z term (D35 moved it into its own
/// operator), so this equals imsrg++'s `Rp2 + PROTON_RCH2 + NEUTRON_RCH2 *
/// (A - Z) / Z + DARWIN_FOLDY` with the old Rp2. Unlike imsrg++'s HF printout
/// there is no special case for |Rp2| < 1e-6.
/// @param rp2 expectation value of `Rp2`, fm^2 (HF value or after the flow).
/// @param rso expectation value of `Rso`, fm^2, in the same state as `rp2`.
/// @param mass_number A, the mass number used to build Rp2.
/// @param proton_number Z, the proton number used to build Rp2.
/// @pre `0 < proton_number <= mass_number` (checked, throws InvariantError).
double ChargeRadiusSquared(double rp2, double rso, int mass_number,
                           int proton_number);

/// The charge radius sqrt(Rch2) in fm, or `std::nullopt` if `rch2` is negative
/// or not finite (a result that left the physical regime).
std::optional<double> ChargeRadius(double rch2);

}  // namespace imsrg::driver::pipeline

#endif  // IMSRG_DRIVER_PIPELINE_CHARGE_RADIUS_H_
