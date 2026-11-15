// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef IMSRG_BASE_CONSTANTS_H_
#define IMSRG_BASE_CONSTANTS_H_
/// @file
/// Physical and mathematical constants, copied literally from imsrg++
/// (`PhysicalConstants.hh`) so that numerical results can be reproduced bit for
/// bit. Each constant cites its line in the reference file and its unit.
/// Energies are in MeV, lengths in fm unless noted.

namespace imsrg::constants {

/// Reduced Planck constant times c, in MeV fm. PhysicalConstants.hh:13.
inline constexpr double kHbarc = 197.3269718;
/// Proton mass in MeV/c^2. PhysicalConstants.hh:14.
inline constexpr double kMProton = 938.2720813;
/// Neutron mass in MeV/c^2. PhysicalConstants.hh:15.
inline constexpr double kMNeutron = 939.5654133;
/// Electron mass in MeV/c^2. PhysicalConstants.hh:16.
inline constexpr double kMElectron = 0.5109989461;
/// Average nucleon mass in MeV/c^2, (m_proton + m_neutron) / 2.0 (same
/// expression as imsrg++). PhysicalConstants.hh:17.
inline constexpr double kMNucleon = (kMProton + kMNeutron) / 2.0;
/// Charged pion mass in MeV/c^2. PhysicalConstants.hh:18.
inline constexpr double kMPionCharged = 139.57018;
/// Neutral pion mass in MeV/c^2. PhysicalConstants.hh:19.
inline constexpr double kMPionNeutral = 134.9766;
/// Nucleon vector g factor (1 by CVC), dimensionless. PhysicalConstants.hh:20.
inline constexpr double kNucleonVectorG = 1.00;
/// Nucleon axial g factor gA, dimensionless. PhysicalConstants.hh:21.
inline constexpr double kNucleonAxialG = 1.27;
/// Proton spin g factor (PDG 2020), dimensionless. PhysicalConstants.hh:22.
inline constexpr double kProtonSpinG = 5.585690569;
/// Neutron spin g factor (PDG 2020), dimensionless. PhysicalConstants.hh:23.
inline constexpr double kNeutronSpinG = -3.826085;
/// Electron spin g factor, dimensionless. PhysicalConstants.hh:24.
inline constexpr double kElectronSpinG = 2.002319;
/// Fine-structure constant, 1.0 / 137.035999, dimensionless.
/// PhysicalConstants.hh:25.
inline constexpr double kAlphaFs = 1.0 / 137.035999;
/// Pion decay constant in MeV. PhysicalConstants.hh:26.
inline constexpr double kFPi = 92.2;
/// One Hartree in eV. PhysicalConstants.hh:27.
inline constexpr double kHartree = 27.21138602;
/// Proton charge radius squared in fm^2 (PDG 2020, Rp = 0.8409 fm).
/// PhysicalConstants.hh:29.
inline constexpr double kProtonRch2 = 0.707;
/// Neutron charge radius squared in fm^2 (PDG 2020). PhysicalConstants.hh:31.
inline constexpr double kNeutronRch2 = -0.1161;
/// Darwin-Foldy relativistic correction to charge radii in fm^2.
/// PhysicalConstants.hh:33.
inline constexpr double kDarwinFoldy = 0.033172;

/// Square root of 2. PhysicalConstants.hh:41.
inline constexpr double kSqrt2 = 1.41421356237309505;
/// 1 / sqrt(2) as written in imsrg++: 0.7071067811865 is TRUNCATED to 13 digits
/// (the exact value is 0.70710678118654752...). Kept only to reproduce
/// reference numerics; new code should use `1.0 / sqrt2`.
/// PhysicalConstants.hh:42.
inline constexpr double kInvSqrt2 = 0.7071067811865;
/// Natural logarithm of 2 (reference name LOG2). PhysicalConstants.hh:43.
inline constexpr double kLn2 = 0.69314718055994530942;
/// Pi. PhysicalConstants.hh:44.
inline constexpr double kPi = 3.14159265358979324;
/// Square root of pi. PhysicalConstants.hh:45.
inline constexpr double kSqrtPi = 1.7724538509055160;

}  // namespace imsrg::constants

#endif  // IMSRG_BASE_CONSTANTS_H_
