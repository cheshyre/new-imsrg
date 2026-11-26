// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_OPSLIB_DETAIL_NREFT_RADIAL_H_
#define IMSRG_OPSLIB_DETAIL_NREFT_RADIAL_H_
/// @file
/// Harmonic-oscillator radial integrals of the dark-matter NREFT operators
/// (port of D. Gazda's Fortran as in the owner's imsrg++ fork,
/// DarkMatterNREFT.cc:60-231 at 79e9330; reference/06 section 2.2).
///
/// All functions take the radial quantum numbers n >= 0 (number of nodes, the
/// imsrg++ convention), orbital angular momenta l, the multipole L and
/// y = (q b / 2)^2 with b the oscillator length. They follow the fork's
/// formulas term by term (same summation order), so that the tables agree
/// with the fork to rounding. Host-only.

#include <cstdint>

namespace imsrg::opslib::detail {

/// Gamma(twice_x / 2) for a positive integer twice_x: (x-1)! for integer x,
/// sqrt(pi) (2x-2)!! / 2^(x-1/2) for half-integer x. Precondition:
/// 1 <= twice_x <= 340.
double GammaOfHalf(std::int32_t twice_x);

/// n! as a double (exact for n <= 22). Precondition: 0 <= n <= 170.
double Factorial(std::int32_t n);

/// n!! as a double (exact for n <= 29). Precondition: -1 <= n <= 300.
double DoubleFactorial(std::int32_t n);

/// Confluent hypergeometric function 1F1(a; c; y) for an integer a, c > 0
/// and y >= 0: a finite polynomial for a <= 0, the (positive-term) Kummer
/// series for a > 0 (summed until the terms drop below 1e-17 of the sum).
double Hyp1F1(std::int32_t a, double c, double y);

/// <na la| j_L(q r) |nb lb> for oscillator functions (fork `jho`).
/// Precondition: la + lb + L even (otherwise the fork multiplies it by a
/// vanishing 3j symbol; this port then has a half-integer first argument of
/// 1F1 and is not defined).
double Jho(std::int32_t na, std::int32_t la, std::int32_t nb, std::int32_t lb,
           std::int32_t L, double y);

/// The fork's `jdmho`: <na la| j_L(q r) (d/dr - ...)/q |nb lb> combination of
/// Gazda's jdmHO. Precondition: la + lb + L odd. y = 0 with L = 0 gives
/// inf/NaN, as in the fork.
double Jdmho(std::int32_t na, std::int32_t la, std::int32_t nb, std::int32_t lb,
             std::int32_t L, double y);

/// The fork's `jdpho` (Gazda's jdpHO). Preconditions as `Jdmho`.
double Jdpho(std::int32_t na, std::int32_t la, std::int32_t nb, std::int32_t lb,
             std::int32_t L, double y);

}  // namespace imsrg::opslib::detail

#endif  // IMSRG_OPSLIB_DETAIL_NREFT_RADIAL_H_
