#pragma once
/// @file
/// Wigner 3j, 6j and 9j symbols and Clebsch-Gordan coefficients (design/10 section 2, layer 1).
///
/// Replaces GSL's `gsl_sf_coupling_{3j,6j,9j}` as used by imsrg++ (`AngMom.cc:107-135`, see
/// design/reference/01 section 6.1 and design/reference/05 section 4.4). All angular momenta and
/// projections are passed doubled (`TwoJ{3}` is j = 3/2), so there is no conversion from
/// floating point and no `int(2*j)` truncation.
///
/// Conventions: standard Wigner symbols (Edmonds); Clebsch-Gordan coefficients in the
/// Condon-Shortley phase convention, `<j1 m1 j2 m2|J M> = (-1)^(j1-j2+M) sqrt(2J+1)
/// (j1 j2 J; m1 m2 -M)`, as imsrg++ (`AngMom.cc:114-117`).
///
/// Every function returns exactly 0.0 for arguments that do not form a valid symbol (a triangle
/// condition fails, |m| > j, j and m of different integrality, projections that do not sum
/// correctly, or a negative 2j). Evaluation is host-only, in double precision with Racah's
/// single-sum formulas over a factorial table. Accuracy, validated against sympy's exact values
/// for every argument needed up to emax 8 (tests/unit/angmom): |error| <= max(1e-12 |x|, 1e-14)
/// (typically ~1e-16 absolute; ~1e-14 for <l1 0 l2 0|L 0> with l <= 16); it degrades for larger
/// j (~1e-11 at l ~ 30). Arguments whose factorials would exceed 170! give 0. Kernels never
/// evaluate symbols: they read precomputed tables (design/12 section 4.4).

#include "imsrg/base/device.hpp"
#include "imsrg/base/types.hpp"

namespace imsrg::angmom {

/// True if a, b, c (doubled) can couple: |a - b| <= c <= a + b and a + b + c even. O(1).
IMSRG_INLINE_FUNCTION constexpr bool triangle(TwoJ a, TwoJ b, TwoJ c) noexcept {
  const auto d = a.twice - b.twice;
  return a.twice >= 0 && b.twice >= 0 && c.twice >= 0 && (a.twice + b.twice + c.twice) % 2 == 0 &&
         (d < 0 ? -d : d) <= c.twice && c.twice <= a.twice + b.twice;
}

/// Wigner 3j symbol (j1 j2 j3; m1 m2 m3). Zero unless m1 + m2 + m3 = 0, |mi| <= ji, ji - mi
/// integer, and triangle(j1, j2, j3). O(min(j)) operations.
double wigner_3j(TwoJ j1, TwoJ j2, TwoJ j3, TwoM m1, TwoM m2, TwoM m3) noexcept;

/// Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (Condon-Shortley). Zero unless M = m1 + m2
/// and the corresponding 3j symbol is valid.
double clebsch_gordan(TwoJ j1, TwoM m1, TwoJ j2, TwoM m2, TwoJ J, TwoM M) noexcept;

/// Wigner 6j symbol {j1 j2 j3; j4 j5 j6}. Zero unless the four triads (j1 j2 j3), (j1 j5 j6),
/// (j4 j2 j6), (j4 j5 j3) satisfy the triangle condition. O(min(j)) operations.
double wigner_6j(TwoJ j1, TwoJ j2, TwoJ j3, TwoJ j4, TwoJ j5, TwoJ j6) noexcept;

/// Wigner 9j symbol {j11 j12 j13; j21 j22 j23; j31 j32 j33} (rows as written). Zero unless every
/// row and column triad satisfies the triangle condition. Computed as the standard sum over x of
/// (-1)^(2x) (2x+1) {j11 j21 j31; j32 j33 x} {j12 j22 j32; j21 x j23} {j13 j23 j33; x j11 j12}.
double wigner_9j(TwoJ j11, TwoJ j12, TwoJ j13, TwoJ j21, TwoJ j22, TwoJ j23, TwoJ j31, TwoJ j32,
                 TwoJ j33) noexcept;

/// Normalized 9j symbol: sqrt((2 j13 + 1)(2 j23 + 1)(2 j31 + 1)(2 j32 + 1)) times the 9j symbol,
/// as imsrg++ `AngMom::NormNineJ` (`AngMom.cc:132-135`); e.g. the LS-jj recoupling coefficient
/// <(la lb) L, (sa sb) S; J | (la sa) ja, (lb sb) jb; J> = norm_9j(la, sa, ja, lb, sb, jb, L, S,
/// J).
double norm_9j(TwoJ j11, TwoJ j12, TwoJ j13, TwoJ j21, TwoJ j22, TwoJ j23, TwoJ j31, TwoJ j32,
               TwoJ j33) noexcept;

}  // namespace imsrg::angmom
