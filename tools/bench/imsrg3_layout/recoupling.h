// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_RECOUPLING_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_RECOUPLING_H_
/// @file
/// Device-callable ports of imsrg++'s three-body permutation and recoupling
/// helpers (ThreeBodyStorage.cc:66-140; design/reference/01 section 5.2).
///
/// Conventions (identical to imsrg++, so canonical matrix elements compare
/// element by element):
/// * a three-body ket |(ab) Jab, c; J> couples orbits a and b to Jab, then c;
///   canonical kets have a <= b <= c (orbit indices);
/// * `Permute(p, a, b, c)` reorders (a, b, c) as named by `p` (CBA gives (c,
///   b, a));
/// * `SortOrbits(a_in, b_in, c_in)` sorts to a <= b <= c and returns the
///   permutation p with (a_in, b_in, c_in) = Permute(p, a, b, c);
/// * |(a_in b_in) Jab_in, c_in; J> = PermutationPhase(p) * sum_Jab
///   RecouplingCoefficient(p, ja, jb, jc, Jab_in, Jab, 2J) |(ab) Jab, c; J>
///   for antisymmetrized states, with (ja, jb, jc) the j of the SORTED orbits
///   (ThreeBodyStorage::GetKetIndex_withRecoupling, ThreeBodyStorage.cc:236).

#include <cstdint>

#include "imsrg/base/device.h"
#include "imsrg/base/types.h"
#include "tools/bench/imsrg3_layout/sixj3.h"

namespace imsrg::layout3 {

/// The six orderings of three orbits (imsrg++ `ThreeBodyStorage::Permutation`).
enum class Perm : std::int8_t { kABC, kBCA, kCAB, kACB, kBAC, kCBA };

/// (-1)^n for any integer n.
IMSRG_INLINE_FUNCTION constexpr int Phase(int n) noexcept {
  return (n & 1) == 0 ? 1 : -1;
}

/// Triangle rule for doubled angular momenta: |a - b| <= c <= a + b and a + b
/// + c even.
IMSRG_INLINE_FUNCTION constexpr bool Triangle2(int two_a, int two_b,
                                               int two_c) noexcept {
  const int d = two_a > two_b ? two_a - two_b : two_b - two_a;
  return two_c >= d && two_c <= two_a + two_b &&
         ((two_a + two_b + two_c) & 1) == 0;
}

/// +1 for the even permutations ABC, BCA, CAB and -1 for the odd ones
/// (ThreeBodyStorage.cc:123-126).
IMSRG_INLINE_FUNCTION constexpr int PermutationPhase(Perm p) noexcept {
  return (p == Perm::kABC || p == Perm::kBCA || p == Perm::kCAB) ? 1 : -1;
}

/// (o1, o2, o3) = (a, b, c) reordered by `p` (ThreeBodyStorage::Permute).
IMSRG_INLINE_FUNCTION constexpr void Permute(Perm p, int a, int b, int c,
                                             int& o1, int& o2,
                                             int& o3) noexcept {
  switch (p) {
    case Perm::kABC:
      o1 = a;
      o2 = b;
      o3 = c;
      break;
    case Perm::kBAC:
      o1 = b;
      o2 = a;
      o3 = c;
      break;
    case Perm::kBCA:
      o1 = b;
      o2 = c;
      o3 = a;
      break;
    case Perm::kCAB:
      o1 = c;
      o2 = a;
      o3 = b;
      break;
    case Perm::kACB:
      o1 = a;
      o2 = c;
      o3 = b;
      break;
    case Perm::kCBA:
      o1 = c;
      o2 = b;
      o3 = a;
      break;
  }
}

/// Sorts (a_in, b_in, c_in) into a <= b <= c and returns the permutation p
/// with (a_in, b_in, c_in) = Permute(p, a, b, c) (ThreeBodyStorage.cc:66-85,
/// including its choice among equal orbits).
IMSRG_INLINE_FUNCTION constexpr Perm SortOrbits(int a_in, int b_in, int c_in,
                                                int& a, int& b,
                                                int& c) noexcept {
  a = a_in;
  b = b_in;
  c = c_in;
  int t = 0;
  if (a > b) {
    t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    t = b;
    b = c;
    c = t;
  }
  if (a > b) {
    t = a;
    a = b;
    b = t;
  }
  if (a_in == a) return (b_in == b) ? Perm::kABC : Perm::kACB;
  if (a_in == b) return (b_in == a) ? Perm::kBAC : Perm::kBCA;
  return (b_in == a) ? Perm::kCAB : Perm::kCBA;
}

/// imsrg++ `ThreeBodyStorage::RecouplingCoefficient(p, ja, jb, jc, Jab_in,
/// Jab, twoJ)` (ThreeBodyStorage.cc:97-120) WITHOUT the permutation phase:
///
///   ABC: delta(Jab, Jab_in)            BAC: delta * (-1)^(ja + jb - Jab)
///   BCA: (-1)^(jb + jc + Jab_in + 1)    hh' {ja jb Jab; jc J Jab_in}
///   CAB: (-1)^(ja + jb - Jab + 1)       hh' {jb ja Jab; jc J Jab_in}
///   ACB: (-1)^(jb + jc + Jab_in - Jab)  hh' {jb ja Jab; jc J Jab_in}
///   CBA: -1                             hh' {ja jb Jab; jc J Jab_in}
///
/// with hh' = sqrt((2 Jab_in + 1)(2 Jab + 1)). `Jab_in` and `Jab` are integer
/// two-body J (not doubled), `two_J` is the doubled three-body J. Returns 0.0
/// unless triangle(ja, jb, Jab) and triangle(jc, J, Jab) hold, and for BCA/CBA
/// triangle(ja, J, Jab_in) and triangle(jc, jb, Jab_in), for CAB/ACB
/// triangle(jb, J, Jab_in) and triangle(jc, ja, Jab_in). O(1).
IMSRG_INLINE_FUNCTION double RecouplingCoefficient(
    Perm p, TwoJ ja, TwoJ jb, TwoJ jc, int Jab_in, int Jab, TwoJ two_J,
    const SixJ3View& sixj) noexcept {
  const int a2 = ja.twice;
  const int b2 = jb.twice;
  const int c2 = jc.twice;
  const int J2 = two_J.twice;
  if (!(Triangle2(a2, b2, 2 * Jab) && Triangle2(c2, J2, 2 * Jab))) return 0.0;
  if (p == Perm::kBCA || p == Perm::kCBA) {
    if (!(Triangle2(a2, J2, 2 * Jab_in) && Triangle2(c2, b2, 2 * Jab_in))) {
      return 0.0;
    }
  } else if (p == Perm::kCAB || p == Perm::kACB) {
    if (!(Triangle2(b2, J2, 2 * Jab_in) && Triangle2(c2, a2, 2 * Jab_in))) {
      return 0.0;
    }
  }
  switch (p) {
    case Perm::kABC:
      return Jab == Jab_in ? 1.0 : 0.0;
    case Perm::kBAC:
      return Jab == Jab_in ? Phase((a2 + b2) / 2 - Jab) : 0.0;
    default:
      break;
  }
  const double hats = sixj.Hat(Jab_in) * sixj.Hat(Jab);
  switch (p) {
    case Perm::kBCA:
      return Phase((b2 + c2) / 2 + Jab_in + 1) * hats *
             sixj(ja, jb, TwoJ{2 * Jab}, jc, two_J, TwoJ{2 * Jab_in});
    case Perm::kCAB:
      return Phase((a2 + b2) / 2 - Jab + 1) * hats *
             sixj(jb, ja, TwoJ{2 * Jab}, jc, two_J, TwoJ{2 * Jab_in});
    case Perm::kACB:
      return Phase((b2 + c2) / 2 + Jab_in - Jab) * hats *
             sixj(jb, ja, TwoJ{2 * Jab}, jc, two_J, TwoJ{2 * Jab_in});
    case Perm::kCBA:
      return -hats * sixj(ja, jb, TwoJ{2 * Jab}, jc, two_J, TwoJ{2 * Jab_in});
    default:
      return 0.0;
  }
}

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_RECOUPLING_H_
