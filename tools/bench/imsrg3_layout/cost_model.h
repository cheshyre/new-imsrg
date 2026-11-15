// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#ifndef TOOLS_BENCH_IMSRG3_LAYOUT_COST_MODEL_H_
#define TOOLS_BENCH_IMSRG3_LAYOUT_COST_MODEL_H_
/// @file
/// Operation-count models of the [2,2]->3 kernels of the layout benchmark
/// (benchmark prototype, design/14 section 2). They are computed from the
/// canonical space metadata alone (no operators, no 6j table), so they run at
/// emax 10-12 where the kernels cannot.

#include "tools/bench/imsrg3_layout/canonical_space.h"

namespace imsrg::layout3 {

/// Operation counts summed over every channel and every triple pair (t, t' <=
/// t) of the channel. All are numbers of multiply-adds (doubles, because they
/// exceed 2^63 at large emax).
struct CostModels {
  /// W work of K1: sum over (key k of t, key k' of t') of n_J(k) n_J(k')
  /// n_g(k, k'), where a key is a distinct (pair, third) splitting of a triple
  /// among the perms ABC, CBA, ACB (pair_third.h), n_J(k) the number of Jpq
  /// that pass Pauli and triangle(Jpq, j_third, J), and n_g the number of
  /// orbits a of the parity (l_p + l_q + l_w) % 2 and 2tz = tz2_p + tz2_q -
  /// tz2_w (0 unless +-1), (p, q) the pair of k and w the third of k'.
  double w_cost = 0.0;
  /// Work of K0 (imsrg++ comm223ss loop structure): sum over the perms p1 of
  /// t and p2 of t' among ABC, CBA, ACB of n_t n_t' nJ'(p1) nJ'(p2) n_g, with
  /// n_t the canonical rows of t, nJ'(ABC) = 1, nJ'(p) the J' range of the
  /// permuted pair, and n_g from the orbits (I1, I2, I6) as in K0.
  double k0_cost = 0.0;
  /// Dense-per-triple epilogue L_T W L_T'^T: sum over pairs of m_t nL_t' m_t'
  /// + nL_t m_t nL_t', with m the pair-third states of a triple and nL = n_t
  /// (layout A), n_t (layout B, approximation), m_t (layout C).
  double epi_cost_a = 0.0;
  double epi_cost_b = 0.0;  ///< See `epi_cost_a`.
  double epi_cost_c = 0.0;  ///< See `epi_cost_a`.
};

/// Computes the cost models of `space`. The pair sums are factorized over the
/// orbit classes (pair parity and tz, third parity and tz) and accumulated
/// with running sums, so the cost is O(number of triples) after the key
/// enumeration, not O(pairs).
CostModels ComputeCostModels(const CanonicalSpace& space);

}  // namespace imsrg::layout3

#endif  // TOOLS_BENCH_IMSRG3_LAYOUT_COST_MODEL_H_
