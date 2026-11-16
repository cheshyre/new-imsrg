// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_FLOW_TRIPLES_H_
#define IMSRG_FLOW_TRIPLES_H_
/// @file
/// The perturbative-triples correction after an IMSRG(2) Magnus flow (imsrg++
/// `IMSRGSolver::CalculatePerturbativeTriples`, IMSRGSolver.cc:1172-1214,
/// `perturbative_triples=true`): dE = [Omega, Htilde]_3-body contracted with
/// itself over the energy denominators of the flowing Hamiltonian. The
/// combination of the Omega stack, the shifted BCH series of Htilde, the
/// denominators and the 3B space are numerics-defining (design/00 D12).
#include <cstdint>
#include <limits>

#include "imsrg/flow/magnus.h"

namespace imsrg::flow {

/// Truncations of the three-body space of the triples (imsrg++ `dE3max`,
/// `OccNat3Cut`; the triples always use E3max = 3 emax).
struct TriplesOptions {
  /// Largest sum |e_i - e_fermi| of a kept 3B ket; infinity = no cut.
  double de3max = std::numeric_limits<double>::infinity();
  /// Smallest prod n_i (1 - n_i) of a kept ket; negative = no cut.
  double occnat3_cut = -1.0;
};

/// Result of `PerturbativeTriples`.
struct TriplesResult {
  double de = 0.0;              ///< Energy correction dE (MeV).
  std::int64_t kept_rows = 0;   ///< 3B rows (kets) kept by the cuts.
  std::int64_t total_kets = 0;  ///< 3B kets before the dE3max/OccNat3 cuts.
};

/// dE of the triples for the state of `solver` after `Solve` (imsrg++
/// `CalculatePerturbativeTriples`):
///   omega = Omega_0; omega = BchProduct(omega, Omega_n) for n = 1, 2, ...
///     (imsrg++'s operand order, design/QUESTIONS.md M10-Q40);
///   Htilde = Transform(H_0) with `BchOptions::skip_first_commutator`;
///   epsilon_a = diagonal one-body matrix element a a of H(s);
///   dE = `commutator::PerturbativeTriplesEnergy(omega, Htilde, epsilon)` on
///   the model space with three-body channels of E3max = 3 emax and the cuts
///   of `options`.
/// Does not change `solver`. Preconditions: Magnus solver after at least one
/// `Solve`, an IMSRG(2) Hamiltonian (no 3B part), scalar number-conserving
/// operators, an initialized exec::Runtime. Complexity: the Omega product and
/// transform (a few dozen commutators), the 3B space enumeration, and the
/// triples kernel.
TriplesResult PerturbativeTriples(MagnusSolver& solver,
                                  const TriplesOptions& options = {});

}  // namespace imsrg::flow

#endif  // IMSRG_FLOW_TRIPLES_H_
