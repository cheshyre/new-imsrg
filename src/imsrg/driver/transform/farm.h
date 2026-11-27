// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_TRANSFORM_FARM_H_
#define IMSRG_DRIVER_TRANSFORM_FARM_H_
/// @file
/// Work distribution of `run.task = "transform"` (design/50 C10 and section
/// 6): the work list, the cost model, the static assignment of work items to
/// ranks and the failure manifest. Pure host code without MPI.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/driver/operators/operator_request.h"

namespace imsrg::driver::transform {

/// Reads a work list: one operator name per line; leading/trailing blanks are
/// stripped, empty lines and lines starting with `#` are skipped, and a TAB
/// ends the name (so a failure manifest, `name<TAB>reason`, is a valid work
/// list). Order is kept.
/// @throws InputError if the file cannot be read.
std::vector<std::string> ReadWorkList(const std::filesystem::path& path);

/// Relative cost of ONE commutator [scalar Omega, O] for an operator O of
/// spherical rank `rank_J` and parity `parity` on `ms` (IMSRG(2), two-body
/// dominated): sum over the stored two-body blocks (ch_bra <= ch_ket,
/// |J_bra - J_ket| <= rank_J <= J_bra + J_ket, equal Tz, parity rule) of
/// n_bra n_ket (n_bra + n_ket), the multiply-adds of the block products
/// Omega_bra O + O Omega_ket of the pp/hh ladders; the particle-hole term
/// scales the same way per block pair. Scalars (rank 0, even parity) count
/// every channel once, 2 n^3. O(number of channel pairs).
double CommutatorWeight(const basis::ModelSpace& ms, std::int32_t rank_J,
                        Parity parity);

/// Measured cost of one tensor commutator relative to `CommutatorWeight`,
/// normalized to the scalar commutator: 1 for rank 0, 1.67 + 1.43 rank_J
/// for rank_J >= 1 (the tensor particle-hole recoupling grows with the rank;
/// fit of the emax-10 farm items on one GCD, design/benchmarks/al27-farm.md,
/// AL-F3).
double TensorCostFactor(std::int32_t rank_J);

/// Host preparation cost of TCM, once per item (independent of the number of
/// Omegas), in units of `CommutatorWeight` of a scalar (its two-body part is
/// built at basis.emax from the Moshinsky table; 27Al: ~750 s on 7 threads,
/// ~240x a scalar NREFT item with two Omegas). A 27Al fit (emax 14 basis,
/// emax 10 IMSRG space, one GCD with 7 host threads); refit for other spaces.
inline constexpr double kTcmPreparationCost = 500.0;

/// Cost model of one work item (design/50 section 6): the number of nested
/// commutators is num_omegas x (BCH terms per Omega), the BCH terms are taken
/// equal for all operators of a stack (they depend on ||Omega_k|| and less
/// on O; measured: rank 5 at q = 1050 MeV takes 0.4x the time of q = 150),
/// so the cost is num_omegas x `CommutatorWeight` x `TensorCostFactor` of
/// the operator's rank and parity (TCM: plus its preparation,
/// `kTcmPreparationCost` x the scalar weight). Rank and parity
/// come from the request: NREFT shorthands `F<family><L>...` have rank L and
/// even parity (the stored tensors are parity even, opslib/nreft.h); other
/// operators the catalog shape.
double ItemCost(const OperatorRequest& request, const basis::ModelSpace& ms,
                std::size_t num_omegas);

/// Static cost-weighted assignment (longest processing time first): items in
/// order of decreasing cost (ties: lower index first) each go to the rank
/// with the smallest assigned cost so far (ties: lower rank). Deterministic,
/// so every rank computes the same table without communication. Returns the
/// owning rank of every item; makespan <= 4/3 of the optimum + 1 item.
/// Precondition: num_ranks >= 1, costs >= 0.
std::vector<int> AssignStatic(std::span<const double> costs, int num_ranks);

/// A work item that did not finish.
struct Failure {
  std::string name;    ///< the operator name of the work list
  std::string reason;  ///< exception text (newlines replaced by blanks)
};

/// Writes the failure manifest: a header comment, then `name<TAB>reason`
/// per failure in the given order; an empty list gives the header only.
/// Written atomically (temporary file + rename).
/// @throws RuntimeError if it cannot be written.
void WriteFailureManifest(const std::filesystem::path& path,
                          std::span<const Failure> failures);

}  // namespace imsrg::driver::transform

#endif  // IMSRG_DRIVER_TRANSFORM_FARM_H_
