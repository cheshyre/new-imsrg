// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_TRIPLES_H_
#define IMSRG_COMMUTATOR_TRIPLES_H_

/// @file
/// The perturbative-triples energy on a dedicated, reduced context (plan
/// section 3.11, D49, M10-Q46).
///
/// `PerturbativeTriplesEnergy(X, Y, epsilon, const ThreeBodyContext&)`
/// (three_body.h) needs the full kept three-body space of the model space (E
/// and R maps of every triple, the comm232 tables, ...), which is far too large
/// at emax >= 8. The triples read only
///  * the channels (2J, parity, 2Tz) that hold a kept triple with an
///    occupation product n_a n_b n_c >= 1e-8 (hhh type): in any other channel
///    every tile factor n_T nbar_T' - n_T' nbar_T is below imsrg++'s skip
///    1e-8 (`detail::kTriplesSkip`), so it contributes nothing;
///  * within them, the triples with n_T >= 1e-8 or nbar_T >= 1e-8 (hhh or ppp
///    type; the same test as the kernel's skip of a bra triple, and a ket
///    triple outside it has a factor below the skip as well);
///  * for each of those triples its kept rows (the kets that pass E3max,
///    emax_3body and the dE3max / OccNat3 cuts, applied at the enumeration,
///    owner decision D34), its pair-third states and its R block.
/// `TriplesContext` enumerates exactly these (no global ket list, no E map, no
/// comm232 tables, no three-body operator) and calls the same tile kernel as
/// the full context (`detail::RunTriplesTiles`), so the energy agrees with the
/// full-context energy up to the order of the parallel reduction.
///
/// Host-side construction; the tables live in `exec::DeviceBuffer`s, so an
/// initialized `exec::Runtime` is required. No Kokkos in this header.

#include <cstdint>
#include <memory>
#include <span>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/three_body_channels.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator {

/// Sizes of a `TriplesContext` (the reduced space the triples tiles run on).
struct TriplesCounts {
  /// Channels holding a kept hhh triple (the channels of the context).
  std::int64_t channels = 0;
  /// Kept (channel, orbit triple) pairs of the context (hhh or ppp type).
  std::int64_t triples = 0;
  /// Of them with n_T >= 1e-8 (hhh type; a triple may be both for a
  /// fractional reference).
  std::int64_t triples_hhh = 0;
  /// Of them with nbar_T >= 1e-8 (ppp type).
  std::int64_t triples_ppp = 0;
  /// Kept three-body rows (kets) of those triples.
  std::int64_t rows = 0;
  /// Kets the same triples would have without the dE3max and OccNat3 cuts
  /// (the E3max and emax_3body limits still apply).
  std::int64_t rows_before_cuts = 0;
  /// Pair-third states of the kept triples.
  std::int64_t states = 0;
};

/// Device tables of the perturbative-triples tiles for one model space and one
/// set of three-body cuts: the 3B 6j table, the reduced kept space (see the
/// file comment) with its pair-third states and the recoupling map R only,
/// the orbit groups and the leg gather tables, and the work order of the bra
/// triples. Built once; read-only afterwards (D10), so concurrent use of a
/// const context is safe. Needs an initialized `exec::Runtime`; destroy it
/// before the Runtime. Movable, not copyable.
class TriplesContext {
 public:
  /// Enumerates the reduced space and builds the tables for `ms`, whose
  /// reference occupations select the hhh/ppp triples (a new reference needs
  /// a new context). `ms` needs no three-body space. `cuts` as for
  /// `ModelSpace::WithThreeBody`: `e3max`, `emax_3body` (< 0: the basis
  /// emax), `de3max`, `occnat3_cut` and `natural_occupations` (empty: the
  /// reference's), so the kept rows of a channel are those of the full space
  /// restricted to the kept triples.
  /// @throws InputError if `ms` is null.
  /// Cost: the 6j table (O(emax^6) symbols), the enumeration (O(orbits^3 +
  /// kept triples x J range)), the pair-third states and R (O(states x J
  /// range)), and the leg tables (O(orbits^2 emax)).
  TriplesContext(std::shared_ptr<const basis::ModelSpace> ms,
                 const basis::ThreeBodyCuts& cuts);

  TriplesContext(TriplesContext&&) noexcept;
  TriplesContext& operator=(TriplesContext&&) noexcept;
  TriplesContext(const TriplesContext&) = delete;
  TriplesContext& operator=(const TriplesContext&) = delete;
  ~TriplesContext();

  /// The model space the tables were built for.
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr()
      const noexcept {
    return ms_;
  }
  /// The cuts used, with `emax_3body` resolved.
  const basis::ThreeBodyCuts& cuts() const noexcept { return cuts_; }
  /// The sizes of the reduced space. O(1).
  const TriplesCounts& counts() const noexcept { return counts_; }

  /// Bytes of the device tables owned by the context (6j table, pair-third
  /// states with R, reduced-space tables, orbit groups, leg gather tables,
  /// work order), for reports. 0 for a moved-from context.
  std::int64_t TableBytes() const noexcept;

  /// Bytes of team scratch one team of the tile kernel requests, for reports.
  /// 0 for a moved-from context.
  std::int64_t ScratchBytesPerTeam() const noexcept;

  /// Multiply-adds of the tiles the triples evaluate: over the tiles (T, T'),
  /// T' < T in a channel with a nonzero occupation factor, the sum over the
  /// pair-third states r of T and s of T' of the number of internal orbits a
  /// that couple (the group of a, and the triangles of the 6j symbol
  /// {j_w j_a J_r; j_z J J_s}), i.e. the work of W; the epilogue R W R^T is not
  /// included. Host computation, O(sum over tiles of m m'): use it for
  /// reports, not in loops. Returns 0 for a moved-from context.
  double TileMultiplyAdds() const;

  /// Internal tables (defined in triples.cc).
  struct Impl;
  /// Internal tables, for the functions of this header. Precondition: not
  /// moved from (IMSRG_CHECK).
  const Impl& impl() const;

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  basis::ThreeBodyCuts cuts_;
  TriplesCounts counts_;
  std::unique_ptr<Impl> impl_;
};

/// The perturbative-triples energy of `PerturbativeTriplesEnergy(X, Y, epsilon,
/// const ThreeBodyContext&)` (formula and conventions in three_body.h), on the
/// reduced `TriplesContext`: the same tile kernel on the hhh/ppp triples of
/// the channels with an hhh triple. The result agrees with the full-context
/// energy (same cuts) up to the order of the parallel reduction (relative
/// 1e-14); the sum is a parallel reduction, so the last digits depend on the
/// thread count.
/// @throws InputError if `epsilon.size()` is not the number of orbits, if X or
///         Y is empty or has no two-body part.
/// @throws NotImplementedError for non-scalar or non-number-conserving X or Y.
/// @throws InvariantError if X or Y lives on a model space whose basis,
///         reference or partition differs from the context's, or the context
///         was moved from.
/// Complexity: `TriplesContext::TileMultiplyAdds()` multiply-adds for W plus
/// the gathers O((m + m') n_a) and the epilogue per tile; one team per bra
/// triple.
double PerturbativeTriplesEnergy(const op::Operator& X, const op::Operator& Y,
                                 std::span<const double> epsilon,
                                 const TriplesContext& ctx);

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_TRIPLES_H_
