// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_DETAIL_WORKSPACE_IMPL_H_
#define IMSRG_COMMUTATOR_DETAIL_WORKSPACE_IMPL_H_

/// @file
/// Internal data of `commutator::Workspace` (no Kokkos): device tables, 6j
/// table, one-body channel CSR, two-body block offsets and the term-specific
/// precomputed data. Shared by `workspace.cc`, `commutator.cc` and the kernel
/// TUs.

#include <cstdint>
#include <memory>
#include <vector>

#include "imsrg/angmom/tables.h"
#include "imsrg/basis/device_tables.h"
#include "imsrg/commutator/detail/device_views.h"
#include "imsrg/commutator/detail/ladder.h"
#include "imsrg/commutator/detail/lifted.h"
#include "imsrg/commutator/detail/pandya.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/workspace.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator {

namespace detail {

/// The gamma-dependent one-body intermediates of the factorized double
/// commutator (factorized.h; the eta-only ones live in `FactorizedEta`):
/// dense column-major num_orbits x num_orbits device matrices. Only the
/// entries (d, e) of equal (l, 2j, 2tz) are written (each call overwrites all
/// of them); the others stay zero.
struct FactorizedScratch {
  exec::DeviceBuffer<double> chi2;  ///< chi2 = B_pp + B_hh (from gamma P eta).
  exec::DeviceBuffer<double> chib;  ///< chib (from gamma P eta).
};

}  // namespace detail

/// Internal data of a `Workspace`; built once per model space by the
/// `Workspace` constructor.
struct Workspace::Impl {
  /// Builds all tables for `ms` (non-null). Needs an initialized
  /// `exec::Runtime`.
  explicit Impl(const std::shared_ptr<const basis::ModelSpace>& ms);

  basis::DeviceTables
      tables;              ///< Device copies of orbit, ket and channel tables.
  angmom::SixJTable sixj;  ///< Dense 6j table up to the model space's emax.

  /// One-body channel CSR (see `detail::OneBodyChannelView`).
  exec::DeviceBuffer<std::int32_t> ob_offsets;  ///< num_orbits + 1 entries.
  exec::DeviceBuffer<std::int32_t> ob_orbits;   ///< CSR entries: the orbit q.
  exec::DeviceBuffer<std::int32_t>
      ob_owner;  ///< CSR entries: the orbit p owning the entry.

  /// Offset of each channel's block in the two-body operator buffer
  /// (num_channels + 1 entries, the last one the total size), as stored by
  /// `op::TwoBody`.
  exec::DeviceBuffer<std::int64_t> block_offsets;
  std::vector<std::int64_t>
      block_offsets_host;  ///< Host copy of `block_offsets`.
  std::int64_t two_body_size =
      0;  ///< Total number of stored two-body elements.

  /// For each entry of the flat `channel_kets` table, the channel it belongs
  /// to.
  exec::DeviceBuffer<std::int32_t> entry_channel;

  detail::LadderData ladder;  ///< Data of the pp/hh ladder terms (T3b).
  detail::PandyaData pandya;  ///< Data of the particle-hole term (T4).
  detail::LiftedData lifted;  ///< Lifted one-body lists of comm122ss.
  /// This rank's share of the ladder, comm122 and particle-hole work and its
  /// device lists (`detail::AssignWork`; everything owned on one rank).
  detail::WorkAssignment work;
  detail::WorkDeviceData work_device;  ///< Device copies of the owned lists.
  detail::FactorizedScratch
      factorized;  ///< One-body scratch of the factorized double commutator.

  /// View of the one-body channel CSR. O(1).
  detail::OneBodyChannelView one_body_channels() const noexcept {
    return {ob_offsets.device_data(), ob_orbits.device_data()};
  }

  /// View of the two-body part of `op`; `data == nullptr` if the part is not
  /// allocated. The operator must live on this workspace's model space. O(1).
  detail::TwoBodyView two_body_view(const op::Operator& op) const;
};

namespace detail {

/// Shared precondition check of every `terms::` function and of `Commutator()`:
/// X, Y and Z are non-empty on the workspace's model space
/// (`Workspace::Compatible`), Z is neither X nor Y, and Z's hermiticity is that
/// of [X, Y] (`CommutatorSpec`: NonHermitian if X or Y is), so the kernels may
/// derive it from X and Y alone.
/// @throws InvariantError on a violated precondition (incl. a wrong hermiticity
///         of Z). O(1).
void CheckTermInputs(const op::Operator& X, const op::Operator& Y,
                     const op::Operator& Z, const Workspace& ws);

/// True if X or Y is NonHermitian: the kernels then take their general paths
/// (explicit X P Y - Y P X products, no use of X^T = hX X); otherwise they use
/// the (anti-)Hermiticity of both. O(1).
bool IsGeneral(const op::Operator& X, const op::Operator& Y);

/// Hermiticity sign hZ = -hX hY (+1 Hermitian, -1 anti-Hermitian) of [X, Y] for
/// X, Y of definite hermiticity: [X,Y]^dagger = hX hY [Y,X] = -hX hY [X,Y].
/// Kernels use this instead of Z's flag. Precondition: `!IsGeneral(X, Y)`
/// (asserted in debug builds).
int CommutatorHermiticitySign(const op::Operator& X, const op::Operator& Y);

/// Combines the per-part results of a commutator over the ranks of
/// `ws.comm()` (D14 phase 1): Z0, Z1 and Z2 hold each part's partial sum and
/// become the sum over all parts by `Communicator::ReduceBcastSum` (reduce to
/// rank 0, then broadcast, so every rank holds rank 0's bits, M12b-Q2); Z3,
/// computed on rank 0 alone in phase 1, is broadcast from rank 0. Nothing
/// happens on a one-rank communicator (fake parts of a serial workspace stay
/// partial). Collective over `ws.comm()`: every rank calls it once per
/// commutator, in the same order. Precondition (InvariantError): on more than
/// one rank the work assignment is (comm.rank(), comm.size()). Host buffers
/// (the operator is synced to the host and back). Complexity: O(size of Z)
/// communication per rank, independent of the number of ranks.
void CombinePartials(op::Operator& Z, const Workspace& ws);

}  // namespace detail

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_DETAIL_WORKSPACE_IMPL_H_
