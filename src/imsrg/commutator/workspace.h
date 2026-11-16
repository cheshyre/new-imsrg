// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_COMMUTATOR_WORKSPACE_H_
#define IMSRG_COMMUTATOR_WORKSPACE_H_

/// @file
/// `Workspace`: the per-model-space precomputed data and preallocated scratch
/// buffers of the commutator kernels (design/10 section 3.4, design/12 section
/// 4).

#include <memory>

#include "imsrg/basis/model_space.h"
#include "imsrg/comm/communicator.h"
#include "imsrg/operator/operator.h"

namespace imsrg::commutator {

class ThreeBodyContext;

/// Construction options of a `Workspace`.
struct WorkspaceOptions {
  /// The ranks that share the commutator work (design/12 section 6, D14 phase
  /// 1): every rank holds the full one- and two-body operators, and a
  /// workspace on more than one rank lets the expensive terms compute this
  /// rank's share, which `Commutator()` combines (a collective over `comm`).
  /// The default, the serial communicator, distributes nothing.
  comm::Communicator comm;
};

/// Everything the IMSRG(2) kernels need besides X, Y and Z, built once per
/// model space and reused by every commutator call on that model space (a flow
/// performs thousands):
/// * device copies of the basis index tables (`basis::DeviceTables`) and the
///   dense Pandya 6j table (`angmom::SixJTable`, design/12 section 4.4);
/// * per-channel ket lists with occupation weights for the ladder (pp/hh)
///   GEMMs;
/// * the particle-hole (Pandya) recoupling data of every cross-coupled channel;
/// * scratch buffers (gathered GEMM operands, ladder intermediates,
///   Pandya-transformed blocks), allocated in the constructor so the flow loop
///   does not allocate.
///
/// Nothing in it is a lazy cache (design/00 D10): construction does all the
/// work, afterwards the precomputed data are read-only. The occupations are
/// part of the model space, so a new reference (e.g. after Hartree-Fock) needs
/// a new Workspace.
///
/// Thread safety: the scratch buffers are mutable, so a Workspace serves one
/// `commutator` call at a time; give concurrent calls their own Workspaces.
/// Needs an initialized `exec::Runtime`; destroy it before the Runtime.
/// Movable, not copyable.
class Workspace {
 public:
  /// Builds the precomputed data and scratch buffers for `ms`. Cost: dominated
  /// by the 6j table (O((emax+1)^4 (2 emax+2)^2)) and the scratch allocation.
  /// Scratch footprint (M12a, M7-Q4): the ladder keeps about 2 (measured 2.0x
  /// at O16 emax 4 and 6: the pp operands of closed-shell channels are read in
  /// place, not gathered; 3.5x and 3.8x before) and the Pandya term about 3.5
  /// to 4.5 (4.3x at emax 4, 3.5x at emax 6: no ZB buffer and half of YK; 8.0x
  /// and 6.6x before) two-body operators of memory; the exact bytes are printed
  /// by the `commutator_robustness.scratch_footprint` test. `opts.comm` is
  /// kept (see `WorkspaceOptions`); construction itself is not collective.
  /// @throws InvariantError if `ms` is null.
  explicit Workspace(std::shared_ptr<const basis::ModelSpace> ms,
                     const WorkspaceOptions& opts = {});

  ~Workspace();
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  /// The model space the workspace was built for.
  const basis::ModelSpace& model_space() const noexcept { return *ms_; }

  /// The shared model space.
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr()
      const noexcept {
    return ms_;
  }

  /// True if `op` lives on this workspace's model space (the same object or a
  /// structurally equal one, `basis::ModelSpace::operator==`). O(1) for the
  /// same object.
  bool Compatible(const op::Operator& op) const noexcept;

  /// The communicator of `WorkspaceOptions::comm` (serial by default).
  const comm::Communicator& comm() const noexcept { return comm_; }

  /// The tables of the IMSRG(3) terms (`ThreeBodyContext`, three_body.h),
  /// built by the constructor iff the model space has a three-body space
  /// (`ModelSpace::WithThreeBody`); null otherwise.
  const ThreeBodyContext* three_body_context() const noexcept {
    return three_body_.get();
  }

  /// Internal data of the kernels (defined in `detail/workspace_impl.h`, not a
  /// public API).
  struct Impl;
  /// Internal data, for the kernels of this module.
  Impl& impl() noexcept { return *impl_; }
  /// Internal data, read-only.
  const Impl& impl() const noexcept { return *impl_; }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  comm::Communicator comm_;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<ThreeBodyContext> three_body_;
};

}  // namespace imsrg::commutator

#endif  // IMSRG_COMMUTATOR_WORKSPACE_H_
