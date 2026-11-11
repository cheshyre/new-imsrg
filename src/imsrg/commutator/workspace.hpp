#pragma once

/// @file
/// `Workspace`: the per-model-space precomputed data and preallocated scratch buffers of the
/// commutator kernels (design/10 section 3.4, design/12 section 4).

#include <memory>

#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::commutator {

/// Everything the IMSRG(2) kernels need besides X, Y and Z, built once per model space and
/// reused by every commutator call on that model space (a flow performs thousands):
/// * device copies of the basis index tables (`basis::DeviceTables`) and the dense Pandya 6j
///   table (`angmom::SixJTable`, design/12 section 4.4);
/// * per-channel ket lists with occupation weights for the ladder (pp/hh) GEMMs;
/// * the particle-hole (Pandya) recoupling data of every cross-coupled channel;
/// * scratch buffers (gathered GEMM operands, ladder intermediates, Pandya-transformed blocks),
///   allocated in the constructor so the flow loop does not allocate.
///
/// Nothing in it is a lazy cache (design/00 D10): construction does all the work, afterwards
/// the precomputed data are read-only. The occupations are part of the model space, so a new
/// reference (e.g. after Hartree-Fock) needs a new Workspace.
///
/// Thread safety: the scratch buffers are mutable, so a Workspace serves one `commutator` call
/// at a time; give concurrent calls their own Workspaces. Needs an initialized `exec::Runtime`;
/// destroy it before the Runtime. Movable, not copyable.
class Workspace {
 public:
  /// Builds the precomputed data and scratch buffers for `ms`. Cost: dominated by the 6j table
  /// (O((emax+1)^4 (2 emax+2)^2)) and the scratch allocation. Scratch footprint: the ladder keeps
  /// about 4 (measured 3.5x at O16 emax 4, 3.8x at emax 6) and the Pandya term about 6 to 8
  /// (8.0x, 6.6x) two-body operators of memory; the exact bytes are printed by the
  /// `commutator_robustness.scratch_footprint` test).
  /// @throws InvariantError if `ms` is null.
  explicit Workspace(std::shared_ptr<const basis::ModelSpace> ms);

  ~Workspace();
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  /// The model space the workspace was built for.
  const basis::ModelSpace& model_space() const noexcept { return *ms_; }

  /// The shared model space.
  const std::shared_ptr<const basis::ModelSpace>& model_space_ptr() const noexcept { return ms_; }

  /// True if `op` lives on this workspace's model space (the same object or a structurally
  /// equal one, `basis::ModelSpace::operator==`). O(1) for the same object.
  bool compatible(const op::Operator& op) const noexcept;

  /// Internal data of the kernels (defined in `detail/workspace_impl.hpp`, not a public API).
  struct Impl;
  /// Internal data, for the kernels of this module.
  Impl& impl() noexcept { return *impl_; }
  /// Internal data, read-only.
  const Impl& impl() const noexcept { return *impl_; }

 private:
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace imsrg::commutator
