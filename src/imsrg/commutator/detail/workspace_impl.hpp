#pragma once

/// @file
/// Internal data of `commutator::Workspace` (no Kokkos): device tables, 6j table, one-body
/// channel CSR, two-body block offsets and the term-specific precomputed data. Shared by
/// `workspace.cpp`, `commutator.cpp` and the kernel TUs.

#include <cstdint>
#include <memory>
#include <vector>

#include "imsrg/angmom/tables.hpp"
#include "imsrg/basis/device_tables.hpp"
#include "imsrg/commutator/detail/device_views.hpp"
#include "imsrg/commutator/detail/ladder.hpp"
#include "imsrg/commutator/detail/pandya.hpp"
#include "imsrg/commutator/workspace.hpp"
#include "imsrg/exec/device_buffer.hpp"
#include "imsrg/operator/operator.hpp"

namespace imsrg::commutator {

/// Internal data of a `Workspace`; built once per model space by the `Workspace` constructor.
struct Workspace::Impl {
  /// Builds all tables for `ms` (non-null). Needs an initialized `exec::Runtime`.
  explicit Impl(const std::shared_ptr<const basis::ModelSpace>& ms);

  basis::DeviceTables tables;  ///< Device copies of orbit, ket and channel tables.
  angmom::SixJTable sixj;      ///< Dense 6j table up to the model space's emax.

  /// One-body channel CSR (see `detail::OneBodyChannelView`).
  exec::DeviceBuffer<std::int32_t> ob_offsets;  ///< num_orbits + 1 entries.
  exec::DeviceBuffer<std::int32_t> ob_orbits;   ///< CSR entries: the orbit q.
  exec::DeviceBuffer<std::int32_t> ob_owner;    ///< CSR entries: the orbit p owning the entry.

  /// Offset of each channel's block in the two-body operator buffer (num_channels + 1 entries,
  /// the last one the total size), as stored by `op::TwoBody`.
  exec::DeviceBuffer<std::int64_t> block_offsets;
  std::vector<std::int64_t> block_offsets_host;  ///< Host copy of `block_offsets`.
  std::int64_t two_body_size = 0;                ///< Total number of stored two-body elements.

  /// For each entry of the flat `channel_kets` table, the channel it belongs to.
  exec::DeviceBuffer<std::int32_t> entry_channel;

  detail::LadderData ladder;  ///< Data of the pp/hh ladder terms (T3b).
  detail::PandyaData pandya;  ///< Data of the particle-hole term (T4).

  /// View of the one-body channel CSR. O(1).
  detail::OneBodyChannelView one_body_channels() const noexcept {
    return {ob_offsets.device_data(), ob_orbits.device_data()};
  }

  /// View of the two-body part of `op`; `data == nullptr` if the part is not allocated. The
  /// operator must live on this workspace's model space. O(1).
  detail::TwoBodyView two_body_view(const op::Operator& op) const;
};

namespace detail {

/// Shared precondition check of every `terms::` function and of `commutator()`: X, Y and Z are
/// non-empty on the workspace's model space (`Workspace::compatible`), Z is neither X nor Y, and
/// X, Y and Z have a definite hermiticity.
/// @throws InvariantError on a violated precondition, NotImplementedError if X or Y is
///         NonHermitian (the fast kernels use the (anti-)Hermiticity). O(1).
void check_term_inputs(const op::Operator& X, const op::Operator& Y, const op::Operator& Z,
                       const Workspace& ws);

}  // namespace detail

}  // namespace imsrg::commutator
