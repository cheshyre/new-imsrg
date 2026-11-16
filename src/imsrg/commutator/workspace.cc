// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/workspace.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"
#include "imsrg/commutator/detail/work_items.h"
#include "imsrg/commutator/detail/workspace_impl.h"
#include "imsrg/commutator/three_body.h"

namespace imsrg::commutator {

namespace {

/// Builds the one-body channel CSR: for each orbit p the orbits with equal
/// (l, 2j, 2tz) (`orbits`, delimited by `offsets`) and the owner of each entry.
void BuildOneBodyChannels(const basis::ModelSpace& ms,
                          std::vector<std::int32_t>& offsets,
                          std::vector<std::int32_t>& orbits,
                          std::vector<std::int32_t>& owner) {
  const std::size_t norb = static_cast<std::size_t>(ms.num_orbits());
  offsets.assign(norb + 1, 0);
  for (std::size_t p = 0; p < norb; ++p) {
    for (const OrbitIndex q :
         ms.basis().SameLjtz(OrbitIndex{static_cast<std::int32_t>(p)})) {
      orbits.push_back(static_cast<std::int32_t>(q.value()));
      owner.push_back(static_cast<std::int32_t>(p));
    }
    offsets[p + 1] = static_cast<std::int32_t>(orbits.size());
  }
}

/// Returns the channel of each entry of the flat channel-ket table.
std::vector<std::int32_t> EntryChannels(const basis::ModelSpace& ms) {
  const auto& channels = ms.two_body();
  std::vector<std::int32_t> ec;
  for (std::int32_t ch = 0; ch < channels.num_channels(); ++ch) {
    const auto n = channels.channel_kets(ChannelIndex{ch}).size();
    ec.insert(ec.end(), n, ch);
  }
  return ec;
}

}  // namespace

Workspace::Impl::Impl(const std::shared_ptr<const basis::ModelSpace>& ms)
    : tables(*ms),
      sixj(ms->basis().emax()),
      ladder(detail::BuildLadderData(*ms)),
      pandya(detail::BuildPandyaData(*ms)),
      lifted(detail::BuildLiftedData(*ms)) {
  const std::size_t norb = static_cast<std::size_t>(ms->num_orbits());

  // One-body channel CSR: orbits with equal (l, 2j, 2tz).
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> orbits;
  std::vector<std::int32_t> owner;
  BuildOneBodyChannels(*ms, offsets, orbits, owner);
  ob_offsets = exec::DeviceBuffer<std::int32_t>(offsets.size(),
                                                "commutator::ob_offsets");
  ob_offsets.CopyFromHost(offsets);
  ob_orbits =
      exec::DeviceBuffer<std::int32_t>(orbits.size(), "commutator::ob_orbits");
  ob_orbits.CopyFromHost(orbits);
  ob_owner =
      exec::DeviceBuffer<std::int32_t>(owner.size(), "commutator::ob_owner");
  ob_owner.CopyFromHost(owner);

  // One-body intermediates of the factorized double commutator (zeroed).
  factorized.chi2 =
      exec::DeviceBuffer<double>(norb * norb, "commutator::factorized_chi2");
  factorized.chib =
      exec::DeviceBuffer<double>(norb * norb, "commutator::factorized_chib");

  // Block offsets exactly as `op::TwoBody` stores a scalar rank-2 operator.
  {
    const op::TwoBody tmp(ms, op::OperatorSpec::Scalar(2));
    const auto& m = tmp.matrix();
    block_offsets_host.reserve(m.num_blocks() + 1);
    for (std::size_t k = 0; k < m.num_blocks(); ++k)
      block_offsets_host.push_back(m.block_info(k).offset);
    two_body_size = m.total_size();
    block_offsets_host.push_back(two_body_size);
    block_offsets = exec::DeviceBuffer<std::int64_t>(
        block_offsets_host.size(), "commutator::block_offsets");
    block_offsets.CopyFromHost(block_offsets_host);
  }

  // Channel of each entry of the flat channel-ket table.
  {
    const std::vector<std::int32_t> ec = EntryChannels(*ms);
    entry_channel = exec::DeviceBuffer<std::int32_t>(
        ec.size(), "commutator::entry_channel");
    entry_channel.CopyFromHost(ec);
  }
}

detail::TwoBodyView Workspace::Impl::two_body_view(
    const op::Operator& op) const {
  detail::TwoBodyView v;
  v.tb = tables.two_body();
  v.block_offset = block_offsets.device_data();
  v.data = op.two_body().allocated()
               ? op.two_body().matrix().buffer().device_data()
               : nullptr;
  return v;
}

Workspace::Workspace(std::shared_ptr<const basis::ModelSpace> ms,
                     const WorkspaceOptions& opts)
    : ms_(std::move(ms)), comm_(opts.comm) {
  IMSRG_CHECK(ms_ != nullptr, "Workspace: model space is null");
  impl_ = std::make_unique<Impl>(ms_);
  detail::AssignWork(*this, static_cast<std::int32_t>(comm_.rank()),
                     static_cast<std::int32_t>(comm_.size()));
  if (opts.three_body_context == ThreeBodyContextMode::kAuto &&
      ms_->three_body_space() != nullptr) {
    BuildThreeBodyContext();
  }
}

void Workspace::BuildThreeBodyContext() {
  if (three_body_ != nullptr) return;
  if (ms_->three_body_space() == nullptr)
    throw InputError(
        "Workspace::BuildThreeBodyContext: the model space has no three-body "
        "space (ModelSpace::WithThreeBody)");
  three_body_ = std::make_unique<ThreeBodyContext>(ms_);
}

Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;

bool Workspace::Compatible(const op::Operator& op) const noexcept {
  if (op.empty()) return false;
  return op.model_space_ptr() == ms_ || op.model_space() == *ms_;
}

}  // namespace imsrg::commutator
