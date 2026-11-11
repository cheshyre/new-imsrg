#include "imsrg/commutator/workspace.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "imsrg/base/assert.hpp"
#include "imsrg/commutator/detail/workspace_impl.hpp"

namespace imsrg::commutator {

Workspace::Impl::Impl(const std::shared_ptr<const basis::ModelSpace>& ms)
    : tables(*ms),
      sixj(ms->basis().emax()),
      ladder(detail::build_ladder_data(*ms)),
      pandya(detail::build_pandya_data(*ms)) {
  const std::size_t norb = static_cast<std::size_t>(ms->num_orbits());

  // One-body channel CSR: orbits with equal (l, 2j, 2tz).
  std::vector<std::int32_t> offsets(norb + 1, 0);
  std::vector<std::int32_t> orbits;
  std::vector<std::int32_t> owner;
  for (std::size_t p = 0; p < norb; ++p) {
    for (const OrbitIndex q : ms->basis().same_ljtz(OrbitIndex{static_cast<std::int32_t>(p)})) {
      orbits.push_back(static_cast<std::int32_t>(q.value()));
      owner.push_back(static_cast<std::int32_t>(p));
    }
    offsets[p + 1] = static_cast<std::int32_t>(orbits.size());
  }
  ob_offsets = exec::DeviceBuffer<std::int32_t>(offsets.size(), "commutator::ob_offsets");
  ob_offsets.copy_from_host(offsets);
  ob_orbits = exec::DeviceBuffer<std::int32_t>(orbits.size(), "commutator::ob_orbits");
  ob_orbits.copy_from_host(orbits);
  ob_owner = exec::DeviceBuffer<std::int32_t>(owner.size(), "commutator::ob_owner");
  ob_owner.copy_from_host(owner);

  // Block offsets exactly as `op::TwoBody` stores a scalar rank-2 operator.
  {
    const op::TwoBody tmp(ms, op::OperatorSpec::scalar(2));
    const auto& m = tmp.matrix();
    block_offsets_host.reserve(m.num_blocks() + 1);
    for (std::size_t k = 0; k < m.num_blocks(); ++k)
      block_offsets_host.push_back(m.block_info(k).offset);
    two_body_size = m.total_size();
    block_offsets_host.push_back(two_body_size);
    block_offsets =
        exec::DeviceBuffer<std::int64_t>(block_offsets_host.size(), "commutator::block_offsets");
    block_offsets.copy_from_host(block_offsets_host);
  }

  // Channel of each entry of the flat channel-ket table.
  {
    const auto& channels = ms->two_body();
    std::vector<std::int32_t> ec;
    for (std::int32_t ch = 0; ch < channels.num_channels(); ++ch) {
      const auto n = channels.channel_kets(ChannelIndex{ch}).size();
      ec.insert(ec.end(), n, ch);
    }
    entry_channel = exec::DeviceBuffer<std::int32_t>(ec.size(), "commutator::entry_channel");
    entry_channel.copy_from_host(ec);
  }
}

detail::TwoBodyView Workspace::Impl::two_body_view(const op::Operator& op) const {
  detail::TwoBodyView v;
  v.tb = tables.two_body();
  v.block_offset = block_offsets.device_data();
  v.data = op.two_body().allocated() ? op.two_body().matrix().buffer().device_data() : nullptr;
  return v;
}

Workspace::Workspace(std::shared_ptr<const basis::ModelSpace> ms) : ms_(std::move(ms)) {
  IMSRG_CHECK(ms_ != nullptr, "Workspace: model space is null");
  impl_ = std::make_unique<Impl>(ms_);
}

Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;

bool Workspace::compatible(const op::Operator& op) const noexcept {
  if (op.empty()) return false;
  return op.model_space_ptr() == ms_ || op.model_space() == *ms_;
}

}  // namespace imsrg::commutator
