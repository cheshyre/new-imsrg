// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "imsrg/commutator/detail/ladder.h"
#include "imsrg/commutator/detail/workspace_impl.h"

namespace imsrg::commutator::detail {

namespace {

/// Copies the host lists of a projector to device buffers and allocates its
/// gathered-operand and product buffers.
void UploadProjector(LadderProjector& proj,
                     const std::vector<std::int32_t>& list,
                     const std::vector<double>& w,
                     const std::vector<std::int64_t>& list_offset,
                     const char* tag, std::int64_t total) {
  const std::string t = std::string("commutator::ladder_") + tag;
  proj.list = exec::DeviceBuffer<std::int32_t>(list.size(), t + "_list");
  if (!list.empty()) proj.list.CopyFromHost(list);
  proj.weight = exec::DeviceBuffer<double>(w.size(), t + "_weight");
  if (!w.empty()) proj.weight.CopyFromHost(w);
  proj.list_offset =
      exec::DeviceBuffer<std::int64_t>(list_offset.size(), t + "_list_offset");
  proj.list_offset.CopyFromHost(list_offset);
  proj.gather_offset = exec::DeviceBuffer<std::int64_t>(
      proj.gather_offset_host.size(), t + "_gather_offset");
  proj.gather_offset.CopyFromHost(proj.gather_offset_host);
  proj.direct_first = exec::DeviceBuffer<std::int32_t>(
      proj.direct_first_host.size(), t + "_direct_first");
  if (!proj.direct_first_host.empty())
    proj.direct_first.CopyFromHost(proj.direct_first_host);
  const auto gsize = static_cast<std::size_t>(proj.gather_offset_host.back());
  proj.xg = exec::DeviceBuffer<double>(gsize, t + "_xg");
  proj.yg = exec::DeviceBuffer<double>(gsize, t + "_yg");
  proj.product = exec::DeviceBuffer<double>(static_cast<std::size_t>(total),
                                            t + "_product");
}

/// Fills `proj` for the weight function `weight(n_p, n_q)`; `total` is the size
/// of a two-body operator buffer.
template <class WeightFn>
void FillProjector(LadderProjector& proj, const basis::ModelSpace& ms,
                   std::int64_t total, const char* tag, WeightFn weight) {
  const auto& channels = ms.two_body();
  const auto occ = ms.reference().occupations();
  const auto nch = static_cast<std::size_t>(channels.num_channels());

  std::vector<std::int32_t> list;
  std::vector<double> w;
  std::vector<std::int64_t> list_offset(nch + 1, 0);
  proj.count_host.assign(nch, 0);
  proj.gather_offset_host.assign(nch + 1, 0);
  proj.direct_first_host.assign(nch, -1);
  for (std::size_t ch = 0; ch < nch; ++ch) {
    const auto kets =
        channels.channel_kets(ChannelIndex{static_cast<std::int32_t>(ch)});
    std::int32_t count = 0;
    bool direct = true;  // listed kets contiguous and every weight == 1.0
    const auto first_entry = static_cast<std::int64_t>(list.size());
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const auto& k = channels.ket(kets[i]);
      const double wk = weight(occ[static_cast<std::size_t>(k.p.value())],
                               occ[static_cast<std::size_t>(k.q.value())]);
      if (wk == 0.0)
        continue;  // exact test: fractional references keep every non-zero
                   // weight
      if (static_cast<std::int64_t>(list.size()) != first_entry &&
          static_cast<std::int32_t>(i) != list.back() + 1)
        direct = false;
      if (wk != 1.0) direct = false;
      list.push_back(static_cast<std::int32_t>(i));
      w.push_back(wk);
      ++count;
    }
    proj.count_host[ch] = count;
    list_offset[ch + 1] = static_cast<std::int64_t>(list.size());
    if (count > 0 && direct)
      proj.direct_first_host[ch] = list[static_cast<std::size_t>(first_entry)];
    // Direct channels read X and Y in place: no gathered operands.
    proj.gather_offset_host[ch + 1] =
        proj.gather_offset_host[ch] +
        (proj.direct_first_host[ch] >= 0
             ? 0
             : static_cast<std::int64_t>(kets.size()) * count);
  }

  UploadProjector(proj, list, w, list_offset, tag, total);
}

}  // namespace

LadderData BuildLadderData(const basis::ModelSpace& ms) {
  std::int64_t total = 0;
  for (const auto& ch : ms.two_body().channels())
    total += static_cast<std::int64_t>(ch.num_kets) * ch.num_kets;
  LadderData d;
  FillProjector(d.pp, ms, total, "pp",
                [](double np, double nq) { return (1.0 - np) * (1.0 - nq); });
  FillProjector(d.hh, ms, total, "hh",
                [](double np, double nq) { return np * nq; });
  d.problems.reserve(2 *
                     static_cast<std::size_t>(ms.two_body().num_channels()));
  return d;
}

void AttachLadderWork(Workspace::Impl& impl) {
  const std::size_t nch =
      impl.block_offsets_host.empty() ? 0 : impl.block_offsets_host.size() - 1;
  const std::size_t n = std::max(nch, impl.work.ladder_blocks().size());
  impl.ladder.problems.reserve(2 * n);
}

}  // namespace imsrg::commutator::detail
