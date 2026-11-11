#include <cstddef>
#include <span>
#include <string>

#include "imsrg/commutator/detail/ladder.hpp"

namespace imsrg::commutator::detail {

namespace {

/// Fills `proj` for the weight function `weight(n_p, n_q)`; `total` is the size of a two-body
/// operator buffer.
template <class WeightFn>
void fill_projector(LadderProjector& proj, const basis::ModelSpace& ms, std::int64_t total,
                    const char* tag, WeightFn weight) {
  const auto& channels = ms.two_body();
  const auto occ = ms.reference().occupations();
  const auto nch = static_cast<std::size_t>(channels.num_channels());

  std::vector<std::int32_t> list;
  std::vector<double> w;
  std::vector<std::int64_t> list_offset(nch + 1, 0);
  proj.count_host.assign(nch, 0);
  proj.gather_offset_host.assign(nch + 1, 0);
  for (std::size_t ch = 0; ch < nch; ++ch) {
    const auto kets = channels.channel_kets(ChannelIndex{static_cast<std::int32_t>(ch)});
    std::int32_t count = 0;
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const auto& k = channels.ket(kets[i]);
      const double wk = weight(occ[static_cast<std::size_t>(k.p.value())],
                               occ[static_cast<std::size_t>(k.q.value())]);
      if (wk == 0.0) continue;  // exact test: fractional references keep every non-zero weight
      list.push_back(static_cast<std::int32_t>(i));
      w.push_back(wk);
      ++count;
    }
    proj.count_host[ch] = count;
    list_offset[ch + 1] = static_cast<std::int64_t>(list.size());
    proj.gather_offset_host[ch + 1] =
        proj.gather_offset_host[ch] + static_cast<std::int64_t>(kets.size()) * count;
  }

  const std::string t = std::string("commutator::ladder_") + tag;
  proj.list = exec::DeviceBuffer<std::int32_t>(list.size(), t + "_list");
  if (!list.empty()) proj.list.copy_from_host(list);
  proj.weight = exec::DeviceBuffer<double>(w.size(), t + "_weight");
  if (!w.empty()) proj.weight.copy_from_host(w);
  proj.list_offset = exec::DeviceBuffer<std::int64_t>(list_offset.size(), t + "_list_offset");
  proj.list_offset.copy_from_host(list_offset);
  proj.gather_offset =
      exec::DeviceBuffer<std::int64_t>(proj.gather_offset_host.size(), t + "_gather_offset");
  proj.gather_offset.copy_from_host(proj.gather_offset_host);
  const auto gsize = static_cast<std::size_t>(proj.gather_offset_host.back());
  proj.xg = exec::DeviceBuffer<double>(gsize, t + "_xg");
  proj.yg = exec::DeviceBuffer<double>(gsize, t + "_yg");
  proj.product = exec::DeviceBuffer<double>(static_cast<std::size_t>(total), t + "_product");
}

}  // namespace

LadderData build_ladder_data(const basis::ModelSpace& ms) {
  std::int64_t total = 0;
  for (const auto& ch : ms.two_body().channels())
    total += static_cast<std::int64_t>(ch.num_kets) * ch.num_kets;
  LadderData d;
  fill_projector(d.pp, ms, total, "pp",
                 [](double np, double nq) { return (1.0 - np) * (1.0 - nq); });
  fill_projector(d.hh, ms, total, "hh", [](double np, double nq) { return np * nq; });
  d.problems.reserve(2 * static_cast<std::size_t>(ms.two_body().num_channels()));
  return d;
}

}  // namespace imsrg::commutator::detail
