#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "imsrg/base/constants.hpp"
#include "imsrg/basis/two_body_channels.hpp"
#include "imsrg/commutator/detail/lifted.hpp"

namespace imsrg::commutator::detail {

LiftedData build_lifted_data(const basis::ModelSpace& ms) {
  const auto& channels = ms.two_body();
  const auto& sp = ms.basis();
  const auto nch = static_cast<std::size_t>(channels.num_channels());

  std::vector<std::int32_t> row_base(nch + 1, 0);
  std::vector<std::int32_t> entry_offset{0};
  std::vector<std::int32_t> m_list, from_list, to_list;
  std::vector<double> coef_list;

  for (std::size_t ch = 0; ch < nch; ++ch) {
    const ChannelIndex cidx{static_cast<std::int32_t>(ch)};
    const auto kets = channels.channel_kets(cidx);
    const TwoJ two_j = channels.channel(cidx).two_j;
    for (const KetIndex kidx : kets) {
      const auto& ket = channels.ket(kidx);
      const int i = ket.p.value();
      const int j = ket.q.value();
      const double norm = (i == j) ? 1.0 / constants::sqrt2 : 1.0;
      // Adds the entry for the lifted pair (a, b) (any order); `one_from` -> `one_to` is the
      // one-body element.
      auto add = [&](int a, int b, int one_from, int one_to) {
        const bool swap = a > b;
        const int lo = swap ? b : a;
        const int hi = swap ? a : b;
        const auto loc =
            channels.local_index(channels.ket_index(OrbitIndex{lo}, OrbitIndex{hi}), two_j);
        if (!loc) return;
        double c = norm * ((a == b) ? constants::sqrt2 : 1.0);
        if (swap)
          c *= basis::exchange_phase(sp.orbit(OrbitIndex{a}).j2, sp.orbit(OrbitIndex{b}).j2, two_j);
        m_list.push_back(static_cast<std::int32_t>(loc->value()));
        from_list.push_back(one_from);
        to_list.push_back(one_to);
        coef_list.push_back(c);
      };
      for (const OrbitIndex c : sp.same_ljtz(OrbitIndex{i})) add(c.value(), j, i, c.value());
      for (const OrbitIndex c : sp.same_ljtz(OrbitIndex{j})) add(i, c.value(), j, c.value());
      entry_offset.push_back(static_cast<std::int32_t>(m_list.size()));
    }
    row_base[ch + 1] = row_base[ch] + static_cast<std::int32_t>(kets.size());
  }

  LiftedData d;
  auto upload = [](auto& buf, const auto& host, const char* tag) {
    using T = typename std::decay_t<decltype(host)>::value_type;
    buf = exec::DeviceBuffer<T>(host.size(), tag);
    if (!host.empty()) buf.copy_from_host(host);
  };
  upload(d.row_base, row_base, "commutator::lifted_row_base");
  upload(d.entry_offset, entry_offset, "commutator::lifted_entry_offset");
  upload(d.m, m_list, "commutator::lifted_m");
  upload(d.from, from_list, "commutator::lifted_from");
  upload(d.to, to_list, "commutator::lifted_to");
  upload(d.coef, coef_list, "commutator::lifted_coef");
  return d;
}

}  // namespace imsrg::commutator::detail
