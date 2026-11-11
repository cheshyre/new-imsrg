#include "imsrg/basis/particle_hole_channels.hpp"

#include <algorithm>
#include <cstdlib>

#include "imsrg/base/assert.hpp"

namespace imsrg::basis {

namespace {
constexpr std::int32_t kChannelsPerJ = 6;
}

ParticleHoleChannels::ParticleHoleChannels(const TwoBodyChannels& two_body)
    : num_orbits_(two_body.num_orbits()), num_kets_(two_body.num_kets()), max_J_(two_body.max_J()) {
  for (const Orbit& o : two_body.orbits()) {
    orbit_tz2_.push_back(o.tz2.twice);
    orbit_parity_.push_back(static_cast<std::int32_t>(o.parity()));
    orbit_j2_.push_back(o.j2.twice);
  }
  const std::int32_t nch = kChannelsPerJ * (max_J_ + 1);
  channels_.resize(static_cast<std::size_t>(nch));
  for (std::int32_t ch = 0; ch < nch; ++ch) {
    TwoBodyChannel& c = channels_[static_cast<std::size_t>(ch)];
    c.two_j = TwoJ{2 * (ch / kChannelsPerJ)};
    c.two_tz = TwoTz{2 * ((ch % kChannelsPerJ) / 2 - 1)};
    c.parity = (ch % 2 == 0) ? Parity::Even : Parity::Odd;
  }

  std::vector<std::vector<KetIndex>> members(static_cast<std::size_t>(nch));
  info_.reserve(static_cast<std::size_t>(num_kets_));
  ket_J_offsets_.assign(static_cast<std::size_t>(num_kets_) + 1, 0);
  for (std::int32_t k = 0; k < num_kets_; ++k) {
    const Ket& kt = two_body.ket(KetIndex{k});
    const auto p = static_cast<std::size_t>(kt.p.value());
    const auto q = static_cast<std::size_t>(kt.q.value());
    KetInfo ki;
    ki.jmin = std::abs(orbit_j2_[p] - orbit_j2_[q]) / 2;
    ki.jmax = (orbit_j2_[p] + orbit_j2_[q]) / 2;
    ki.parity = static_cast<Parity>(orbit_parity_[p] ^ orbit_parity_[q]);
    ki.tz_diff = std::abs(orbit_tz2_[p] - orbit_tz2_[q]) / 2;
    info_.push_back(ki);
    ket_J_offsets_[static_cast<std::size_t>(k) + 1] =
        ket_J_offsets_[static_cast<std::size_t>(k)] + (ki.jmax - ki.jmin + 1);
    for (std::int32_t J = ki.jmin; J <= ki.jmax; ++J) {
      const ChannelIndex ch = imsrgpp_channel_index(TwoJ{2 * J}, ki.parity, TwoTz{2 * ki.tz_diff});
      IMSRG_CHECK(ch.value() >= 0 && ch.value() < nch, "ph ket channel out of range");
      auto& list = members[static_cast<std::size_t>(ch.value())];
      local_index_table_.push_back(static_cast<std::int32_t>(list.size()));
      list.push_back(KetIndex{k});
    }
  }

  std::int32_t offset = 0;
  for (std::int32_t ch = 0; ch < nch; ++ch) {
    const auto& list = members[static_cast<std::size_t>(ch)];
    TwoBodyChannel& c = channels_[static_cast<std::size_t>(ch)];
    c.num_kets = static_cast<std::int32_t>(list.size());
    c.ket_offset = offset;
    offset += c.num_kets;
    channel_ket_table_.insert(channel_ket_table_.end(), list.begin(), list.end());
    if (c.num_kets > 0) channels_by_size_.push_back(ChannelIndex{ch});
  }
  std::stable_sort(channels_by_size_.begin(), channels_by_size_.end(),
                   [this](ChannelIndex x, ChannelIndex y) {
                     return channels_[static_cast<std::size_t>(x.value())].num_kets >
                            channels_[static_cast<std::size_t>(y.value())].num_kets;
                   });
}

const TwoBodyChannel& ParticleHoleChannels::channel(ChannelIndex ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels(), "channel index out of range");
  return channels_[static_cast<std::size_t>(ch.value())];
}

std::optional<ChannelIndex> ParticleHoleChannels::find_channel(TwoJ two_j, Parity parity,
                                                               TwoTz two_tz) const noexcept {
  if (two_j.twice < 0 || two_j.twice % 2 != 0 || two_j.twice / 2 > max_J_) return std::nullopt;
  if (two_tz.twice != -2 && two_tz.twice != 0 && two_tz.twice != 2) return std::nullopt;
  return imsrgpp_channel_index(two_j, parity, two_tz);
}

ChannelIndex ParticleHoleChannels::channel_of(OrbitIndex p, OrbitIndex q, TwoJ two_j) const {
  IMSRG_ASSERT(
      p.value() >= 0 && p.value() < num_orbits_ && q.value() >= 0 && q.value() < num_orbits_,
      "orbit index out of range");
  IMSRG_ASSERT(two_j.twice >= 0 && two_j.twice % 2 == 0 && two_j.twice / 2 <= max_J_,
               "J out of range");
  const auto a = static_cast<std::size_t>(p.value());
  const auto b = static_cast<std::size_t>(q.value());
  return imsrgpp_channel_index(two_j, static_cast<Parity>(orbit_parity_[a] ^ orbit_parity_[b]),
                               TwoTz{std::abs(orbit_tz2_[a] - orbit_tz2_[b])});
}

std::span<const KetIndex> ParticleHoleChannels::channel_kets(ChannelIndex ch) const {
  const TwoBodyChannel& c = channel(ch);
  return std::span<const KetIndex>(channel_ket_table_)
      .subspan(static_cast<std::size_t>(c.ket_offset), static_cast<std::size_t>(c.num_kets));
}

std::optional<LocalKetIndex> ParticleHoleChannels::local_index(KetIndex k, TwoJ two_j) const {
  IMSRG_ASSERT(k.value() >= 0 && k.value() < num_kets_, "ket index out of range");
  if (two_j.twice < 0 || two_j.twice % 2 != 0) return std::nullopt;
  const KetInfo& ki = info_[static_cast<std::size_t>(k.value())];
  const std::int32_t J = two_j.twice / 2;
  if (J < ki.jmin || J > ki.jmax) return std::nullopt;
  return LocalKetIndex{local_index_table_[static_cast<std::size_t>(
      ket_J_offsets_[static_cast<std::size_t>(k.value())] + (J - ki.jmin))]};
}

std::optional<LocalKetIndex> ParticleHoleChannels::local_index(ChannelIndex ch, KetIndex k) const {
  const TwoBodyChannel& c = channel(ch);
  IMSRG_ASSERT(k.value() >= 0 && k.value() < num_kets_, "ket index out of range");
  const KetInfo& ki = info_[static_cast<std::size_t>(k.value())];
  if (ki.parity != c.parity || ki.tz_diff != c.Tz()) return std::nullopt;
  return local_index(k, c.two_j);
}

std::optional<std::int32_t> ParticleHoleChannels::pair_index(ChannelIndex ch, OrbitIndex p,
                                                             OrbitIndex q) const {
  IMSRG_ASSERT(
      p.value() >= 0 && p.value() < num_orbits_ && q.value() >= 0 && q.value() < num_orbits_,
      "orbit index out of range");
  const bool sorted = p <= q;
  const KetIndex k =
      sorted ? imsrgpp_ket_index(p, q, num_orbits_) : imsrgpp_ket_index(q, p, num_orbits_);
  const auto local = local_index(ch, k);
  if (!local) return std::nullopt;
  return sorted ? local->value() : local->value() + channel(ch).num_kets;
}

}  // namespace imsrg::basis
