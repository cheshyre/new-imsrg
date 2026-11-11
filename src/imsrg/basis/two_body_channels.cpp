#include "imsrg/basis/two_body_channels.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::basis {

namespace {

constexpr std::int32_t kNumParities = 2;
constexpr std::int32_t kChannelsPerJ = 6;

/// Integer J range [jmin, jmax] of a pair of orbits allowed by the triangle rule.
struct JRange {
  std::int32_t jmin;
  std::int32_t jmax;
};

JRange j_range(const Orbit& a, const Orbit& b) noexcept {
  return {std::abs(a.j2.twice - b.j2.twice) / 2, (a.j2.twice + b.j2.twice) / 2};
}

}  // namespace

std::string_view ket_class_name(KetClass c) noexcept {
  switch (c) {
    case KetClass::PP:
      return "pp";
    case KetClass::HH:
      return "hh";
    case KetClass::PH:
      return "ph";
    case KetClass::CC:
      return "cc";
    case KetClass::VC:
      return "vc";
    case KetClass::QC:
      return "qc";
    case KetClass::VV:
      return "vv";
    case KetClass::QV:
      return "qv";
    case KetClass::QQ:
      return "qq";
  }
  return "?";
}

TwoBodyChannels::TwoBodyChannels(const SingleParticleBasis& basis)
    : orbits_(basis.orbits().begin(), basis.orbits().end()), max_J_(basis.max_two_body_J()) {
  const std::int32_t n = num_orbits();

  // Kets in imsrg++ order (lexicographic, p <= q).
  kets_.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) / 2);
  for (std::int32_t p = 0; p < n; ++p) {
    for (std::int32_t q = p; q < n; ++q) {
      IMSRG_CHECK(imsrgpp_ket_index(OrbitIndex{p}, OrbitIndex{q}, n).value() ==
                      static_cast<std::int32_t>(kets_.size()),
                  "ket position differs from the imsrg++ Index2 formula");
      kets_.push_back(Ket{OrbitIndex{p}, OrbitIndex{q}});
    }
  }

  const std::int32_t nch = kChannelsPerJ * (max_J_ + 1);
  channels_.resize(static_cast<std::size_t>(nch));
  for (std::int32_t ch = 0; ch < nch; ++ch) {
    TwoBodyChannel& c = channels_[static_cast<std::size_t>(ch)];
    c.two_j = TwoJ{2 * (ch / kChannelsPerJ)};
    c.two_tz = TwoTz{2 * ((ch % kChannelsPerJ) / kNumParities - 1)};
    c.parity = (ch % kNumParities == 0) ? Parity::Even : Parity::Odd;
  }

  // One pass over the kets in ascending order: each channel list is ascending (the local order).
  std::vector<std::vector<KetIndex>> members(static_cast<std::size_t>(nch));
  ket_J_offsets_.assign(kets_.size() + 1, 0);
  for (std::size_t k = 0; k < kets_.size(); ++k) {
    const Orbit& a = orbits_[static_cast<std::size_t>(kets_[k].p.value())];
    const Orbit& b = orbits_[static_cast<std::size_t>(kets_[k].q.value())];
    const bool same = kets_[k].p == kets_[k].q;
    const JRange jr = j_range(a, b);
    ket_J_offsets_[k + 1] = ket_J_offsets_[k] + (jr.jmax - jr.jmin + 1);
    for (std::int32_t J = jr.jmin; J <= jr.jmax; ++J) {
      if (same && (J % 2 != 0)) {
        local_index_table_.push_back(-1);
        continue;
      }
      const ChannelIndex ch = imsrgpp_channel_index(TwoJ{2 * J}, a.parity() * b.parity(),
                                                    TwoTz{a.tz2.twice + b.tz2.twice});
      IMSRG_CHECK(ch.value() >= 0 && ch.value() < nch, "ket channel out of range");
      auto& list = members[static_cast<std::size_t>(ch.value())];
      local_index_table_.push_back(static_cast<std::int32_t>(list.size()));
      list.push_back(KetIndex{static_cast<std::int32_t>(k)});
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

const Ket& TwoBodyChannels::ket(KetIndex k) const {
  IMSRG_ASSERT(k.value() >= 0 && k.value() < num_kets(), "ket index out of range");
  return kets_[static_cast<std::size_t>(k.value())];
}

KetIndex TwoBodyChannels::ket_index(OrbitIndex p, OrbitIndex q) const {
  IMSRG_ASSERT(p.value() >= 0 && p <= q && q.value() < num_orbits(), "need 0 <= p <= q < N");
  return imsrgpp_ket_index(p, q, num_orbits());
}

const TwoBodyChannel& TwoBodyChannels::channel(ChannelIndex ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels(), "channel index out of range");
  return channels_[static_cast<std::size_t>(ch.value())];
}

std::optional<ChannelIndex> TwoBodyChannels::find_channel(TwoJ two_j, Parity parity,
                                                          TwoTz two_tz) const noexcept {
  if (two_j.twice < 0 || two_j.twice % 2 != 0 || two_j.twice / 2 > max_J_) return std::nullopt;
  if (two_tz.twice != -2 && two_tz.twice != 0 && two_tz.twice != 2) return std::nullopt;
  return imsrgpp_channel_index(two_j, parity, two_tz);
}

ChannelIndex TwoBodyChannels::channel_of(OrbitIndex p, OrbitIndex q, TwoJ two_j) const {
  IMSRG_ASSERT(
      p.value() >= 0 && p.value() < num_orbits() && q.value() >= 0 && q.value() < num_orbits(),
      "orbit index out of range");
  IMSRG_ASSERT(two_j.twice >= 0 && two_j.twice % 2 == 0 && two_j.twice / 2 <= max_J_,
               "J out of range");
  const Orbit& a = orbits_[static_cast<std::size_t>(p.value())];
  const Orbit& b = orbits_[static_cast<std::size_t>(q.value())];
  return imsrgpp_channel_index(two_j, a.parity() * b.parity(), TwoTz{a.tz2.twice + b.tz2.twice});
}

std::span<const KetIndex> TwoBodyChannels::channel_kets(ChannelIndex ch) const {
  const TwoBodyChannel& c = channel(ch);
  return std::span<const KetIndex>(channel_ket_table_)
      .subspan(static_cast<std::size_t>(c.ket_offset), static_cast<std::size_t>(c.num_kets));
}

KetIndex TwoBodyChannels::global_ket(ChannelIndex ch, LocalKetIndex i) const {
  const TwoBodyChannel& c = channel(ch);
  IMSRG_ASSERT(i.value() >= 0 && i.value() < c.num_kets, "local ket index out of range");
  return channel_ket_table_[static_cast<std::size_t>(c.ket_offset + i.value())];
}

std::optional<LocalKetIndex> TwoBodyChannels::local_index(KetIndex k, TwoJ two_j) const {
  IMSRG_ASSERT(k.value() >= 0 && k.value() < num_kets(), "ket index out of range");
  if (two_j.twice < 0 || two_j.twice % 2 != 0) return std::nullopt;
  const Ket& kt = kets_[static_cast<std::size_t>(k.value())];
  const JRange jr = j_range(orbits_[static_cast<std::size_t>(kt.p.value())],
                            orbits_[static_cast<std::size_t>(kt.q.value())]);
  const std::int32_t J = two_j.twice / 2;
  if (J < jr.jmin || J > jr.jmax) return std::nullopt;
  const std::int32_t v = local_index_table_[static_cast<std::size_t>(
      ket_J_offsets_[static_cast<std::size_t>(k.value())] + (J - jr.jmin))];
  if (v < 0) return std::nullopt;
  return LocalKetIndex{v};
}

std::optional<LocalKetIndex> TwoBodyChannels::local_index(ChannelIndex ch, KetIndex k) const {
  const TwoBodyChannel& c = channel(ch);
  IMSRG_ASSERT(k.value() >= 0 && k.value() < num_kets(), "ket index out of range");
  const Ket& kt = kets_[static_cast<std::size_t>(k.value())];
  if (channel_of(kt.p, kt.q, c.two_j) != ch) return std::nullopt;
  return local_index(k, c.two_j);
}

// ---------------------------------------------------------------------------------------------

TwoBodyKetSubsets::TwoBodyKetSubsets(const TwoBodyChannels& channels,
                                     std::span<const double> occupations,
                                     std::span<const OrbitClass> classes)
    : num_channels_(channels.num_channels()) {
  const auto n = static_cast<std::size_t>(channels.num_orbits());
  if (occupations.size() != n) {
    throw InputError("TwoBodyKetSubsets: occupations has size " +
                     std::to_string(occupations.size()) + ", expected " + std::to_string(n));
  }
  if (classes.size() != n) {
    throw InputError("TwoBodyKetSubsets: classes has size " + std::to_string(classes.size()) +
                     ", expected " + std::to_string(n));
  }

  subset_offsets_.reserve(static_cast<std::size_t>(num_channels_) * kNumKetClasses + 1);
  subset_offsets_.push_back(0);
  for (std::int32_t ch = 0; ch < num_channels_; ++ch) {
    std::array<std::vector<LocalKetIndex>, kNumKetClasses> lists;
    std::array<std::vector<double>, kNumKetClasses> occ_prod;
    std::array<std::vector<double>, kNumKetClasses> vac_prod;
    const auto kets = channels.channel_kets(ChannelIndex{ch});
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const Ket& kt = channels.ket(kets[i]);
      const auto p = static_cast<std::size_t>(kt.p.value());
      const auto q = static_cast<std::size_t>(kt.q.value());
      const LocalKetIndex li{static_cast<std::int32_t>(i)};
      const bool hole_p = occupations[p] >= kOccupationCut;
      const bool hole_q = occupations[q] >= kOccupationCut;
      auto add = [&](KetClass c) {
        const auto ci = static_cast<std::size_t>(c);
        lists[ci].push_back(li);
        const bool track = (c == KetClass::HH || c == KetClass::PH);
        occ_prod[ci].push_back(track ? occupations[p] * occupations[q] : 0.0);
        vac_prod[ci].push_back(track ? (1.0 - occupations[p]) * (1.0 - occupations[q]) : 0.0);
      };
      if (!hole_p && !hole_q) add(KetClass::PP);
      if (hole_p && hole_q) add(KetClass::HH);
      if (hole_p != hole_q) add(KetClass::PH);
      const std::int32_t cp = cvq(classes[p]);
      const std::int32_t cq = cvq(classes[q]);
      if (cp + cq == 0) add(KetClass::CC);
      if (cp + cq == 1) add(KetClass::VC);
      if (std::abs(cp - cq) == 2) add(KetClass::QC);
      if (cp * cq == 1) add(KetClass::VV);
      if (cp + cq == 3) add(KetClass::QV);
      if (cp + cq == 4) add(KetClass::QQ);
    }
    for (std::size_t c = 0; c < kNumKetClasses; ++c) {
      subset_table_.insert(subset_table_.end(), lists[c].begin(), lists[c].end());
      occupation_products_.insert(occupation_products_.end(), occ_prod[c].begin(),
                                  occ_prod[c].end());
      vacancy_products_.insert(vacancy_products_.end(), vac_prod[c].begin(), vac_prod[c].end());
      subset_offsets_.push_back(static_cast<std::int32_t>(subset_table_.size()));
    }
  }
}

namespace {
/// Slot of (ch, c) in the offsets table.
std::size_t subset_slot(ChannelIndex ch, KetClass c) noexcept {
  return static_cast<std::size_t>(ch.value() * kNumKetClasses + static_cast<std::int32_t>(c));
}
}  // namespace

std::span<const LocalKetIndex> TwoBodyKetSubsets::kets(ChannelIndex ch, KetClass c) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels_, "channel index out of range");
  const auto slot = subset_slot(ch, c);
  const auto b = static_cast<std::size_t>(subset_offsets_[slot]);
  const auto e = static_cast<std::size_t>(subset_offsets_[slot + 1]);
  return std::span<const LocalKetIndex>(subset_table_).subspan(b, e - b);
}

std::span<const double> TwoBodyKetSubsets::occupation_products(ChannelIndex ch, KetClass c) const {
  IMSRG_CHECK(c == KetClass::HH || c == KetClass::PH, "occupation_products needs class HH or PH");
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels_, "channel index out of range");
  const auto slot = subset_slot(ch, c);
  const auto b = static_cast<std::size_t>(subset_offsets_[slot]);
  const auto e = static_cast<std::size_t>(subset_offsets_[slot + 1]);
  return std::span<const double>(occupation_products_).subspan(b, e - b);
}

std::span<const double> TwoBodyKetSubsets::vacancy_products(ChannelIndex ch, KetClass c) const {
  IMSRG_CHECK(c == KetClass::HH || c == KetClass::PH, "vacancy_products needs class HH or PH");
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels_, "channel index out of range");
  const auto slot = subset_slot(ch, c);
  const auto b = static_cast<std::size_t>(subset_offsets_[slot]);
  const auto e = static_cast<std::size_t>(subset_offsets_[slot + 1]);
  return std::span<const double>(vacancy_products_).subspan(b, e - b);
}

}  // namespace imsrg::basis
