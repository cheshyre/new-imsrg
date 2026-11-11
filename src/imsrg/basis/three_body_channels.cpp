#include "imsrg/basis/three_body_channels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <tuple>

#include "imsrg/base/assert.hpp"

namespace imsrg::basis {

namespace {

// ThreeBodyChannel::CheckChannel_ket (ThreeBodyChannel.cc:53-70) for a ket of orbits (a, b, c)
// with twice-j values (ja, jb, jc), l and tz2 given through the orbit list, coupled to
// 2 Jpq = two_jpq, and a channel (two_j, parity, two_tz). All rules are copied exactly,
// including the ones imsrg++ marks as redundant.
bool check_channel_ket(const Orbit& oa, const Orbit& ob, const Orbit& oc, OrbitIndex a,
                       OrbitIndex b, OrbitIndex c, std::int32_t two_jpq, std::int32_t two_j,
                       std::int32_t parity, std::int32_t two_tz) {
  const std::int32_t ja = oa.j2.twice;
  const std::int32_t jb = ob.j2.twice;
  const std::int32_t jc = oc.j2.twice;
  if ((oa.l + ob.l + oc.l) % 2 != parity) return false;
  if ((oa.tz2.twice + ob.tz2.twice + oc.tz2.twice) != two_tz) return false;
  if (std::abs(two_jpq - jc) > two_j || (two_jpq + jc) < two_j) return false;
  if (two_j == (ja + jb + jc)) {
    if (a == b || a == c || b == c) return false;
  }
  if (two_j == (ja + jb + jc) && (a == b || b == c)) return false;
  if (a == b && std::abs(two_j - jc) == 2 * ja) return false;
  if (b == c && std::abs(two_j - ja) == 2 * jb) return false;
  if (a == b && a == c && two_j > (3 * ja - 3)) return false;
  return true;
}

}  // namespace

ThreeBodyChannels::ThreeBodyChannels(const SingleParticleBasis& basis, const ThreeBodyCuts& cuts)
    : cuts_(cuts) {
  if (cuts_.emax_3body < 0) cuts_.emax_3body = basis.emax();
  IMSRG_CHECK(cuts_.emax_3body <= basis.emax(), "emax_3body exceeds the basis emax");
  IMSRG_CHECK(cuts_.e3max >= 0, "e3max must be non-negative");

  const std::span<const Orbit> orbits = basis.orbits();
  std::vector<std::int32_t> space;  // orbits of the three-body space, ascending index
  for (std::int32_t i = 0; i < basis.num_orbits(); ++i) {
    if (orbits[static_cast<std::size_t>(i)].e() <= cuts_.emax_3body) space.push_back(i);
  }

  // Global kets, in the loop order of Setup3bKets.
  for (std::size_t ip = 0; ip < space.size(); ++ip) {
    const Orbit& op = orbits[static_cast<std::size_t>(space[ip])];
    for (std::size_t iq = ip; iq < space.size(); ++iq) {
      const Orbit& oq = orbits[static_cast<std::size_t>(space[iq])];
      const std::int32_t jpq_min = std::abs(op.j2.twice - oq.j2.twice) / 2;
      const std::int32_t jpq_max = (op.j2.twice + oq.j2.twice) / 2;
      for (std::size_t ir = iq; ir < space.size(); ++ir) {
        const Orbit& orr = orbits[static_cast<std::size_t>(space[ir])];
        if (2 * (op.n + oq.n + orr.n) + op.l + oq.l + orr.l > cuts_.e3max) continue;
        for (std::int32_t jpq = jpq_min; jpq <= jpq_max; ++jpq) {
          if (ip == iq && jpq % 2 > 0) continue;
          if (ip == iq && iq == ir && op.j2.twice == 1) continue;
          kets_.push_back(Ket3{OrbitIndex{space[ip]}, OrbitIndex{space[iq]}, OrbitIndex{space[ir]},
                               TwoJ{2 * jpq}});
        }
      }
    }
  }
  IMSRG_CHECK(kets_.size() < static_cast<std::size_t>(INT32_MAX), "too many three-body kets");

  // Channels, in std::set order of (2J, parity, 2Tz).
  const std::int32_t num_two_j = 3 * cuts_.emax_3body + 1;
  const std::int32_t nch = 8 * num_two_j;
  channels_.resize(static_cast<std::size_t>(nch));
  for (std::int32_t ch = 0; ch < nch; ++ch) {
    ThreeBodyChannel& c = channels_[static_cast<std::size_t>(ch)];
    c.two_j = TwoJ{2 * (ch / 8) + 1};
    c.parity = ((ch % 8) / 4 == 0) ? Parity::Even : Parity::Odd;
    c.two_tz = TwoTz{2 * (ch % 4) - 3};
  }

  // Membership: for every ket, test the channels its 2J range reaches (parity and 2Tz are fixed
  // by the ket), and record the local indices in the compact per-ket table.
  std::vector<std::vector<Ket3Index>> members(static_cast<std::size_t>(nch));
  ket_twoJ_offsets_.assign(kets_.size() + 1, 0);
  ket_twoJ_min_.resize(kets_.size());
  ket_channel_base_.resize(kets_.size());
  for (std::size_t k = 0; k < kets_.size(); ++k) {
    const Ket3& kt = kets_[k];
    const Orbit& oa = orbits[static_cast<std::size_t>(kt.p.value())];
    const Orbit& ob = orbits[static_cast<std::size_t>(kt.q.value())];
    const Orbit& oc = orbits[static_cast<std::size_t>(kt.r.value())];
    const std::int32_t parity = (oa.l + ob.l + oc.l) % 2;
    const std::int32_t two_tz = oa.tz2.twice + ob.tz2.twice + oc.tz2.twice;
    const std::int32_t jr = oc.j2.twice;
    const std::int32_t jlo = std::abs(kt.two_jpq.twice - jr);
    const std::int32_t jhi = kt.two_jpq.twice + jr;
    ket_twoJ_min_[k] = jlo;
    ket_channel_base_[k] = parity * 4 + (two_tz + 3) / 2;
    ket_twoJ_offsets_[k + 1] = ket_twoJ_offsets_[k] + (jhi - jlo) / 2 + 1;
    for (std::int32_t two_j = jlo; two_j <= jhi; two_j += 2) {
      std::int32_t local = -1;
      const std::int32_t ch = ((two_j - 1) / 2) * 8 + ket_channel_base_[k];
      if (two_j >= 1 && ch < nch &&
          check_channel_ket(oa, ob, oc, kt.p, kt.q, kt.r, kt.two_jpq.twice, two_j, parity,
                            two_tz)) {
        auto& list = members[static_cast<std::size_t>(ch)];
        local = static_cast<std::int32_t>(list.size());
        list.push_back(Ket3Index{static_cast<std::int32_t>(k)});
      }
      local_index_table_.push_back(local);
    }
  }

  std::int64_t offset = 0;
  for (std::int32_t ch = 0; ch < nch; ++ch) {
    const auto& list = members[static_cast<std::size_t>(ch)];
    ThreeBodyChannel& c = channels_[static_cast<std::size_t>(ch)];
    c.num_kets = static_cast<std::int64_t>(list.size());
    c.ket_offset = offset;
    offset += c.num_kets;
    channel_ket_table_.insert(channel_ket_table_.end(), list.begin(), list.end());
  }
}

const Ket3& ThreeBodyChannels::ket(Ket3Index k) const {
  IMSRG_ASSERT(k.value() >= 0 && static_cast<std::size_t>(k.value()) < kets_.size(),
               "ket index out of range");
  return kets_[static_cast<std::size_t>(k.value())];
}

std::optional<Ket3Index> ThreeBodyChannels::find_ket(OrbitIndex p, OrbitIndex q, OrbitIndex r,
                                                     TwoJ two_jpq) const noexcept {
  const auto key = std::make_tuple(p.value(), q.value(), r.value(), two_jpq.twice);
  const auto it =
      std::lower_bound(kets_.begin(), kets_.end(), key, [](const Ket3& a, const decltype(key)& b) {
        return std::make_tuple(a.p.value(), a.q.value(), a.r.value(), a.two_jpq.twice) < b;
      });
  if (it == kets_.end() ||
      std::make_tuple(it->p.value(), it->q.value(), it->r.value(), it->two_jpq.twice) != key) {
    return std::nullopt;
  }
  return Ket3Index{static_cast<std::int32_t>(it - kets_.begin())};
}

const ThreeBodyChannel& ThreeBodyChannels::channel(Channel3Index ch) const {
  IMSRG_ASSERT(ch.value() >= 0 && ch.value() < num_channels(), "channel index out of range");
  return channels_[static_cast<std::size_t>(ch.value())];
}

std::optional<Channel3Index> ThreeBodyChannels::find_channel(TwoJ two_j, Parity parity,
                                                             TwoTz two_tz) const noexcept {
  if (two_j.twice < 1 || two_j.twice % 2 == 0 || two_j.twice > 6 * cuts_.emax_3body + 1) {
    return std::nullopt;
  }
  if (two_tz.twice < -3 || two_tz.twice > 3 || two_tz.twice % 2 == 0) return std::nullopt;
  return channel_index(two_j, parity, two_tz);
}

std::span<const Ket3Index> ThreeBodyChannels::channel_kets(Channel3Index ch) const {
  const ThreeBodyChannel& c = channel(ch);
  return std::span<const Ket3Index>(channel_ket_table_)
      .subspan(static_cast<std::size_t>(c.ket_offset), static_cast<std::size_t>(c.num_kets));
}

std::optional<LocalKet3Index> ThreeBodyChannels::local_index(Channel3Index ch, Ket3Index k) const {
  const ThreeBodyChannel& c = channel(ch);
  IMSRG_ASSERT(k.value() >= 0 && static_cast<std::size_t>(k.value()) < kets_.size(),
               "ket index out of range");
  const auto ik = static_cast<std::size_t>(k.value());
  if (ch.value() % 8 != ket_channel_base_[ik]) return std::nullopt;
  const std::int32_t two_j = c.two_j.twice;
  const std::int32_t jlo = ket_twoJ_min_[ik];
  const std::int64_t n = ket_twoJ_offsets_[ik + 1] - ket_twoJ_offsets_[ik];
  if (two_j < jlo || (two_j - jlo) / 2 >= n) return std::nullopt;
  const std::int32_t local =
      local_index_table_[static_cast<std::size_t>(ket_twoJ_offsets_[ik] + (two_j - jlo) / 2)];
  if (local < 0) return std::nullopt;
  return LocalKet3Index{local};
}

bool ThreeBodyChannels::passes_configuration_cut(
    Ket3Index k, const SingleParticleBasis& basis, const Reference& ref, double de3max,
    double occnat3_cut, std::span<const double> natural_occupations) const {
  const Ket3& kt = ket(k);
  const std::span<const double> occ =
      natural_occupations.empty() ? ref.occupations() : natural_occupations;
  IMSRG_ASSERT(static_cast<std::int32_t>(occ.size()) == basis.num_orbits(),
               "occupation count differs from the orbit count");
  double de = 0.0;
  double prod = 1.0;
  for (const OrbitIndex i : {kt.p, kt.q, kt.r}) {
    const Orbit& o = basis.orbit(i);
    de += std::abs(static_cast<double>(o.e()) - ref.e_fermi(o.tz2));
    const double n = occ[static_cast<std::size_t>(i.value())];
    prod *= n * (1.0 - n);
  }
  return de <= de3max && prod >= occnat3_cut;
}

}  // namespace imsrg::basis
