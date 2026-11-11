#include "imsrg/operator/no2b.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "imsrg/angmom/wigner.hpp"
#include "imsrg/base/assert.hpp"
#include "imsrg/base/error.hpp"

namespace imsrg::op {

namespace {

std::uint64_t state_key(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t tab) noexcept {
  return static_cast<std::uint64_t>(a) | (static_cast<std::uint64_t>(b) << 16) |
         (static_cast<std::uint64_t>(c) << 32) | (static_cast<std::uint64_t>(tab) << 48);
}

// Twice-unit triangle condition including the parity of the sum.
bool triangle2(std::int32_t a, std::int32_t b, std::int32_t c) noexcept {
  return (a + b + c) % 2 == 0 && c >= std::abs(a - b) && c <= a + b;
}

// Dense channel table: J2 in 0..J2max, P2 in {0,1}, (J1-1)/2 in 0..lmax, P1 in {0,1}, (T3-1)/2.
std::size_t lookup_index(std::int32_t J2, std::int32_t P2, std::int32_t J1, std::int32_t P1,
                         std::int32_t T3, std::int32_t lmax) noexcept {
  return ((((static_cast<std::size_t>(J2) * 2 + static_cast<std::size_t>(P2)) *
                static_cast<std::size_t>(lmax + 1) +
            static_cast<std::size_t>((J1 - 1) / 2)) *
               2 +
           static_cast<std::size_t>(P1)) *
              2 +
          static_cast<std::size_t>((T3 - 1) / 2));
}

}  // namespace

No2bThreeBody::No2bThreeBody(const No2bTruncation& t) : trunc_(t) {
  if (t.emax < 0 || t.e2max < 0 || t.e3max < 0 || t.lmax < 0) {
    throw InputError("No2bThreeBody: truncations must be non-negative");
  }
  for (std::int32_t e = 0; e <= t.emax; ++e) {
    for (std::int32_t l = e % 2; l <= std::min(e, t.lmax); l += 2) {
      const std::int32_t n = (e - l) / 2;
      for (std::int32_t tj = std::abs(2 * l - 1); tj <= 2 * l + 1; tj += 2) {
        orbits_.push_back(IsospinOrbit{n, l, TwoJ{tj}});
      }
    }
  }
  const auto norb = static_cast<std::int32_t>(orbits_.size());
  const std::int32_t j2max = std::min(2 * t.lmax + 1, t.e2max + 1);
  channel_lookup_.assign(
      static_cast<std::size_t>(j2max + 1) * 2 * static_cast<std::size_t>(t.lmax + 1) * 2 * 2, -1);
  std::size_t offset = 0;
  for (std::int32_t J2 = 0; J2 <= j2max; ++J2) {
    for (std::int32_t P2 : {0, 1}) {
      for (std::int32_t J1 = 1; J1 <= 2 * t.lmax + 1; J1 += 2) {
        for (std::int32_t P1 : {0, 1}) {
          for (std::int32_t T3 : {1, 3}) {
            Channel ch;
            ch.J2 = J2;
            ch.P2 = P2;
            ch.J1 = J1;
            ch.P1 = P1;
            ch.T3 = T3;
            std::int32_t ndim = 0;
            for (std::int32_t ia = 0; ia < norb; ++ia) {
              const auto& oa = orbits_[static_cast<std::size_t>(ia)];
              if (oa.e() > t.emax) continue;
              for (std::int32_t ib = ia; ib < norb; ++ib) {
                const auto& ob = orbits_[static_cast<std::size_t>(ib)];
                if (ob.e() > t.emax) continue;
                if (oa.e() + ob.e() > t.e2max) continue;
                if (!triangle2(oa.j2.twice, ob.j2.twice, 2 * J2)) continue;
                if ((oa.l + ob.l) % 2 != P2) continue;
                for (std::int32_t ic = 0; ic < norb; ++ic) {
                  const auto& oc = orbits_[static_cast<std::size_t>(ic)];
                  if (oc.j2.twice != J1) continue;
                  if (oc.l % 2 != P1) continue;
                  if (oc.e() > t.emax) continue;
                  if (oa.e() + oc.e() > t.e2max) continue;
                  if (ob.e() + oc.e() > t.e2max) continue;
                  if (oa.e() + ob.e() + oc.e() > t.e3max) continue;
                  for (std::int32_t tab : {0, 1}) {
                    if (!triangle2(2 * tab, 1, T3)) continue;
                    if ((J2 + tab) % 2 == 0 && ia == ib) continue;
                    const std::int32_t ph =
                        1 - 2 * (((oa.j2.twice + ob.j2.twice) / 2 + J2 + tab) % 2);
                    ch.states[state_key(ia, ib, ic, tab)] = {ndim, 1};
                    ch.states[state_key(ib, ia, ic, tab)] = {ndim, ph};
                    ++ndim;
                  }
                }
              }
            }
            if (ndim < 1) continue;
            ch.num_states = ndim;
            ch.offset = offset;
            offset += static_cast<std::size_t>(ndim) * (static_cast<std::size_t>(ndim) + 1) / 2;
            channel_lookup_[lookup_index(J2, P2, J1, P1, T3, t.lmax)] =
                static_cast<std::int32_t>(channels_.size());
            channels_.push_back(std::move(ch));
          }
        }
      }
    }
  }
  values_.assign(offset, 0.0F);
}

std::optional<std::int32_t> No2bThreeBody::isospin_orbit_index(std::int32_t n, std::int32_t l,
                                                               TwoJ j2) const noexcept {
  for (std::size_t i = 0; i < orbits_.size(); ++i) {
    if (orbits_[i].n == n && orbits_[i].l == l && orbits_[i].j2 == j2) {
      return static_cast<std::int32_t>(i);
    }
  }
  return std::nullopt;
}

std::optional<std::int32_t> No2bThreeBody::channel_index(std::int32_t J2, std::int32_t P2,
                                                         std::int32_t J1, std::int32_t P1,
                                                         std::int32_t T3) const noexcept {
  if (channel_lookup_.empty()) return std::nullopt;
  const std::int32_t j2max = std::min(2 * trunc_.lmax + 1, trunc_.e2max + 1);
  if (J2 < 0 || J2 > j2max || (P2 != 0 && P2 != 1) || (P1 != 0 && P1 != 1)) return std::nullopt;
  if (J1 < 1 || J1 > 2 * trunc_.lmax + 1 || J1 % 2 == 0) return std::nullopt;
  if (T3 != 1 && T3 != 3) return std::nullopt;
  const std::int32_t c = channel_lookup_[lookup_index(J2, P2, J1, P1, T3, trunc_.lmax)];
  if (c < 0) return std::nullopt;
  return c;
}

void No2bThreeBody::set_isospin(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t Tab,
                                std::int32_t d, std::int32_t e, std::int32_t f, std::int32_t Tde,
                                std::int32_t J2, std::int32_t twoT, double v) {
  const auto norb = static_cast<std::int32_t>(orbits_.size());
  for (std::int32_t i : {a, b, c, d, e, f}) {
    IMSRG_CHECK(i >= 0 && i < norb, "No2bThreeBody::set_isospin: orbit index out of range");
  }
  const auto& oa = orbits_[static_cast<std::size_t>(a)];
  const auto& ob = orbits_[static_cast<std::size_t>(b)];
  const auto& oc = orbits_[static_cast<std::size_t>(c)];
  const auto& od = orbits_[static_cast<std::size_t>(d)];
  const auto& oe = orbits_[static_cast<std::size_t>(e)];
  const auto& of = orbits_[static_cast<std::size_t>(f)];
  const std::int32_t P1 = oc.l % 2;
  IMSRG_CHECK(P1 == of.l % 2, "No2bThreeBody::set_isospin: parity of c and f differs");
  IMSRG_CHECK(oc.j2 == of.j2, "No2bThreeBody::set_isospin: j_c != j_f");
  const std::int32_t P2 = (oa.l + ob.l) % 2;
  IMSRG_CHECK(P2 == (od.l + oe.l) % 2, "No2bThreeBody::set_isospin: pair parities differ");
  const auto ch = channel_index(J2, P2, oc.j2.twice, P1, twoT);
  IMSRG_CHECK(ch.has_value(), "No2bThreeBody::set_isospin: channel not stored");
  const Channel& channel = channels_[static_cast<std::size_t>(*ch)];
  const auto bra = channel.states.find(state_key(a, b, c, Tab));
  const auto ket = channel.states.find(state_key(d, e, f, Tde));
  IMSRG_CHECK(bra != channel.states.end() && ket != channel.states.end(),
              "No2bThreeBody::set_isospin: state not stored (truncation or Pauli)");
  const auto ib = static_cast<std::size_t>(bra->second.first);
  const auto ik = static_cast<std::size_t>(ket->second.first);
  const std::size_t hi = std::max(ib, ik);
  const std::size_t lo = std::min(ib, ik);
  const std::int32_t ph = bra->second.second * ket->second.second;
  values_[channel.offset + hi * (hi + 1) / 2 + lo] = static_cast<float>(v * ph);
}

double No2bThreeBody::get_isospin(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t Tab,
                                  std::int32_t d, std::int32_t e, std::int32_t f, std::int32_t Tde,
                                  std::int32_t J2, std::int32_t twoT) const noexcept {
  const auto norb = static_cast<std::int32_t>(orbits_.size());
  for (std::int32_t i : {a, b, c, d, e, f}) {
    if (i < 0 || i >= norb) return 0.0;
  }
  const auto& oa = orbits_[static_cast<std::size_t>(a)];
  const auto& ob = orbits_[static_cast<std::size_t>(b)];
  const auto& oc = orbits_[static_cast<std::size_t>(c)];
  const auto& od = orbits_[static_cast<std::size_t>(d)];
  const auto& oe = orbits_[static_cast<std::size_t>(e)];
  const auto& of = orbits_[static_cast<std::size_t>(f)];
  const std::int32_t P1 = oc.l % 2;
  if (P1 != of.l % 2 || oc.j2 != of.j2) return 0.0;
  const std::int32_t P2 = (oa.l + ob.l) % 2;
  if (P2 != (od.l + oe.l) % 2) return 0.0;
  const auto ch = channel_index(J2, P2, oc.j2.twice, P1, twoT);
  if (!ch) return 0.0;
  const Channel& channel = channels_[static_cast<std::size_t>(*ch)];
  const auto bra = channel.states.find(state_key(a, b, c, Tab));
  const auto ket = channel.states.find(state_key(d, e, f, Tde));
  if (bra == channel.states.end() || ket == channel.states.end()) return 0.0;
  const auto ib = static_cast<std::size_t>(bra->second.first);
  const auto ik = static_cast<std::size_t>(ket->second.first);
  const std::size_t hi = std::max(ib, ik);
  const std::size_t lo = std::min(ib, ik);
  const std::int32_t ph = bra->second.second * ket->second.second;
  return ph * static_cast<double>(values_[channel.offset + hi * (hi + 1) / 2 + lo]);
}

double No2bThreeBody::get_pn(const basis::Orbit& a, const basis::Orbit& b, const basis::Orbit& c,
                             const basis::Orbit& d, const basis::Orbit& e, const basis::Orbit& f,
                             std::int32_t J2) const noexcept {
  if (c.l % 2 != f.l % 2 || c.j2 != f.j2) return 0.0;
  if ((a.l + b.l) % 2 != (d.l + e.l) % 2) return 0.0;
  const std::int32_t z3 = a.tz2.twice + b.tz2.twice + c.tz2.twice;
  if (z3 != d.tz2.twice + e.tz2.twice + f.tz2.twice) return 0.0;
  const auto i1 = isospin_orbit_index(a.n, a.l, a.j2);
  const auto i2 = isospin_orbit_index(b.n, b.l, b.j2);
  const auto i3 = isospin_orbit_index(c.n, c.l, c.j2);
  const auto i4 = isospin_orbit_index(d.n, d.l, d.j2);
  const auto i5 = isospin_orbit_index(e.n, e.l, e.j2);
  const auto i6 = isospin_orbit_index(f.n, f.l, f.j2);
  if (!i1 || !i2 || !i3 || !i4 || !i5 || !i6) return 0.0;
  const std::int32_t zab = a.tz2.twice + b.tz2.twice;
  const std::int32_t zde = d.tz2.twice + e.tz2.twice;
  const TwoJ half{1};
  double vout = 0.0;
  for (std::int32_t t12 : {0, 1}) {
    if (std::abs(zab) > 2 * t12) continue;
    for (std::int32_t t45 : {0, 1}) {
      if (std::abs(zde) > 2 * t45) continue;
      for (std::int32_t twoT = 1; twoT <= std::min(2 * t12 + 1, 2 * t45 + 1); twoT += 2) {
        const double w = get_isospin(*i1, *i2, *i3, t12, *i4, *i5, *i6, t45, J2, twoT);
        vout += w *
                angmom::clebsch_gordan(half, TwoM{a.tz2.twice}, half, TwoM{b.tz2.twice},
                                       TwoJ{2 * t12}, TwoM{zab}) *
                angmom::clebsch_gordan(half, TwoM{d.tz2.twice}, half, TwoM{e.tz2.twice},
                                       TwoJ{2 * t45}, TwoM{zde}) *
                angmom::clebsch_gordan(TwoJ{2 * t12}, TwoM{zab}, half, TwoM{c.tz2.twice},
                                       TwoJ{twoT}, TwoM{z3}) *
                angmom::clebsch_gordan(TwoJ{2 * t45}, TwoM{zde}, half, TwoM{f.tz2.twice},
                                       TwoJ{twoT}, TwoM{z3});
      }
    }
  }
  return vout;
}

double No2bThreeBody::norm() const noexcept {
  double s = 0.0;
  for (const float v : values_) s += static_cast<double>(v) * static_cast<double>(v);
  return std::sqrt(s);
}

}  // namespace imsrg::op
