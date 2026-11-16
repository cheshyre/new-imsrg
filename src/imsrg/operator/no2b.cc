// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/operator/no2b.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/error.h"

namespace imsrg::op {

namespace {

std::uint64_t StateKey(std::int32_t a, std::int32_t b, std::int32_t c,
                       std::int32_t tab) noexcept {
  return static_cast<std::uint64_t>(a) | (static_cast<std::uint64_t>(b) << 16) |
         (static_cast<std::uint64_t>(c) << 32) |
         (static_cast<std::uint64_t>(tab) << 48);
}

// Twice-unit triangle condition including the parity of the sum.
bool Triangle2(std::int32_t a, std::int32_t b, std::int32_t c) noexcept {
  return (a + b + c) % 2 == 0 && c >= std::abs(a - b) && c <= a + b;
}

// Dense channel table: J2 in 0..J2max, P2 in {0,1}, (J1-1)/2 in 0..lmax, P1 in
// {0,1}, (T3-1)/2.
std::size_t LookupIndex(std::int32_t J2, std::int32_t P2, std::int32_t J1,
                        std::int32_t P1, std::int32_t T3,
                        std::int32_t lmax) noexcept {
  return ((((static_cast<std::size_t>(J2) * 2 + static_cast<std::size_t>(P2)) *
                static_cast<std::size_t>(lmax + 1) +
            static_cast<std::size_t>((J1 - 1) / 2)) *
               2 +
           static_cast<std::size_t>(P1)) *
              2 +
          static_cast<std::size_t>((T3 - 1) / 2));
}

// Isospin Clebsch-Gordan tables for get_pn, filled once by the same
// angmom::ClebschGordan calls (same arguments) as a direct evaluation, so the
// values are bitwise identical. Indices: m = +-1/2 -> (2m + 1) / 2; zab = -2,
// 0, 2 -> (zab + 2) / 2; t in {0, 1}; (2T - 1) / 2 for T in {1/2, 3/2}.
struct IsospinCgTables {
  // (1/2 ma 1/2 mb | t zab): [ma][mb][t].
  double pair[2][2][2];
  // (t zab 1/2 mc | T z3): [t][zab][mc][T].
  double couple[2][3][2][2];
};

const IsospinCgTables& IsospinCg() {
  static const IsospinCgTables* const kTables = [] {
    auto* tab = new IsospinCgTables{};
    const TwoJ half{1};
    for (std::int32_t ia = 0; ia < 2; ++ia) {
      for (std::int32_t ib = 0; ib < 2; ++ib) {
        for (std::int32_t t = 0; t < 2; ++t) {
          const std::int32_t ma = 2 * ia - 1;
          const std::int32_t mb = 2 * ib - 1;
          const std::int32_t zab = ma + mb;
          tab->pair[ia][ib][t] =
              std::abs(zab) > 2 * t
                  ? 0.0
                  : angmom::ClebschGordan(half, TwoM{ma}, half, TwoM{mb},
                                          TwoJ{2 * t}, TwoM{zab});
        }
      }
    }
    for (std::int32_t t = 0; t < 2; ++t) {
      for (std::int32_t iz = 0; iz < 3; ++iz) {
        for (std::int32_t ic = 0; ic < 2; ++ic) {
          for (std::int32_t iT = 0; iT < 2; ++iT) {
            const std::int32_t zab = 2 * iz - 2;
            const std::int32_t mc = 2 * ic - 1;
            tab->couple[t][iz][ic][iT] =
                std::abs(zab) > 2 * t
                    ? 0.0
                    : angmom::ClebschGordan(TwoJ{2 * t}, TwoM{zab}, half,
                                            TwoM{mc}, TwoJ{2 * iT + 1},
                                            TwoM{zab + mc});
          }
        }
      }
    }
    return tab;
  }();
  return *kTables;
}

/// Fills `ch.states` with the basis states (a <= b, c, t_ab) of the channel
/// (J2, P2, J1, P1, T3) in the order a, b, c, t_ab (a state with a == b and
/// odd J2 + t_ab is absent; the swapped key carries the exchange phase) and
/// returns their number.
template <class Channel>
std::int32_t FillChannelStates(const No2bTruncation& t,
                               const std::vector<IsospinOrbit>& orbits,
                               std::int32_t J2, std::int32_t P2,
                               std::int32_t J1, std::int32_t P1,
                               std::int32_t T3, Channel& ch) {
  const auto norb = static_cast<std::int32_t>(orbits.size());
  std::int32_t ndim = 0;
  for (std::int32_t ia = 0; ia < norb; ++ia) {
    const auto& oa = orbits[static_cast<std::size_t>(ia)];
    if (oa.e() > t.emax) continue;
    for (std::int32_t ib = ia; ib < norb; ++ib) {
      const auto& ob = orbits[static_cast<std::size_t>(ib)];
      if (ob.e() > t.emax) continue;
      if (oa.e() + ob.e() > t.e2max) continue;
      if (!Triangle2(oa.j2.twice, ob.j2.twice, 2 * J2)) continue;
      if ((oa.l + ob.l) % 2 != P2) continue;
      for (std::int32_t ic = 0; ic < norb; ++ic) {
        const auto& oc = orbits[static_cast<std::size_t>(ic)];
        if (oc.j2.twice != J1) continue;
        if (oc.l % 2 != P1) continue;
        if (oc.e() > t.emax) continue;
        if (oa.e() + oc.e() > t.e2max) continue;
        if (ob.e() + oc.e() > t.e2max) continue;
        if (oa.e() + ob.e() + oc.e() > t.e3max) continue;
        for (std::int32_t tab : {0, 1}) {
          if (!Triangle2(2 * tab, 1, T3)) continue;
          if ((J2 + tab) % 2 == 0 && ia == ib) continue;
          const std::int32_t ph =
              1 - 2 * (((oa.j2.twice + ob.j2.twice) / 2 + J2 + tab) % 2);
          ch.states[StateKey(ia, ib, ic, tab)] = {ndim, 1};
          ch.states[StateKey(ib, ia, ic, tab)] = {ndim, ph};
          ++ndim;
        }
      }
    }
  }
  return ndim;
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
  orbit_stride_l_ = 2;
  orbit_stride_n_ = (t.lmax + 1) * 2;
  orbit_lookup_.assign(
      static_cast<std::size_t>((t.emax / 2 + 1) * orbit_stride_n_), -1);
  for (std::int32_t i = 0; i < norb; ++i) {
    const auto& o = orbits_[static_cast<std::size_t>(i)];
    orbit_lookup_[static_cast<std::size_t>(o.n * orbit_stride_n_ +
                                           o.l * orbit_stride_l_ +
                                           (o.j2.twice - 2 * o.l + 1) / 2)] = i;
  }
  const std::int32_t j2max = std::min(2 * t.lmax + 1, t.e2max + 1);
  channel_lookup_.assign(static_cast<std::size_t>(j2max + 1) * 2 *
                             static_cast<std::size_t>(t.lmax + 1) * 2 * 2,
                         -1);
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
            const std::int32_t ndim =
                FillChannelStates(t, orbits_, J2, P2, J1, P1, T3, ch);
            if (ndim < 1) continue;
            ch.num_states = ndim;
            ch.offset = offset;
            offset += static_cast<std::size_t>(ndim) *
                      (static_cast<std::size_t>(ndim) + 1) / 2;
            channel_lookup_[LookupIndex(J2, P2, J1, P1, T3, t.lmax)] =
                static_cast<std::int32_t>(channels_.size());
            channels_.push_back(std::move(ch));
          }
        }
      }
    }
  }
  values_.assign(offset, 0.0F);
}

std::optional<std::int32_t> No2bThreeBody::isospin_orbit_index(
    std::int32_t n, std::int32_t l, TwoJ j2) const noexcept {
  if (n < 0 || l < 0 || orbit_lookup_.empty()) return std::nullopt;
  const std::int32_t d =
      j2.twice - 2 * l + 1;  // 0 for j = l - 1/2, 2 for l+1/2
  if (d != 0 && d != 2) return std::nullopt;
  if (l > trunc_.lmax ||
      n >= static_cast<std::int32_t>(orbit_lookup_.size()) / orbit_stride_n_) {
    return std::nullopt;
  }
  const std::int32_t i = orbit_lookup_[static_cast<std::size_t>(
      n * orbit_stride_n_ + l * orbit_stride_l_ + d / 2)];
  if (i < 0) return std::nullopt;
  return i;
}

std::optional<std::int32_t> No2bThreeBody::channel_index(
    std::int32_t J2, std::int32_t P2, std::int32_t J1, std::int32_t P1,
    std::int32_t T3) const noexcept {
  if (channel_lookup_.empty()) return std::nullopt;
  const std::int32_t j2max = std::min(2 * trunc_.lmax + 1, trunc_.e2max + 1);
  if (J2 < 0 || J2 > j2max || (P2 != 0 && P2 != 1) || (P1 != 0 && P1 != 1))
    return std::nullopt;
  if (J1 < 1 || J1 > 2 * trunc_.lmax + 1 || J1 % 2 == 0) return std::nullopt;
  if (T3 != 1 && T3 != 3) return std::nullopt;
  const std::int32_t c =
      channel_lookup_[LookupIndex(J2, P2, J1, P1, T3, trunc_.lmax)];
  if (c < 0) return std::nullopt;
  return c;
}

void No2bThreeBody::set_isospin(std::int32_t a, std::int32_t b, std::int32_t c,
                                std::int32_t Tab, std::int32_t d,
                                std::int32_t e, std::int32_t f,
                                std::int32_t Tde, std::int32_t J2,
                                std::int32_t twoT, double v) {
  const auto norb = static_cast<std::int32_t>(orbits_.size());
  for (std::int32_t i : {a, b, c, d, e, f}) {
    IMSRG_CHECK(i >= 0 && i < norb,
                "No2bThreeBody::set_isospin: orbit index out of range");
  }
  const auto& oa = orbits_[static_cast<std::size_t>(a)];
  const auto& ob = orbits_[static_cast<std::size_t>(b)];
  const auto& oc = orbits_[static_cast<std::size_t>(c)];
  const auto& od = orbits_[static_cast<std::size_t>(d)];
  const auto& oe = orbits_[static_cast<std::size_t>(e)];
  const auto& of = orbits_[static_cast<std::size_t>(f)];
  const std::int32_t P1 = oc.l % 2;
  IMSRG_CHECK(P1 == of.l % 2,
              "No2bThreeBody::set_isospin: parity of c and f differs");
  IMSRG_CHECK(oc.j2 == of.j2, "No2bThreeBody::set_isospin: j_c != j_f");
  const std::int32_t P2 = (oa.l + ob.l) % 2;
  IMSRG_CHECK(P2 == (od.l + oe.l) % 2,
              "No2bThreeBody::set_isospin: pair parities differ");
  const auto ch = channel_index(J2, P2, oc.j2.twice, P1, twoT);
  IMSRG_CHECK(ch.has_value(), "No2bThreeBody::set_isospin: channel not stored");
  const Channel& channel = channels_[static_cast<std::size_t>(*ch)];
  const auto bra = channel.states.find(StateKey(a, b, c, Tab));
  const auto ket = channel.states.find(StateKey(d, e, f, Tde));
  IMSRG_CHECK(
      bra != channel.states.end() && ket != channel.states.end(),
      "No2bThreeBody::set_isospin: state not stored (truncation or Pauli)");
  const auto ib = static_cast<std::size_t>(bra->second.first);
  const auto ik = static_cast<std::size_t>(ket->second.first);
  const std::size_t hi = std::max(ib, ik);
  const std::size_t lo = std::min(ib, ik);
  const std::int32_t ph = bra->second.second * ket->second.second;
  values_[channel.offset + hi * (hi + 1) / 2 + lo] = static_cast<float>(v * ph);
}

double No2bThreeBody::get_isospin(std::int32_t a, std::int32_t b,
                                  std::int32_t c, std::int32_t Tab,
                                  std::int32_t d, std::int32_t e,
                                  std::int32_t f, std::int32_t Tde,
                                  std::int32_t J2,
                                  std::int32_t twoT) const noexcept {
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
  const auto bra = channel.states.find(StateKey(a, b, c, Tab));
  const auto ket = channel.states.find(StateKey(d, e, f, Tde));
  if (bra == channel.states.end() || ket == channel.states.end()) return 0.0;
  const auto ib = static_cast<std::size_t>(bra->second.first);
  const auto ik = static_cast<std::size_t>(ket->second.first);
  const std::size_t hi = std::max(ib, ik);
  const std::size_t lo = std::min(ib, ik);
  const std::int32_t ph = bra->second.second * ket->second.second;
  return ph *
         static_cast<double>(values_[channel.offset + hi * (hi + 1) / 2 + lo]);
}

double No2bThreeBody::get_pn(const basis::Orbit& a, const basis::Orbit& b,
                             const basis::Orbit& c, const basis::Orbit& d,
                             const basis::Orbit& e, const basis::Orbit& f,
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
  const auto& cg = IsospinCg();
  const auto ma = static_cast<std::size_t>((a.tz2.twice + 1) / 2);
  const auto mb = static_cast<std::size_t>((b.tz2.twice + 1) / 2);
  const auto md = static_cast<std::size_t>((d.tz2.twice + 1) / 2);
  const auto me = static_cast<std::size_t>((e.tz2.twice + 1) / 2);
  const auto mc = static_cast<std::size_t>((c.tz2.twice + 1) / 2);
  const auto mf = static_cast<std::size_t>((f.tz2.twice + 1) / 2);
  const auto iab = static_cast<std::size_t>((zab + 2) / 2);
  const auto ide = static_cast<std::size_t>((zde + 2) / 2);
  double vout = 0.0;
  for (std::int32_t t12 : {0, 1}) {
    if (std::abs(zab) > 2 * t12) continue;
    for (std::int32_t t45 : {0, 1}) {
      if (std::abs(zde) > 2 * t45) continue;
      for (std::int32_t twoT = 1; twoT <= std::min(2 * t12 + 1, 2 * t45 + 1);
           twoT += 2) {
        const double w =
            get_isospin(*i1, *i2, *i3, t12, *i4, *i5, *i6, t45, J2, twoT);
        const auto iT = static_cast<std::size_t>((twoT - 1) / 2);
        vout += w * cg.pair[ma][mb][t12] * cg.pair[md][me][t45] *
                cg.couple[t12][iab][mc][iT] * cg.couple[t45][ide][mf][iT];
      }
    }
  }
  return vout;
}

double No2bThreeBody::norm() const noexcept {
  double s = 0.0;
  for (const float v : values_)
    s += static_cast<double>(v) * static_cast<double>(v);
  return std::sqrt(s);
}

}  // namespace imsrg::op
