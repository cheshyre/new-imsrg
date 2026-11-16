// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include "imsrg/commutator/triples.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "imsrg/angmom/sixj3_table.h"
#include "imsrg/base/assert.h"
#include "imsrg/base/constants.h"
#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/three_body_space.h"
#include "imsrg/basis/two_body_channels.h"
#include "imsrg/commutator/detail/three_body_tiles.h"
#include "imsrg/exec/device_buffer.h"
#include "imsrg/operator/pair_third.h"

namespace imsrg::commutator {

namespace {

constexpr std::size_t U(std::int64_t i) { return static_cast<std::size_t>(i); }

template <class T>
exec::DeviceBuffer<T> Upload(const std::vector<T>& v, const char* label) {
  exec::DeviceBuffer<T> buf(v.size(), label);
  if (!v.empty()) buf.CopyFromHost(std::span<const T>(v));
  return buf;
}

template <class T>
std::int64_t BufferBytes(const exec::DeviceBuffer<T>& b) {
  return static_cast<std::int64_t>(b.size() * sizeof(T));
}

// ------------------------------------------------------------ enumeration

/// The kept rows of the reduced space under construction, per channel of the
/// full channel numbering (`ThreeBodyChannels::channel_index`).
struct ReducedRows {
  std::vector<std::vector<std::int32_t>> two_jab;  ///< Doubled Jab per row.
  std::vector<std::vector<basis::OrbitTriple>> triples;
  std::vector<char> relevant;  ///< Channel holds a kept hhh triple.
};

/// Occupation products n_a n_b n_c and nbar_a nbar_b nbar_c of orbits a, b, c.
void Occupations(std::span<const double> occ, std::int32_t a, std::int32_t b,
                 std::int32_t c, double& o, double& u) {
  const double na = occ[U(a)];
  const double nb = occ[U(b)];
  const double nc = occ[U(c)];
  o = na * nb * nc;
  u = (1.0 - na) * (1.0 - nb) * (1.0 - nc);
}

/// Calls `fn(ch, two_jab)` for every ket |(a b) Jab, c; J> of the orbit triple
/// (a <= b <= c) that `ThreeBodyChannels` would hold (loop order of
/// `Setup3bKets`, channel rule `basis::CheckChannelKet`), in ascending Jab and
/// then ascending J, with `ch` the full channel index. `num_channels` is
/// 8 (3 emax_3body + 1). O(Jab range x J range).
template <class Fn>
void ForEachTripleRow(const basis::SingleParticleBasis& basis, std::int32_t a,
                      std::int32_t b, std::int32_t c, std::int32_t num_channels,
                      Fn&& fn) {
  const basis::Orbit& oa = basis.orbit(OrbitIndex{a});
  const basis::Orbit& ob = basis.orbit(OrbitIndex{b});
  const basis::Orbit& oc = basis.orbit(OrbitIndex{c});
  const std::int32_t jpq_min = std::abs(oa.j2.twice - ob.j2.twice) / 2;
  const std::int32_t jpq_max = (oa.j2.twice + ob.j2.twice) / 2;
  const std::int32_t parity = (oa.l + ob.l + oc.l) % 2;
  const std::int32_t two_tz = oa.tz2.twice + ob.tz2.twice + oc.tz2.twice;
  const std::int32_t jr = oc.j2.twice;
  for (std::int32_t jpq = jpq_min; jpq <= jpq_max; ++jpq) {
    if (a == b && jpq % 2 > 0) continue;
    if (a == b && b == c && oa.j2.twice == 1) continue;
    const std::int32_t jlo = std::abs(2 * jpq - jr);
    const std::int32_t jhi = 2 * jpq + jr;
    for (std::int32_t two_j = jlo; two_j <= jhi; two_j += 2) {
      const std::int32_t ch =
          ((two_j - 1) / 2) * 8 + parity * 4 + (two_tz + 3) / 2;
      if (two_j >= 1 && ch < num_channels &&
          basis::CheckChannelKet(oa, ob, oc, OrbitIndex{a}, OrbitIndex{b},
                                 OrbitIndex{c}, 2 * jpq, two_j, parity,
                                 two_tz)) {
        fn(ch, 2 * jpq);
      }
    }
  }
}

/// The reduced kept space of `ms` for `cuts` (see the file comment of
/// triples.h) and its counts. Two passes over the orbit triples a <= b <= c
/// that pass E3max and the configuration cut: the first marks the channels
/// with an hhh triple, the second collects the hhh/ppp triples of those
/// channels with their rows.
basis::ThreeBodySpace EnumerateReducedSpace(const basis::ModelSpace& ms,
                                            const basis::ThreeBodyCuts& cuts,
                                            TriplesCounts& counts) {
  const basis::SingleParticleBasis& basis = ms.basis();
  const std::span<const basis::Orbit> orbits = basis.orbits();
  const std::span<const double> occ = ms.reference().occupations();
  const std::int32_t num_channels = 8 * (3 * cuts.emax_3body + 1);
  std::vector<std::int32_t> space;  // orbits of the three-body space
  for (std::int32_t i = 0; i < basis.num_orbits(); ++i) {
    if (orbits[U(i)].e() <= cuts.emax_3body) space.push_back(i);
  }
  ReducedRows rr;
  rr.two_jab.resize(U(num_channels));
  rr.triples.resize(U(num_channels));
  rr.relevant.assign(U(num_channels), 0);

  // Calls fn(a, b, c, hhh, ppp, passes_cut) for every triple a <= b <= c with
  // n_T >= skip (pass 1) or n_T >= skip or nbar_T >= skip (pass 2).
  const auto visit = [&](bool hhh_only, auto&& fn) {
    for (std::size_t ia = 0; ia < space.size(); ++ia) {
      const std::int32_t a = space[ia];
      const basis::Orbit& oa = orbits[U(a)];
      for (std::size_t ib = ia; ib < space.size(); ++ib) {
        const std::int32_t b = space[ib];
        const basis::Orbit& ob = orbits[U(b)];
        for (std::size_t ic = ib; ic < space.size(); ++ic) {
          const std::int32_t c = space[ic];
          const basis::Orbit& oc = orbits[U(c)];
          if (2 * (oa.n + ob.n + oc.n) + oa.l + ob.l + oc.l > cuts.e3max) {
            continue;
          }
          double o = 0.0;
          double u = 0.0;
          Occupations(occ, a, b, c, o, u);
          const bool hhh = o >= detail::kTriplesSkip;
          const bool ppp = u >= detail::kTriplesSkip;
          if (!(hhh || (!hhh_only && ppp))) continue;
          const bool passes = basis::PassesConfigurationCut(
              OrbitIndex{a}, OrbitIndex{b}, OrbitIndex{c}, basis,
              ms.reference(), cuts.de3max, cuts.occnat3_cut,
              cuts.natural_occupations);
          fn(a, b, c, hhh, ppp, passes);
        }
      }
    }
  };

  visit(true, [&](std::int32_t a, std::int32_t b, std::int32_t c, bool, bool,
                  bool passes) {
    if (!passes) return;
    ForEachTripleRow(
        basis, a, b, c, num_channels,
        [&](std::int32_t ch, std::int32_t) { rr.relevant[U(ch)] = 1; });
  });

  visit(false, [&](std::int32_t a, std::int32_t b, std::int32_t c, bool hhh,
                   bool ppp, bool passes) {
    std::int32_t kept_rows = 0;
    std::int64_t before = 0;
    ForEachTripleRow(
        basis, a, b, c, num_channels, [&](std::int32_t ch, std::int32_t jab) {
          if (rr.relevant[U(ch)] == 0) return;
          ++before;
          if (!passes) return;
          std::vector<basis::OrbitTriple>& tl = rr.triples[U(ch)];
          std::vector<std::int32_t>& rows = rr.two_jab[U(ch)];
          if (tl.empty() || tl.back().a.value() != a ||
              tl.back().b.value() != b || tl.back().c.value() != c) {
            basis::OrbitTriple t;
            t.a = OrbitIndex{a};
            t.b = OrbitIndex{b};
            t.c = OrbitIndex{c};
            t.row_begin =
                basis::KeptRow3Index{static_cast<std::int32_t>(rows.size())};
            t.row_end = t.row_begin;
            t.sym = (a == b && b == c) ? 1 : (a == b || b == c) ? 3 : 6;
            tl.push_back(t);
            ++counts.triples;
            if (hhh) ++counts.triples_hhh;
            if (ppp) ++counts.triples_ppp;
          }
          rows.push_back(jab);
          tl.back().row_end =
              basis::KeptRow3Index{static_cast<std::int32_t>(rows.size())};
          ++kept_rows;
        });
    counts.rows_before_cuts += before;
    counts.rows += kept_rows;
  });

  // Assemble the channels (those with a kept hhh triple, in channel order).
  std::vector<basis::KeptChannel3> channels;
  std::vector<basis::OrbitTriple> triples;
  std::vector<std::int32_t> two_jab;
  for (std::int32_t ch = 0; ch < num_channels; ++ch) {
    const std::vector<basis::OrbitTriple>& tl = rr.triples[U(ch)];
    if (rr.relevant[U(ch)] == 0 || tl.empty()) continue;
    basis::KeptChannel3 kc;
    kc.two_j = TwoJ{2 * (ch / 8) + 1};
    kc.parity = ((ch % 8) / 4 == 0) ? Parity::kEven : Parity::kOdd;
    kc.two_tz = TwoTz{2 * (ch % 4) - 3};
    kc.dim = static_cast<std::int32_t>(rr.two_jab[U(ch)].size());
    kc.row_offset = static_cast<std::int64_t>(two_jab.size());
    kc.triple_begin =
        basis::Triple3Index{static_cast<std::int32_t>(triples.size())};
    const Channel3Index new_ch{static_cast<std::int32_t>(channels.size())};
    for (basis::OrbitTriple t : tl) {
      t.channel = new_ch;
      triples.push_back(t);
    }
    kc.triple_end =
        basis::Triple3Index{static_cast<std::int32_t>(triples.size())};
    two_jab.insert(two_jab.end(), rr.two_jab[U(ch)].begin(),
                   rr.two_jab[U(ch)].end());
    channels.push_back(kc);
  }
  counts.channels = static_cast<std::int64_t>(channels.size());
  return basis::ThreeBodySpace(std::move(channels), std::move(triples),
                               std::move(two_jab));
}

}  // namespace

// ------------------------------------------------------------------ tables

/// The tables of a `TriplesContext`.
struct TriplesContext::Impl {
  /// Builds every table for the reduced `space` of `ms`.
  Impl(const basis::ModelSpace& ms, const basis::ThreeBodySpace& space);

  angmom::SixJ3Table sixj;
  op::PairThirdSpace pair_third;  ///< States and R (no E).

  /// Offsets of the two-body channel blocks of `op::TwoBody`.
  std::vector<std::int64_t> two_body_offset_host;

  exec::DeviceBuffer<std::int32_t> ch_two_j, ch_dim, ch_triple_begin,
      ch_triple_end;
  exec::DeviceBuffer<std::int64_t> ch_row_offset;
  exec::DeviceBuffer<std::int32_t> tr_a, tr_b, tr_c, tr_channel, tr_row_begin,
      tr_row_end, tr_sym;
  /// Orbit table (quantum numbers and reference occupations).
  exec::DeviceBuffer<std::int32_t> orb_n, orb_l, orb_j2, orb_tz2;
  exec::DeviceBuffer<double> orb_occ;
  exec::DeviceBuffer<std::int32_t> group_offset, group_orbits, group_j_pos;
  std::int32_t nj = 0;
  exec::DeviceBuffer<std::int32_t> pair_local;
  exec::DeviceBuffer<double> pair_factor;
  exec::DeviceBuffer<std::int32_t> leg_range;
  exec::DeviceBuffer<std::int64_t> state_bra_base, state_ket_base;
  exec::DeviceBuffer<std::int32_t> state_dim;
  std::int32_t num_j = 0;
  exec::DeviceBuffer<std::int32_t> order_triples;
  std::int32_t num_triples = 0;
  std::int32_t triples_league = 0;
  detail::TileExtents ext;
  std::int64_t table_bytes = 0;
  std::int64_t scratch_bytes = 0;

  /// Host copies for the reports.
  std::vector<std::int64_t> host_row_off;  ///< Flat first row per triple (+1).
  std::vector<std::int32_t> host_tr_channel, host_ch_two_j;
  std::vector<std::int32_t> host_tr_a, host_tr_b, host_tr_c;
  std::vector<double> host_occ;
  std::vector<std::int32_t> host_orb_l, host_orb_j2, host_orb_tz2;
  std::vector<std::int32_t> host_group_orbits, host_group_j_pos;

  /// The kernel views (operands, output and epsilon unset). O(1).
  detail::TileViews Views() const;
};

namespace {

/// Group of orbit `o` (see `detail::OrbitGroupsView`).
std::int32_t GroupOfOrbit(const basis::Orbit& o) {
  return detail::OrbitGroupsView::GroupOf(o.l % 2, o.tz2.twice);
}

}  // namespace

TriplesContext::Impl::Impl(const basis::ModelSpace& ms,
                           const basis::ThreeBodySpace& space)
    : sixj(ms.basis().emax()), pair_third(space, ms.basis(), sixj, false) {
  const basis::SingleParticleBasis& basis = ms.basis();
  const auto orbits = basis.orbits();
  const auto& tbc = ms.two_body();
  const std::int32_t norb = basis.num_orbits();

  // Two-body block offsets as op::TwoBody stores a scalar operator.
  two_body_offset_host.assign(U(tbc.num_channels()) + 1, 0);
  for (std::int32_t ch = 0; ch < tbc.num_channels(); ++ch) {
    const std::int64_t n = tbc.channel(ChannelIndex{ch}).num_kets;
    two_body_offset_host[U(ch) + 1] = two_body_offset_host[U(ch)] + n * n;
  }

  // Reduced kept space.
  const auto channels = space.channels();
  const std::size_t nc = channels.size();
  std::vector<std::int32_t> c_two_j(nc), c_dim(nc), c_tb(nc), c_te(nc);
  std::vector<std::int64_t> c_row(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    c_two_j[c] = channels[c].two_j.twice;
    c_dim[c] = channels[c].dim;
    c_tb[c] = channels[c].triple_begin.value();
    c_te[c] = channels[c].triple_end.value();
    c_row[c] = channels[c].row_offset;
  }
  ch_two_j = Upload(c_two_j, "commutator::triples_ch_two_j");
  ch_dim = Upload(c_dim, "commutator::triples_ch_dim");
  ch_triple_begin = Upload(c_tb, "commutator::triples_ch_triple_begin");
  ch_triple_end = Upload(c_te, "commutator::triples_ch_triple_end");
  ch_row_offset = Upload(c_row, "commutator::triples_ch_row_offset");
  const auto triples = space.triples();
  num_triples = static_cast<std::int32_t>(triples.size());
  const std::size_t nt = triples.size();
  std::vector<std::int32_t> t_a(nt), t_b(nt), t_c(nt), t_ch(nt), t_rb(nt),
      t_re(nt), t_sym(nt);
  for (std::size_t t = 0; t < nt; ++t) {
    t_a[t] = triples[t].a.value();
    t_b[t] = triples[t].b.value();
    t_c[t] = triples[t].c.value();
    t_ch[t] = triples[t].channel.value();
    t_rb[t] = triples[t].row_begin.value();
    t_re[t] = triples[t].row_end.value();
    t_sym[t] = triples[t].sym;
  }
  tr_a = Upload(t_a, "commutator::triples_tr_a");
  tr_b = Upload(t_b, "commutator::triples_tr_b");
  tr_c = Upload(t_c, "commutator::triples_tr_c");
  tr_channel = Upload(t_ch, "commutator::triples_tr_channel");
  tr_row_begin = Upload(t_rb, "commutator::triples_tr_row_begin");
  tr_row_end = Upload(t_re, "commutator::triples_tr_row_end");
  tr_sym = Upload(t_sym, "commutator::triples_tr_sym");

  // Orbit table.
  std::vector<std::int32_t> o_n(U(norb)), o_l(U(norb)), o_j2(U(norb)),
      o_tz2(U(norb));
  for (std::int32_t i = 0; i < norb; ++i) {
    o_n[U(i)] = orbits[U(i)].n;
    o_l[U(i)] = orbits[U(i)].l;
    o_j2[U(i)] = orbits[U(i)].j2.twice;
    o_tz2[U(i)] = orbits[U(i)].tz2.twice;
  }
  const auto occ_span = ms.reference().occupations();
  host_occ.assign(occ_span.begin(), occ_span.end());
  orb_n = Upload(o_n, "commutator::triples_orb_n");
  orb_l = Upload(o_l, "commutator::triples_orb_l");
  orb_j2 = Upload(o_j2, "commutator::triples_orb_j2");
  orb_tz2 = Upload(o_tz2, "commutator::triples_orb_tz2");
  orb_occ = Upload(host_occ, "commutator::triples_orb_occ");

  // Orbit groups by (parity, 2tz), sorted by (2j, index), with the j blocks.
  nj = basis.emax() + 1;
  std::vector<std::int32_t> g_off(5, 0);
  std::vector<std::int32_t> g_orb;
  std::vector<std::int32_t> g_jpos(U(4 * (nj + 1)), 0);
  for (std::int32_t g = 0; g < 4; ++g) {
    std::vector<std::int32_t> members;
    for (std::int32_t i = 0; i < norb; ++i) {
      if (GroupOfOrbit(orbits[U(i)]) == g) members.push_back(i);
    }
    std::stable_sort(members.begin(), members.end(),
                     [&orbits](std::int32_t x, std::int32_t y) {
                       return orbits[U(x)].j2.twice < orbits[U(y)].j2.twice;
                     });
    for (std::int32_t ij = 0; ij <= nj; ++ij) {
      std::int32_t pos = 0;
      while (pos < static_cast<std::int32_t>(members.size()) &&
             (orbits[U(members[U(pos)])].j2.twice - 1) / 2 < ij) {
        ++pos;
      }
      g_jpos[U(g * (nj + 1) + ij)] = pos;
    }
    g_orb.insert(g_orb.end(), members.begin(), members.end());
    g_off[U(g) + 1] = static_cast<std::int32_t>(g_orb.size());
    ext.s_max = std::max(ext.s_max, static_cast<std::int32_t>(members.size()));
  }
  group_offset = Upload(g_off, "commutator::triples_group_offset");
  group_orbits = Upload(g_orb, "commutator::triples_group_orbits");
  group_j_pos = Upload(g_jpos, "commutator::triples_group_j_pos");

  // Leg tables: per (third o, J, grouped position of a) the local ket index
  // and factor (exchange phase for o > a, sqrt 2 for o == a, 0 if absent).
  num_j = 2 * basis.emax() + 2;
  {
    std::vector<std::int32_t> p_local(U(norb) * U(num_j) * U(norb), 0);
    std::vector<double> p_factor(p_local.size(), 0.0);
    for (std::int32_t o = 0; o < norb; ++o) {
      const TwoJ jo = orbits[U(o)].j2;
      for (std::int32_t J = 0; J < num_j; ++J) {
        for (std::int32_t gp = 0; gp < norb; ++gp) {
          const std::int32_t a = g_orb[U(gp)];
          const TwoJ ja = orbits[U(a)].j2;
          const KetIndex ket = tbc.ket_index(OrbitIndex{std::min(o, a)},
                                             OrbitIndex{std::max(o, a)});
          const auto li = tbc.local_index(ket, TwoJ{2 * J});
          if (!li.has_value()) continue;
          const std::size_t e = (U(o) * U(num_j) + U(J)) * U(norb) + U(gp);
          p_local[e] = li->value();
          p_factor[e] = o < a   ? 1.0
                        : o > a ? static_cast<double>(
                                      basis::ExchangePhase(jo, ja, TwoJ{2 * J}))
                                : constants::kSqrt2;
        }
      }
    }
    pair_local = Upload(p_local, "commutator::triples_pair_local");
    pair_factor = Upload(p_factor, "commutator::triples_pair_factor");
    // Leg ranges: positions of the j blocks in the triangle range of (o, J).
    std::vector<std::int32_t> range(U(norb) * U(num_j) * 8, 0);
    for (std::int32_t o = 0; o < norb; ++o) {
      const std::int32_t j2o = orbits[U(o)].j2.twice;
      for (std::int32_t J = 0; J < num_j; ++J) {
        const std::int32_t ij_lo = (std::abs(2 * J - j2o) - 1) / 2;
        const std::int32_t ij_hi = std::min((2 * J + j2o - 1) / 2, nj - 1);
        for (std::int32_t g = 0; g < 4; ++g) {
          const std::size_t e = ((U(o) * U(num_j) + U(J)) * 4 + U(g)) * 2;
          if (ij_lo > ij_hi) continue;
          range[e] = g_jpos[U(g * (nj + 1) + ij_lo)];
          range[e + 1] = g_jpos[U(g * (nj + 1) + ij_hi + 1)];
        }
      }
    }
    leg_range = Upload(range, "commutator::triples_leg_range");
  }

  // Per pair-third state: the two-body bases of its pair (p, q) at Jpq.
  const op::PairThirdView pv = pair_third.host_view();
  {
    const auto ns = static_cast<std::size_t>(pv.num_states);
    std::vector<std::int64_t> bra(ns);
    std::vector<std::int64_t> ket(ns);
    std::vector<std::int32_t> dim(ns);
    for (std::size_t f = 0; f < ns; ++f) {
      const OrbitIndex p{pv.st_p[f]};
      const OrbitIndex q{pv.st_q[f]};
      const TwoJ tj{pv.st_two_jpq[f]};
      const auto li = tbc.local_index(tbc.ket_index(p, q), tj);
      IMSRG_CHECK(li.has_value(),
                  "TriplesContext: pair-third state without a two-body ket");
      const ChannelIndex ch = tbc.channel_of(p, q, tj);
      const std::int64_t n = tbc.channel(ch).num_kets;
      const std::int64_t off = two_body_offset_host[ch.ToSize()];
      bra[f] = off + li->value();
      ket[f] = off + n * li->value();
      dim[f] = static_cast<std::int32_t>(n);
    }
    state_bra_base = Upload(bra, "commutator::triples_state_bra_base");
    state_ket_base = Upload(ket, "commutator::triples_state_ket_base");
    state_dim = Upload(dim, "commutator::triples_state_dim");
  }

  // Tile extents and the work order of the bra triples: estimated work of a
  // bra triple t is m_t times the summed m of its ket triples T' < t with a
  // nonzero occupation factor (a loose, half-skip test; the kernel applies
  // the exact one).
  std::vector<double> cost(nt, 0.0);
  const double loose = 0.5 * detail::kTriplesSkip;
  const auto occupation = [&](std::size_t t, double& o, double& u) {
    Occupations(host_occ, t_a[t], t_b[t], t_c[t], o, u);
  };
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::int32_t t = c_tb[c]; t < c_te[c]; ++t) {
      const double m = pv.tr_state_end[t] - pv.tr_state_begin[t];
      ext.m_max = std::max(ext.m_max, static_cast<std::int32_t>(m));
      ext.k_max = std::max(ext.k_max, t_re[U(t)] - t_rb[U(t)]);
      double o_t = 0.0;
      double u_t = 0.0;
      occupation(U(t), o_t, u_t);
      if (o_t < loose && u_t < loose) continue;
      double work = 0.0;
      for (std::int32_t tp = c_tb[c]; tp < t; ++tp) {
        double o_p = 0.0;
        double u_p = 0.0;
        occupation(U(tp), o_p, u_p);
        if (std::abs(o_t * u_p - o_p * u_t) < loose) continue;
        work += pv.tr_state_end[tp] - pv.tr_state_begin[tp];
      }
      cost[U(t)] = m * work;
    }
  }
  std::vector<std::int32_t> order(nt);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&cost](std::int32_t x, std::int32_t y) {
                     return cost[U(x)] > cost[U(y)];
                   });
  triples_league = static_cast<std::int32_t>(std::count_if(
      cost.begin(), cost.end(), [](double x) { return x > 0.0; }));
  order_triples = Upload(order, "commutator::triples_order");
  scratch_bytes = num_triples > 0 ? detail::TeamScratchBytes(ext) : 0;

  host_row_off.assign(nt + 1, 0);
  for (std::size_t t = 0; t < nt; ++t) {
    host_row_off[t + 1] = host_row_off[t] + (t_re[t] - t_rb[t]);
  }
  host_tr_channel = std::move(t_ch);
  host_tr_a = std::move(t_a);
  host_tr_b = std::move(t_b);
  host_tr_c = std::move(t_c);
  host_ch_two_j = std::move(c_two_j);
  host_orb_l = std::move(o_l);
  host_orb_j2 = std::move(o_j2);
  host_orb_tz2 = std::move(o_tz2);
  host_group_orbits = std::move(g_orb);
  host_group_j_pos = std::move(g_jpos);

  table_bytes = sixj.size_bytes() + pair_third.Bytes() + BufferBytes(ch_two_j) +
                BufferBytes(ch_dim) + BufferBytes(ch_triple_begin) +
                BufferBytes(ch_triple_end) + BufferBytes(ch_row_offset) +
                BufferBytes(tr_a) + BufferBytes(tr_b) + BufferBytes(tr_c) +
                BufferBytes(tr_channel) + BufferBytes(tr_row_begin) +
                BufferBytes(tr_row_end) + BufferBytes(tr_sym) +
                BufferBytes(orb_n) + BufferBytes(orb_l) + BufferBytes(orb_j2) +
                BufferBytes(orb_tz2) + BufferBytes(orb_occ) +
                BufferBytes(group_offset) + BufferBytes(group_orbits) +
                BufferBytes(group_j_pos) + BufferBytes(pair_local) +
                BufferBytes(pair_factor) + BufferBytes(leg_range) +
                BufferBytes(state_bra_base) + BufferBytes(state_ket_base) +
                BufferBytes(state_dim) + BufferBytes(order_triples);
}

detail::TileViews TriplesContext::Impl::Views() const {
  detail::TileViews v;
  v.sp.ch_two_j = ch_two_j.device_data();
  v.sp.ch_dim = ch_dim.device_data();
  v.sp.ch_row_offset = ch_row_offset.device_data();
  v.sp.ch_triple_begin = ch_triple_begin.device_data();
  v.sp.ch_triple_end = ch_triple_end.device_data();
  v.sp.tr_a = tr_a.device_data();
  v.sp.tr_b = tr_b.device_data();
  v.sp.tr_c = tr_c.device_data();
  v.sp.tr_channel = tr_channel.device_data();
  v.sp.tr_row_begin = tr_row_begin.device_data();
  v.sp.tr_row_end = tr_row_end.device_data();
  v.sp.tr_sym = tr_sym.device_data();
  v.sp.num_channels = static_cast<std::int32_t>(ch_dim.size());
  v.sp.num_triples = num_triples;
  v.pt = pair_third.device_view();
  v.sj = sixj.device_view();
  v.ob.n = orb_n.device_data();
  v.ob.l = orb_l.device_data();
  v.ob.j2 = orb_j2.device_data();
  v.ob.tz2 = orb_tz2.device_data();
  v.ob.occupation = orb_occ.device_data();
  v.ob.cvq = nullptr;
  v.ob.num_orbits = static_cast<std::int32_t>(orb_n.size());
  v.groups.offset = group_offset.device_data();
  v.groups.orbits = group_orbits.device_data();
  v.groups.j_pos = group_j_pos.device_data();
  v.groups.nj = nj;
  v.legs.pair_local = pair_local.device_data();
  v.legs.pair_factor = pair_factor.device_data();
  v.legs.leg_range = leg_range.device_data();
  v.legs.state_bra_base = state_bra_base.device_data();
  v.legs.state_ket_base = state_ket_base.device_data();
  v.legs.state_dim = state_dim.device_data();
  v.legs.num_j = num_j;
  v.legs.num_orbits = static_cast<std::int32_t>(group_orbits.size());
  // Strides of SixJ3View's layout ((((i1 nj + i2) nJ + J12) nj + i3) nJ3 + iJ)
  // nJ + J23.
  const std::int64_t e = sixj.emax();
  const std::int64_t n_j = e + 1;
  const std::int64_t n_J = 2 * e + 2;
  const std::int64_t n_J3 = 3 * e + 2;
  v.c5 = n_J;
  v.c4 = n_J3 * n_J;
  v.c3 = n_j * v.c4;
  v.c2 = n_J * v.c3;
  v.c1 = n_j * v.c2;
  v.ext = ext;
  return v;
}

// ----------------------------------------------------------------- context

TriplesContext::TriplesContext(std::shared_ptr<const basis::ModelSpace> ms,
                               const basis::ThreeBodyCuts& cuts)
    : ms_(std::move(ms)), cuts_(cuts) {
  if (ms_ == nullptr) throw InputError("TriplesContext: null model space");
  if (cuts_.emax_3body < 0) cuts_.emax_3body = ms_->basis().emax();
  IMSRG_CHECK(cuts_.emax_3body <= ms_->basis().emax(),
              "emax_3body exceeds the basis emax");
  IMSRG_CHECK(cuts_.e3max >= 0, "e3max must be non-negative");
  const basis::ThreeBodySpace space =
      EnumerateReducedSpace(*ms_, cuts_, counts_);
  impl_ = std::make_unique<Impl>(*ms_, space);
  counts_.states = impl_->pair_third.num_states();
}

TriplesContext::TriplesContext(TriplesContext&&) noexcept = default;
TriplesContext& TriplesContext::operator=(TriplesContext&&) noexcept = default;
TriplesContext::~TriplesContext() = default;

std::int64_t TriplesContext::TableBytes() const noexcept {
  return impl_ != nullptr ? impl_->table_bytes : 0;
}

std::int64_t TriplesContext::ScratchBytesPerTeam() const noexcept {
  return impl_ != nullptr ? impl_->scratch_bytes : 0;
}

const TriplesContext::Impl& TriplesContext::impl() const {
  IMSRG_CHECK(impl_ != nullptr, "TriplesContext: moved-from context");
  return *impl_;
}

TriplesWork TriplesContext::TileWork() const {
  if (impl_ == nullptr) return {};
  const Impl& im = *impl_;
  const op::PairThirdView pv = im.pair_third.host_view();
  const std::int32_t nj = im.nj;
  // cum[g][k]: orbits of group g with (2j - 1) / 2 < k, k in [0, nj].
  const auto count_in = [&](std::int32_t g, std::int32_t two_lo,
                            std::int32_t two_hi) -> std::int64_t {
    if (g < 0 || two_lo > two_hi) return 0;
    const std::int32_t ij_lo = std::max(0, (two_lo - 1) / 2);
    const std::int32_t ij_hi = std::min(nj - 1, (two_hi - 1) / 2);
    if (ij_lo > ij_hi) return 0;
    return im.host_group_j_pos[U(g * (nj + 1) + ij_hi + 1)] -
           im.host_group_j_pos[U(g * (nj + 1) + ij_lo)];
  };
  const auto triangle = [](std::int32_t a, std::int32_t b, std::int32_t c) {
    return std::abs(a - b) <= c && c <= a + b;
  };
  const std::size_t nc = im.host_ch_two_j.size();
  std::vector<std::int32_t> tb(nc, 0);
  std::vector<std::int32_t> te(nc, 0);
  for (std::size_t t = 0; t < im.host_tr_channel.size(); ++t) {
    const std::size_t c = U(im.host_tr_channel[t]);
    if (te[c] == 0) tb[c] = static_cast<std::int32_t>(t);
    te[c] = static_cast<std::int32_t>(t) + 1;
  }
  TriplesWork work;
  for (std::size_t c = 0; c < nc; ++c) {
    const std::int32_t two_J = im.host_ch_two_j[c];
    const std::int32_t st0 = pv.ch_state_offset[c];
    for (std::int32_t t = tb[c]; t < te[c]; ++t) {
      double o_t = 0.0;
      double u_t = 0.0;
      Occupations(im.host_occ, im.host_tr_a[U(t)], im.host_tr_b[U(t)],
                  im.host_tr_c[U(t)], o_t, u_t);
      if (o_t < detail::kTriplesSkip && u_t < detail::kTriplesSkip) continue;
      const std::int32_t ta = im.host_tr_a[U(t)];
      const std::int32_t tbb = im.host_tr_b[U(t)];
      const std::int32_t tcc = im.host_tr_c[U(t)];
      const std::int32_t ch_parity =
          im.host_orb_l[U(ta)] + im.host_orb_l[U(tbb)] + im.host_orb_l[U(tcc)];
      const std::int32_t ch_tz2 = im.host_orb_tz2[U(ta)] +
                                  im.host_orb_tz2[U(tbb)] +
                                  im.host_orb_tz2[U(tcc)];
      for (std::int32_t tp = tb[c]; tp < t; ++tp) {
        double o_p = 0.0;
        double u_p = 0.0;
        Occupations(im.host_occ, im.host_tr_a[U(tp)], im.host_tr_b[U(tp)],
                    im.host_tr_c[U(tp)], o_p, u_p);
        const double num = o_t * u_p - o_p * u_t;
        if (num < detail::kTriplesSkip && num > -detail::kTriplesSkip) continue;
        const std::int64_t row_t = im.host_row_off[U(t)];
        const std::int64_t row_p = im.host_row_off[U(tp)];
        const double nnz_t = static_cast<double>(
            pv.r.ptr[im.host_row_off[U(t) + 1]] - pv.r.ptr[row_t]);
        const double nnz_p = static_cast<double>(
            pv.r.ptr[im.host_row_off[U(tp) + 1]] - pv.r.ptr[row_p]);
        work.epilogue +=
            static_cast<double>(pv.tr_state_end[t] - pv.tr_state_begin[t]) *
                nnz_p +
            static_cast<double>(im.host_row_off[U(tp) + 1] - row_p) * nnz_t;
        for (std::int32_t r = pv.tr_state_begin[t]; r < pv.tr_state_end[t];
             ++r) {
          const std::int32_t z = pv.st_r[st0 + r];
          const std::int32_t two_j1 = pv.st_two_jpq[st0 + r];
          const std::int32_t jz = im.host_orb_j2[U(z)];
          for (std::int32_t s = pv.tr_state_begin[tp]; s < pv.tr_state_end[tp];
               ++s) {
            const std::int32_t w = pv.st_r[st0 + s];
            const std::int32_t two_j2 = pv.st_two_jpq[st0 + s];
            const std::int32_t jw = im.host_orb_j2[U(w)];
            if (!triangle(jw, two_J, two_j2) || !triangle(jz, two_J, two_j1)) {
              continue;
            }
            const std::int32_t g = detail::OrbitGroupsView::GroupOf(
                ch_parity + im.host_orb_l[U(z)] + im.host_orb_l[U(w)],
                ch_tz2 - im.host_orb_tz2[U(z)] - im.host_orb_tz2[U(w)]);
            const std::int32_t lo =
                std::max(std::abs(two_j1 - jw), std::abs(two_j2 - jz));
            const std::int32_t hi = std::min(two_j1 + jw, two_j2 + jz);
            work.w += static_cast<double>(count_in(g, lo, hi));
          }
        }
      }
    }
  }
  return work;
}

namespace {

/// The operand checks of `PerturbativeTriplesEnergy` (as `CheckOperand` of
/// three_body.cc, against the context's model space and two-body layout).
void CheckOperand(const char* caller, const op::Operator& op, const char* name,
                  const TriplesContext& ctx,
                  const std::vector<std::int64_t>& offsets) {
  const std::string who = std::string(caller) + ": " + name;
  if (op.empty()) throw InputError(who + " is an empty operator");
  if (!op.spec().is_scalar() || !op.spec().is_number_conserving()) {
    throw NotImplementedError(who +
                              " must be a scalar, number-conserving operator");
  }
  if (!op.two_body().allocated()) {
    throw InputError(who + " has no two-body part (particle rank >= 2 needed)");
  }
  const basis::ModelSpace& ms = op.model_space();
  const basis::ModelSpace& ms3 = *ctx.model_space_ptr();
  IMSRG_CHECK(&ms == &ms3 || (ms.basis() == ms3.basis() &&
                              ms.reference() == ms3.reference() &&
                              ms.partition() == ms3.partition()),
              who +
                  " lives on a model space whose basis, reference or "
                  "partition differs from the context's");
  const auto& m = op.two_body().matrix();
  IMSRG_CHECK(
      m.num_blocks() + 1 == offsets.size() && m.total_size() == offsets.back(),
      who + ": two-body layout differs from the context's");
}

}  // namespace

double PerturbativeTriplesEnergy(const op::Operator& X, const op::Operator& Y,
                                 std::span<const double> epsilon,
                                 const TriplesContext& ctx) {
  const TriplesContext::Impl& impl = ctx.impl();
  CheckOperand("PerturbativeTriplesEnergy", X, "X", ctx,
               impl.two_body_offset_host);
  CheckOperand("PerturbativeTriplesEnergy", Y, "Y", ctx,
               impl.two_body_offset_host);
  const std::int32_t norb = ctx.model_space_ptr()->num_orbits();
  if (static_cast<std::int64_t>(epsilon.size()) != norb) {
    throw InputError("PerturbativeTriplesEnergy: epsilon has " +
                     std::to_string(epsilon.size()) + " values, expected " +
                     std::to_string(norb) + " (one per orbit)");
  }
  if (impl.triples_league == 0) return 0.0;
  exec::DeviceBuffer<double> eps(epsilon.size(), "commutator::triples_eps");
  eps.CopyFromHost(epsilon);
  detail::TileViews v = impl.Views();
  v.x2 = X.two_body().matrix().buffer().device_data();
  v.y2 = Y.two_body().matrix().buffer().device_data();
  v.epsilon = eps.device_data();
  return detail::RunTriplesTiles(v, impl.order_triples.device_data(),
                                 impl.triples_league);
}

}  // namespace imsrg::commutator
