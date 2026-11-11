// Unit tests of ThreeBodyChannels (design/reference/01 section 3). Kets are rebuilt by brute
// force; the golden comparison lives in test_golden_basis.cpp.
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/nucleus.hpp"
#include "imsrg/basis/three_body_channels.hpp"

namespace imsrg::basis {
namespace {

SingleParticleBasis make_basis(std::int32_t emax) {
  return SingleParticleBasis::harmonic_oscillator(BasisTruncation{emax}, 16.0);
}

std::int64_t brute_force_ket_count(const SingleParticleBasis& b, std::int32_t e3max) {
  std::int64_t n = 0;
  const std::int32_t no = b.num_orbits();
  for (std::int32_t p = 0; p < no; ++p) {
    for (std::int32_t q = p; q < no; ++q) {
      for (std::int32_t r = q; r < no; ++r) {
        const Orbit& a = b.orbit(OrbitIndex{p});
        const Orbit& c = b.orbit(OrbitIndex{q});
        const Orbit& d = b.orbit(OrbitIndex{r});
        if (2 * (a.n + c.n + d.n) + a.l + c.l + d.l > e3max) continue;
        for (std::int32_t j = std::abs(a.j2.twice - c.j2.twice) / 2;
             j <= (a.j2.twice + c.j2.twice) / 2; ++j) {
          if (p == q && j % 2 == 1) continue;
          if (p == q && q == r && a.j2.twice == 1) continue;
          ++n;
        }
      }
    }
  }
  return n;
}

TEST(ThreeBodyChannels, CutsAreResolved) {
  const auto b = make_basis(3);
  EXPECT_EQ(ThreeBodyChannels(b, ThreeBodyCuts{6}).cuts().emax_3body, 3);
  EXPECT_EQ(ThreeBodyChannels(b, ThreeBodyCuts{6, 2}).cuts().emax_3body, 2);
}

TEST(ThreeBodyChannels, KetCountMatchesBruteForceAndOrderIsLexicographic) {
  const auto b = make_basis(3);
  for (const std::int32_t e3max : {0, 3, 6, 9}) {
    const ThreeBodyChannels tb(b, ThreeBodyCuts{e3max});
    EXPECT_EQ(tb.num_kets(), brute_force_ket_count(b, e3max)) << e3max;
    const auto kets = tb.kets();
    for (std::size_t i = 1; i < kets.size(); ++i) {
      const auto& x = kets[i - 1];
      const auto& y = kets[i];
      EXPECT_TRUE(std::make_tuple(x.p.value(), x.q.value(), x.r.value(), x.two_jpq.twice) <
                  std::make_tuple(y.p.value(), y.q.value(), y.r.value(), y.two_jpq.twice));
    }
  }
}

TEST(ThreeBodyChannels, EmaxThreeBodyRestrictsOrbits) {
  const auto b = make_basis(3);
  const ThreeBodyChannels tb(b, ThreeBodyCuts{9, 1});
  EXPECT_EQ(tb.num_channels(), 8 * (3 * 1 + 1));
  for (const Ket3& k : tb.kets()) {
    EXPECT_LE(b.orbit(k.p).e(), 1);
    EXPECT_LE(b.orbit(k.q).e(), 1);
    EXPECT_LE(b.orbit(k.r).e(), 1);
  }
}

TEST(ThreeBodyChannels, FindKetRoundTripsAndRejectsAbsentKets) {
  const auto b = make_basis(2);
  const ThreeBodyChannels tb(b, ThreeBodyCuts{6});
  for (std::int64_t i = 0; i < tb.num_kets(); ++i) {
    const Ket3& k = tb.kets()[static_cast<std::size_t>(i)];
    const auto found = tb.find_ket(k.p, k.q, k.r, k.two_jpq);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->value(), i);
    EXPECT_TRUE(tb.ket(*found) == k);
  }
  const Ket3& k0 = tb.kets()[0];
  EXPECT_FALSE(tb.find_ket(k0.p, k0.q, k0.r, TwoJ{k0.two_jpq.twice + 100}).has_value());
  // Unordered orbits are never present.
  EXPECT_FALSE(tb.find_ket(OrbitIndex{3}, OrbitIndex{2}, OrbitIndex{5}, TwoJ{0}).has_value());
  // p == q with odd Jpq is dropped: orbit 2 (0p1/2 or 0p3/2) with itself and Jpq = 1.
  EXPECT_FALSE(tb.find_ket(OrbitIndex{2}, OrbitIndex{2}, OrbitIndex{4}, TwoJ{2}).has_value());
}

TEST(ThreeBodyChannels, ChannelIndexFormulaAndLookup) {
  const auto b = make_basis(2);
  const ThreeBodyChannels tb(b, ThreeBodyCuts{6});
  EXPECT_EQ(tb.num_channels(), 8 * (3 * 2 + 1));
  for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
    const ThreeBodyChannel& c = tb.channel(Channel3Index{ch});
    EXPECT_EQ(ThreeBodyChannels::channel_index(c.two_j, c.parity, c.two_tz).value(), ch);
    const auto f = tb.find_channel(c.two_j, c.parity, c.two_tz);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->value(), ch);
  }
  EXPECT_FALSE(tb.find_channel(TwoJ{2}, Parity::Even, TwoTz{1}).has_value());
  EXPECT_FALSE(tb.find_channel(TwoJ{1}, Parity::Even, TwoTz{0}).has_value());
  EXPECT_FALSE(tb.find_channel(TwoJ{6 * 2 + 3}, Parity::Even, TwoTz{1}).has_value());
  EXPECT_EQ(tb.find_channel(TwoJ{3}, Parity::Odd, TwoTz{-1})->value(), 1 * 8 + 4 + 1);
}

TEST(ThreeBodyChannels, MembershipAndLocalIndexAreConsistent) {
  const auto b = make_basis(2);
  const ThreeBodyChannels tb(b, ThreeBodyCuts{6});
  std::int64_t total = 0;
  for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
    const Channel3Index c{ch};
    const ThreeBodyChannel& info = tb.channel(c);
    const auto kets = tb.channel_kets(c);
    ASSERT_EQ(static_cast<std::int64_t>(kets.size()), info.num_kets);
    total += info.num_kets;
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const Ket3& k = tb.ket(kets[i]);
      const Orbit& a = b.orbit(k.p);
      const Orbit& bb = b.orbit(k.q);
      const Orbit& d = b.orbit(k.r);
      EXPECT_EQ((a.l + bb.l + d.l) % 2, static_cast<int>(info.parity));
      EXPECT_EQ(a.tz2.twice + bb.tz2.twice + d.tz2.twice, info.two_tz.twice);
      EXPECT_LE(std::abs(k.two_jpq.twice - d.j2.twice), info.two_j.twice);
      EXPECT_GE(k.two_jpq.twice + d.j2.twice, info.two_j.twice);
      if (i > 0) EXPECT_LT(kets[i - 1].value(), kets[i].value());
      const auto local = tb.local_index(c, kets[i]);
      ASSERT_TRUE(local.has_value());
      EXPECT_EQ(local->value(), static_cast<std::int32_t>(i));
    }
  }
  EXPECT_EQ(total, tb.sum_channel_kets());
  // A ket is absent from a channel of the wrong parity.
  const Ket3& k0 = tb.kets()[0];
  const Orbit& o1 = b.orbit(k0.p);
  const Orbit& o2 = b.orbit(k0.q);
  const Orbit& o3 = b.orbit(k0.r);
  const int parity = (o1.l + o2.l + o3.l) % 2;
  const auto wrong = static_cast<Parity>(1 - parity);
  const auto ch =
      tb.find_channel(TwoJ{1}, wrong, TwoTz{o1.tz2.twice + o2.tz2.twice + o3.tz2.twice});
  ASSERT_TRUE(ch.has_value());
  EXPECT_FALSE(tb.local_index(*ch, Ket3Index{0}).has_value());
}

TEST(ThreeBodyChannels, StretchedStateRulesDropRedundantKets) {
  // A ket of three identical orbits is absent from every channel above 2J = 3 (2j) - 3
  // (rule copied from imsrg++).
  const auto b = make_basis(2);
  const ThreeBodyChannels tb(b, ThreeBodyCuts{6});
  int checked = 0;
  for (std::int64_t i = 0; i < tb.num_kets(); ++i) {
    const Ket3& k = tb.kets()[static_cast<std::size_t>(i)];
    if (!(k.p == k.q && k.q == k.r)) continue;
    const std::int32_t j = b.orbit(k.p).j2.twice;
    for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
      if (tb.channel(Channel3Index{ch}).two_j.twice > 3 * j - 3) {
        EXPECT_FALSE(
            tb.local_index(Channel3Index{ch}, Ket3Index{static_cast<std::int32_t>(i)}).has_value());
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

TEST(ThreeBodyChannels, ConfigurationCutOnAHandCheckedKet) {
  const auto b = make_basis(2);
  const auto ref = Reference::from_nucleus(b, Nucleus::parse("O16"));
  const ThreeBodyChannels tb(b, ThreeBodyCuts{6});
  // Ket 0 holds the two s1/2 orbits (e = 0) and one more orbit. O16 has e_fermi = 1 for both
  // species, so sum |e - e_F| = 1 + 1 + |e_r - 1|.
  const Ket3Index k{0};
  const Ket3& kt = tb.ket(k);
  ASSERT_EQ(b.orbit(kt.p).e(), 0);
  ASSERT_EQ(b.orbit(kt.q).e(), 0);
  const auto dist = [&](OrbitIndex o) {
    const Orbit& orb = b.orbit(o);
    return std::abs(static_cast<double>(orb.e()) - ref.e_fermi(orb.tz2));
  };
  const double de = dist(kt.p) + dist(kt.q) + dist(kt.r);
  ASSERT_GT(de, 0.5);
  EXPECT_TRUE(tb.passes_configuration_cut(k, b, ref, de, 0.0));
  EXPECT_FALSE(tb.passes_configuration_cut(k, b, ref, de - 0.1, 0.0));
  // The s orbits are fully occupied: n (1 - n) = 0, so any positive occupation cut rejects the
  // ket with the reference occupations ...
  EXPECT_FALSE(tb.passes_configuration_cut(k, b, ref, 10.0, 1e-3));
  // ... while natural occupations of 1/2 give a product of (1/4)^3.
  const std::vector<double> half(static_cast<std::size_t>(b.num_orbits()), 0.5);
  EXPECT_TRUE(tb.passes_configuration_cut(k, b, ref, 10.0, 1.0 / 64.0, half));
  EXPECT_FALSE(tb.passes_configuration_cut(k, b, ref, 10.0, 1.0 / 64.0 + 1e-6, half));
}

}  // namespace
}  // namespace imsrg::basis
