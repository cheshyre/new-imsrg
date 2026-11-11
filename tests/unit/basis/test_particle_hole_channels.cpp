// Unit tests of ParticleHoleChannels (design/reference/01 sections 2.2-2.3). The expected
// structure is rebuilt by brute force from the orbits; the golden comparison lives in
// test_golden_basis.cpp.
#include <algorithm>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/basis/particle_hole_channels.hpp"

namespace imsrg::basis {
namespace {

SingleParticleBasis make_basis(std::int32_t emax) {
  return SingleParticleBasis::harmonic_oscillator(BasisTruncation{emax}, 16.0);
}

TEST(ParticleHoleChannels, CountsMatchTheNormalChannels) {
  for (const std::int32_t emax : {2, 4}) {
    const auto basis = make_basis(emax);
    const TwoBodyChannels tb(basis);
    const ParticleHoleChannels ph(tb);
    EXPECT_EQ(ph.num_orbits(), tb.num_orbits());
    EXPECT_EQ(ph.num_kets(), tb.num_kets());
    EXPECT_EQ(ph.max_J(), tb.max_J());
    EXPECT_EQ(ph.num_channels(), 6 * (tb.max_J() + 1));
    EXPECT_EQ(static_cast<std::int32_t>(ph.channels().size()), ph.num_channels());
  }
}

TEST(ParticleHoleChannels, MembershipFollowsTheTriangleParityAndTzRules) {
  const auto basis = make_basis(2);
  const TwoBodyChannels tb(basis);
  const ParticleHoleChannels ph(tb);
  std::int64_t total = 0;
  for (std::int32_t ch = 0; ch < ph.num_channels(); ++ch) {
    const TwoBodyChannel& c = ph.channel(ChannelIndex{ch});
    EXPECT_EQ(c.J(), ch / 6);
    EXPECT_EQ(c.Tz(), (ch % 6) / 2 - 1);
    EXPECT_EQ(static_cast<std::int32_t>(c.parity), ch % 2);
    if (c.Tz() < 0) EXPECT_EQ(c.num_kets, 0) << "Tz = -1 slots are empty";
    const auto kets = ph.channel_kets(ChannelIndex{ch});
    ASSERT_EQ(static_cast<std::int32_t>(kets.size()), c.num_kets);
    total += c.num_kets;
    for (std::size_t i = 0; i < kets.size(); ++i) {
      const Ket& k = tb.ket(kets[i]);
      const Orbit& a = basis.orbit(k.p);
      const Orbit& b = basis.orbit(k.q);
      EXPECT_LE(std::abs(a.j2.twice - b.j2.twice), 2 * c.J());
      EXPECT_GE(a.j2.twice + b.j2.twice, 2 * c.J());
      EXPECT_EQ((a.l + b.l) % 2, ch % 2);
      EXPECT_EQ(std::abs(a.tz2.twice - b.tz2.twice) / 2, c.Tz());
      if (i > 0) EXPECT_LT(kets[i - 1].value(), kets[i].value());
      const auto local = ph.local_index(ChannelIndex{ch}, kets[i]);
      ASSERT_TRUE(local.has_value());
      EXPECT_EQ(local->value(), static_cast<std::int32_t>(i));
      EXPECT_EQ(ph.local_index(kets[i], c.two_j), local);
    }
  }
  // Every ket appears once per J of its triangle range.
  std::int64_t expect = 0;
  for (std::int32_t k = 0; k < tb.num_kets(); ++k) {
    const Ket& kt = tb.ket(KetIndex{k});
    const auto ja = basis.orbit(kt.p).j2.twice;
    const auto jb = basis.orbit(kt.q).j2.twice;
    expect += (ja + jb) / 2 - std::abs(ja - jb) / 2 + 1;
  }
  EXPECT_EQ(total, expect);
  EXPECT_EQ(static_cast<std::int64_t>(ph.channel_ket_table().size()), expect);
}

TEST(ParticleHoleChannels, DiagonalKetsAppearForEveryJ) {
  const auto basis = make_basis(2);
  const TwoBodyChannels tb(basis);
  const ParticleHoleChannels ph(tb);
  // Orbit 0 is 0s1/2: the diagonal ket |00> has J = 0 and 1 (no Pauli rule), and nothing else.
  const OrbitIndex o{0};
  const KetIndex k = imsrgpp_ket_index(o, o, tb.num_orbits());
  for (const std::int32_t J : {0, 1}) {
    const ChannelIndex ch = ph.channel_of(o, o, TwoJ{2 * J});
    EXPECT_TRUE(ph.local_index(ch, k).has_value()) << J;
  }
  EXPECT_FALSE(ph.local_index(k, TwoJ{4}).has_value());
}

TEST(ParticleHoleChannels, FindChannelAndChannelOf) {
  const auto basis = make_basis(2);
  const TwoBodyChannels tb(basis);
  const ParticleHoleChannels ph(tb);
  const auto ch = ph.find_channel(TwoJ{4}, Parity::Odd, TwoTz{2});
  ASSERT_TRUE(ch.has_value());
  EXPECT_EQ(ch->value(), 6 * 2 + 2 * 2 + 1);
  EXPECT_FALSE(ph.find_channel(TwoJ{3}, Parity::Odd, TwoTz{2}).has_value());
  EXPECT_FALSE(ph.find_channel(TwoJ{-2}, Parity::Odd, TwoTz{2}).has_value());
  EXPECT_FALSE(ph.find_channel(TwoJ{2 * (ph.max_J() + 1)}, Parity::Even, TwoTz{0}).has_value());
  EXPECT_FALSE(ph.find_channel(TwoJ{2}, Parity::Even, TwoTz{1}).has_value());
}

TEST(ParticleHoleChannels, PairIndexHandlesBothOrdersAndAbsence) {
  const auto basis = make_basis(2);
  const TwoBodyChannels tb(basis);
  const ParticleHoleChannels ph(tb);
  const std::int32_t n = tb.num_orbits();
  for (std::int32_t ch = 0; ch < ph.num_channels(); ++ch) {
    const ChannelIndex c{ch};
    const std::int32_t nk = ph.channel(c).num_kets;
    for (std::int32_t p = 0; p < n; ++p) {
      for (std::int32_t q = 0; q < n; ++q) {
        const KetIndex k =
            imsrgpp_ket_index(OrbitIndex{std::min(p, q)}, OrbitIndex{std::max(p, q)}, n);
        const auto sorted = ph.local_index(c, k);
        const auto got = ph.pair_index(c, OrbitIndex{p}, OrbitIndex{q});
        if (!sorted) {
          EXPECT_FALSE(got.has_value());
        } else {
          ASSERT_TRUE(got.has_value());
          EXPECT_EQ(*got, p <= q ? sorted->value() : sorted->value() + nk);
        }
      }
    }
  }
}

TEST(ParticleHoleChannels, ChannelsBySizeAreDescendingAndNonEmpty) {
  const auto basis = make_basis(4);
  const TwoBodyChannels tb(basis);
  const ParticleHoleChannels ph(tb);
  const auto order = ph.channels_by_size();
  ASSERT_FALSE(order.empty());
  for (std::size_t i = 0; i < order.size(); ++i) {
    EXPECT_GT(ph.channel(order[i]).num_kets, 0);
    if (i > 0) {
      const auto a = ph.channel(order[i - 1]).num_kets;
      const auto b = ph.channel(order[i]).num_kets;
      EXPECT_TRUE(a > b || (a == b && order[i - 1].value() < order[i].value()));
    }
  }
}

}  // namespace
}  // namespace imsrg::basis
