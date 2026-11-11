// Unit tests of TwoBodyChannels and TwoBodyKetSubsets (design/reference/01 sections 2.1-2.4).
// Hardcoded values come from tests/golden/basis/{emax2_He4,emax2_sd-shell,emax4_O16}.txt.
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/two_body_channels.hpp"

namespace imsrg::basis {
namespace {

SingleParticleBasis make_basis(std::int32_t emax) {
  return SingleParticleBasis::harmonic_oscillator(BasisTruncation{emax}, 16.0);
}

std::vector<std::int32_t> kets_of(const TwoBodyChannels& tb, std::int32_t ch) {
  std::vector<std::int32_t> v;
  for (KetIndex k : tb.channel_kets(ChannelIndex{ch})) v.push_back(k.value());
  return v;
}

std::vector<std::int32_t> locals(std::span<const LocalKetIndex> s) {
  std::vector<std::int32_t> v;
  for (LocalKetIndex i : s) v.push_back(i.value());
  return v;
}

using V = std::vector<std::int32_t>;

TEST(TwoBodyChannels, CountsAndKetOrder) {
  for (auto [emax, norb, nket, nch] : {std::tuple{2, 12, 78, 36}, std::tuple{4, 30, 465, 60}}) {
    const auto basis = make_basis(emax);
    const TwoBodyChannels tb(basis);
    ASSERT_EQ(tb.num_orbits(), norb);
    EXPECT_EQ(tb.num_kets(), nket);
    EXPECT_EQ(tb.num_channels(), nch);
    EXPECT_EQ(tb.max_J(), basis.max_two_body_J());
    EXPECT_TRUE(tb == TwoBodyChannels(basis));
    for (std::int32_t k = 0; k < tb.num_kets(); ++k) {
      const Ket& kt = tb.ket(KetIndex{k});
      ASSERT_LE(kt.p, kt.q);
      EXPECT_EQ(imsrgpp_ket_index(kt.p, kt.q, norb).value(), k);
      EXPECT_EQ(tb.ket_index(kt.p, kt.q).value(), k);
    }
  }
}

TEST(TwoBodyChannels, GoldenChannelsEmax2) {
  const TwoBodyChannels tb(make_basis(2));
  EXPECT_EQ(kets_of(tb, 0), (V{0, 10, 23, 42, 57, 68, 75}));
  EXPECT_EQ(tb.channel(ChannelIndex{0}).J(), 0);
  EXPECT_EQ(tb.channel(ChannelIndex{0}).Tz(), -1);
  EXPECT_EQ(tb.channel(ChannelIndex{0}).parity, Parity::Even);
  EXPECT_EQ(kets_of(tb, 1), (V{4, 29, 48}));
  EXPECT_EQ(kets_of(tb, 2), (V{1, 11, 21, 24, 43, 58, 69, 76}));
  EXPECT_EQ(kets_of(tb, 8), (V{1, 9, 11, 19, 21, 24, 26, 34, 43, 58, 60, 64, 69, 71, 73, 76}));
  EXPECT_EQ(tb.channel(ChannelIndex{8}).J(), 1);
  EXPECT_EQ(tb.channel(ChannelIndex{8}).Tz(), 0);
  EXPECT_EQ(tb.channel(ChannelIndex{35}).J(), 5);
  EXPECT_EQ(tb.channel(ChannelIndex{35}).parity, Parity::Odd);
  EXPECT_EQ(tb.channel(ChannelIndex{35}).Tz(), 1);
  EXPECT_EQ(tb.channel(ChannelIndex{35}).num_kets, 0);
  EXPECT_TRUE(tb.channel_kets(ChannelIndex{35}).empty());
}

TEST(TwoBodyChannels, GoldenChannelEmax4) {
  const TwoBodyChannels tb(make_basis(4));
  EXPECT_EQ(kets_of(tb, 1), (V{4, 18, 65, 83, 120, 138, 173, 220, 263, 304, 339, 370, 397}));
  EXPECT_EQ(tb.channel(ChannelIndex{1}).num_kets, 13);
}

TEST(TwoBodyChannels, LookupsAndMembership) {
  for (std::int32_t emax : {2, 3, 4}) {
    const TwoBodyChannels tb(make_basis(emax));
    std::int64_t allowed_pairs = 0;
    for (std::int32_t k = 0; k < tb.num_kets(); ++k) {
      const Ket& kt = tb.ket(KetIndex{k});
      const Orbit& a = tb.orbits()[static_cast<std::size_t>(kt.p.value())];
      const Orbit& b = tb.orbits()[static_cast<std::size_t>(kt.q.value())];
      for (std::int32_t J = 0; J <= tb.max_J(); ++J) {
        const bool allowed = std::abs(a.j2.twice - b.j2.twice) <= 2 * J &&
                             2 * J <= a.j2.twice + b.j2.twice && !(kt.p == kt.q && J % 2 != 0);
        allowed_pairs += allowed ? 1 : 0;
        const auto li = tb.local_index(KetIndex{k}, TwoJ{2 * J});
        EXPECT_EQ(li.has_value(), allowed);
        const ChannelIndex ch = tb.channel_of(kt.p, kt.q, TwoJ{2 * J});
        const auto li2 = tb.local_index(ch, KetIndex{k});
        EXPECT_EQ(li2.has_value(), allowed);
        if (allowed) {
          ASSERT_TRUE(li && li2);
          EXPECT_EQ(li->value(), li2->value());
          EXPECT_EQ(tb.global_ket(ch, *li).value(), k);
        }
      }
      EXPECT_FALSE(tb.local_index(KetIndex{k}, TwoJ{1}).has_value());
      EXPECT_FALSE(tb.local_index(KetIndex{k}, TwoJ{-2}).has_value());
    }

    std::int64_t total = 0;
    for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
      const TwoBodyChannel& c = tb.channel(ChannelIndex{ch});
      EXPECT_EQ(imsrgpp_channel_index(c.two_j, c.parity, c.two_tz).value(), ch);
      EXPECT_EQ(tb.find_channel(c.two_j, c.parity, c.two_tz)->value(), ch);
      total += c.num_kets;
      std::int32_t prev = -1;
      for (std::int32_t i = 0; i < c.num_kets; ++i) {
        const KetIndex k = tb.global_ket(ChannelIndex{ch}, LocalKetIndex{i});
        EXPECT_GT(k.value(), prev);
        prev = k.value();
        EXPECT_EQ(tb.local_index(ChannelIndex{ch}, k)->value(), i);
        EXPECT_EQ(tb.local_index(k, c.two_j)->value(), i);
        const Ket& kt = tb.ket(k);
        const Orbit& a = tb.orbits()[static_cast<std::size_t>(kt.p.value())];
        const Orbit& b = tb.orbits()[static_cast<std::size_t>(kt.q.value())];
        EXPECT_EQ(a.parity() * b.parity(), c.parity);
        EXPECT_EQ(a.tz2.twice + b.tz2.twice, c.two_tz.twice);
        EXPECT_GE(a.j2.twice + b.j2.twice, c.two_j.twice);
        EXPECT_LE(std::abs(a.j2.twice - b.j2.twice), c.two_j.twice);
        if (kt.p == kt.q) EXPECT_EQ(c.J() % 2, 0);
      }
    }
    EXPECT_EQ(total, allowed_pairs);
    EXPECT_EQ(tb.channel_ket_table().size(), static_cast<std::size_t>(total));
    EXPECT_EQ(tb.local_index_table().size(), static_cast<std::size_t>(tb.ket_J_offsets().back()));
    EXPECT_EQ(tb.ket_J_offsets().size(), static_cast<std::size_t>(tb.num_kets() + 1));
  }
}

TEST(TwoBodyChannels, FindChannelRejects) {
  const TwoBodyChannels tb(make_basis(2));
  EXPECT_FALSE(tb.find_channel(TwoJ{1}, Parity::Even, TwoTz{0}));
  EXPECT_FALSE(tb.find_channel(TwoJ{-2}, Parity::Even, TwoTz{0}));
  EXPECT_FALSE(tb.find_channel(TwoJ{2 * (tb.max_J() + 1)}, Parity::Even, TwoTz{0}));
  EXPECT_FALSE(tb.find_channel(TwoJ{0}, Parity::Even, TwoTz{1}));
  EXPECT_FALSE(tb.find_channel(TwoJ{0}, Parity::Even, TwoTz{4}));
  EXPECT_EQ(tb.find_channel(TwoJ{2}, Parity::Even, TwoTz{0})->value(), 8);
}

TEST(TwoBodyChannels, ChannelsBySize) {
  const TwoBodyChannels tb(make_basis(2));
  const auto order = tb.channels_by_size();
  std::int32_t nonempty = 0;
  for (const auto& c : tb.channels()) nonempty += c.num_kets > 0 ? 1 : 0;
  ASSERT_EQ(static_cast<std::int32_t>(order.size()), nonempty);
  for (std::size_t i = 1; i < order.size(); ++i) {
    const auto a = tb.channel(order[i - 1]).num_kets;
    const auto b = tb.channel(order[i]).num_kets;
    EXPECT_TRUE(a > b || (a == b && order[i - 1] < order[i]));
  }
  EXPECT_EQ(tb.channel(order.front()).num_kets, 16);
}

TEST(TwoBodyChannels, ExchangePhase) {
  EXPECT_EQ(exchange_phase(TwoJ{1}, TwoJ{1}, TwoJ{0}), 1);
  EXPECT_EQ(exchange_phase(TwoJ{1}, TwoJ{1}, TwoJ{2}), -1);
  // j = 3/2 and 1/2, J = 1: (-1)^(2+1+1) = +1; J = 2: -1.
  EXPECT_EQ(exchange_phase(TwoJ{3}, TwoJ{1}, TwoJ{2}), 1);
  EXPECT_EQ(exchange_phase(TwoJ{3}, TwoJ{1}, TwoJ{4}), -1);
}

TEST(TwoBodyChannels, ClassNames) {
  const char* names[] = {"pp", "hh", "ph", "cc", "vc", "qc", "vv", "qv", "qq"};
  for (std::size_t i = 0; i < kAllKetClasses.size(); ++i) {
    EXPECT_EQ(ket_class_name(kAllKetClasses[i]), names[i]);
  }
}

// --- subsets ---------------------------------------------------------------------------------

struct He4Setup {
  TwoBodyChannels tb{make_basis(2)};
  std::vector<double> occ = std::vector<double>(12, 0.0);
  std::vector<OrbitClass> cls = std::vector<OrbitClass>(12, OrbitClass::QSpace);
  He4Setup() {
    occ[0] = occ[1] = 1.0;
    cls[0] = cls[1] = OrbitClass::Core;
  }
};

TEST(TwoBodyKetSubsets, He4Channel0) {
  He4Setup s;
  const TwoBodyKetSubsets sub(s.tb, s.occ, s.cls);
  const ChannelIndex ch{0};
  EXPECT_EQ(sub.num_channels(), 36);
  EXPECT_EQ(locals(sub.kets(ch, KetClass::PP)), (V{2, 3, 4, 5, 6}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::HH)), (V{0}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::PH)), (V{1}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::CC)), (V{0}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::QC)), (V{1}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::QQ)), (V{2, 3, 4, 5, 6}));
  EXPECT_TRUE(sub.kets(ch, KetClass::VC).empty());
  EXPECT_TRUE(sub.kets(ch, KetClass::VV).empty());
  EXPECT_TRUE(sub.kets(ch, KetClass::QV).empty());
  EXPECT_EQ(sub.occupation_products(ch, KetClass::HH).size(), 1u);
  EXPECT_DOUBLE_EQ(sub.occupation_products(ch, KetClass::HH)[0], 1.0);
  EXPECT_DOUBLE_EQ(sub.vacancy_products(ch, KetClass::HH)[0], 0.0);
  EXPECT_DOUBLE_EQ(sub.occupation_products(ch, KetClass::PH)[0], 0.0);
  EXPECT_DOUBLE_EQ(sub.vacancy_products(ch, KetClass::PH)[0], 0.0);
}

TEST(TwoBodyKetSubsets, He4GoldenChannels) {
  He4Setup s;
  const TwoBodyKetSubsets sub(s.tb, s.occ, s.cls);
  const ChannelIndex ch2{2};
  EXPECT_EQ(locals(sub.kets(ch2, KetClass::PP)), (V{3, 4, 5, 6, 7}));
  EXPECT_EQ(locals(sub.kets(ch2, KetClass::HH)), (V{0}));
  EXPECT_EQ(locals(sub.kets(ch2, KetClass::PH)), (V{1, 2}));
  EXPECT_EQ(locals(sub.kets(ch2, KetClass::QC)), (V{1, 2}));
  const ChannelIndex ch8{8};
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::PP)), (V{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::HH)), (V{0}));
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::PH)), (V{1, 2, 3, 4}));
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::CC)), (V{0}));
  EXPECT_TRUE(sub.kets(ch8, KetClass::VC).empty());
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::QC)), (V{1, 2, 3, 4}));
  EXPECT_EQ(locals(sub.kets(ch8, KetClass::QQ)), (V{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  for (KetClass c : kAllKetClasses) EXPECT_TRUE(sub.kets(ChannelIndex{35}, c).empty());

  // Flat tables are consistent with the accessors.
  EXPECT_EQ(sub.subset_offsets().size(), static_cast<std::size_t>(36 * kNumKetClasses + 1));
  EXPECT_EQ(static_cast<std::size_t>(sub.subset_offsets().back()), sub.subset_table().size());
}

TEST(TwoBodyKetSubsets, SdShellChannel8) {
  const TwoBodyChannels tb(make_basis(2));
  std::vector<double> occ(12, 0.0);
  std::vector<OrbitClass> cls(12, OrbitClass::Valence);
  for (int i = 0; i < 6; ++i) {
    occ[static_cast<std::size_t>(i)] = 1.0;
    cls[static_cast<std::size_t>(i)] = OrbitClass::Core;
  }
  const TwoBodyKetSubsets sub(tb, occ, cls);
  const ChannelIndex ch{8};
  EXPECT_EQ(locals(sub.kets(ch, KetClass::PP)), (V{9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::HH)), (V{0, 5, 6, 7, 8}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::PH)), (V{1, 2, 3, 4}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::CC)), (V{0, 5, 6, 7, 8}));
  EXPECT_EQ(locals(sub.kets(ch, KetClass::VC)), (V{1, 2, 3, 4}));
  EXPECT_TRUE(sub.kets(ch, KetClass::QC).empty());
  EXPECT_EQ(locals(sub.kets(ch, KetClass::VV)), (V{9, 10, 11, 12, 13, 14, 15}));
  EXPECT_TRUE(sub.kets(ch, KetClass::QV).empty());
  EXPECT_TRUE(sub.kets(ch, KetClass::QQ).empty());
}

TEST(TwoBodyKetSubsets, FractionalOccupations) {
  const TwoBodyChannels tb(make_basis(2));
  std::vector<double> occ(12, 0.0);
  std::vector<OrbitClass> cls(12, OrbitClass::Valence);
  for (int i = 0; i < 6; ++i) occ[static_cast<std::size_t>(i)] = 1.0;
  occ[6] = 1.0 / 3.0;
  occ[7] = 0.5;
  const TwoBodyKetSubsets sub(tb, occ, cls);
  bool saw_frac_hh = false;
  for (std::int32_t ch = 0; ch < tb.num_channels(); ++ch) {
    const ChannelIndex c{ch};
    for (KetClass kc : {KetClass::HH, KetClass::PH}) {
      const auto l = sub.kets(c, kc);
      const auto op = sub.occupation_products(c, kc);
      const auto vp = sub.vacancy_products(c, kc);
      ASSERT_EQ(l.size(), op.size());
      ASSERT_EQ(l.size(), vp.size());
      for (std::size_t i = 0; i < l.size(); ++i) {
        const Ket& kt = tb.ket(tb.global_ket(c, l[i]));
        const double np = occ[static_cast<std::size_t>(kt.p.value())];
        const double nq = occ[static_cast<std::size_t>(kt.q.value())];
        EXPECT_EQ(op[i], np * nq);
        EXPECT_EQ(vp[i], (1.0 - np) * (1.0 - nq));
        if (kc == KetClass::HH && (kt.p.value() >= 6 || kt.q.value() >= 6)) {
          saw_frac_hh = true;
          EXPECT_LT(op[i], 1.0);
        }
        EXPECT_EQ(kc == KetClass::HH, np > 0 && nq > 0);
      }
    }
  }
  EXPECT_TRUE(saw_frac_hh);
}

TEST(TwoBodyKetSubsets, Errors) {
  He4Setup s;
  const TwoBodyKetSubsets sub(s.tb, s.occ, s.cls);
  EXPECT_THROW(sub.occupation_products(ChannelIndex{0}, KetClass::PP), InvariantError);
  EXPECT_THROW(sub.vacancy_products(ChannelIndex{0}, KetClass::QQ), InvariantError);
  std::vector<double> short_occ(11, 0.0);
  EXPECT_THROW(TwoBodyKetSubsets(s.tb, short_occ, s.cls), InputError);
  std::vector<OrbitClass> short_cls(5, OrbitClass::Core);
  EXPECT_THROW(TwoBodyKetSubsets(s.tb, s.occ, short_cls), InputError);
}

}  // namespace
}  // namespace imsrg::basis
