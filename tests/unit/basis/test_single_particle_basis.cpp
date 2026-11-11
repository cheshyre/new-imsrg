// The harmonic-oscillator single-particle basis and its truncations.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/single_particle_basis.hpp"

namespace imsrg::basis {
namespace {

SingleParticleBasis make_basis(int emax, int lmax = -1, int emax_unocc = -1, int lmax_unocc = -1,
                               double hw = 16.0) {
  return SingleParticleBasis::harmonic_oscillator(
      BasisTruncation{emax, lmax, emax_unocc, lmax_unocc}, hw);
}

// The untruncated orbit list generated independently of the class, in imsrg++ order.
std::vector<Orbit> reference_orbits(int emax) {
  std::vector<Orbit> v;
  for (int N = 0; N <= emax; ++N) {
    for (int l = N; l >= 0; l -= 2) {
      for (const int j2 : {2 * l + 1, 2 * l - 1}) {
        if (j2 <= 0) continue;
        for (const int tz2 : {-1, 1}) v.push_back(Orbit{(N - l) / 2, l, TwoJ{j2}, TwoTz{tz2}});
      }
    }
  }
  return v;
}

TEST(SingleParticleBasis, UntruncatedMatchesGenerationOrderAndHash) {
  for (int emax = 0; emax <= 8; ++emax) {
    const auto b = make_basis(emax);
    const auto ref = reference_orbits(emax);
    ASSERT_EQ(b.num_orbits(), num_orbits_untruncated(emax));
    ASSERT_EQ(static_cast<std::size_t>(b.num_orbits()), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
      EXPECT_EQ(b.orbits()[i], ref[i]) << emax << " " << i;
      EXPECT_EQ(b.orbit(OrbitIndex{static_cast<int>(i)}), ref[i]);
      EXPECT_EQ(imsrgpp_orbit_hash(ref[i]), static_cast<int>(i));
    }
    EXPECT_EQ(b.emax(), emax);
    EXPECT_EQ(b.lmax(), emax);
  }
}

TEST(SingleParticleBasis, HbarOmegaAndTruncationAreStored) {
  const auto b = make_basis(4, 2, -1, -1, 20.0);
  EXPECT_EQ(b.hbar_omega(), 20.0);
  EXPECT_EQ(b.truncation(), (BasisTruncation{4, 2, -1, -1}));
  // hw does not change the orbit list.
  EXPECT_EQ(b.orbits().size(), make_basis(4, 2, -1, -1, 8.0).orbits().size());
}

TEST(SingleParticleBasis, LmaxTruncation) {
  const auto full = make_basis(4);
  const auto b = make_basis(4, 1);
  EXPECT_EQ(b.lmax(), 1);
  // Orbits with l <= 1 up to e = 4: s (n=0,1,2), p (n=0,1): 2 j-values each for p, 1 for s, x2 tz.
  // s: 3 orbits x 1 j x 2 = 6; p: 2 x 2 j x 2 = 8.
  EXPECT_EQ(b.num_orbits(), 14);
  std::vector<Orbit> expected;
  for (const Orbit& o : full.orbits()) {
    if (o.l <= 1) expected.push_back(o);
  }
  ASSERT_EQ(expected.size(), 14u);
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(b.orbits()[i], expected[i]) << i;
  // The first orbit with l > lmax is not found; proton/neutron parity of the index is kept.
  EXPECT_FALSE(b.contains(Orbit{0, 2, TwoJ{5}, TwoTz{-1}}));
  for (int i = 0; i < b.num_orbits(); ++i) {
    EXPECT_EQ(i % 2 == 0, b.orbit(OrbitIndex{i}).is_proton());
  }
}

TEST(SingleParticleBasis, LmaxLargerThanEmaxIsNoCut) {
  EXPECT_EQ(make_basis(3, 50), make_basis(3));
  EXPECT_EQ(make_basis(3, 50).lmax(), 3);
  EXPECT_EQ(make_basis(3, 3).lmax(), 3);
  EXPECT_EQ(make_basis(3, 0).num_orbits(), 2 * 2);  // 0s and 1s, both tz
}

TEST(SingleParticleBasis, EmaxUnoccTruncation) {
  const int emax = 6, emax_unocc = 3, lmax_unocc = 2;
  const auto full = make_basis(emax);
  const auto b = make_basis(emax, -1, emax_unocc, lmax_unocc);
  std::vector<Orbit> expected;
  for (const Orbit& o : full.orbits()) {
    if (o.e() > emax_unocc && o.l > lmax_unocc) continue;
    expected.push_back(o);
  }
  ASSERT_EQ(static_cast<std::size_t>(b.num_orbits()), expected.size());
  EXPECT_LT(b.num_orbits(), full.num_orbits());
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(b.orbits()[i], expected[i]) << i;

  // Removed orbits are not found and not contained; kept ones are.
  for (const Orbit& o : full.orbits()) {
    const bool removed = o.e() > emax_unocc && o.l > lmax_unocc;
    EXPECT_EQ(b.find(o).has_value(), !removed);
    EXPECT_EQ(b.contains(o), !removed);
    if (removed) {
      EXPECT_EQ(b.find(o), std::nullopt);
      EXPECT_THROW((void)b.index_of(o), InputError);
    }
  }
  // Specific cases: 0g9/2 (e = 4, l = 4 > 2) is gone; 1d5/2 (e = 4, l = 2) stays.
  EXPECT_FALSE(b.contains(Orbit{0, 4, TwoJ{9}, TwoTz{-1}}));
  EXPECT_TRUE(b.contains(Orbit{1, 2, TwoJ{5}, TwoTz{-1}}));
  // Protons still even, neutron partner at +1.
  for (int i = 0; i < b.num_orbits(); ++i) {
    EXPECT_EQ(i % 2 == 0, b.orbit(OrbitIndex{i}).is_proton());
  }
  // Orbit order is preserved (hash is strictly increasing).
  for (int i = 1; i < b.num_orbits(); ++i) {
    EXPECT_LT(imsrgpp_orbit_hash(b.orbit(OrbitIndex{i - 1})),
              imsrgpp_orbit_hash(b.orbit(OrbitIndex{i})));
  }
}

TEST(SingleParticleBasis, EmaxUnoccAtOrAboveEmaxIsNoCut) {
  EXPECT_EQ(make_basis(4, -1, 4, 1), make_basis(4));
  EXPECT_EQ(make_basis(4, -1, 7, 1), make_basis(4));
}

TEST(SingleParticleBasis, EmaxUnoccCombinedWithLmax) {
  const auto b = make_basis(6, 4, 2, 1);
  for (const Orbit& o : b.orbits()) {
    EXPECT_LE(o.l, 4);
    EXPECT_TRUE(o.e() <= 2 || o.l <= 1) << o.e() << " " << o.l;
  }
}

TEST(SingleParticleBasis, InvalidArguments) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)make_basis(-1), InputError);
  EXPECT_THROW((void)make_basis(2, -1, -1, -1, 0.0), InputError);
  EXPECT_THROW((void)make_basis(2, -1, -1, -1, -5.0), InputError);
  EXPECT_THROW((void)make_basis(2, -1, -1, -1, nan), InputError);
  EXPECT_THROW((void)make_basis(2, -1, -1, -1, inf), InputError);
  // emax_unocc cut active (0 <= emax_unocc < emax) without lmax_unocc.
  EXPECT_THROW((void)make_basis(4, -1, 2, -1), InputError);
  EXPECT_NO_THROW((void)make_basis(4, -1, 2, 0));
  EXPECT_NO_THROW((void)make_basis(0));
}

TEST(SingleParticleBasis, FindIndexOfContains) {
  const auto b = make_basis(4);
  const Orbit d5{0, 2, TwoJ{5}, TwoTz{-1}};
  const Orbit d5n{0, 2, TwoJ{5}, TwoTz{1}};
  ASSERT_TRUE(b.find(d5).has_value());
  EXPECT_EQ(*b.find(d5), OrbitIndex{6});
  EXPECT_EQ(b.index_of(d5n), OrbitIndex{7});
  EXPECT_TRUE(b.contains(d5));
  for (int i = 0; i < b.num_orbits(); ++i) {
    const Orbit& o = b.orbit(OrbitIndex{i});
    EXPECT_EQ(b.index_of(o), OrbitIndex{i});
    EXPECT_EQ(*b.find(o), OrbitIndex{i});
  }
  // e = 5 is outside emax = 4; invalid orbits are simply not found.
  const Orbit e5{0, 5, TwoJ{11}, TwoTz{1}};
  EXPECT_FALSE(b.contains(e5));
  EXPECT_EQ(b.find(e5), std::nullopt);
  try {
    (void)b.index_of(e5);
    FAIL() << "no exception";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find("n0h11"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(b.contains(Orbit{0, 0, TwoJ{2}, TwoTz{1}}));
  EXPECT_FALSE(b.contains(Orbit{0, 0, TwoJ{1}, TwoTz{0}}));
  EXPECT_FALSE(b.contains(Orbit{-1, 0, TwoJ{1}, TwoTz{1}}));
}

TEST(SingleParticleBasis, SameLjtz) {
  const auto b = make_basis(4);
  // p0d5 (index 6): same l, 2j, tz = p1d5? e = 2+2 = 4: 1d5/2 proton is in the basis.
  const auto same = b.same_ljtz(OrbitIndex{6});
  std::vector<OrbitIndex> expected;
  for (int i = 0; i < b.num_orbits(); ++i) {
    const Orbit& o = b.orbit(OrbitIndex{i});
    if (o.l == 2 && o.j2.twice == 5 && o.tz2.twice == -1) expected.push_back(OrbitIndex{i});
  }
  ASSERT_EQ(same.size(), 2u);  // 0d5/2 and 1d5/2
  EXPECT_EQ(same, expected);
  EXPECT_TRUE(std::is_sorted(same.begin(), same.end()));
  EXPECT_EQ(same.front(), OrbitIndex{6});
  // 0s1/2 neutron: 0s, 1s, 2s.
  const auto s = b.same_ljtz(OrbitIndex{1});
  ASSERT_EQ(s.size(), 3u);
  for (const auto i : s) {
    EXPECT_EQ(b.orbit(i).l, 0);
    EXPECT_TRUE(b.orbit(i).is_neutron());
  }
  // Every orbit is in its own class, classes are ascending.
  for (int i = 0; i < b.num_orbits(); ++i) {
    const auto c = b.same_ljtz(OrbitIndex{i});
    EXPECT_TRUE(std::find(c.begin(), c.end(), OrbitIndex{i}) != c.end());
    EXPECT_TRUE(std::is_sorted(c.begin(), c.end()));
  }
  // With an l truncation the class shrinks.
  EXPECT_EQ(make_basis(4, 1).same_ljtz(OrbitIndex{0}).size(), 3u);
}

TEST(SingleParticleBasis, MaxJ2AndMaxTwoBodyJ) {
  // Largest 2j of shell e: l = e, 2j = 2e + 1.
  for (int emax = 0; emax <= 10; ++emax) {
    const auto b = make_basis(emax);
    EXPECT_EQ(b.max_j2().twice, 2 * emax + 1);
    EXPECT_EQ(b.max_two_body_J(), 2 * emax + 1);
  }
  // With an l cut the maximum follows lmax.
  EXPECT_EQ(make_basis(6, 2).max_j2().twice, 5);
  EXPECT_EQ(make_basis(6, 2).max_two_body_J(), 5);
  EXPECT_EQ(make_basis(6, 0).max_j2().twice, 1);
  // unoccupied-shell cut: 6 -> highest kept is max(l <= lmax_unocc at high e, any l at e <=
  // emax_unocc)
  const auto b = make_basis(6, -1, 3, 1);
  EXPECT_EQ(b.max_j2().twice, 7);  // 0f7/2 at e = 3
}

TEST(SingleParticleBasis, ChannelCountFormula) {
  // The number of two-body channels is 6 (Jmax + 1).
  EXPECT_EQ(6 * (make_basis(2).max_two_body_J() + 1), 36);
  EXPECT_EQ(6 * (make_basis(4).max_two_body_J() + 1), 60);
}

TEST(SingleParticleBasis, Equality) {
  EXPECT_EQ(make_basis(4), make_basis(4));
  EXPECT_NE(make_basis(4), make_basis(3));
  EXPECT_NE(make_basis(4), make_basis(4, 2));
  EXPECT_NE(make_basis(4, -1, -1, -1, 16.0), make_basis(4, -1, -1, -1, 20.0));
  // Same orbits from different truncation parameters compare equal.
  EXPECT_EQ(make_basis(4, 4), make_basis(4));
  EXPECT_EQ(make_basis(4, -1, 4, 0), make_basis(4));
  // Copies are independent values.
  const auto a = make_basis(3);
  const auto c = a;
  EXPECT_EQ(a, c);
}

TEST(SingleParticleBasis, EmaxZero) {
  const auto b = make_basis(0);
  ASSERT_EQ(b.num_orbits(), 2);
  EXPECT_EQ(b.orbit(OrbitIndex{0}), (Orbit{0, 0, TwoJ{1}, TwoTz{-1}}));
  EXPECT_EQ(b.orbit(OrbitIndex{1}), (Orbit{0, 0, TwoJ{1}, TwoTz{1}}));
  EXPECT_EQ(b.max_j2().twice, 1);
}

}  // namespace
}  // namespace imsrg::basis
