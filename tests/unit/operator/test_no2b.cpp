// Tests of the NO2B three-nucleon input container `No2bThreeBody` (layout of imsrg++
// ThreeBodyStorage_no2b<float>): orbits, channel/state counts against a brute-force enumeration,
// set/get round trips with ordering phases, float storage, and the pn recoupling.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/operator/no2b.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

struct BfOrbit {
  int e, l, tj;
};

struct BfCounts {
  int channels = 0;
  std::size_t size = 0;
};

// Independent enumeration from the rules (half-integers as doubles, no J2 cap).
BfCounts brute_force(const No2bTruncation& t) {
  std::vector<BfOrbit> orbs;
  for (int e = 0; e <= t.emax; ++e) {
    for (int l = 0; l <= e && l <= t.lmax; ++l) {
      if ((e - l) % 2 != 0) continue;
      orbs.push_back({e, l, 2 * l - 1 > 0 ? 2 * l - 1 : 1});
      if (l > 0) orbs.push_back({e, l, 2 * l + 1});
    }
  }
  // l = 0 gets only 2j = 1; for l > 0 the loop above pushed 2l-1 then 2l+1.
  BfCounts out;
  for (int J2 = 0; J2 <= 2 * t.lmax + 8; ++J2) {
    for (int P2 = 0; P2 < 2; ++P2) {
      for (int tjc = 1; tjc <= 2 * t.lmax + 1; tjc += 2) {
        for (int P1 = 0; P1 < 2; ++P1) {
          for (int T3 = 1; T3 <= 3; T3 += 2) {
            std::size_t n = 0;
            for (std::size_t ia = 0; ia < orbs.size(); ++ia) {
              for (std::size_t ib = ia; ib < orbs.size(); ++ib) {
                const auto& a = orbs[ia];
                const auto& b = orbs[ib];
                if (a.e + b.e > t.e2max) continue;
                const double ja = a.tj / 2.0;
                const double jb = b.tj / 2.0;
                if (std::abs(ja - jb) > J2 || ja + jb < J2) continue;
                if ((a.l + b.l) % 2 != P2) continue;
                for (const auto& c : orbs) {
                  if (c.tj != tjc || c.l % 2 != P1) continue;
                  if (a.e + c.e > t.e2max || b.e + c.e > t.e2max) continue;
                  if (a.e + b.e + c.e > t.e3max) continue;
                  for (int Tab = 0; Tab < 2; ++Tab) {
                    const double tt = T3 / 2.0;
                    if (std::abs(Tab - 0.5) > tt || Tab + 0.5 < tt) continue;
                    if (ia == ib && (J2 + Tab) % 2 == 0) continue;
                    ++n;
                  }
                }
              }
            }
            if (n > 0) {
              ++out.channels;
              out.size += n * (n + 1) / 2;
            }
          }
        }
      }
    }
  }
  return out;
}

basis::Orbit orb(int n, int l, int tj, int tz) { return basis::Orbit{n, l, TwoJ{tj}, TwoTz{tz}}; }

TEST(No2b, OrbitList) {
  const No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  const auto orbs = w.isospin_orbits();
  ASSERT_EQ(orbs.size(), 6U);
  const int expect[6][3] = {{0, 0, 1}, {0, 1, 1}, {0, 1, 3}, {1, 0, 1}, {0, 2, 3}, {0, 2, 5}};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(orbs[static_cast<std::size_t>(i)].n, expect[i][0]);
    EXPECT_EQ(orbs[static_cast<std::size_t>(i)].l, expect[i][1]);
    EXPECT_EQ(orbs[static_cast<std::size_t>(i)].j2.twice, expect[i][2]);
    EXPECT_EQ(w.isospin_orbit_index(expect[i][0], expect[i][1], TwoJ{expect[i][2]}), i);
  }
  EXPECT_FALSE(w.isospin_orbit_index(0, 3, TwoJ{5}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(0, 0, TwoJ{3}).has_value());
}

TEST(No2b, NegativeTruncationThrows) {
  EXPECT_THROW(No2bThreeBody(No2bTruncation{-1, 4, 6, 2}), InputError);
  EXPECT_THROW(No2bThreeBody(No2bTruncation{2, -4, 6, 2}), InputError);
  EXPECT_THROW(No2bThreeBody(No2bTruncation{2, 4, -6, 2}), InputError);
  EXPECT_THROW(No2bThreeBody(No2bTruncation{2, 4, 6, -2}), InputError);
}

TEST(No2b, CountsMatchBruteForce) {
  for (const auto& t : {No2bTruncation{2, 4, 6, 2}, No2bTruncation{3, 4, 5, 3}}) {
    const No2bThreeBody w(t);
    const BfCounts bf = brute_force(t);
    EXPECT_GT(bf.channels, 0);
    EXPECT_EQ(w.num_channels(), bf.channels);
    EXPECT_EQ(w.size(), bf.size);
  }
}

TEST(No2b, EmptyContainer) {
  const No2bThreeBody w;
  EXPECT_EQ(w.num_channels(), 0);
  EXPECT_EQ(w.size(), 0U);
  EXPECT_EQ(w.get_isospin(0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 0.0);
  EXPECT_EQ(w.norm(), 0.0);
}

TEST(No2b, SetGetRoundTrip) {
  No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  // a = s1/2 (0), b = p1/2 (1), c = s1/2 (0); J2 = 1, 2T = 1.
  const double v = 0.1;
  w.set_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 1, v);
  const double vf = static_cast<double>(static_cast<float>(v));
  EXPECT_NE(vf, v);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 1), vf);
  // (j_a + j_b)/2 + J2 + Tab = 1 + 1 + 1 odd: swapping a, b flips the sign.
  EXPECT_EQ(w.get_isospin(1, 0, 0, 1, 0, 1, 0, 1, 1, 1), -vf);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 1, 0, 0, 1, 1, 1), -vf);
  EXPECT_EQ(w.get_isospin(1, 0, 0, 1, 1, 0, 0, 1, 1, 1), vf);
  // Different Tab: another element, still zero.
  EXPECT_EQ(w.get_isospin(0, 1, 0, 0, 0, 1, 0, 1, 1, 1), 0.0);

  // Off-diagonal element and its transpose; Tab = 0 has even sum: no sign on swap.
  w.set_isospin(0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0.25);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 0, 0, 1, 0, 1, 1, 1), 0.25);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 0, 0, 1, 1), 0.25);
  EXPECT_EQ(w.get_isospin(1, 0, 0, 0, 0, 1, 0, 1, 1, 1), 0.25);
  EXPECT_EQ(w.get_isospin(1, 0, 0, 0, 0, 1, 0, 1, 1, 1), 0.25);
  EXPECT_EQ(w.get_isospin(1, 0, 0, 1, 0, 1, 0, 0, 1, 1), -0.25);

  // norm: diagonal element once, off-diagonal pair stored once.
  const double expect = std::sqrt(vf * vf + 0.25 * 0.25);
  EXPECT_NEAR(w.norm(), expect, 1e-15);
}

TEST(No2b, AbsentElementsAreZero) {
  const No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 1), 0.0);  // stored zero
  EXPECT_EQ(w.get_isospin(-1, 1, 0, 1, 0, 1, 0, 1, 1, 1), 0.0);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 99, 1, 1, 1), 0.0);
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 0, 1, 99, 1), 0.0);  // J2 out of range
  EXPECT_EQ(w.get_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 5), 0.0);   // bad T3
  EXPECT_EQ(w.get_isospin(0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 0.0);   // Pauli-forbidden
  EXPECT_EQ(w.get_isospin(0, 1, 1, 1, 0, 1, 2, 1, 1, 1), 0.0);   // j_c != j_f
}

TEST(No2b, InvalidSetThrows) {
  No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  // a == b with J2 + Tab even: Pauli forbidden (J2 = 0, Tab = 0 on s1/2 s1/2).
  EXPECT_THROW(w.set_isospin(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0), InvariantError);
  // j_c != j_f (p1/2 vs p3/2).
  EXPECT_THROW(w.set_isospin(0, 3, 1, 1, 0, 3, 2, 1, 1, 1, 1.0), InvariantError);
  // Parity of c, f differs.
  EXPECT_THROW(w.set_isospin(0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1.0), InvariantError);
  // Out-of-range orbit index and channel.
  EXPECT_THROW(w.set_isospin(0, 1, 0, 1, 0, 1, 77, 1, 1, 1, 1.0), InvariantError);
  EXPECT_THROW(w.set_isospin(0, 1, 0, 1, 0, 1, 0, 1, 99, 1, 1.0), InvariantError);
  EXPECT_EQ(w.norm(), 0.0);
}

TEST(No2b, PnDiagonalCases) {
  No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  w.set_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 3, 0.75);  // Tab = Tde = 1, T = 3/2
  const double v = 0.75;
  for (int tz : {-1, +1}) {
    const auto s = orb(0, 0, 1, tz);
    const auto p = orb(0, 1, 1, tz);
    EXPECT_NEAR(w.get_pn(s, p, s, s, p, s, 1), v, 1e-15) << tz;
  }
  // Mixed Tz in the isospin-3/2 element only: ppn gets CGs < 1 but T = 3/2 only, Tab = 1:
  // <1/2 -1/2 1/2 -1/2|1 -1> = 1, <1 -1, 1/2 +1/2 | 3/2 -1/2> = sqrt(1/3).
  EXPECT_NEAR(w.get_pn(orb(0, 0, 1, -1), orb(0, 1, 1, -1), orb(0, 0, 1, +1), orb(0, 0, 1, -1),
                       orb(0, 1, 1, -1), orb(0, 0, 1, +1), 1),
              v / 3.0, 1e-7);
  // Tz mismatch between bra and ket.
  EXPECT_EQ(w.get_pn(orb(0, 0, 1, -1), orb(0, 1, 1, -1), orb(0, 0, 1, -1), orb(0, 0, 1, -1),
                     orb(0, 1, 1, -1), orb(0, 0, 1, +1), 1),
            0.0);
  // Parity mismatch of the pair (a b = s p, d e = s s).
  EXPECT_EQ(w.get_pn(orb(0, 0, 1, -1), orb(0, 1, 1, -1), orb(0, 0, 1, -1), orb(0, 0, 1, -1),
                     orb(0, 0, 1, -1), orb(0, 0, 1, -1), 1),
            0.0);
  // j_c != j_f (p1/2 vs p3/2, same parity).
  EXPECT_EQ(w.get_pn(orb(0, 0, 1, -1), orb(0, 0, 1, +1), orb(0, 1, 1, -1), orb(0, 0, 1, -1),
                     orb(0, 0, 1, +1), orb(0, 1, 3, -1), 0),
            0.0);
  // Orbit not stored (l = 3 > lmax).
  EXPECT_EQ(w.get_pn(orb(0, 3, 5, -1), orb(0, 1, 1, -1), orb(0, 0, 1, -1), orb(0, 0, 1, -1),
                     orb(0, 1, 1, -1), orb(0, 0, 1, -1), 1),
            0.0);
}

TEST(No2b, PnAgainstHandCgProducts) {
  No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  // a = s1/2, b = p1/2, c = s1/2; J2 = 1. Elements (Tab, Tde, 2T).
  const double w00 = 0.5, w01 = -0.25, w11a = 0.125, w11b = 1.5;
  w.set_isospin(0, 1, 0, 0, 0, 1, 0, 0, 1, 1, w00);
  w.set_isospin(0, 1, 0, 0, 0, 1, 0, 1, 1, 1, w01);
  w.set_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 1, w11a);
  w.set_isospin(0, 1, 0, 1, 0, 1, 0, 1, 1, 3, w11b);
  // proton, neutron, proton -> proton, neutron, proton (Z3 = -1/2).
  // Pair CGs <1/2 -1/2 1/2 +1/2 | T12 0>: T12 = 0 -> -1/sqrt2, T12 = 1 -> +1/sqrt2.
  // Third CGs with M = -1/2, (T12, 0; 1/2, -1/2): (0, 1/2) -> 1; (1, 1/2) -> +sqrt(1/3);
  // (1, 3/2) -> sqrt(2/3).
  const double r2 = std::sqrt(0.5);
  const double c0 = -r2, c1 = r2;
  const double t0 = 1.0, t1_half = std::sqrt(1.0 / 3.0), t1_32 = std::sqrt(2.0 / 3.0);
  const double expect = w00 * c0 * c0 * t0 * t0 + w01 * c0 * c1 * t0 * t1_half +
                        w01 * c1 * c0 * t1_half * t0 + w11a * c1 * c1 * t1_half * t1_half +
                        w11b * c1 * c1 * t1_32 * t1_32;
  const auto got = w.get_pn(orb(0, 0, 1, -1), orb(0, 1, 1, +1), orb(0, 0, 1, -1), orb(0, 0, 1, -1),
                            orb(0, 1, 1, +1), orb(0, 0, 1, -1), 1);
  EXPECT_NEAR(got, expect, 1e-7);
  EXPECT_GT(std::abs(got), 0.1);
}

}  // namespace
