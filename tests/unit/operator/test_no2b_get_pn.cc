// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Bitwise comparison of `No2bThreeBody::get_pn` (dense orbit table, tabulated
// isospin Clebsch-Gordan coefficients) with a test-local copy of the original
// algorithm (linear orbit scan, direct angmom::ClebschGordan calls).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.h"
#include "imsrg/base/error.h"
#include "imsrg/operator/no2b.h"

namespace imsrg::op {
namespace {

basis::Orbit Orb(int n, int l, int tj, int tz) {
  return basis::Orbit{n, l, TwoJ{tj}, TwoTz{tz}};
}

std::optional<std::int32_t> LinearIndex(const No2bThreeBody& w,
                                        const basis::Orbit& o) {
  const auto orbs = w.isospin_orbits();
  for (std::size_t i = 0; i < orbs.size(); ++i) {
    if (orbs[i].n == o.n && orbs[i].l == o.l && orbs[i].j2 == o.j2) {
      return static_cast<std::int32_t>(i);
    }
  }
  return std::nullopt;
}

// The pre-optimisation get_pn.
double OldGetPn(const No2bThreeBody& w, const basis::Orbit& a,
                const basis::Orbit& b, const basis::Orbit& c,
                const basis::Orbit& d, const basis::Orbit& e,
                const basis::Orbit& f, std::int32_t J2) {
  if (c.l % 2 != f.l % 2 || c.j2 != f.j2) return 0.0;
  if ((a.l + b.l) % 2 != (d.l + e.l) % 2) return 0.0;
  const std::int32_t z3 = a.tz2.twice + b.tz2.twice + c.tz2.twice;
  if (z3 != d.tz2.twice + e.tz2.twice + f.tz2.twice) return 0.0;
  const auto i1 = LinearIndex(w, a);
  const auto i2 = LinearIndex(w, b);
  const auto i3 = LinearIndex(w, c);
  const auto i4 = LinearIndex(w, d);
  const auto i5 = LinearIndex(w, e);
  const auto i6 = LinearIndex(w, f);
  if (!i1 || !i2 || !i3 || !i4 || !i5 || !i6) return 0.0;
  const std::int32_t zab = a.tz2.twice + b.tz2.twice;
  const std::int32_t zde = d.tz2.twice + e.tz2.twice;
  const TwoJ half{1};
  double vout = 0.0;
  for (std::int32_t t12 : {0, 1}) {
    if (std::abs(zab) > 2 * t12) continue;
    for (std::int32_t t45 : {0, 1}) {
      if (std::abs(zde) > 2 * t45) continue;
      for (std::int32_t twoT = 1; twoT <= std::min(2 * t12 + 1, 2 * t45 + 1);
           twoT += 2) {
        const double wv =
            w.get_isospin(*i1, *i2, *i3, t12, *i4, *i5, *i6, t45, J2, twoT);
        vout +=
            wv *
            angmom::ClebschGordan(half, TwoM{a.tz2.twice}, half,
                                  TwoM{b.tz2.twice}, TwoJ{2 * t12}, TwoM{zab}) *
            angmom::ClebschGordan(half, TwoM{d.tz2.twice}, half,
                                  TwoM{e.tz2.twice}, TwoJ{2 * t45}, TwoM{zde}) *
            angmom::ClebschGordan(TwoJ{2 * t12}, TwoM{zab}, half,
                                  TwoM{c.tz2.twice}, TwoJ{twoT}, TwoM{z3}) *
            angmom::ClebschGordan(TwoJ{2 * t45}, TwoM{zde}, half,
                                  TwoM{f.tz2.twice}, TwoJ{twoT}, TwoM{z3});
      }
    }
  }
  return vout;
}

TEST(No2bGetPn, MatchesOldAlgorithmBitwise) {
  No2bThreeBody w(No2bTruncation{2, 4, 6, 2});
  const auto orbs = w.isospin_orbits();
  const auto norb = static_cast<std::int32_t>(orbs.size());
  // Synthetic values on every element the storage accepts.
  std::int32_t nset = 0;
  for (std::int32_t a = 0; a < norb; ++a)
    for (std::int32_t b = 0; b < norb; ++b)
      for (std::int32_t c = 0; c < norb; ++c)
        for (std::int32_t d = 0; d < norb; ++d)
          for (std::int32_t e = 0; e < norb; ++e)
            for (std::int32_t f = 0; f < norb; ++f)
              for (std::int32_t tab : {0, 1})
                for (std::int32_t tde : {0, 1})
                  for (std::int32_t J2 = 0; J2 <= 5; ++J2)
                    for (std::int32_t tt : {1, 3}) {
                      if (tt > 2 * tab + 1 || tt > 2 * tde + 1) continue;
                      const double v = 0.01 * (1 + (nset * 7919) % 1013) *
                                       ((nset % 3 == 0) ? -1.0 : 1.0);
                      try {
                        w.set_isospin(a, b, c, tab, d, e, f, tde, J2, tt, v);
                        ++nset;
                      } catch (const InvariantError&) {
                      }
                    }
  ASSERT_GT(nset, 100);
  std::vector<basis::Orbit> all;
  for (const auto& o : orbs)
    for (int tz : {-1, 1}) all.push_back(Orb(o.n, o.l, o.j2.twice, tz));
  int nonzero = 0;
  for (const auto& a : all)
    for (const auto& b : all)
      for (const auto& c : all)
        for (const auto& d : all)
          for (const auto& e : all)
            for (const auto& f : all)
              for (std::int32_t J2 = 0; J2 <= 5; ++J2) {
                const double got = w.get_pn(a, b, c, d, e, f, J2);
                const double ref = OldGetPn(w, a, b, c, d, e, f, J2);
                ASSERT_EQ(got, ref);
                if (ref != 0.0) ++nonzero;
              }
  EXPECT_GT(nonzero, 100);
}

TEST(No2bGetPn, OrbitIndexPresentAndAbsent) {
  const No2bThreeBody w(No2bTruncation{3, 6, 6, 3});
  const auto orbs = w.isospin_orbits();
  for (std::size_t i = 0; i < orbs.size(); ++i) {
    EXPECT_EQ(w.isospin_orbit_index(orbs[i].n, orbs[i].l, orbs[i].j2),
              static_cast<std::int32_t>(i));
  }
  EXPECT_FALSE(w.isospin_orbit_index(0, 0, TwoJ{3}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(0, 1, TwoJ{5}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(0, 4, TwoJ{7}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(2, 0, TwoJ{1}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(1, 2, TwoJ{3}).has_value());  // e = 4
  EXPECT_FALSE(w.isospin_orbit_index(-1, 0, TwoJ{1}).has_value());
  EXPECT_FALSE(w.isospin_orbit_index(0, -1, TwoJ{1}).has_value());
  EXPECT_FALSE(No2bThreeBody().isospin_orbit_index(0, 0, TwoJ{1}).has_value());
}

}  // namespace
}  // namespace imsrg::op
