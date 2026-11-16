// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Unit tests of ThreeBodySpace (the kept canonical three-body space) and its
// ModelSpace integration. The O16 sizes were measured by the imsrg3 layout
// benchmark (design/benchmarks/imsrg3-layout.md).
#include <cstdint>
#include <memory>
#include <set>
#include <tuple>

#include <gtest/gtest.h>

#include "imsrg/basis/model_space.h"
#include "imsrg/basis/nucleus.h"
#include "imsrg/basis/three_body_space.h"

namespace imsrg::basis {
namespace {

std::shared_ptr<const ModelSpace> O16(std::int32_t emax, ThreeBodyCuts cuts) {
  auto ms = ModelSpace::FromNames(BasisTruncation{emax}, 16.0, "O16", "O16");
  return ms->WithThreeBody(cuts);
}

ThreeBodyCuts Cuts(std::int32_t e3max, double de3max = 1.0e30) {
  ThreeBodyCuts c;
  c.e3max = e3max;
  c.de3max = de3max;
  return c;
}

struct Expected {
  std::int32_t emax;
  std::int32_t e3max;
  double de3max;
  std::int64_t sum;
  std::int32_t max_dim;
  std::int64_t packed;  // 0: not checked
};

TEST(ThreeBodySpaceTest, O16SizesMatchBenchmark) {
  const Expected cases[] = {
      {2, 6, 1.0e30, 2099, 113, 0},
      {3, 9, 1.0e30, 13223, 554, 2261446},
      {3, 7, 1.0e30, 8909, 449, 0},
      {3, 9, 3.0, 5832, 270, 0},
  };
  for (const Expected& e : cases) {
    const auto ms = O16(e.emax, Cuts(e.e3max, e.de3max));
    const ThreeBodySpaceCounts c = ms->three_body_space()->Counts();
    EXPECT_EQ(c.sum_dim, e.sum) << e.emax << " " << e.e3max << " " << e.de3max;
    EXPECT_EQ(c.max_dim, e.max_dim);
    if (e.packed != 0) EXPECT_EQ(c.sum_packed, e.packed);
  }
}

TEST(ThreeBodySpaceTest, NoCutKeepsEveryKetAndStructureIsConsistent) {
  const auto ms = O16(2, Cuts(6));
  const ThreeBodyChannels& tbc = *ms->three_body();
  const ThreeBodySpace& sp = *ms->three_body_space();
  EXPECT_EQ(sp.Counts().sum_dim, tbc.SumChannelKets());
  EXPECT_EQ(sp.NumRows(), tbc.SumChannelKets());
  ASSERT_EQ(sp.num_channels(), tbc.num_channels());
  std::int64_t sum2 = 0;
  for (std::int32_t c = 0; c < sp.num_channels(); ++c) {
    const Channel3Index ch{c};
    EXPECT_EQ(sp.dim(ch), tbc.channel(ch).num_kets);
    EXPECT_EQ(sp.channel(ch).two_j, tbc.channel(ch).two_j);
    EXPECT_EQ(sp.channel(ch).parity, tbc.channel(ch).parity);
    EXPECT_EQ(sp.channel(ch).two_tz, tbc.channel(ch).two_tz);
    sum2 += static_cast<std::int64_t>(sp.dim(ch)) * sp.dim(ch);
  }
  EXPECT_EQ(sp.Counts().sum_dim2, sum2);
}

TEST(ThreeBodySpaceTest, CutKeepsExactlyTheKetsPassingIt) {
  ThreeBodyCuts cuts = Cuts(9, 3.0);
  // Natural occupations 0.5 (n (1 - n) = 0.25) on even orbits and 0.9 (0.09)
  // on odd ones: a ket with three even orbits has 0.0156, mixed ones less.
  cuts.occnat3_cut = 0.01;
  const auto ms0 = O16(3, cuts);
  for (std::int32_t i = 0; i < ms0->num_orbits(); ++i) {
    cuts.natural_occupations.push_back(i % 2 == 0 ? 0.5 : 0.9);
  }
  const auto ms = O16(3, cuts);
  const ThreeBodyChannels& tbc = *ms->three_body();
  const ThreeBodySpace& sp = *ms->three_body_space();
  std::int64_t kept = 0;
  for (std::int32_t c = 0; c < tbc.num_channels(); ++c) {
    const Channel3Index ch{c};
    std::int32_t local = 0;
    for (const Ket3Index k : tbc.channel_kets(ch)) {
      const bool pass = tbc.PassesConfigurationCut(
          k, ms->basis(), ms->reference(), cuts.de3max, cuts.occnat3_cut,
          cuts.natural_occupations);
      const Ket3& ket = tbc.ket(k);
      const auto row = sp.find_row(ch, ket.p, ket.q, ket.r, ket.two_jpq);
      ASSERT_EQ(row.has_value(), pass);
      if (pass) {
        EXPECT_EQ(row->value(), local);
        EXPECT_EQ(sp.row_ket(sp.FlatRow(ch, *row)), k);
        ++local;
        ++kept;
      }
    }
    EXPECT_EQ(sp.dim(ch), local);
  }
  EXPECT_EQ(sp.NumRows(), kept);
  EXPECT_LT(kept, tbc.SumChannelKets());
  EXPECT_GT(kept, 0);
}

TEST(ThreeBodySpaceTest, TriplesAndRowsRoundTrip) {
  const auto ms = O16(2, Cuts(6, 3.0));
  const ThreeBodySpace& sp = *ms->three_body_space();
  std::int64_t rows_seen = 0;
  std::int32_t num_empty = 0;
  for (std::int32_t c = 0; c < sp.num_channels(); ++c) {
    const Channel3Index ch{c};
    const KeptChannel3& kc = sp.channel(ch);
    if (kc.dim == 0) ++num_empty;
    EXPECT_EQ(kc.row_offset, rows_seen);
    std::int32_t next_row = 0;
    std::set<std::tuple<int, int, int>> seen;
    for (std::int32_t t = kc.triple_begin.value(); t < kc.triple_end.value();
         ++t) {
      const OrbitTriple& tr = sp.triple(Triple3Index{t});
      EXPECT_LE(tr.a.value(), tr.b.value());
      EXPECT_LE(tr.b.value(), tr.c.value());
      EXPECT_EQ(tr.channel, ch);
      EXPECT_EQ(tr.row_begin.value(), next_row);  // contiguous
      EXPECT_GT(tr.row_end.value(), tr.row_begin.value());
      const int expected_sym = (tr.a == tr.b && tr.b == tr.c)   ? 1
                               : (tr.a == tr.b || tr.b == tr.c) ? 3
                                                                : 6;
      EXPECT_EQ(tr.sym, expected_sym);
      EXPECT_TRUE(
          seen.insert({tr.a.value(), tr.b.value(), tr.c.value()}).second);
      for (std::int32_t r = tr.row_begin.value(); r < tr.row_end.value(); ++r) {
        const std::int64_t flat = sp.FlatRow(ch, KeptRow3Index{r});
        EXPECT_EQ(sp.row_triple(flat), Triple3Index{t});
        const Ket3& ket = ms->three_body()->ket(sp.row_ket(flat));
        EXPECT_EQ(ket.p, tr.a);
        EXPECT_EQ(ket.q, tr.b);
        EXPECT_EQ(ket.r, tr.c);
        EXPECT_EQ(ket.two_jpq, sp.row_two_jab(flat));
        const auto found = sp.find_row(ch, tr.a, tr.b, tr.c, ket.two_jpq);
        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(found->value(), r);
        const ChannelRow3 cr = sp.ChannelRowOf(flat);
        EXPECT_EQ(cr.channel, ch);
        EXPECT_EQ(cr.row.value(), r);
      }
      next_row = tr.row_end.value();
    }
    EXPECT_EQ(next_row, kc.dim);
    rows_seen += kc.dim;
  }
  EXPECT_EQ(rows_seen, sp.NumRows());
  EXPECT_GT(num_empty, 0);  // the dE3max cut empties some channels
  EXPECT_EQ(sp.Counts().num_triples, sp.num_triples());
}

TEST(ThreeBodySpaceTest, FindRowRejectsNonCanonicalAndUnknown) {
  const auto ms = O16(2, Cuts(6));
  const ThreeBodySpace& sp = *ms->three_body_space();
  // Non-canonical order (a > b) and a bad Jab are not found.
  for (std::int32_t c = 0; c < sp.num_channels(); ++c) {
    const Channel3Index chan{c};
    if (sp.dim(chan) == 0) continue;
    const OrbitTriple& tr = sp.triple(sp.channel(chan).triple_begin);
    if (tr.a != tr.c) {
      EXPECT_FALSE(sp.find_row(chan, tr.c, tr.b, tr.a, TwoJ{0}).has_value());
    }
    EXPECT_FALSE(sp.find_row(chan, tr.a, tr.b, tr.c, TwoJ{-2}).has_value());
    EXPECT_FALSE(sp.find_row(chan, tr.a, tr.b, tr.c, TwoJ{100}).has_value());
  }
  EXPECT_FALSE(sp.find_row(Channel3Index{-1}, OrbitIndex{0}, OrbitIndex{0},
                           OrbitIndex{0}, TwoJ{0})
                   .has_value());
}

TEST(ThreeBodySpaceTest, WithReferenceRebuildsKeptSpaceSharesChannels) {
  const auto ms = O16(3, Cuts(9, 3.0));
  const Nucleus ca40 = Nucleus::Parse("Ca40");
  const Reference ref = Reference::FromNucleus(ms->basis(), ca40);
  const auto other = ms->WithReference(ref);
  ASSERT_NE(other->three_body_space(), nullptr);
  EXPECT_EQ(other->three_body(), ms->three_body());  // shared object
  EXPECT_NE(other->three_body_space(), ms->three_body_space());
  EXPECT_NE(other->three_body_space()->Counts().sum_dim,
            ms->three_body_space()->Counts().sum_dim);
  // Same reference again: identical kept space.
  const auto same = ms->WithReference(ms->reference());
  EXPECT_EQ(same->three_body_space()->Counts().sum_dim,
            ms->three_body_space()->Counts().sum_dim);
}

TEST(ThreeBodySpaceTest, ModelSpaceWithoutThreeBodyHasNoSpace) {
  const auto ms = ModelSpace::FromNames(BasisTruncation{2}, 16.0, "O16", "O16");
  EXPECT_EQ(ms->three_body_space(), nullptr);
  EXPECT_EQ(ms->WithReference(ms->reference())->three_body_space(), nullptr);
}

TEST(ThreeBodySpaceTest, EqualityComparesFullCuts) {
  const auto a = O16(2, Cuts(6, 3.0));
  const auto b = O16(2, Cuts(6, 3.0));
  EXPECT_TRUE(*a == *b);
  EXPECT_FALSE(*a == *O16(2, Cuts(6, 4.0)));
  ThreeBodyCuts occ = Cuts(6, 3.0);
  occ.occnat3_cut = 1.0e-3;
  EXPECT_FALSE(*a == *O16(2, occ));
  ThreeBodyCuts nat = Cuts(6, 3.0);
  nat.natural_occupations.assign(static_cast<std::size_t>(a->num_orbits()),
                                 0.5);
  EXPECT_FALSE(*a == *O16(2, nat));
  EXPECT_TRUE(Cuts(6) == Cuts(6));
}

}  // namespace
}  // namespace imsrg::basis
