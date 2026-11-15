// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Orbit quantum numbers, the imsrg++ hash and the label format.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/orbit.h"

namespace imsrg::basis {
namespace {

Orbit Make(int n, int l, int j2, int tz2) {
  return Orbit{n, l, TwoJ{j2}, TwoTz{tz2}};
}

// The untruncated orbit list in imsrg++ generation order (ModelSpace::Init).
std::vector<Orbit> Generate(int emax) {
  std::vector<Orbit> v;
  for (int N = 0; N <= emax; ++N) {
    for (int l = N; l >= 0; l -= 2) {
      const int n = (N - l) / 2;
      for (const int j2 : {2 * l + 1, 2 * l - 1}) {
        if (j2 <= 0) continue;
        for (const int tz2 : {-1, 1}) v.push_back(Make(n, l, j2, tz2));
      }
    }
  }
  return v;
}

TEST(Orbit, DefaultIsProton0s1) {
  const Orbit o;
  EXPECT_EQ(o, Make(0, 0, 1, -1));
  EXPECT_TRUE(o.is_proton());
  EXPECT_FALSE(o.is_neutron());
  EXPECT_EQ(o.e(), 0);
  EXPECT_TRUE(IsValid(o));
}

TEST(Orbit, ShellAndParity) {
  const Orbit o = Make(2, 3, 7, 1);  // n2 f7/2 neutron
  EXPECT_EQ(o.e(), 7);
  EXPECT_TRUE(o.is_neutron());
  EXPECT_EQ(o.parity(), ParityOfL(3));
  EXPECT_NE(o.parity(), ParityOfL(2));
}

TEST(Orbit, Validity) {
  EXPECT_TRUE(IsValid(Make(0, 0, 1, 1)));
  EXPECT_FALSE(IsValid(Make(0, 0, -1, 1)));  // 2j = 2l - 1 = -1
  EXPECT_FALSE(IsValid(Make(0, 1, 2, 1)));
  EXPECT_FALSE(IsValid(Make(-1, 0, 1, 1)));
  EXPECT_FALSE(IsValid(Make(0, -1, 1, 1)));
  EXPECT_FALSE(IsValid(Make(0, 0, 1, 0)));
  EXPECT_FALSE(IsValid(Make(0, 0, 1, 3)));
}

TEST(Orbit, NumOrbitsUntruncated) {
  EXPECT_EQ(NumOrbitsUntruncated(-1), 0);
  EXPECT_EQ(NumOrbitsUntruncated(0), 2);
  EXPECT_EQ(NumOrbitsUntruncated(2), 12);
  EXPECT_EQ(NumOrbitsUntruncated(6), 56);
  for (int emax = 0; emax <= 6; ++emax) {
    EXPECT_EQ(static_cast<int>(Generate(emax).size()),
              NumOrbitsUntruncated(emax));
  }
}

TEST(Orbit, HashEqualsIndexInUntruncatedBasis) {
  for (int emax = 0; emax <= 6; ++emax) {
    const auto orbits = Generate(emax);
    for (std::size_t i = 0; i < orbits.size(); ++i) {
      const Orbit& o = orbits[i];
      ASSERT_TRUE(IsValid(o));
      EXPECT_EQ(ImsrgppOrbitHash(o), static_cast<int>(i)) << OrbitLabel(o);
      // Protons even, neutron = proton + 1.
      EXPECT_EQ(i % 2, o.is_proton() ? 0u : 1u);
      if (o.is_neutron()) {
        Orbit p = o;
        p.tz2 = TwoTz{-1};
        EXPECT_EQ(ImsrgppOrbitHash(p) + 1, ImsrgppOrbitHash(o));
      }
    }
  }
}

TEST(Orbit, FirstOrbitsOfImsrgpp) {
  const std::vector<std::string> expected = {
      "p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1", "p0d5",
      "n0d5", "p0d3", "n0d3", "p1s1", "n1s1", "p0f7"};
  const auto orbits = Generate(3);
  ASSERT_GE(orbits.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(OrbitLabel(orbits[i]), expected[i]) << i;
  }
  EXPECT_EQ(ImsrgppOrbitHash(ParseOrbitLabel("p0d5")), 6);
  EXPECT_EQ(ImsrgppOrbitHash(ParseOrbitLabel("p0f7")), 12);
}

TEST(Orbit, LabelsRoundTripUpToEmax12) {
  for (const Orbit& o : Generate(12)) {
    const std::string label = OrbitLabel(o);
    EXPECT_EQ(ParseOrbitLabel(label), o) << label;
  }
}

TEST(Orbit, LabelLetters) {
  const std::string letters = "spdfghijklmno";
  for (int l = 0; l <= 12; ++l) {
    const std::string label = OrbitLabel(Make(0, l, 2 * l + 1, 1));
    EXPECT_EQ(label, "n0" +
                         std::string(1, letters[static_cast<std::size_t>(l)]) +
                         std::to_string(2 * l + 1));
  }
  // imsrg++ writes j for l = 7.
  EXPECT_EQ(OrbitLabel(Make(0, 7, 15, 1)), "n0j15");
  EXPECT_EQ(ParseOrbitLabel("n0j15"), Make(0, 7, 15, 1));
  EXPECT_EQ(ParseOrbitLabel("p0s1"), Make(0, 0, 1, -1));
  EXPECT_EQ(ParseOrbitLabel("n1s1"), Make(1, 0, 1, 1));
  EXPECT_EQ(ParseOrbitLabel("p0d5"), Make(0, 2, 5, -1));
  EXPECT_EQ(ParseOrbitLabel("p0d3"), Make(0, 2, 3, -1));
}

TEST(Orbit, MultiDigitN) {
  const Orbit o = Make(12, 0, 1, 1);
  EXPECT_EQ(OrbitLabel(o), "n12s1");
  EXPECT_EQ(ParseOrbitLabel("n12s1"), o);
  EXPECT_EQ(ParseOrbitLabel("p10p3"), Make(10, 1, 3, -1));
  EXPECT_EQ(ParseOrbitLabel("p10p3").e(), 21);
}

TEST(Orbit, LabelOfInvalidOrbitThrows) {
  EXPECT_THROW((void)OrbitLabel(Make(0, 0, 2, 1)), InputError);
  EXPECT_THROW((void)OrbitLabel(Make(0, 0, 1, 0)), InputError);
  EXPECT_THROW((void)OrbitLabel(Make(-1, 0, 1, 1)), InputError);
  EXPECT_THROW((void)OrbitLabel(Make(0, 13, 27, 1)),
               InputError);  // no letter for l = 13
}

TEST(Orbit, ParseErrors) {
  EXPECT_THROW((void)ParseOrbitLabel(""), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("x0s1"), InputError);  // bad first char
  EXPECT_THROW((void)ParseOrbitLabel("P0s1"), InputError);  // case sensitive
  EXPECT_THROW((void)ParseOrbitLabel("0s1"), InputError);   // no species
  EXPECT_THROW((void)ParseOrbitLabel("p0z1"), InputError);  // unknown letter
  EXPECT_THROW((void)ParseOrbitLabel("p0d4"),
               InputError);  // 2j inconsistent with l
  EXPECT_THROW((void)ParseOrbitLabel("p0d7"),
               InputError);  // 2j inconsistent with l
  EXPECT_THROW((void)ParseOrbitLabel("p0s3"),
               InputError);  // 2j inconsistent with l
  EXPECT_THROW((void)ParseOrbitLabel("p0s-1"), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("p0d5x"), InputError);  // trailing junk
  EXPECT_THROW((void)ParseOrbitLabel("p0d5 "), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("p0d5d"), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("p"), InputError);  // missing parts
  EXPECT_THROW((void)ParseOrbitLabel("p0"), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("p0d"), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("pd5"), InputError);
  EXPECT_THROW((void)ParseOrbitLabel("p-1s1"), InputError);
}

TEST(Orbit, ParseErrorNamesTheLabel) {
  try {
    (void)ParseOrbitLabel("p0d4");
    FAIL() << "no exception";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find("p0d4"), std::string::npos)
        << e.what();
  }
}

TEST(Orbit, Equality) {
  EXPECT_EQ(Make(0, 2, 5, -1), Make(0, 2, 5, -1));
  EXPECT_NE(Make(0, 2, 5, -1), Make(0, 2, 5, 1));
  EXPECT_NE(Make(0, 2, 5, -1), Make(0, 2, 3, -1));
  EXPECT_NE(Make(0, 2, 5, -1), Make(1, 2, 5, -1));
  EXPECT_NE(Make(0, 1, 3, -1), Make(0, 2, 5, -1));
}

}  // namespace
}  // namespace imsrg::basis
