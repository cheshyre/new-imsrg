// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Nucleus names, element symbols and the naive shell-model filling.
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/nucleus.h"

namespace imsrg::basis {
namespace {

// Occupation of the orbit with the given label in `filling`, or -1 if it is
// absent.
double OccupationOf(const std::vector<OrbitOccupation>& filling,
                    const char* label) {
  const Orbit o = ParseOrbitLabel(label);
  for (const auto& f : filling) {
    if (f.orbit == o) return f.occupation;
  }
  return -1.0;
}

TEST(Nucleus, ParseElements) {
  const Nucleus o16 = Nucleus::Parse("O16");
  EXPECT_EQ(o16.A, 16.0);
  EXPECT_EQ(o16.Z, 8.0);
  EXPECT_EQ(o16.N(), 8.0);
  const Nucleus he4 = Nucleus::Parse("He4");
  EXPECT_EQ(he4.A, 4.0);
  EXPECT_EQ(he4.Z, 2.0);
  const Nucleus ca48 = Nucleus::Parse("Ca48");
  EXPECT_EQ(ca48.A, 48.0);
  EXPECT_EQ(ca48.Z, 20.0);
  EXPECT_EQ(ca48.N(), 28.0);
  const Nucleus pb208 = Nucleus::Parse("Pb208");
  EXPECT_EQ(pb208.A, 208.0);
  EXPECT_EQ(pb208.Z, 82.0);
  EXPECT_EQ(Nucleus::Parse("H1"), (Nucleus{1.0, 1.0}));
}

TEST(Nucleus, ParseNeutronSymbol) {
  // "n" is the element with Z = 0; as in imsrg++ the A is then part of the
  // name, and a bare "n0" parses to A = Z = 0 without error (documented
  // behavior, same as "vacuum").
  const Nucleus n0 = Nucleus::Parse("n0");
  EXPECT_EQ(n0.A, 0.0);
  EXPECT_EQ(n0.Z, 0.0);
  const Nucleus n1 = Nucleus::Parse("n1");
  EXPECT_EQ(n1.A, 1.0);
  EXPECT_EQ(n1.Z, 0.0);
}

TEST(Nucleus, ParseVacuumAndMixture) {
  EXPECT_EQ(Nucleus::Parse("vacuum"), (Nucleus{0.0, 0.0}));
  const Nucleus m = Nucleus::Parse("mix_A4.5_Z2.5");
  EXPECT_EQ(m.A, 4.5);
  EXPECT_EQ(m.Z, 2.5);
  EXPECT_EQ(m.N(), 2.0);
  EXPECT_EQ(Nucleus::Parse("mix_A16_Z8"), (Nucleus{16.0, 8.0}));
}

TEST(Nucleus, ParseErrors) {
  EXPECT_THROW((void)Nucleus::Parse("Xx12"), InputError);  // unknown element
  EXPECT_THROW((void)Nucleus::Parse("O"), InputError);     // missing A
  EXPECT_THROW((void)Nucleus::Parse("16O"), InputError);   // A first
  EXPECT_THROW((void)Nucleus::Parse("O16x"),
               InputError);                              // trailing characters
  EXPECT_THROW((void)Nucleus::Parse("O4"), InputError);  // A < Z
  EXPECT_THROW((void)Nucleus::Parse(""), InputError);
  EXPECT_THROW((void)Nucleus::Parse("o16"), InputError);  // case sensitive
  EXPECT_THROW((void)Nucleus::Parse("mix_A4.5"), InputError);
  EXPECT_THROW((void)Nucleus::Parse("mix_A4.5_Z"), InputError);
  EXPECT_THROW((void)Nucleus::Parse("mix_A2_Z3"), InputError);  // A < Z
  EXPECT_THROW((void)Nucleus::Parse("mix_A4_Z2x"), InputError);
}

TEST(Nucleus, ParseErrorNamesTheInput) {
  try {
    (void)Nucleus::Parse("Xx12");
    FAIL() << "no exception";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find("Xx12"), std::string::npos)
        << e.what();
  }
}

TEST(Nucleus, NameRoundTrip) {
  for (const char* n :
       {"O16", "He4", "Ca48", "Pb208", "H1", "vacuum", "mix_A4.5_Z2.5"}) {
    const Nucleus nuc = Nucleus::Parse(n);
    EXPECT_EQ(nuc.name(), n);
    EXPECT_EQ(Nucleus::Parse(nuc.name()), nuc);
  }
  EXPECT_EQ((Nucleus{0.0, 0.0}).name(), "vacuum");
  EXPECT_EQ((Nucleus{16.0, 8.0}).name(), "O16");
  // Fractional values go through the mix_ form and survive the round trip.
  for (const Nucleus n :
       {Nucleus{4.5, 2.5}, Nucleus{17.25, 8.0}, Nucleus{0.1, 0.05}}) {
    EXPECT_EQ(Nucleus::Parse(n.name()), n) << n.name();
  }
}

TEST(Nucleus, ElementSymbolsAndProtonNumbers) {
  EXPECT_EQ(ElementSymbol(0), "n");
  EXPECT_EQ(ElementSymbol(1), "H");
  EXPECT_EQ(ElementSymbol(2), "He");
  EXPECT_EQ(ElementSymbol(8), "O");
  EXPECT_EQ(ElementSymbol(20), "Ca");
  EXPECT_EQ(ElementSymbol(82), "Pb");
  EXPECT_EQ(ElementSymbol(118), "Og");
  std::set<std::string> seen;
  for (int Z = 0; Z <= 118; ++Z) {
    const std::string s(ElementSymbol(Z));
    EXPECT_FALSE(s.empty()) << Z;
    EXPECT_TRUE(seen.insert(s).second) << "duplicate symbol " << s;
    EXPECT_EQ(ElementProtonNumber(s), Z) << s;
  }
  EXPECT_EQ(seen.size(), 119u);
  EXPECT_THROW((void)ElementSymbol(-1), InputError);
  EXPECT_THROW((void)ElementSymbol(119), InputError);
  EXPECT_THROW((void)ElementProtonNumber("Xx"), InputError);
  EXPECT_THROW((void)ElementProtonNumber(""), InputError);
  EXPECT_THROW((void)ElementProtonNumber("he"), InputError);
  EXPECT_THROW((void)ElementProtonNumber("HE"), InputError);
}

TEST(NaiveFilling, O16) {
  const auto f = NaiveFilling(Nucleus::Parse("O16"), 4);
  // 0s1/2, 0p3/2, 0p1/2 for protons and neutrons: six full orbits.
  ASSERT_EQ(f.size(), 6u);
  const std::vector<std::string> order = {"p0s1", "n0s1", "p0p3",
                                          "n0p3", "p0p1", "n0p1"};
  for (std::size_t i = 0; i < f.size(); ++i) {
    EXPECT_EQ(OrbitLabel(f[i].orbit), order[i]);
    EXPECT_EQ(f[i].occupation, 1.0);
  }
}

TEST(NaiveFilling, Ne20FractionalOccupationIsBitwiseExact) {
  const auto f = NaiveFilling(Nucleus::Parse("Ne20"), 4);
  ASSERT_EQ(f.size(), 8u);  // 6 full + p0d5 and n0d5
  EXPECT_EQ(OccupationOf(f, "p0d5"), 2.0 / 6.0);
  EXPECT_EQ(OccupationOf(f, "n0d5"), 2.0 / 6.0);
  EXPECT_EQ(OccupationOf(f, "p0d5"), 0.33333333333333331);
  EXPECT_EQ(OccupationOf(f, "p0p1"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p0d3"), -1.0);  // not occupied
}

TEST(NaiveFilling, Ca48AtEmax4) {
  const auto f = NaiveFilling(Nucleus::Parse("Ca48"), 4);
  // Protons: 20 = 2 + 4 + 2 + 6 + 4 + 2 fills s1 p3 p1 d5 d3 s1 (six orbits).
  EXPECT_EQ(OccupationOf(f, "p0d3"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p1s1"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p0f7"), -1.0);
  // Neutrons: 28 = 20 + 8, the f7/2 neutrons are full, nothing above.
  EXPECT_EQ(OccupationOf(f, "n0f7"), 1.0);
  EXPECT_EQ(OccupationOf(f, "n0p3") + OccupationOf(f, "n0d3"), 2.0);
  EXPECT_EQ(OccupationOf(f, "n1p3"), -1.0);
  EXPECT_EQ(OccupationOf(f, "n0f5"), -1.0);
  EXPECT_EQ(
      f.size(),
      13u);  // 6 proton orbits (s1 p3 p1 d5 d3 s1) + 7 neutron (adds n0f7)
  double protons = 0, neutrons = 0;
  for (const auto& o : f) {
    (o.orbit.is_proton() ? protons : neutrons) +=
        o.occupation * (o.orbit.j2.twice + 1);
  }
  EXPECT_EQ(protons, 20.0);
  EXPECT_EQ(neutrons, 28.0);
}

TEST(NaiveFilling, MixedFractionalNucleus) {
  const auto f = NaiveFilling(Nucleus::Parse("mix_A4.5_Z2.5"), 4);
  // 2.5 protons: s1/2 full (2), the remaining 0.5 in p3/2 -> 0.5 / 4.
  EXPECT_EQ(OccupationOf(f, "p0s1"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p0p3"), 0.125);
  EXPECT_EQ(OccupationOf(f, "n0s1"), 1.0);
  EXPECT_EQ(OccupationOf(f, "n0p3"), -1.0);
  EXPECT_EQ(f.size(), 3u);
}

TEST(NaiveFilling, Vacuum) {
  EXPECT_TRUE(NaiveFilling(Nucleus::Parse("vacuum"), 4).empty());
  EXPECT_TRUE(NaiveFilling(Nucleus{0.0, 0.0}, 0).empty());
}

TEST(NaiveFilling, He4AndSingleShell) {
  const auto f = NaiveFilling(Nucleus::Parse("He4"), 0);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0].orbit, ParseOrbitLabel("p0s1"));
  EXPECT_EQ(f[1].orbit, ParseOrbitLabel("n0s1"));
  EXPECT_EQ(f[0].occupation, 1.0);
}

TEST(NaiveFilling, SkipList) {
  const Orbit skip[] = {ParseOrbitLabel("p0p1"), ParseOrbitLabel("n0p1")};
  const auto f = NaiveFilling(Nucleus::Parse("O16"), 4, skip);
  // The p1/2 orbits are passed over, so the remaining 2 nucleons of each
  // species go to d5/2.
  EXPECT_EQ(OccupationOf(f, "p0p1"), -1.0);
  EXPECT_EQ(OccupationOf(f, "n0p1"), -1.0);
  EXPECT_EQ(OccupationOf(f, "p0s1"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p0p3"), 1.0);
  EXPECT_EQ(OccupationOf(f, "p0d5"), 2.0 / 6.0);
  EXPECT_EQ(OccupationOf(f, "n0d5"), 2.0 / 6.0);
  EXPECT_EQ(f.size(), 6u);
}

TEST(NaiveFilling, DoesNotFit) {
  // Ca48: 28 neutrons need shells up to N = 3, but emax = 2 holds only 20.
  EXPECT_THROW((void)NaiveFilling(Nucleus::Parse("Ca48"), 2), InputError);
  EXPECT_NO_THROW((void)NaiveFilling(Nucleus::Parse("Ca40"), 2));
  EXPECT_THROW((void)NaiveFilling(Nucleus::Parse("O16"), 0), InputError);
  EXPECT_NO_THROW((void)NaiveFilling(Nucleus::Parse("O16"), 1));
}

}  // namespace
}  // namespace imsrg::basis
