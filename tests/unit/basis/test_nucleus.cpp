// Nucleus names, element symbols and the naive shell-model filling.
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/nucleus.hpp"

namespace imsrg::basis {
namespace {

// Occupation of the orbit with the given label in `filling`, or -1 if it is absent.
double occupation_of(const std::vector<OrbitOccupation>& filling, const char* label) {
  const Orbit o = parse_orbit_label(label);
  for (const auto& f : filling) {
    if (f.orbit == o) return f.occupation;
  }
  return -1.0;
}

TEST(Nucleus, ParseElements) {
  const Nucleus o16 = Nucleus::parse("O16");
  EXPECT_EQ(o16.A, 16.0);
  EXPECT_EQ(o16.Z, 8.0);
  EXPECT_EQ(o16.N(), 8.0);
  const Nucleus he4 = Nucleus::parse("He4");
  EXPECT_EQ(he4.A, 4.0);
  EXPECT_EQ(he4.Z, 2.0);
  const Nucleus ca48 = Nucleus::parse("Ca48");
  EXPECT_EQ(ca48.A, 48.0);
  EXPECT_EQ(ca48.Z, 20.0);
  EXPECT_EQ(ca48.N(), 28.0);
  const Nucleus pb208 = Nucleus::parse("Pb208");
  EXPECT_EQ(pb208.A, 208.0);
  EXPECT_EQ(pb208.Z, 82.0);
  EXPECT_EQ(Nucleus::parse("H1"), (Nucleus{1.0, 1.0}));
}

TEST(Nucleus, ParseNeutronSymbol) {
  // "n" is the element with Z = 0; as in imsrg++ the A is then part of the name, and a bare "n0"
  // parses to A = Z = 0 without error (documented behavior, same as "vacuum").
  const Nucleus n0 = Nucleus::parse("n0");
  EXPECT_EQ(n0.A, 0.0);
  EXPECT_EQ(n0.Z, 0.0);
  const Nucleus n1 = Nucleus::parse("n1");
  EXPECT_EQ(n1.A, 1.0);
  EXPECT_EQ(n1.Z, 0.0);
}

TEST(Nucleus, ParseVacuumAndMixture) {
  EXPECT_EQ(Nucleus::parse("vacuum"), (Nucleus{0.0, 0.0}));
  const Nucleus m = Nucleus::parse("mix_A4.5_Z2.5");
  EXPECT_EQ(m.A, 4.5);
  EXPECT_EQ(m.Z, 2.5);
  EXPECT_EQ(m.N(), 2.0);
  EXPECT_EQ(Nucleus::parse("mix_A16_Z8"), (Nucleus{16.0, 8.0}));
}

TEST(Nucleus, ParseErrors) {
  EXPECT_THROW((void)Nucleus::parse("Xx12"), InputError);  // unknown element
  EXPECT_THROW((void)Nucleus::parse("O"), InputError);     // missing A
  EXPECT_THROW((void)Nucleus::parse("16O"), InputError);   // A first
  EXPECT_THROW((void)Nucleus::parse("O16x"), InputError);  // trailing characters
  EXPECT_THROW((void)Nucleus::parse("O4"), InputError);    // A < Z
  EXPECT_THROW((void)Nucleus::parse(""), InputError);
  EXPECT_THROW((void)Nucleus::parse("o16"), InputError);  // case sensitive
  EXPECT_THROW((void)Nucleus::parse("mix_A4.5"), InputError);
  EXPECT_THROW((void)Nucleus::parse("mix_A4.5_Z"), InputError);
  EXPECT_THROW((void)Nucleus::parse("mix_A2_Z3"), InputError);  // A < Z
  EXPECT_THROW((void)Nucleus::parse("mix_A4_Z2x"), InputError);
}

TEST(Nucleus, ParseErrorNamesTheInput) {
  try {
    (void)Nucleus::parse("Xx12");
    FAIL() << "no exception";
  } catch (const InputError& e) {
    EXPECT_NE(std::string(e.what()).find("Xx12"), std::string::npos) << e.what();
  }
}

TEST(Nucleus, NameRoundTrip) {
  for (const char* n : {"O16", "He4", "Ca48", "Pb208", "H1", "vacuum", "mix_A4.5_Z2.5"}) {
    const Nucleus nuc = Nucleus::parse(n);
    EXPECT_EQ(nuc.name(), n);
    EXPECT_EQ(Nucleus::parse(nuc.name()), nuc);
  }
  EXPECT_EQ((Nucleus{0.0, 0.0}).name(), "vacuum");
  EXPECT_EQ((Nucleus{16.0, 8.0}).name(), "O16");
  // Fractional values go through the mix_ form and survive the round trip.
  for (const Nucleus n : {Nucleus{4.5, 2.5}, Nucleus{17.25, 8.0}, Nucleus{0.1, 0.05}}) {
    EXPECT_EQ(Nucleus::parse(n.name()), n) << n.name();
  }
}

TEST(Nucleus, ElementSymbolsAndProtonNumbers) {
  EXPECT_EQ(element_symbol(0), "n");
  EXPECT_EQ(element_symbol(1), "H");
  EXPECT_EQ(element_symbol(2), "He");
  EXPECT_EQ(element_symbol(8), "O");
  EXPECT_EQ(element_symbol(20), "Ca");
  EXPECT_EQ(element_symbol(82), "Pb");
  EXPECT_EQ(element_symbol(118), "Og");
  std::set<std::string> seen;
  for (int Z = 0; Z <= 118; ++Z) {
    const std::string s(element_symbol(Z));
    EXPECT_FALSE(s.empty()) << Z;
    EXPECT_TRUE(seen.insert(s).second) << "duplicate symbol " << s;
    EXPECT_EQ(element_proton_number(s), Z) << s;
  }
  EXPECT_EQ(seen.size(), 119u);
  EXPECT_THROW((void)element_symbol(-1), InputError);
  EXPECT_THROW((void)element_symbol(119), InputError);
  EXPECT_THROW((void)element_proton_number("Xx"), InputError);
  EXPECT_THROW((void)element_proton_number(""), InputError);
  EXPECT_THROW((void)element_proton_number("he"), InputError);
  EXPECT_THROW((void)element_proton_number("HE"), InputError);
}

TEST(NaiveFilling, O16) {
  const auto f = naive_filling(Nucleus::parse("O16"), 4);
  // 0s1/2, 0p3/2, 0p1/2 for protons and neutrons: six full orbits.
  ASSERT_EQ(f.size(), 6u);
  const std::vector<std::string> order = {"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"};
  for (std::size_t i = 0; i < f.size(); ++i) {
    EXPECT_EQ(orbit_label(f[i].orbit), order[i]);
    EXPECT_EQ(f[i].occupation, 1.0);
  }
}

TEST(NaiveFilling, Ne20FractionalOccupationIsBitwiseExact) {
  const auto f = naive_filling(Nucleus::parse("Ne20"), 4);
  ASSERT_EQ(f.size(), 8u);  // 6 full + p0d5 and n0d5
  EXPECT_EQ(occupation_of(f, "p0d5"), 2.0 / 6.0);
  EXPECT_EQ(occupation_of(f, "n0d5"), 2.0 / 6.0);
  EXPECT_EQ(occupation_of(f, "p0d5"), 0.33333333333333331);
  EXPECT_EQ(occupation_of(f, "p0p1"), 1.0);
  EXPECT_EQ(occupation_of(f, "p0d3"), -1.0);  // not occupied
}

TEST(NaiveFilling, Ca48AtEmax4) {
  const auto f = naive_filling(Nucleus::parse("Ca48"), 4);
  // Protons: 20 = 2 + 4 + 2 + 6 + 4 + 2 fills s1 p3 p1 d5 d3 s1 (six orbits).
  EXPECT_EQ(occupation_of(f, "p0d3"), 1.0);
  EXPECT_EQ(occupation_of(f, "p1s1"), 1.0);
  EXPECT_EQ(occupation_of(f, "p0f7"), -1.0);
  // Neutrons: 28 = 20 + 8, the f7/2 neutrons are full, nothing above.
  EXPECT_EQ(occupation_of(f, "n0f7"), 1.0);
  EXPECT_EQ(occupation_of(f, "n0p3") + occupation_of(f, "n0d3"), 2.0);
  EXPECT_EQ(occupation_of(f, "n1p3"), -1.0);
  EXPECT_EQ(occupation_of(f, "n0f5"), -1.0);
  EXPECT_EQ(f.size(), 13u);  // 6 proton orbits (s1 p3 p1 d5 d3 s1) + 7 neutron (adds n0f7)
  double protons = 0, neutrons = 0;
  for (const auto& o : f) {
    (o.orbit.is_proton() ? protons : neutrons) += o.occupation * (o.orbit.j2.twice + 1);
  }
  EXPECT_EQ(protons, 20.0);
  EXPECT_EQ(neutrons, 28.0);
}

TEST(NaiveFilling, MixedFractionalNucleus) {
  const auto f = naive_filling(Nucleus::parse("mix_A4.5_Z2.5"), 4);
  // 2.5 protons: s1/2 full (2), the remaining 0.5 in p3/2 -> 0.5 / 4.
  EXPECT_EQ(occupation_of(f, "p0s1"), 1.0);
  EXPECT_EQ(occupation_of(f, "p0p3"), 0.125);
  EXPECT_EQ(occupation_of(f, "n0s1"), 1.0);
  EXPECT_EQ(occupation_of(f, "n0p3"), -1.0);
  EXPECT_EQ(f.size(), 3u);
}

TEST(NaiveFilling, Vacuum) {
  EXPECT_TRUE(naive_filling(Nucleus::parse("vacuum"), 4).empty());
  EXPECT_TRUE(naive_filling(Nucleus{0.0, 0.0}, 0).empty());
}

TEST(NaiveFilling, He4AndSingleShell) {
  const auto f = naive_filling(Nucleus::parse("He4"), 0);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0].orbit, parse_orbit_label("p0s1"));
  EXPECT_EQ(f[1].orbit, parse_orbit_label("n0s1"));
  EXPECT_EQ(f[0].occupation, 1.0);
}

TEST(NaiveFilling, SkipList) {
  const Orbit skip[] = {parse_orbit_label("p0p1"), parse_orbit_label("n0p1")};
  const auto f = naive_filling(Nucleus::parse("O16"), 4, skip);
  // The p1/2 orbits are passed over, so the remaining 2 nucleons of each species go to d5/2.
  EXPECT_EQ(occupation_of(f, "p0p1"), -1.0);
  EXPECT_EQ(occupation_of(f, "n0p1"), -1.0);
  EXPECT_EQ(occupation_of(f, "p0s1"), 1.0);
  EXPECT_EQ(occupation_of(f, "p0p3"), 1.0);
  EXPECT_EQ(occupation_of(f, "p0d5"), 2.0 / 6.0);
  EXPECT_EQ(occupation_of(f, "n0d5"), 2.0 / 6.0);
  EXPECT_EQ(f.size(), 6u);
}

TEST(NaiveFilling, DoesNotFit) {
  // Ca48: 28 neutrons need shells up to N = 3, but emax = 2 holds only 20.
  EXPECT_THROW((void)naive_filling(Nucleus::parse("Ca48"), 2), InputError);
  EXPECT_NO_THROW((void)naive_filling(Nucleus::parse("Ca40"), 2));
  EXPECT_THROW((void)naive_filling(Nucleus::parse("O16"), 0), InputError);
  EXPECT_NO_THROW((void)naive_filling(Nucleus::parse("O16"), 1));
}

}  // namespace
}  // namespace imsrg::basis
