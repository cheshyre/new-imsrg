// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/reference.h"

namespace imsrg::basis {
namespace {

constexpr TwoTz kProton{-1};
constexpr TwoTz kNeutron{+1};

std::vector<int> Ints(std::span<const OrbitIndex> v) {
  std::vector<int> out;
  for (const OrbitIndex i : v) out.push_back(i.value());
  return out;
}

std::vector<int> Range(int lo, int hi) {  // [lo, hi]
  std::vector<int> out;
  for (int i = lo; i <= hi; ++i) out.push_back(i);
  return out;
}

SingleParticleBasis MakeBasis(int emax) {
  return SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{emax, -1, -1, -1}, 16.0);
}

TEST(Reference, O16Emax2) {
  const auto basis = MakeBasis(2);
  const Reference ref = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  EXPECT_EQ(Ints(ref.holes()), Range(0, 5));
  EXPECT_EQ(Ints(ref.particles()), Range(6, 11));
  EXPECT_DOUBLE_EQ(ref.A(), 16.0);
  EXPECT_DOUBLE_EQ(ref.Z(), 8.0);
  EXPECT_DOUBLE_EQ(ref.N(), 8.0);
  EXPECT_EQ(ref.target_mass(), 16);
  EXPECT_EQ(ref.target_Z(), 8);
  EXPECT_EQ(ref.name(), "O16");
  EXPECT_DOUBLE_EQ(ref.e_fermi(kProton), 1.5);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kNeutron), 1.5);
  for (int a = 0; a < 12; ++a)
    EXPECT_EQ(ref.occupation(OrbitIndex{a}), a < 6 ? 1.0 : 0.0);
  EXPECT_TRUE(ref.is_hole(OrbitIndex{5}));
  EXPECT_FALSE(ref.is_hole(OrbitIndex{6}));
}

TEST(Reference, He4Emax2) {
  const auto basis = MakeBasis(2);
  const Reference ref = Reference::FromNucleus(basis, Nucleus::Parse("He4"));
  EXPECT_EQ(Ints(ref.holes()), Range(0, 1));
  EXPECT_EQ(Ints(ref.particles()), Range(2, 11));
  EXPECT_DOUBLE_EQ(ref.A(), 4.0);
  EXPECT_DOUBLE_EQ(ref.Z(), 2.0);
  EXPECT_DOUBLE_EQ(ref.N(), 2.0);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kProton), 0.5);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kNeutron), 0.5);
}

TEST(Reference, Ne20FractionalSd) {
  const auto basis = MakeBasis(2);
  const Reference ref = Reference::FromNucleus(basis, Nucleus::Parse("Ne20"));
  EXPECT_EQ(Ints(ref.holes()), Range(0, 7));
  EXPECT_EQ(Ints(ref.particles()), Range(8, 11));
  EXPECT_EQ(ref.occupation(OrbitIndex{6}), 2.0 / 6.0);  // p0d5/2
  EXPECT_EQ(ref.occupation(OrbitIndex{7}), 2.0 / 6.0);  // n0d5/2
  EXPECT_DOUBLE_EQ(ref.A(), 20.0);
  EXPECT_DOUBLE_EQ(ref.Z(), 10.0);
  EXPECT_DOUBLE_EQ(ref.N(), 10.0);
  EXPECT_EQ(ref.target_mass(), 20);
  EXPECT_EQ(ref.target_Z(), 10);
  // occmax = 1/3 < 0.99: e_fermi is the shell of the last hole.
  EXPECT_DOUBLE_EQ(ref.e_fermi(kProton), 2.0);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kNeutron), 2.0);
}

TEST(Reference, VacuumHasNoHoles) {
  const auto basis = MakeBasis(2);
  const Reference ref = Reference::FromNucleus(basis, Nucleus::Parse("vacuum"));
  EXPECT_TRUE(ref.holes().empty());
  EXPECT_EQ(Ints(ref.particles()), Range(0, 11));
  EXPECT_DOUBLE_EQ(ref.A(), 0.0);
  EXPECT_DOUBLE_EQ(ref.Z(), 0.0);
  EXPECT_EQ(ref.target_mass(), 0);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kProton), -1.0);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kNeutron), -1.0);
}

TEST(Reference, EnsembleMix) {
  // mix_A4.5_Z2.5: protons 2.5 = 0s1/2 (2) + 0.5 in 0p3/2 (occ 0.125); neutrons
  // 2 = 0s1/2 full. Protons: occmax 0.125, last hole shell 1 -> e_fermi 1.
  // Neutrons: occmax 1, last hole shell 0, lowest neutron particle shell 1 ->
  // 0.5.
  const auto basis = MakeBasis(2);
  const Reference ref =
      Reference::FromNucleus(basis, Nucleus::Parse("mix_A4.5_Z2.5"));
  EXPECT_DOUBLE_EQ(ref.A(), 4.5);
  EXPECT_DOUBLE_EQ(ref.Z(), 2.5);
  EXPECT_DOUBLE_EQ(ref.N(), 2.0);
  EXPECT_EQ(ref.target_mass(), 4);
  EXPECT_EQ(ref.target_Z(), 2);
  EXPECT_EQ(Ints(ref.holes()), (std::vector<int>{0, 1, 2}));
  EXPECT_DOUBLE_EQ(ref.occupation(OrbitIndex{2}), 0.125);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kProton), 1.0);
  EXPECT_DOUBLE_EQ(ref.e_fermi(kNeutron), 0.5);
}

TEST(Reference, Emax4) {
  const auto basis = MakeBasis(4);
  const Reference o16 = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  EXPECT_EQ(Ints(o16.holes()), Range(0, 5));
  EXPECT_EQ(o16.particles().size(),
            static_cast<std::size_t>(basis.num_orbits() - 6));
  EXPECT_DOUBLE_EQ(o16.e_fermi(kProton), 1.5);
  EXPECT_DOUBLE_EQ(o16.A(), 16.0);

  const Reference ca48 = Reference::FromNucleus(basis, Nucleus::Parse("Ca48"));
  EXPECT_DOUBLE_EQ(ca48.A(), 48.0);
  EXPECT_DOUBLE_EQ(ca48.Z(), 20.0);

  const Reference vac = Reference::FromNucleus(basis, Nucleus::Parse("vacuum"));
  EXPECT_TRUE(vac.holes().empty());
  EXPECT_DOUBLE_EQ(vac.e_fermi(kNeutron), -1.0);

  const Reference he4 = Reference::FromNucleus(basis, Nucleus::Parse("He4"));
  EXPECT_DOUBLE_EQ(he4.e_fermi(kProton), 0.5);
  const Reference ne20 = Reference::FromNucleus(basis, Nucleus::Parse("Ne20"));
  EXPECT_DOUBLE_EQ(ne20.e_fermi(kNeutron), 2.0);
  EXPECT_EQ(ne20.occupation(OrbitIndex{6}), 2.0 / 6.0);
}

TEST(Reference, FromOccupationsErrors) {
  const auto basis = MakeBasis(2);
  EXPECT_THROW(Reference::FromOccupations(basis, std::vector<double>(11, 0.0)),
               InputError);
  EXPECT_THROW(Reference::FromOccupations(basis, std::vector<double>(13, 0.0)),
               InputError);
  std::vector<double> occ(12, 0.0);
  occ[3] = 1.5;
  EXPECT_THROW(Reference::FromOccupations(basis, occ), InputError);
  occ[3] = -0.1;
  EXPECT_THROW(Reference::FromOccupations(basis, occ), InputError);
  occ[3] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Reference::FromOccupations(basis, occ), InputError);
  occ[3] = 1.0;
  const Reference ok = Reference::FromOccupations(basis, occ);
  EXPECT_EQ(ok.name(), "custom");
  EXPECT_EQ(Ints(ok.holes()), (std::vector<int>{3}));
}

TEST(Reference, TinyOccupationIsParticle) {
  const auto basis = MakeBasis(2);
  std::vector<double> occ(12, 0.0);
  occ[0] = 1.0;
  occ[1] = 0.9e-6;  // below OCC_CUT
  const Reference ref = Reference::FromOccupations(basis, occ, "x");
  EXPECT_EQ(Ints(ref.holes()), (std::vector<int>{0}));
  EXPECT_EQ(ref.name(), "x");
}

TEST(Reference, FromNucleusErrors) {
  // lmax = 0 removes p orbits: O16 cannot be filled.
  const auto s_only = SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{2, 0, -1, -1}, 16.0);
  EXPECT_THROW(Reference::FromNucleus(s_only, Nucleus::Parse("O16")),
               InputError);
  // Ca48 does not fit into emax 2.
  EXPECT_THROW(Reference::FromNucleus(MakeBasis(2), Nucleus::Parse("Ca48")),
               InputError);
}

TEST(Reference, FromOrbitOccupations) {
  const auto basis = MakeBasis(2);
  const Orbit p0s1 = ParseOrbitLabel("p0s1");
  const Orbit n0p3 = ParseOrbitLabel("n0p3");
  const std::vector<OrbitOccupation> entries = {
      {p0s1, 0.5}, {n0p3, 1.0e-9}, {ParseOrbitLabel("n0s1"), 1.0}, {p0s1, 1.0}};
  const Reference ref = Reference::FromOrbitOccupations(basis, entries);
  EXPECT_EQ(ref.name(), "custom");
  EXPECT_EQ(Ints(ref.holes()),
            (std::vector<int>{0, 1}));            // |occ| < 1e-8 ignored
  EXPECT_EQ(ref.occupation(OrbitIndex{0}), 1.0);  // later entry overrides
  EXPECT_DOUBLE_EQ(ref.A(), 4.0);

  const Reference named =
      Reference::FromOrbitOccupations(basis, entries, "mine");
  EXPECT_EQ(named.name(), "mine");

  // Orbit outside the basis.
  const std::vector<OrbitOccupation> missing = {{ParseOrbitLabel("p0f7"), 1.0}};
  EXPECT_THROW(Reference::FromOrbitOccupations(basis, missing), InputError);
  // Occupation out of range.
  const std::vector<OrbitOccupation> bad = {{p0s1, 1.5}};
  EXPECT_THROW(Reference::FromOrbitOccupations(basis, bad), InputError);
}

TEST(Reference, EFermiBadSpecies) {
  const auto basis = MakeBasis(2);
  const Reference ref = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  EXPECT_THROW(ref.e_fermi(TwoTz{0}), InputError);
}

TEST(Reference, Equality) {
  const auto basis = MakeBasis(2);
  const Reference a = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  const Reference b = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  const Reference c = Reference::FromNucleus(basis, Nucleus::Parse("He4"));
  EXPECT_EQ(a, b);
  EXPECT_FALSE(a == c);
}

}  // namespace
}  // namespace imsrg::basis
