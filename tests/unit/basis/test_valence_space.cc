// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/valence_space.h"

namespace imsrg::basis {
namespace {

using Labels = std::set<std::string>;

Labels LabelsOf(const std::vector<Orbit>& v) {
  Labels out;
  for (const Orbit& o : v) out.insert(OrbitLabel(o));
  return out;
}

SingleParticleBasis MakeBasis(int emax) {
  return SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{emax, -1, -1, -1}, 16.0);
}

ValenceSpaceDefinition Define(std::string_view spec, int emax,
                              std::string_view ref) {
  return DefineValenceSpace(spec, MakeBasis(emax), Nucleus::Parse(ref));
}

TEST(ValenceSpaceTable, SizeSortedAndParsable) {
  const auto table = NamedValenceSpaces();
  EXPECT_EQ(table.size(), 39u);
  EXPECT_TRUE(std::is_sorted(
      table.begin(), table.end(),
      [](const NamedValenceSpace& a, const NamedValenceSpace& b) {
        return a.name < b.name;
      }));
  for (const NamedValenceSpace& s : table) {
    for (const std::string_view l : s.orbits) {
      EXPECT_NO_THROW((void)ParseOrbitLabel(l)) << s.name << " " << l;
    }
    EXPECT_NO_THROW((void)Nucleus::Parse(s.core)) << s.name;
  }
}

TEST(ValenceSpaceTable, Find) {
  const NamedValenceSpace* sd = FindNamedValenceSpace("sd-shell");
  ASSERT_NE(sd, nullptr);
  EXPECT_EQ(sd->core, "O16");
  EXPECT_EQ(sd->orbits.size(), 6u);
  EXPECT_EQ(FindNamedValenceSpace("nonsense"), nullptr);
  ASSERT_NE(FindNamedValenceSpace("p-shell"), nullptr);
  EXPECT_EQ(FindNamedValenceSpace("p-shell")->core, "He4");
}

TEST(ValenceSpace, SdShell) {
  const auto def = Define("sd-shell", 2, "O16");
  EXPECT_EQ(def.core_name, "O16");
  EXPECT_EQ(LabelsOf(def.core),
            (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"}));
  EXPECT_EQ(LabelsOf(def.valence),
            (Labels{"p0d5", "n0d5", "p0d3", "n0d3", "p1s1", "n1s1"}));
  EXPECT_FALSE(def.is_single_reference());
}

TEST(ValenceSpace, PShell) {
  const auto def = Define("p-shell", 2, "He4");
  EXPECT_EQ(def.core_name, "He4");
  EXPECT_EQ(LabelsOf(def.core), (Labels{"p0s1", "n0s1"}));
  EXPECT_EQ(LabelsOf(def.valence), (Labels{"p0p3", "n0p3", "p0p1", "n0p1"}));
}

TEST(ValenceSpace, PsdShell) {
  const auto def = Define("psd-shell", 2, "He4");
  EXPECT_EQ(LabelsOf(def.core), (Labels{"p0s1", "n0s1"}));
  EXPECT_EQ(def.valence.size(), 10u);
  EXPECT_EQ(LabelsOf(def.valence).count("p0p3"), 1u);
  EXPECT_EQ(LabelsOf(def.valence).count("n1s1"), 1u);
}

TEST(ValenceSpace, CommaFormEqualsNamed) {
  const auto comma = Define("O16,p0d5,n0d5,p0d3,n0d3,p1s1,n1s1", 2, "O16");
  const auto named = Define("sd-shell", 2, "O16");
  EXPECT_EQ(comma.core_name, "O16");
  EXPECT_EQ(LabelsOf(comma.core), LabelsOf(named.core));
  EXPECT_EQ(LabelsOf(comma.valence), LabelsOf(named.valence));
}

TEST(ValenceSpace, CommaFormSkipsValenceInCore) {
  // The core filling skips p0p1/n0p1: s1/2, p3/2 and a partial d5/2 (2 of 6)
  // per species.
  const auto def = Define("O16,p0p1,n0p1", 2, "O16");
  EXPECT_EQ(LabelsOf(def.valence), (Labels{"p0p1", "n0p1"}));
  EXPECT_EQ(LabelsOf(def.core),
            (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0d5", "n0d5"}));
}

TEST(ValenceSpace, NamedDoesNotSkip) {
  // For p-shell the core is He4 and the valence orbits are not part of it
  // anyway; use a space whose valence overlaps the core filling: sdfp-shell on
  // O16 core has no overlap either, so check the defining property directly:
  // core = filling of the core nucleus.
  const auto def = Define("sd-shell", 4, "O16");
  EXPECT_EQ(def.core.size(), 6u);
}

TEST(ValenceSpace, ZeroHwShellC20) {
  const auto def = Define("0hw-shell", 2, "C20");  // Z=6, N=14
  EXPECT_EQ(LabelsOf(def.core), (Labels{"p0s1", "n0s1", "n0p3", "n0p1"}));
  EXPECT_EQ(LabelsOf(def.valence),
            (Labels{"p0p3", "p0p1", "n0d5", "n0d3", "n1s1"}));
  EXPECT_EQ(def.core_name, "He10");  // core Z=2, N=8
}

TEST(ValenceSpace, ZeroHwShellNe20) {
  const auto def = Define("0hw-shell", 2, "Ne20");
  EXPECT_EQ(def.core_name, "O16");
  EXPECT_EQ(LabelsOf(def.core),
            (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"}));
  EXPECT_EQ(LabelsOf(def.valence),
            (Labels{"p0d5", "p0d3", "p1s1", "n0d5", "n0d3", "n1s1"}));
}

TEST(ValenceSpace, ZeroHwShellClosedShellHasNoValence) {
  const auto def = Define("0hw-shell", 2, "O16");
  EXPECT_TRUE(def.valence.empty());
  EXPECT_EQ(def.core.size(), 6u);
}

TEST(ValenceSpace, JjShellNe20) {
  // Z = N = 10: largest jj closure <= 10 is 6, so the core is C12 and the
  // valence orbits per species are the largest-j intruder of the next shell
  // (0d5/2) and 0p1/2.
  const auto def = Define("jj-shell", 2, "Ne20");
  EXPECT_EQ(def.core_name, "C12");
  EXPECT_EQ(LabelsOf(def.core), (Labels{"p0s1", "n0s1", "p0p3", "n0p3"}));
  EXPECT_EQ(LabelsOf(def.valence), (Labels{"p0d5", "p0p1", "n0d5", "n0p1"}));
}

TEST(ValenceSpace, JjShellOutOfRange) {
  // Z = 1 is below the first closure.
  EXPECT_THROW(Define("jj-shell", 2, "H2"), InputError);
}

TEST(ValenceSpace, Fci) {
  const auto basis = MakeBasis(2);
  const auto def = DefineValenceSpace("FCI", basis, Nucleus::Parse("He4"));
  EXPECT_TRUE(def.core.empty());
  EXPECT_TRUE(def.core_name.empty());
  EXPECT_EQ(def.valence.size(), static_cast<std::size_t>(basis.num_orbits()));
}

TEST(ValenceSpace, PlainNucleusName) {
  const auto def = Define("Ca48", 4, "Ca48");
  EXPECT_TRUE(def.valence.empty());
  EXPECT_TRUE(def.is_single_reference());
  EXPECT_EQ(def.core_name, "Ca48");
  EXPECT_FALSE(def.core.empty());
}

TEST(ValenceSpace, Errors) {
  EXPECT_THROW(Define("O16,q0d5", 2, "O16"), InputError);
  EXPECT_THROW(Define("Xx99", 2, "O16"), InputError);
  EXPECT_THROW(Define("Ca48", 2, "Ca48"), InputError);  // does not fit emax 2
}

TEST(DefaultReference, Cases) {
  const Nucleus sd = DefaultReferenceNucleus("sd-shell");
  EXPECT_DOUBLE_EQ(sd.A, 16.0);
  EXPECT_DOUBLE_EQ(sd.Z, 8.0);
  const Nucleus comma = DefaultReferenceNucleus("O16,p0d5,n0d5");
  EXPECT_DOUBLE_EQ(comma.A, 16.0);
  EXPECT_DOUBLE_EQ(comma.Z, 8.0);
  const Nucleus ca = DefaultReferenceNucleus("Ca48");
  EXPECT_DOUBLE_EQ(ca.A, 48.0);
  EXPECT_DOUBLE_EQ(ca.Z, 20.0);
}

TEST(DefaultReference, Throws) {
  EXPECT_THROW(DefaultReferenceNucleus("0hw-shell"), InputError);
  EXPECT_THROW(DefaultReferenceNucleus("jj-shell"), InputError);
  EXPECT_THROW(DefaultReferenceNucleus("FCI"), InputError);
  EXPECT_THROW(DefaultReferenceNucleus("Xx99"), InputError);
}

}  // namespace
}  // namespace imsrg::basis
