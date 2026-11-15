// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/partition.h"
#include "imsrg/basis/reference.h"
#include "imsrg/basis/valence_space.h"

namespace imsrg::basis {
namespace {

using V = std::vector<int>;

V Ints(std::span<const OrbitIndex> v) {
  V out;
  for (const OrbitIndex i : v) out.push_back(i.value());
  return out;
}

V Range(int lo, int hi) {  // [lo, hi]; empty if hi < lo
  V out;
  for (int i = lo; i <= hi; ++i) out.push_back(i);
  return out;
}

struct Golden {
  const char* reference;  // "default" = DefaultReferenceNucleus(space)
  const char* space;
  double A, Z, A_core, Z_core;
  V holes, particles, core, valence, qspace;
};

class PartitionGolden : public ::testing::TestWithParam<Golden> {};

TEST_P(PartitionGolden, Emax2) {
  const Golden& g = GetParam();
  const auto basis = SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{2, -1, -1, -1}, 16.0);
  const Nucleus refnuc = std::string(g.reference) == "default"
                             ? DefaultReferenceNucleus(g.space)
                             : Nucleus::Parse(g.reference);
  const Reference ref = Reference::FromNucleus(basis, refnuc);
  const Partition part = Partition::FromDefinition(
      basis, DefineValenceSpace(g.space, basis, refnuc));
  EXPECT_DOUBLE_EQ(ref.A(), g.A);
  EXPECT_DOUBLE_EQ(ref.Z(), g.Z);
  EXPECT_DOUBLE_EQ(part.A_core(), g.A_core);
  EXPECT_DOUBLE_EQ(part.Z_core(), g.Z_core);
  EXPECT_EQ(Ints(ref.holes()), g.holes);
  EXPECT_EQ(Ints(ref.particles()), g.particles);
  EXPECT_EQ(Ints(part.core()), g.core);
  EXPECT_EQ(Ints(part.valence()), g.valence);
  EXPECT_EQ(Ints(part.qspace()), g.qspace);
  EXPECT_EQ(part.is_single_reference(), g.valence.empty());
}

INSTANTIATE_TEST_SUITE_P(Setups, PartitionGolden,
                         ::testing::Values(Golden{"He4",
                                                  "He4",
                                                  4,
                                                  2,
                                                  4,
                                                  2,
                                                  Range(0, 1),
                                                  Range(2, 11),
                                                  Range(0, 1),
                                                  {},
                                                  Range(2, 11)},
                                           Golden{"O16",
                                                  "O16",
                                                  16,
                                                  8,
                                                  16,
                                                  8,
                                                  Range(0, 5),
                                                  Range(6, 11),
                                                  Range(0, 5),
                                                  {},
                                                  Range(6, 11)},
                                           Golden{"Ne20",
                                                  "Ne20",
                                                  20,
                                                  10,
                                                  28,
                                                  14,
                                                  Range(0, 7),
                                                  Range(8, 11),
                                                  Range(0, 7),
                                                  {},
                                                  Range(8, 11)},
                                           Golden{"Ne20",
                                                  "sd-shell",
                                                  20,
                                                  10,
                                                  16,
                                                  8,
                                                  Range(0, 7),
                                                  Range(8, 11),
                                                  Range(0, 5),
                                                  Range(6, 11),
                                                  {}},
                                           Golden{"default", "p-shell", 4, 2, 4,
                                                  2, Range(0, 1), Range(2, 11),
                                                  Range(0, 1), Range(2, 5),
                                                  Range(6, 11)},
                                           Golden{"default",
                                                  "sd-shell",
                                                  16,
                                                  8,
                                                  16,
                                                  8,
                                                  Range(0, 5),
                                                  Range(6, 11),
                                                  Range(0, 5),
                                                  Range(6, 11),
                                                  {}}));

SingleParticleBasis Basis2() {
  return SingleParticleBasis::HarmonicOscillator(BasisTruncation{2, -1, -1, -1},
                                                 16.0);
}

TEST(Partition, FromClassesAndAccessors) {
  const auto basis = Basis2();
  std::vector<OrbitClass> cls(12, OrbitClass::kQSpace);
  cls[0] = OrbitClass::kCore;
  cls[1] = OrbitClass::kCore;
  cls[6] = OrbitClass::kValence;
  const Partition p = Partition::FromClasses(basis, cls);
  EXPECT_EQ(p.num_orbits(), 12);
  EXPECT_EQ(p.orbit_class(OrbitIndex{0}), OrbitClass::kCore);
  EXPECT_EQ(p.orbit_class(OrbitIndex{6}), OrbitClass::kValence);
  EXPECT_EQ(p.orbit_class(OrbitIndex{7}), OrbitClass::kQSpace);
  EXPECT_DOUBLE_EQ(p.A_core(), 4.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 2.0);
  EXPECT_THROW(Partition::FromClasses(
                   basis, std::vector<OrbitClass>(11, OrbitClass::kCore)),
               InputError);
  EXPECT_EQ(OrbitClassName(OrbitClass::kCore), "core");
  EXPECT_EQ(OrbitClassName(OrbitClass::kValence), "valence");
  EXPECT_EQ(OrbitClassName(OrbitClass::kQSpace), "qspace");
}

TEST(Partition, CoreWinsOnOverlap) {
  const auto basis = Basis2();
  ValenceSpaceDefinition def;
  def.core = {ParseOrbitLabel("p0s1"), ParseOrbitLabel("p0p3")};
  def.valence = {ParseOrbitLabel("p0p3"), ParseOrbitLabel("p0p1")};
  const Partition p = Partition::FromDefinition(basis, def);
  EXPECT_EQ(Ints(p.core()), (V{0, 2}));
  EXPECT_EQ(Ints(p.valence()), (V{4}));
  EXPECT_DOUBLE_EQ(p.A_core(), 6.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 6.0);
}

TEST(Partition, ValenceOrbitMissingFromBasisIsDropped) {
  // lmax = 0 keeps only the s orbits: p0s1, n0s1, p1s1, n1s1.
  const auto basis = SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{2, 0, -1, -1}, 16.0);
  ValenceSpaceDefinition def;
  def.core = {ParseOrbitLabel("p0s1"), ParseOrbitLabel("n0s1")};
  def.valence = {ParseOrbitLabel("p0d5"), ParseOrbitLabel("p1s1")};
  const Partition p = Partition::FromDefinition(basis, def);
  ASSERT_EQ(p.num_orbits(), 4);
  EXPECT_EQ(p.core().size(), 2u);
  ASSERT_EQ(p.valence().size(), 1u);
  EXPECT_EQ(basis.orbit(p.valence()[0]), ParseOrbitLabel("p1s1"));
  EXPECT_EQ(p.qspace().size(), 1u);
}

TEST(Partition, CoreOrbitMissingFromBasisThrows) {
  const auto basis = SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{2, 0, -1, -1}, 16.0);
  ValenceSpaceDefinition def;
  def.core = {ParseOrbitLabel("p0p3")};
  EXPECT_THROW(Partition::FromDefinition(basis, def), InputError);
}

TEST(Partition, WithHolesAsCore) {
  const auto basis = Basis2();
  // Single-reference He4 partition, new reference O16: the holes become the
  // core.
  const Reference he4 = Reference::FromNucleus(basis, Nucleus::Parse("He4"));
  const Reference o16 = Reference::FromNucleus(basis, Nucleus::Parse("O16"));
  const Partition prev = Partition::FromDefinition(
      basis, DefineValenceSpace("He4", basis, Nucleus::Parse("He4")));
  const Partition p = Partition::WithHolesAsCore(prev, o16, basis);
  EXPECT_EQ(Ints(p.core()), Range(0, 5));
  EXPECT_TRUE(p.valence().empty());
  EXPECT_EQ(Ints(p.qspace()), Range(6, 11));
  EXPECT_DOUBLE_EQ(p.A_core(), 16.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 8.0);

  // Valence orbits of the previous partition stay valence.
  const Reference ne20 = Reference::FromNucleus(basis, Nucleus::Parse("Ne20"));
  const Partition sd = Partition::FromDefinition(
      basis, DefineValenceSpace("sd-shell", basis, Nucleus::Parse("O16")));
  const Partition q = Partition::WithHolesAsCore(sd, ne20, basis);
  EXPECT_EQ(Ints(q.core()), Range(0, 5));
  EXPECT_EQ(Ints(q.valence()), Range(6, 11));
  EXPECT_TRUE(q.qspace().empty());

  // Size mismatch.
  const auto basis4 = SingleParticleBasis::HarmonicOscillator(
      BasisTruncation{4, -1, -1, -1}, 16.0);
  EXPECT_THROW(
      Partition::WithHolesAsCore(
          prev, Reference::FromNucleus(basis4, Nucleus::Parse("He4")), basis4),
      InputError);
  (void)he4;
}

TEST(Partition, Equality) {
  const auto basis = Basis2();
  const auto def = DefineValenceSpace("sd-shell", basis, Nucleus::Parse("O16"));
  EXPECT_EQ(Partition::FromDefinition(basis, def),
            Partition::FromDefinition(basis, def));
  const auto def2 = DefineValenceSpace("O16", basis, Nucleus::Parse("O16"));
  EXPECT_FALSE(Partition::FromDefinition(basis, def) ==
               Partition::FromDefinition(basis, def2));
}

}  // namespace
}  // namespace imsrg::basis
