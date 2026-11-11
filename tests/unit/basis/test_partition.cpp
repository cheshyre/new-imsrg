#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/partition.hpp"
#include "imsrg/basis/reference.hpp"
#include "imsrg/basis/valence_space.hpp"

namespace imsrg::basis {
namespace {

using V = std::vector<int>;

V ints(std::span<const OrbitIndex> v) {
  V out;
  for (const OrbitIndex i : v) out.push_back(i.value());
  return out;
}

V range(int lo, int hi) {  // [lo, hi]; empty if hi < lo
  V out;
  for (int i = lo; i <= hi; ++i) out.push_back(i);
  return out;
}

struct Golden {
  const char* reference;  // "default" = default_reference_nucleus(space)
  const char* space;
  double A, Z, A_core, Z_core;
  V holes, particles, core, valence, qspace;
};

class PartitionGolden : public ::testing::TestWithParam<Golden> {};

TEST_P(PartitionGolden, Emax2) {
  const Golden& g = GetParam();
  const auto basis = SingleParticleBasis::harmonic_oscillator(BasisTruncation{2, -1, -1, -1}, 16.0);
  const Nucleus refnuc = std::string(g.reference) == "default" ? default_reference_nucleus(g.space)
                                                               : Nucleus::parse(g.reference);
  const Reference ref = Reference::from_nucleus(basis, refnuc);
  const Partition part =
      Partition::from_definition(basis, define_valence_space(g.space, basis, refnuc));
  EXPECT_DOUBLE_EQ(ref.A(), g.A);
  EXPECT_DOUBLE_EQ(ref.Z(), g.Z);
  EXPECT_DOUBLE_EQ(part.A_core(), g.A_core);
  EXPECT_DOUBLE_EQ(part.Z_core(), g.Z_core);
  EXPECT_EQ(ints(ref.holes()), g.holes);
  EXPECT_EQ(ints(ref.particles()), g.particles);
  EXPECT_EQ(ints(part.core()), g.core);
  EXPECT_EQ(ints(part.valence()), g.valence);
  EXPECT_EQ(ints(part.qspace()), g.qspace);
  EXPECT_EQ(part.is_single_reference(), g.valence.empty());
}

INSTANTIATE_TEST_SUITE_P(
    Setups, PartitionGolden,
    ::testing::Values(
        Golden{"He4", "He4", 4, 2, 4, 2, range(0, 1), range(2, 11), range(0, 1), {}, range(2, 11)},
        Golden{
            "O16", "O16", 16, 8, 16, 8, range(0, 5), range(6, 11), range(0, 5), {}, range(6, 11)},
        Golden{"Ne20",
               "Ne20",
               20,
               10,
               28,
               14,
               range(0, 7),
               range(8, 11),
               range(0, 7),
               {},
               range(8, 11)},
        Golden{"Ne20",
               "sd-shell",
               20,
               10,
               16,
               8,
               range(0, 7),
               range(8, 11),
               range(0, 5),
               range(6, 11),
               {}},
        Golden{"default", "p-shell", 4, 2, 4, 2, range(0, 1), range(2, 11), range(0, 1),
               range(2, 5), range(6, 11)},
        Golden{"default",
               "sd-shell",
               16,
               8,
               16,
               8,
               range(0, 5),
               range(6, 11),
               range(0, 5),
               range(6, 11),
               {}}));

SingleParticleBasis basis2() {
  return SingleParticleBasis::harmonic_oscillator(BasisTruncation{2, -1, -1, -1}, 16.0);
}

TEST(Partition, FromClassesAndAccessors) {
  const auto basis = basis2();
  std::vector<OrbitClass> cls(12, OrbitClass::QSpace);
  cls[0] = OrbitClass::Core;
  cls[1] = OrbitClass::Core;
  cls[6] = OrbitClass::Valence;
  const Partition p = Partition::from_classes(basis, cls);
  EXPECT_EQ(p.num_orbits(), 12);
  EXPECT_EQ(p.orbit_class(OrbitIndex{0}), OrbitClass::Core);
  EXPECT_EQ(p.orbit_class(OrbitIndex{6}), OrbitClass::Valence);
  EXPECT_EQ(p.orbit_class(OrbitIndex{7}), OrbitClass::QSpace);
  EXPECT_DOUBLE_EQ(p.A_core(), 4.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 2.0);
  EXPECT_THROW(Partition::from_classes(basis, std::vector<OrbitClass>(11, OrbitClass::Core)),
               InputError);
  EXPECT_EQ(orbit_class_name(OrbitClass::Core), "core");
  EXPECT_EQ(orbit_class_name(OrbitClass::Valence), "valence");
  EXPECT_EQ(orbit_class_name(OrbitClass::QSpace), "qspace");
}

TEST(Partition, CoreWinsOnOverlap) {
  const auto basis = basis2();
  ValenceSpaceDefinition def;
  def.core = {parse_orbit_label("p0s1"), parse_orbit_label("p0p3")};
  def.valence = {parse_orbit_label("p0p3"), parse_orbit_label("p0p1")};
  const Partition p = Partition::from_definition(basis, def);
  EXPECT_EQ(ints(p.core()), (V{0, 2}));
  EXPECT_EQ(ints(p.valence()), (V{4}));
  EXPECT_DOUBLE_EQ(p.A_core(), 6.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 6.0);
}

TEST(Partition, ValenceOrbitMissingFromBasisIsDropped) {
  // lmax = 0 keeps only the s orbits: p0s1, n0s1, p1s1, n1s1.
  const auto basis = SingleParticleBasis::harmonic_oscillator(BasisTruncation{2, 0, -1, -1}, 16.0);
  ValenceSpaceDefinition def;
  def.core = {parse_orbit_label("p0s1"), parse_orbit_label("n0s1")};
  def.valence = {parse_orbit_label("p0d5"), parse_orbit_label("p1s1")};
  const Partition p = Partition::from_definition(basis, def);
  ASSERT_EQ(p.num_orbits(), 4);
  EXPECT_EQ(p.core().size(), 2u);
  ASSERT_EQ(p.valence().size(), 1u);
  EXPECT_EQ(basis.orbit(p.valence()[0]), parse_orbit_label("p1s1"));
  EXPECT_EQ(p.qspace().size(), 1u);
}

TEST(Partition, CoreOrbitMissingFromBasisThrows) {
  const auto basis = SingleParticleBasis::harmonic_oscillator(BasisTruncation{2, 0, -1, -1}, 16.0);
  ValenceSpaceDefinition def;
  def.core = {parse_orbit_label("p0p3")};
  EXPECT_THROW(Partition::from_definition(basis, def), InputError);
}

TEST(Partition, WithHolesAsCore) {
  const auto basis = basis2();
  // Single-reference He4 partition, new reference O16: the holes become the core.
  const Reference he4 = Reference::from_nucleus(basis, Nucleus::parse("He4"));
  const Reference o16 = Reference::from_nucleus(basis, Nucleus::parse("O16"));
  const Partition prev =
      Partition::from_definition(basis, define_valence_space("He4", basis, Nucleus::parse("He4")));
  const Partition p = Partition::with_holes_as_core(prev, o16, basis);
  EXPECT_EQ(ints(p.core()), range(0, 5));
  EXPECT_TRUE(p.valence().empty());
  EXPECT_EQ(ints(p.qspace()), range(6, 11));
  EXPECT_DOUBLE_EQ(p.A_core(), 16.0);
  EXPECT_DOUBLE_EQ(p.Z_core(), 8.0);

  // Valence orbits of the previous partition stay valence.
  const Reference ne20 = Reference::from_nucleus(basis, Nucleus::parse("Ne20"));
  const Partition sd = Partition::from_definition(
      basis, define_valence_space("sd-shell", basis, Nucleus::parse("O16")));
  const Partition q = Partition::with_holes_as_core(sd, ne20, basis);
  EXPECT_EQ(ints(q.core()), range(0, 5));
  EXPECT_EQ(ints(q.valence()), range(6, 11));
  EXPECT_TRUE(q.qspace().empty());

  // Size mismatch.
  const auto basis4 =
      SingleParticleBasis::harmonic_oscillator(BasisTruncation{4, -1, -1, -1}, 16.0);
  EXPECT_THROW(Partition::with_holes_as_core(
                   prev, Reference::from_nucleus(basis4, Nucleus::parse("He4")), basis4),
               InputError);
  (void)he4;
}

TEST(Partition, Equality) {
  const auto basis = basis2();
  const auto def = define_valence_space("sd-shell", basis, Nucleus::parse("O16"));
  EXPECT_EQ(Partition::from_definition(basis, def), Partition::from_definition(basis, def));
  const auto def2 = define_valence_space("O16", basis, Nucleus::parse("O16"));
  EXPECT_FALSE(Partition::from_definition(basis, def) == Partition::from_definition(basis, def2));
}

}  // namespace
}  // namespace imsrg::basis
