#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/valence_space.hpp"

namespace imsrg::basis {
namespace {

using Labels = std::set<std::string>;

Labels labels(const std::vector<Orbit>& v) {
  Labels out;
  for (const Orbit& o : v) out.insert(orbit_label(o));
  return out;
}

SingleParticleBasis make_basis(int emax) {
  return SingleParticleBasis::harmonic_oscillator(BasisTruncation{emax, -1, -1, -1}, 16.0);
}

ValenceSpaceDefinition define(std::string_view spec, int emax, std::string_view ref) {
  return define_valence_space(spec, make_basis(emax), Nucleus::parse(ref));
}

TEST(ValenceSpaceTable, SizeSortedAndParsable) {
  const auto table = named_valence_spaces();
  EXPECT_EQ(table.size(), 39u);
  EXPECT_TRUE(std::is_sorted(
      table.begin(), table.end(),
      [](const NamedValenceSpace& a, const NamedValenceSpace& b) { return a.name < b.name; }));
  for (const NamedValenceSpace& s : table) {
    for (const std::string_view l : s.orbits) {
      EXPECT_NO_THROW((void)parse_orbit_label(l)) << s.name << " " << l;
    }
    EXPECT_NO_THROW((void)Nucleus::parse(s.core)) << s.name;
  }
}

TEST(ValenceSpaceTable, Find) {
  const NamedValenceSpace* sd = find_named_valence_space("sd-shell");
  ASSERT_NE(sd, nullptr);
  EXPECT_EQ(sd->core, "O16");
  EXPECT_EQ(sd->orbits.size(), 6u);
  EXPECT_EQ(find_named_valence_space("nonsense"), nullptr);
  ASSERT_NE(find_named_valence_space("p-shell"), nullptr);
  EXPECT_EQ(find_named_valence_space("p-shell")->core, "He4");
}

TEST(ValenceSpace, SdShell) {
  const auto def = define("sd-shell", 2, "O16");
  EXPECT_EQ(def.core_name, "O16");
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"}));
  EXPECT_EQ(labels(def.valence), (Labels{"p0d5", "n0d5", "p0d3", "n0d3", "p1s1", "n1s1"}));
  EXPECT_FALSE(def.is_single_reference());
}

TEST(ValenceSpace, PShell) {
  const auto def = define("p-shell", 2, "He4");
  EXPECT_EQ(def.core_name, "He4");
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1"}));
  EXPECT_EQ(labels(def.valence), (Labels{"p0p3", "n0p3", "p0p1", "n0p1"}));
}

TEST(ValenceSpace, PsdShell) {
  const auto def = define("psd-shell", 2, "He4");
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1"}));
  EXPECT_EQ(def.valence.size(), 10u);
  EXPECT_EQ(labels(def.valence).count("p0p3"), 1u);
  EXPECT_EQ(labels(def.valence).count("n1s1"), 1u);
}

TEST(ValenceSpace, CommaFormEqualsNamed) {
  const auto comma = define("O16,p0d5,n0d5,p0d3,n0d3,p1s1,n1s1", 2, "O16");
  const auto named = define("sd-shell", 2, "O16");
  EXPECT_EQ(comma.core_name, "O16");
  EXPECT_EQ(labels(comma.core), labels(named.core));
  EXPECT_EQ(labels(comma.valence), labels(named.valence));
}

TEST(ValenceSpace, CommaFormSkipsValenceInCore) {
  // The core filling skips p0p1/n0p1: s1/2, p3/2 and a partial d5/2 (2 of 6) per species.
  const auto def = define("O16,p0p1,n0p1", 2, "O16");
  EXPECT_EQ(labels(def.valence), (Labels{"p0p1", "n0p1"}));
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0d5", "n0d5"}));
}

TEST(ValenceSpace, NamedDoesNotSkip) {
  // For p-shell the core is He4 and the valence orbits are not part of it anyway; use a space
  // whose valence overlaps the core filling: sdfp-shell on O16 core has no overlap either, so
  // check the defining property directly: core = filling of the core nucleus.
  const auto def = define("sd-shell", 4, "O16");
  EXPECT_EQ(def.core.size(), 6u);
}

TEST(ValenceSpace, ZeroHwShellC20) {
  const auto def = define("0hw-shell", 2, "C20");  // Z=6, N=14
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1", "n0p3", "n0p1"}));
  EXPECT_EQ(labels(def.valence), (Labels{"p0p3", "p0p1", "n0d5", "n0d3", "n1s1"}));
  EXPECT_EQ(def.core_name, "He10");  // core Z=2, N=8
}

TEST(ValenceSpace, ZeroHwShellNe20) {
  const auto def = define("0hw-shell", 2, "Ne20");
  EXPECT_EQ(def.core_name, "O16");
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1", "p0p3", "n0p3", "p0p1", "n0p1"}));
  EXPECT_EQ(labels(def.valence), (Labels{"p0d5", "p0d3", "p1s1", "n0d5", "n0d3", "n1s1"}));
}

TEST(ValenceSpace, ZeroHwShellClosedShellHasNoValence) {
  const auto def = define("0hw-shell", 2, "O16");
  EXPECT_TRUE(def.valence.empty());
  EXPECT_EQ(def.core.size(), 6u);
}

TEST(ValenceSpace, JjShellNe20) {
  // Z = N = 10: largest jj closure <= 10 is 6, so the core is C12 and the valence orbits per
  // species are the largest-j intruder of the next shell (0d5/2) and 0p1/2.
  const auto def = define("jj-shell", 2, "Ne20");
  EXPECT_EQ(def.core_name, "C12");
  EXPECT_EQ(labels(def.core), (Labels{"p0s1", "n0s1", "p0p3", "n0p3"}));
  EXPECT_EQ(labels(def.valence), (Labels{"p0d5", "p0p1", "n0d5", "n0p1"}));
}

TEST(ValenceSpace, JjShellOutOfRange) {
  // Z = 1 is below the first closure.
  EXPECT_THROW(define("jj-shell", 2, "H2"), InputError);
}

TEST(ValenceSpace, Fci) {
  const auto basis = make_basis(2);
  const auto def = define_valence_space("FCI", basis, Nucleus::parse("He4"));
  EXPECT_TRUE(def.core.empty());
  EXPECT_TRUE(def.core_name.empty());
  EXPECT_EQ(def.valence.size(), static_cast<std::size_t>(basis.num_orbits()));
}

TEST(ValenceSpace, PlainNucleusName) {
  const auto def = define("Ca48", 4, "Ca48");
  EXPECT_TRUE(def.valence.empty());
  EXPECT_TRUE(def.is_single_reference());
  EXPECT_EQ(def.core_name, "Ca48");
  EXPECT_FALSE(def.core.empty());
}

TEST(ValenceSpace, Errors) {
  EXPECT_THROW(define("O16,q0d5", 2, "O16"), InputError);
  EXPECT_THROW(define("Xx99", 2, "O16"), InputError);
  EXPECT_THROW(define("Ca48", 2, "Ca48"), InputError);  // does not fit emax 2
}

TEST(DefaultReference, Cases) {
  const Nucleus sd = default_reference_nucleus("sd-shell");
  EXPECT_DOUBLE_EQ(sd.A, 16.0);
  EXPECT_DOUBLE_EQ(sd.Z, 8.0);
  const Nucleus comma = default_reference_nucleus("O16,p0d5,n0d5");
  EXPECT_DOUBLE_EQ(comma.A, 16.0);
  EXPECT_DOUBLE_EQ(comma.Z, 8.0);
  const Nucleus ca = default_reference_nucleus("Ca48");
  EXPECT_DOUBLE_EQ(ca.A, 48.0);
  EXPECT_DOUBLE_EQ(ca.Z, 20.0);
}

TEST(DefaultReference, Throws) {
  EXPECT_THROW(default_reference_nucleus("0hw-shell"), InputError);
  EXPECT_THROW(default_reference_nucleus("jj-shell"), InputError);
  EXPECT_THROW(default_reference_nucleus("FCI"), InputError);
  EXPECT_THROW(default_reference_nucleus("Xx99"), InputError);
}

}  // namespace
}  // namespace imsrg::basis
