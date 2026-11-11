// The single schema table and the enum spellings (design/16 sections 2 and 6.1; 6.3 item 7 for
// the enum round trip).

#include <algorithm>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/driver/config/schema.hpp"
#include "imsrg/driver/config/schema_keys.hpp"

namespace imsrg::driver {
namespace {

template <ConfigEnum E>
void expect_enum_round_trip() {
  std::set<std::string_view> seen;
  for (const auto& entry : EnumTraits<E>::entries) {
    EXPECT_EQ(to_string(entry.value), entry.name);
    const auto parsed = enum_from_string<E>(entry.name);
    ASSERT_TRUE(parsed.has_value()) << entry.name;
    EXPECT_EQ(*parsed, entry.value);
    EXPECT_EQ(enum_phase(entry.value), entry.phase);
    EXPECT_TRUE(seen.insert(entry.name).second) << "duplicate spelling " << entry.name;
    // Canonical spellings are lower snake_case.
    EXPECT_TRUE(std::all_of(entry.name.begin(), entry.name.end(), [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    })) << entry.name;
  }
  EXPECT_FALSE(enum_from_string<E>("no_such_value").has_value());
  EXPECT_EQ(enum_names<E>().size(), EnumTraits<E>::entries.size());
}

TEST(DriverSchema, EveryEnumValueRoundTripsThroughItsString) {
  expect_enum_round_trip<Task>();
  expect_enum_round_trip<LogLevel>();
  expect_enum_round_trip<BasisKind>();
  expect_enum_round_trip<KineticEnergy>();
  expect_enum_round_trip<TwoBodyFormat>();
  expect_enum_round_trip<ThreeBodyType>();
  expect_enum_round_trip<ThreeBodyFormat>();
  expect_enum_round_trip<No2bPrecision>();
  expect_enum_round_trip<NatOrder>();
  expect_enum_round_trip<Method>();
  expect_enum_round_trip<Generator>();
  expect_enum_round_trip<Partitioning>();
  expect_enum_round_trip<OmegaStorage>();
  expect_enum_round_trip<Imsrg3Variant>();
  expect_enum_round_trip<ValenceFileFormat>();
  expect_enum_round_trip<OperatorFileFormat>();
}

TEST(DriverSchema, PhasesFollowDesign16) {
  EXPECT_EQ(enum_phase(Method::Magnus), Phase::P1);
  EXPECT_EQ(enum_phase(Method::Flow), Phase::P1);
  EXPECT_EQ(enum_phase(Method::FlowRk4), Phase::P1);
  EXPECT_EQ(enum_phase(Method::MagnusBackoff), Phase::Planned);
  EXPECT_EQ(enum_phase(Generator::ShellModelImaginaryTime), Phase::P1);
  EXPECT_EQ(enum_phase(Generator::ShellModelWegner), Phase::Planned);
  EXPECT_EQ(enum_phase(Generator::OnePa), Phase::Planned);
  EXPECT_EQ(enum_phase(TwoBodyFormat::Me2j), Phase::P1);
  EXPECT_EQ(enum_phase(TwoBodyFormat::Navratil), Phase::Planned);
  EXPECT_EQ(enum_phase(Task::FciExport), Phase::Planned);
  EXPECT_EQ(enum_phase(ValenceFileFormat::Antoine), Phase::Planned);
}

TEST(DriverSchema, KeysAreUniqueSnakeCaseAndFindable) {
  const auto keys = schema_keys();
  ASSERT_GT(keys.size(), 80u);
  std::set<std::string_view> paths;
  for (const KeyInfo& key : keys) {
    EXPECT_TRUE(paths.insert(key.path).second) << "duplicate key " << key.path;
    EXPECT_TRUE(std::all_of(key.path.begin(), key.path.end(), [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.';
    })) << key.path;
    EXPECT_FALSE(key.doc.empty()) << key.path;
    EXPECT_FALSE(key.default_text.empty()) << key.path;
    const KeyInfo* found = find_schema_key(key.path);
    ASSERT_NE(found, nullptr) << key.path;
    EXPECT_EQ(found->path, key.path);
  }
  EXPECT_EQ(find_schema_key("basis.emaxx"), nullptr);
  EXPECT_EQ(find_schema_key("basis"), nullptr);
  ASSERT_NE(find_schema_key("interaction.three_body.e3max"), nullptr);
  EXPECT_TRUE(find_schema_key("interaction.three_body.e3max")->required);
  EXPECT_TRUE(find_schema_key("interaction.two_body.file")->string_or_array);
  EXPECT_EQ(find_schema_key("imsrg.goose_tank")->phase, Phase::Planned);
  EXPECT_TRUE(find_schema_key("output.stem")->derived);
}

TEST(DriverSchema, TablesAndChildren) {
  const auto tables = schema_tables();
  const std::vector<std::string_view> expected = {"run",
                                                  "basis",
                                                  "reference",
                                                  "valence",
                                                  "hamiltonian",
                                                  "interaction",
                                                  "interaction.two_body",
                                                  "interaction.three_body",
                                                  "hartree_fock",
                                                  "natural_orbitals",
                                                  "imsrg",
                                                  "imsrg.generator",
                                                  "imsrg.denominator",
                                                  "imsrg.step",
                                                  "imsrg.omega",
                                                  "imsrg.space",
                                                  "imsrg3",
                                                  "operators",
                                                  "output"};
  EXPECT_EQ(std::vector<std::string_view>(tables.begin(), tables.end()), expected);

  const auto imsrg = schema_children("imsrg");
  for (std::string_view child : {"method", "steps", "generator", "step", "space"}) {
    EXPECT_NE(std::find(imsrg.begin(), imsrg.end(), child), imsrg.end()) << child;
  }
  const auto top = schema_children("");
  EXPECT_NE(std::find(top.begin(), top.end(), "interaction"), top.end());
  EXPECT_EQ(std::find(top.begin(), top.end(), "two_body"), top.end());
}

TEST(DriverSchema, RunConfigEqualityIgnoresProvenance) {
  RunConfig a;
  RunConfig b;
  b.provenance.command_line = "imsrg emax=4";
  EXPECT_EQ(a, b);
  b.imsrg.step.smax = 100.0;
  EXPECT_NE(a, b);
}

TEST(DriverSchema, DidYouMean) {
  const std::vector<std::string> keys = {"emax", "e3max", "hw", "lmax"};
  EXPECT_EQ(did_you_mean("emaxx", keys), "emax");
  EXPECT_EQ(did_you_mean("hw", keys), std::nullopt);  // exact match is never suggested
  EXPECT_EQ(did_you_mean("xyzzy", keys), std::nullopt);
  const std::vector<std::string> methods = {"magnus", "magnus_euler", "flow"};
  EXPECT_EQ(did_you_mean("magnus_eular", methods), "magnus_euler");
  EXPECT_EQ(edit_distance("ab", "ba"), 1);
  EXPECT_EQ(edit_distance("", "abc"), 3);
}

}  // namespace
}  // namespace imsrg::driver
