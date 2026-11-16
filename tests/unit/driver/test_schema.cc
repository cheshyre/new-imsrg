// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// The single schema table and the enum spellings (design/16 sections 2 and 6.1;
// 6.3 item 7 for the enum round trip).

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/config/schema.h"
#include "imsrg/driver/config/schema_keys.h"

namespace imsrg::driver {
namespace {

template <class E>
  requires detail::ConfigEnum<E>
void ExpectEnumRoundTrip() {
  std::set<std::string_view> seen;
  for (const auto& entry : EnumTraits<E>::entries) {
    EXPECT_EQ(ToString(entry.value), entry.name);
    const auto parsed = EnumFromString<E>(entry.name);
    ASSERT_TRUE(parsed.has_value()) << entry.name;
    EXPECT_EQ(*parsed, entry.value);
    EXPECT_EQ(EnumPhase(entry.value), entry.phase);
    EXPECT_TRUE(seen.insert(entry.name).second)
        << "duplicate spelling " << entry.name;
    // Canonical spellings are lower snake_case.
    EXPECT_TRUE(std::all_of(entry.name.begin(), entry.name.end(), [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    })) << entry.name;
  }
  EXPECT_FALSE(EnumFromString<E>("no_such_value").has_value());
  EXPECT_EQ(EnumNames<E>().size(), EnumTraits<E>::entries.size());
}

TEST(DriverSchema, EveryEnumValueRoundTripsThroughItsString) {
  ExpectEnumRoundTrip<Task>();
  ExpectEnumRoundTrip<LogLevel>();
  ExpectEnumRoundTrip<BasisKind>();
  ExpectEnumRoundTrip<KineticEnergy>();
  ExpectEnumRoundTrip<TwoBodyFormat>();
  ExpectEnumRoundTrip<ThreeBodyType>();
  ExpectEnumRoundTrip<ThreeBodyFormat>();
  ExpectEnumRoundTrip<No2bPrecision>();
  ExpectEnumRoundTrip<NatOrder>();
  ExpectEnumRoundTrip<Method>();
  ExpectEnumRoundTrip<Generator>();
  ExpectEnumRoundTrip<Partitioning>();
  ExpectEnumRoundTrip<OmegaStorage>();
  ExpectEnumRoundTrip<Imsrg3Variant>();
  ExpectEnumRoundTrip<ValenceFileFormat>();
  ExpectEnumRoundTrip<OperatorFileFormat>();
}

TEST(DriverSchema, PhasesFollowDesign16) {
  EXPECT_EQ(EnumPhase(Method::kMagnus), Phase::kP1);
  EXPECT_EQ(EnumPhase(Method::kFlow), Phase::kP1);
  EXPECT_EQ(EnumPhase(Method::kFlowRk4), Phase::kP1);
  EXPECT_EQ(EnumPhase(Method::kMagnusBackoff), Phase::kPlanned);
  EXPECT_EQ(EnumPhase(Generator::kShellModelImaginaryTime), Phase::kP1);
  EXPECT_EQ(EnumPhase(Generator::kShellModelWegner), Phase::kPlanned);
  EXPECT_EQ(EnumPhase(Generator::kOnePa), Phase::kPlanned);
  EXPECT_EQ(EnumPhase(TwoBodyFormat::kMe2j), Phase::kP1);
  EXPECT_EQ(EnumPhase(TwoBodyFormat::kNavratil), Phase::kPlanned);
  EXPECT_EQ(EnumPhase(Task::kFciExport), Phase::kPlanned);
  EXPECT_EQ(EnumPhase(ValenceFileFormat::kAntoine), Phase::kPlanned);
}

TEST(DriverSchema, KeysAreUniqueSnakeCaseAndFindable) {
  const auto keys = SchemaKeys();
  ASSERT_GT(keys.size(), 80u);
  std::set<std::string_view> paths;
  for (const KeyInfo& key : keys) {
    EXPECT_TRUE(paths.insert(key.path).second) << "duplicate key " << key.path;
    EXPECT_TRUE(std::all_of(key.path.begin(), key.path.end(), [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
             c == '.';
    })) << key.path;
    EXPECT_FALSE(key.doc.empty()) << key.path;
    EXPECT_FALSE(key.default_text.empty()) << key.path;
    const KeyInfo* found = FindSchemaKey(key.path);
    ASSERT_NE(found, nullptr) << key.path;
    EXPECT_EQ(found->path, key.path);
  }
  EXPECT_EQ(FindSchemaKey("basis.emaxx"), nullptr);
  EXPECT_EQ(FindSchemaKey("basis"), nullptr);
  ASSERT_NE(FindSchemaKey("interaction.three_body.e3max"), nullptr);
  EXPECT_TRUE(FindSchemaKey("interaction.three_body.e3max")->required);
  EXPECT_TRUE(FindSchemaKey("interaction.two_body.file")->string_or_array);
  EXPECT_EQ(FindSchemaKey("imsrg.goose_tank")->phase, Phase::kPlanned);
  EXPECT_TRUE(FindSchemaKey("output.stem")->derived);
}

TEST(DriverSchema, TablesAndChildren) {
  const auto tables = SchemaTables();
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
  EXPECT_EQ(std::vector<std::string_view>(tables.begin(), tables.end()),
            expected);

  const auto imsrg = SchemaChildren("imsrg");
  for (std::string_view child :
       {"method", "steps", "generator", "step", "space"}) {
    EXPECT_NE(std::find(imsrg.begin(), imsrg.end(), child), imsrg.end())
        << child;
  }
  const auto top = SchemaChildren("");
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
  EXPECT_EQ(DidYouMean("emaxx", keys), "emax");
  EXPECT_EQ(DidYouMean("hw", keys),
            std::nullopt);  // exact match is never suggested
  EXPECT_EQ(DidYouMean("xyzzy", keys), std::nullopt);
  const std::vector<std::string> methods = {"magnus", "magnus_euler", "flow"};
  EXPECT_EQ(DidYouMean("magnus_eular", methods), "magnus_euler");
  EXPECT_EQ(EditDistance("ab", "ba"), 1);
  EXPECT_EQ(EditDistance("", "abc"), 3);
}

}  // namespace
}  // namespace imsrg::driver
