// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of the operator-name grammar and catalog (design/16 section 6.3 item
// 9).
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/driver/operators/operator_request.h"

namespace imsrg::driver {
namespace {

const ModelSpaceInfo kInfo{4};

RawValue Str(std::string s) {
  return RawValue{RawValue::Data{std::move(s)}, SourceRef{}};
}
RawValue Num(double d) { return RawValue{RawValue::Data{d}, SourceRef{}}; }
RawValue Integer(std::int64_t i) {
  return RawValue{RawValue::Data{i}, SourceRef{}};
}
RawValue Table(RawTable t) {
  return RawValue{RawValue::Data{std::move(t)}, SourceRef{}};
}

/// Parses one string; the result must be a single request.
OperatorRequest ParseOne(const std::string& spec) {
  auto r = ParseOperatorSpec(spec, kInfo);
  EXPECT_TRUE(r.has_value()) << spec;
  if (!r.has_value()) return {};
  EXPECT_EQ(r.value().size(), 1u) << spec;
  return r.value().front();
}

ConfigIssues ErrorsOf(const std::string& spec) {
  auto r = ParseOperatorSpec(spec, kInfo);
  EXPECT_FALSE(r.has_value()) << spec;
  return r.has_value() ? ConfigIssues{} : r.error();
}

struct P1Case {
  const char* spec;
  OperatorKind kind;
  std::vector<OpParam> params;
  const char* canonical;
};

TEST(OperatorRequest, EveryP1RowParsesAndRoundTrips) {
  const std::vector<P1Case> cases = {
      {"Trel", OperatorKind::kTrel, {}, "Trel"},
      {"TCM", OperatorKind::kTcm, {}, "TCM"},
      {"Tlab", OperatorKind::kTlab, {}, "Tlab"},
      {"TrelMassCorrection",
       OperatorKind::kTrelMassCorrection,
       {},
       "TrelMassCorrection"},
      {"HCM", OperatorKind::kHcm, {}, "HCM"},
      {"HCM_20", OperatorKind::kHcm, {20.0}, "HCM_20"},
      {"HCM_16.5", OperatorKind::kHcm, {16.5}, "HCM_16.5"},
      {"VCM_20", OperatorKind::kVcm, {20.0}, "VCM_20"},
      {"R2CM", OperatorKind::kR2cm, {}, "R2CM"},
      {"Rp2", OperatorKind::kRp2, {}, "Rp2"},
      {"Rn2", OperatorKind::kRn2, {}, "Rn2"},
      {"Rm2", OperatorKind::kRm2, {}, "Rm2"},
      {"Rm2lab", OperatorKind::kRm2lab, {}, "Rm2lab"},
      {"Rp2Z_8", OperatorKind::kRp2Z, {std::int64_t{8}}, "Rp2Z_8"},
      {"Rp2AZ_16_8",
       OperatorKind::kRp2Az,
       {std::int64_t{16}, std::int64_t{8}},
       "Rp2AZ_16_8"},
      {"Rn2Z_8", OperatorKind::kRn2Z, {std::int64_t{8}}, "Rn2Z_8"},
      {"R2_p1", OperatorKind::kR2p1, {}, "R2_p1"},
      {"R2_n1", OperatorKind::kR2n1, {}, "R2_n1"},
      {"R2_p2", OperatorKind::kR2p2, {}, "R2_p2"},
      {"R2_n2", OperatorKind::kR2n2, {}, "R2_n2"},
      {"Rso", OperatorKind::kRso, {}, "Rso"},
      {"E0", OperatorKind::kE0, {}, "E0"},
      {"OccRef", OperatorKind::kOccRef, {}, "OccRef"},
      {"OneOcc_p0s1",
       OperatorKind::kOneOcc,
       {std::string("p0s1")},
       "OneOcc_p0s1"},
      {"AllOcc_p0d5",
       OperatorKind::kAllOcc,
       {std::string("p0d5")},
       "AllOcc_p0d5"},
      {"OBD_p0s1_n0p3",
       OperatorKind::kObd,
       {std::string("p0s1"), std::string("n0p3")},
       "OBD_p0s1_n0p3"},
      {"rhop_3.0", OperatorKind::kRhoP, {3.0}, "rhop_3"},
      {"rhop_0.2", OperatorKind::kRhoP, {0.2}, "rhop_0.2"},
      {"rhon_1.5", OperatorKind::kRhoN, {1.5}, "rhon_1.5"},
      {"rhocentralp_2.5", OperatorKind::kRhoCentralP, {2.5}, "rhocentralp_2.5"},
      {"rhocentraln_2", OperatorKind::kRhoCentralN, {2.0}, "rhocentraln_2"},
      {"Iso2", OperatorKind::kIso2, {}, "Iso2"},
      {"Tz2", OperatorKind::kTz2, {}, "Tz2"},
      {"LdotS", OperatorKind::kLdotS, {}, "LdotS"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.spec);
    const auto req = ParseOne(c.spec);
    EXPECT_EQ(req.kind, c.kind);
    EXPECT_EQ(req.params, c.params);
    EXPECT_EQ(req.canonical_name, c.canonical);
    EXPECT_EQ(req.phase, Phase::kP1);
    EXPECT_EQ(req.shape.rank_j, 0);
    EXPECT_EQ(req.shape.dtz, 0);
    EXPECT_EQ(req.shape.parity, 0);
    // The canonical name re-parses to an equal request.
    EXPECT_EQ(ParseOne(req.canonical_name), req);
  }
}

TEST(OperatorRequest, OrbitIsStoredCanonically) {
  const auto req = ParseOne("OneOcc_p0s1");
  ASSERT_EQ(req.params.size(), 1u);
  EXPECT_EQ(std::get<std::string>(req.params[0]), "p0s1");
}

TEST(OperatorRequest, InlineTableEqualsStringForm) {
  const auto from_string = ParseOne("rhop_3.0");
  EXPECT_EQ(from_string, ParseOne("rhop_3"));
  const RawValue specs[] = {Table({{"name", Str("rhop")}, {"r", Num(3.0)}})};
  auto r = ParseOperatorList(specs, kInfo, "operators.list");
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r.value().size(), 1u);
  EXPECT_EQ(r.value()[0], from_string);

  // Integers are accepted for numbers.
  const RawValue ints[] = {Table({{"name", Str("rhop")}, {"r", Integer(3)}})};
  auto ri = ParseOperatorList(ints, kInfo, "operators.list");
  ASSERT_TRUE(ri.has_value());
  EXPECT_EQ(ri.value()[0], from_string);

  // Orbit and integer parameters, and an omitted optional parameter.
  const RawValue more[] = {
      Table({{"name", Str("OBD")},
             {"orbit1", Str("p0s1")},
             {"orbit2", Str("n0p3")}}),
      Table({{"name", Str("Rp2AZ")}, {"a", Integer(16)}, {"z", Integer(8)}}),
      Table({{"name", Str("HCM")}}),
  };
  auto rm = ParseOperatorList(more, kInfo, "operators.list");
  ASSERT_TRUE(rm.has_value());
  ASSERT_EQ(rm.value().size(), 3u);
  EXPECT_EQ(rm.value()[0], ParseOne("OBD_p0s1_n0p3"));
  EXPECT_EQ(rm.value()[1], ParseOne("Rp2AZ_16_8"));
  EXPECT_EQ(rm.value()[2], ParseOne("HCM"));
}

TEST(OperatorRequest, InlineTableErrors) {
  const auto code_of = [](RawTable t) {
    const RawValue specs[] = {Table(std::move(t))};
    auto r = ParseOperatorList(specs, kInfo, "operators.list");
    EXPECT_FALSE(r.has_value());
    return r.has_value() || r.error().empty() ? std::string()
                                              : r.error().front().code;
  };
  EXPECT_EQ(code_of({{"name", Str("rhop")}}), "E-OP-PARAMS");  // missing
  EXPECT_EQ(code_of({{"name", Str("rhop")}, {"r", Num(1.0)}, {"x", Num(1.0)}}),
            "E-OP-PARAMS");  // extra
  EXPECT_EQ(code_of({{"name", Str("rhop")}, {"r", Str("a")}}),
            "E-OP-PARAMS");  // type
  EXPECT_EQ(code_of({{"name", Str("rhop")}, {"r", Num(-1.0)}}),
            "E-OP-PARAMS");                              // sign
  EXPECT_EQ(code_of({{"r", Num(1.0)}}), "E-OP-PARAMS");  // no name
  EXPECT_EQ(code_of({{"name", Str("Rp3")}}), "E-OP-UNKNOWN");
  EXPECT_EQ(code_of({{"name", Str("E2")}}), "E-PLANNED");
  EXPECT_EQ(code_of({{"name", Str("OneOcc")}, {"orbit", Str("p9s1")}}),
            "E-OP-ORBIT");
  EXPECT_EQ(code_of({{"name", Str("Rp2Z")}, {"z", Num(8.5)}}), "E-OP-PARAMS");
}

TEST(OperatorRequest, WrongArityAndType) {
  const auto issues = ErrorsOf("rhop");
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-OP-PARAMS");
  EXPECT_NE(issues[0].message.find(
                "'rhop' expects 1 parameter (r: number >= 0), got 0"),
            std::string::npos)
      << issues[0].message;

  const auto typed = ErrorsOf("rhop_x");
  ASSERT_EQ(typed.size(), 1u);
  EXPECT_EQ(typed[0].code, "E-OP-PARAMS");
  EXPECT_NE(typed[0].message.find("'rhop_x': 'x' is not a number"),
            std::string::npos)
      << typed[0].message;

  EXPECT_EQ(ErrorsOf("rhop_1_2")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("rhop_-1")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("rhop_nan")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("rhop_1.5x")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("Rp2Z_8.0")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("Rp2Z")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("Rp2AZ_16")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("Trel_3")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("HCM_1_2")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("OBD_p0s1")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("")[0].code, "E-OP-PARAMS");
}

TEST(OperatorRequest, OrbitErrors) {
  // Outside emax = 4: p0s1 has e = 0, p3s1 has e = 6.
  const auto out = ErrorsOf("OneOcc_p3s1");
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].code, "E-OP-ORBIT");
  // Malformed label: reported as E-OP-PARAMS (a syntax error, not a model-space
  // error).
  EXPECT_EQ(ErrorsOf("OneOcc_xyz")[0].code, "E-OP-PARAMS");
  EXPECT_EQ(ErrorsOf("OBD_p0s1_p3s1")[0].code, "E-OP-ORBIT");
  // Inside emax.
  EXPECT_TRUE(ParseOperatorSpec("OneOcc_p2s1", kInfo).has_value());
}

TEST(OperatorRequest, UnknownNamesGetSuggestions) {
  const auto issues = ErrorsOf("Rp3");
  ASSERT_EQ(issues.size(), 1u);
  EXPECT_EQ(issues[0].code, "E-OP-UNKNOWN");
  EXPECT_NE(issues[0].message.find("operator 'Rp3' is not known"),
            std::string::npos);
  ASSERT_TRUE(issues[0].suggestion.has_value());
  EXPECT_EQ(*issues[0].suggestion, "Rp2");

  const auto far = ErrorsOf("completely_different");
  ASSERT_EQ(far.size(), 1u);
  EXPECT_EQ(far[0].code, "E-OP-UNKNOWN");
  EXPECT_FALSE(far[0].suggestion.has_value());

  // Case sensitive.
  EXPECT_EQ(ErrorsOf("trel")[0].code, "E-OP-UNKNOWN");
}

TEST(OperatorRequest, MultiTokenNamesAreNotParameters) {
  EXPECT_EQ(ParseOne("R2_p1").kind, OperatorKind::kR2p1);
  EXPECT_EQ(ParseOne("R2_n2").kind, OperatorKind::kR2n2);
  // "rhop_all" is a macro name, not rhop with parameter "all".
  const auto all = ParseOperatorSpec("rhop_all", kInfo);
  ASSERT_TRUE(all.has_value());
  EXPECT_EQ(all.value().size(), 51u);
  // Sigma_p is a planned multi-token name.
  EXPECT_EQ(ErrorsOf("Sigma_p")[0].code, "E-PLANNED");
  EXPECT_EQ(ErrorsOf("R2_p3")[0].code, "E-OP-UNKNOWN");
}

TEST(OperatorRequest, RhoMacros) {
  for (const char* name : {"rhop_all", "rhon_all"}) {
    SCOPED_TRACE(name);
    auto r = ParseOperatorSpec(name, kInfo);
    ASSERT_TRUE(r.has_value());
    const auto& v = r.value();
    ASSERT_EQ(v.size(), 51u);
    const bool proton = std::string(name) == "rhop_all";
    EXPECT_EQ(v.front().canonical_name, proton ? "rhop_0" : "rhon_0");
    EXPECT_EQ(v.back().canonical_name, proton ? "rhop_10" : "rhon_10");
    EXPECT_EQ(v.front().kind,
              proton ? OperatorKind::kRhoP : OperatorKind::kRhoN);
    std::set<std::string> names;
    for (const auto& q : v) names.insert(q.canonical_name);
    EXPECT_EQ(names.size(), 51u);
    EXPECT_TRUE(names.count(proton ? "rhop_0.6" : "rhon_0.6"));
    // Each expanded request re-parses to itself.
    for (const auto& q : v) EXPECT_EQ(ParseOne(q.canonical_name), q);
  }
}

TEST(OperatorRequest, Duplicates) {
  const RawValue dup[] = {Str("Trel"), Str("Rp2"), Str("Trel")};
  auto r = ParseOperatorList(dup, kInfo, "operators.list");
  ASSERT_FALSE(r.has_value());
  ASSERT_EQ(r.error().size(), 1u);
  EXPECT_EQ(r.error()[0].code, "E-OP-DUPLICATE");

  // Spelling variants with the same canonical name collide.
  const RawValue variants[] = {Str("rhop_3"), Str("rhop_3.0")};
  auto v = ParseOperatorList(variants, kInfo, "operators.list");
  ASSERT_FALSE(v.has_value());
  EXPECT_EQ(v.error()[0].code, "E-OP-DUPLICATE");

  // A macro collides with an explicit element it contains.
  const RawValue macro[] = {Str("rhop_0.6"), Str("rhop_all")};
  auto m = ParseOperatorList(macro, kInfo, "operators.list");
  ASSERT_FALSE(m.has_value());
  EXPECT_EQ(m.error()[0].code, "E-OP-DUPLICATE");
}

TEST(OperatorRequest, PlannedAndNotSupported) {
  std::size_t planned = 0;
  for (const auto& e : OperatorCatalog()) {
    if (e.phase == Phase::kPlanned) {
      ++planned;
      const std::string name =
          e.name == "QdotQ" ? "QdotQ_2" : std::string(e.name);
      SCOPED_TRACE(name);
      const auto issues = ErrorsOf(name);
      ASSERT_EQ(issues.size(), 1u);
      EXPECT_EQ(issues[0].code, "E-PLANNED");
    }
  }
  EXPECT_GT(planned, 60u);
  const auto e2 = ErrorsOf("E2");
  EXPECT_NE(e2[0].message.find("operator 'E2' is not available in this build"),
            std::string::npos);
  // Parameters on planned operators are accepted at this stage.
  EXPECT_EQ(ErrorsOf("M0nu_GT_1_2")[0].code, "E-PLANNED");
  EXPECT_EQ(ErrorsOf("QdotQ_whatever")[0].code, "E-PLANNED");
  EXPECT_EQ(ErrorsOf("hfsNMS")[0].code, "E-OP-NS");
  EXPECT_EQ(ErrorsOf("hfsSMS")[0].code, "E-OP-NS");
}

TEST(OperatorRequest, PlannedShapes) {
  const auto find = [](std::string_view name) {
    for (const auto& e : OperatorCatalog()) {
      if (e.name == name) return e;
    }
    ADD_FAILURE() << name;
    return OperatorCatalogEntry{};
  };
  EXPECT_EQ(find("E2").shape.rank_j, 2);
  EXPECT_EQ(find("E2").shape.parity, 0);
  EXPECT_EQ(find("E3").shape.parity, 1);
  EXPECT_EQ(find("M1").shape.rank_j, 1);
  EXPECT_EQ(find("M1").shape.parity, 0);
  EXPECT_EQ(find("M2").shape.parity, 1);
  EXPECT_NE(find("Fermi").shape.dtz, 0);
  EXPECT_NE(find("GamowTeller").shape.dtz, 0);
  EXPECT_NE(find("M0nu_GT").shape.dtz, 0);
}

TEST(OperatorRequest, ErrorsAccumulateOverAList) {
  const RawValue specs[] = {Str("Trel"), Str("Rp3"),    Str("rhop_x"),
                            Str("E2"),   Str("hfsNMS"), Str("Rp2"),
                            Integer(7),  Str("Trel"),   Str("")};
  auto r = ParseOperatorList(specs, kInfo, "operators.list");
  ASSERT_FALSE(r.has_value());
  std::vector<std::string> codes;
  for (const auto& i : r.error()) {
    codes.push_back(i.code);
    EXPECT_EQ(i.key_path, "operators.list");
  }
  const std::vector<std::string> expected = {
      "E-OP-UNKNOWN", "E-OP-PARAMS",    "E-PLANNED",  "E-OP-NS",
      "E-OP-PARAMS",  "E-OP-DUPLICATE", "E-OP-PARAMS"};
  EXPECT_EQ(codes, expected);
}

TEST(OperatorRequest, EmptyListIsEmptyResult) {
  auto r = ParseOperatorList({}, kInfo, "operators.list");
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r.value().empty());
}

TEST(OperatorRequest, ListPreservesOrder) {
  const RawValue specs[] = {Str("Rp2"), Str("Trel"), Str("rhop_1")};
  auto r = ParseOperatorList(specs, kInfo, "operators.list");
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r.value().size(), 3u);
  EXPECT_EQ(r.value()[0].canonical_name, "Rp2");
  EXPECT_EQ(r.value()[1].canonical_name, "Trel");
  EXPECT_EQ(r.value()[2].canonical_name, "rhop_1");
}

TEST(OperatorCatalog, Sanity) {
  const auto cat = OperatorCatalog();
  ASSERT_FALSE(cat.empty());
  std::set<std::string_view> names;
  bool saw_p1 = false, saw_planned = false, saw_ns = false;
  for (const auto& e : cat) {
    EXPECT_TRUE(names.insert(e.name).second)
        << "duplicate catalog name " << e.name;
    EXPECT_GE(e.min_params, 0);
    EXPECT_LE(static_cast<std::size_t>(e.min_params), e.params.size())
        << e.name;
    switch (e.phase) {
      case Phase::kP1:
        saw_p1 = true;
        EXPECT_TRUE(e.kind.has_value()) << e.name;
        EXPECT_FALSE(e.variadic) << e.name;
        break;
      case Phase::kPlanned:
        saw_planned = true;
        EXPECT_FALSE(e.kind.has_value()) << e.name;
        EXPECT_TRUE(e.variadic) << e.name;
        break;
      case Phase::kNotSupported:
        saw_ns = true;
        EXPECT_FALSE(e.kind.has_value()) << e.name;
        break;
    }
    EXPECT_FALSE(e.doc.empty()) << e.name;
  }
  EXPECT_TRUE(saw_p1 && saw_planned && saw_ns);
}

TEST(OperatorCatalog, TextListsEveryRowInGroups) {
  const auto text = OperatorCatalogText();
  const auto p1 = text.find("P1");
  const auto planned = text.find("planned");
  const auto ns = text.find("not supported");
  ASSERT_NE(p1, std::string::npos);
  ASSERT_NE(planned, std::string::npos);
  ASSERT_NE(ns, std::string::npos);
  EXPECT_LT(p1, planned);
  EXPECT_LT(planned, ns);
  for (const auto& e : OperatorCatalog()) {
    EXPECT_NE(text.find(std::string(e.name)), std::string::npos) << e.name;
  }
  EXPECT_NE(text.find("rhop_<r>"), std::string::npos);
  EXPECT_NE(text.find("Rp2AZ_<a>_<z>"), std::string::npos);
  EXPECT_NE(text.find("J=0 dTz=0 P=+ rank=1"), std::string::npos);
}

}  // namespace
}  // namespace imsrg::driver
