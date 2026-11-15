// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Norms of every golden OperatorDump against the imsrg-devel norms recorded
// next to it (design/30 section 2: norms on fixed inputs agree to 1e-10
// relative; the actual deviations are printed). The dumps are read with our
// reader, so this also checks that reading loses nothing.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/operator/norms.h"
#include "imsrg/operator/operator_dump.h"

namespace fs = std::filesystem;
using namespace imsrg;

namespace {

struct GoldenNorms {
  double zero_body = 0;
  double one_body = 0;
  double two_body = 0;
  double three_body = 0;
  double norm = 0;
};

std::string Slurp(const fs::path& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Every norms object written by the harness (`norms_json`) has the same five
// fields in the same order, so a regex over the flat JSON text is enough.
std::map<std::string, GoldenNorms> ParseNorms(const std::string& text) {
  static const std::regex kRe(
      R"re("([A-Za-z0-9_]+)":\s*\{\s*"zero_body":\s*([^,\s]+),\s*"one_body_norm":\s*([^,\s]+),)re"
      R"re(\s*"two_body_norm":\s*([^,\s]+),\s*"three_body_norm":\s*([^,\s]+),\s*"norm":\s*([^,\s}]+))re");
  std::map<std::string, GoldenNorms> out;
  for (std::sregex_iterator it(text.begin(), text.end(), kRe), end; it != end;
       ++it) {
    const auto& m = *it;
    out[m[1].str()] = {std::stod(m[2].str()), std::stod(m[3].str()),
                       std::stod(m[4].str()), std::stod(m[5].str()),
                       std::stod(m[6].str())};
  }
  return out;
}

struct Case {
  fs::path dump;
  GoldenNorms golden;
};

std::vector<Case> GoldenCases() {
  const fs::path root = IMSRG_GOLDEN_DIR;
  std::vector<Case> cases;
  for (const char* dir :
       {"hamiltonian/emax2_O16", "commutator/emax2_O16", "operator/emax2"}) {
    const auto norms = ParseNorms(Slurp(root / dir / "norms.json"));
    for (const auto& e : fs::directory_iterator(root / dir)) {
      if (e.path().extension() != ".dump") continue;
      const auto it = norms.find(e.path().stem().string());
      if (it != norms.end()) cases.push_back({e.path(), it->second});
    }
  }
  for (const char* dir : {"hf/emax2_O16_2N", "hf/emax2_O16_2N3N"}) {
    const auto norms = ParseNorms(Slurp(root / dir / "hf.json"));
    cases.push_back({root / dir / "HNO.dump", norms.at("HNO_norms")});
  }
  std::sort(cases.begin(), cases.end(),
            [](const Case& a, const Case& b) { return a.dump < b.dump; });
  return cases;
}

double RelDev(double ours, double ref) {
  if (ref == 0.0) return std::abs(ours);
  return std::abs(ours - ref) / std::abs(ref);
}

}  // namespace

TEST(OperatorGoldenNorms, EveryGoldenDumpMatchesImsrgDevelNorms) {
  const auto cases = GoldenCases();
  ASSERT_GE(cases.size(), 35u)
      << "golden dumps with recorded norms are missing";
  double worst = 0.0;
  std::string worst_name;
  for (const auto& c : cases) {
    SCOPED_TRACE(c.dump.string());
    const auto header = op::ReadOperatorDumpHeader(c.dump);
    const auto op = op::ReadOperatorDump(c.dump, op::ModelSpaceForDump(header));
    const double d0 = RelDev(op.zero_body(), c.golden.zero_body);
    const double d1 = RelDev(op::OneBodyNorm(op), c.golden.one_body);
    const double d2 = RelDev(op::TwoBodyNorm(op), c.golden.two_body);
    const double d3 = RelDev(op::ThreeBodyNorm(op), c.golden.three_body);
    const double dn = RelDev(op::Norm(op), c.golden.norm);
    EXPECT_LE(d0, 1.0e-10);
    EXPECT_LE(d1, 1.0e-10);
    EXPECT_LE(d2, 1.0e-10);
    EXPECT_LE(d3, 1.0e-10);
    EXPECT_LE(dn, 1.0e-10);
    const double d = std::max({d0, d1, d2, d3, dn});
    if (d >= worst) {
      worst = d;
      worst_name = c.dump.filename().string();
    }
  }
  std::printf("[golden norms] %zu dumps, max relative deviation %.3g (%s)\n",
              cases.size(), worst, worst_name.c_str());
}
