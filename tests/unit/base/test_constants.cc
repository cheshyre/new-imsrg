// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/constants.h"

namespace {

namespace c = imsrg::constants;

// Reference name -> our value, for the constants written as a plain numeric
// literal.
const std::map<std::string, double>& LiteralTable() {
  static const auto* const kT = new std::map<std::string, double>{
      {"HBARC", c::kHbarc},
      {"M_PROTON", c::kMProton},
      {"M_NEUTRON", c::kMNeutron},
      {"M_ELECTRON", c::kMElectron},
      {"M_PION_CHARGED", c::kMPionCharged},
      {"M_PION_NEUTRAL", c::kMPionNeutral},
      {"NUCLEON_VECTOR_G", c::kNucleonVectorG},
      {"NUCLEON_AXIAL_G", c::kNucleonAxialG},
      {"PROTON_SPIN_G", c::kProtonSpinG},
      {"NEUTRON_SPIN_G", c::kNeutronSpinG},
      {"ELECTRON_SPIN_G", c::kElectronSpinG},
      {"F_PI", c::kFPi},
      {"HARTREE", c::kHartree},
      {"PROTON_RCH2", c::kProtonRch2},
      {"NEUTRON_RCH2", c::kNeutronRch2},
      {"DARWIN_FOLDY", c::kDarwinFoldy},
      {"SQRT2", c::kSqrt2},
      {"INVSQRT2", c::kInvSqrt2},
      {"LOG2", c::kLn2},
      {"PI", c::kPi},
      {"SQRTPI", c::kSqrtPi},
  };
  return *kT;
}

TEST(Constants, EqualReferenceLiterals) {
  EXPECT_EQ(c::kHbarc, 197.3269718);
  EXPECT_EQ(c::kMProton, 938.2720813);
  EXPECT_EQ(c::kMNeutron, 939.5654133);
  EXPECT_EQ(c::kMElectron, 0.5109989461);
  EXPECT_EQ(c::kMNucleon, (938.2720813 + 939.5654133) / 2.0);
  EXPECT_EQ(c::kMPionCharged, 139.57018);
  EXPECT_EQ(c::kMPionNeutral, 134.9766);
  EXPECT_EQ(c::kNucleonVectorG, 1.00);
  EXPECT_EQ(c::kNucleonAxialG, 1.27);
  EXPECT_EQ(c::kProtonSpinG, 5.585690569);
  EXPECT_EQ(c::kNeutronSpinG, -3.826085);
  EXPECT_EQ(c::kElectronSpinG, 2.002319);
  EXPECT_EQ(c::kAlphaFs, 1.0 / 137.035999);
  EXPECT_EQ(c::kFPi, 92.2);
  EXPECT_EQ(c::kHartree, 27.21138602);
  EXPECT_EQ(c::kProtonRch2, 0.707);
  EXPECT_EQ(c::kNeutronRch2, -0.1161);
  EXPECT_EQ(c::kDarwinFoldy, 0.033172);
  EXPECT_EQ(c::kSqrt2, 1.41421356237309505);
  EXPECT_EQ(c::kInvSqrt2, 0.7071067811865);
  EXPECT_EQ(c::kLn2, 0.69314718055994530942);
  EXPECT_EQ(c::kPi, 3.14159265358979324);
  EXPECT_EQ(c::kSqrtPi, 1.7724538509055160);
}

TEST(Constants, InvSqrt2IsTruncatedReferenceValue) {
  EXPECT_NE(c::kInvSqrt2, 1.0 / c::kSqrt2);
  EXPECT_NEAR(c::kInvSqrt2, 1.0 / c::kSqrt2, 1.0e-12);
}

TEST(Constants, MatchReferenceFile) {
  std::ifstream in(IMSRG_REFERENCE_PHYSICAL_CONSTANTS);
  if (!in) GTEST_SKIP() << "reference file not available";
  const std::regex line_re(
      R"(^const double\s+([A-Z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*;)");
  std::size_t matched = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::smatch m;
    if (!std::regex_search(line, m, line_re)) continue;
    const auto it = LiteralTable().find(m[1].str());
    ASSERT_NE(it, LiteralTable().end())
        << "constant not covered: " << m[1].str();
    const double ref = std::strtod(m[2].str().c_str(), nullptr);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(it->second),
              std::bit_cast<std::uint64_t>(ref))
        << m[1].str();
    ++matched;
  }
  // Every plain-literal constant of the reference was found (M_NUCLEON and
  // ALPHA_FS are expressions and covered above).
  EXPECT_EQ(matched, LiteralTable().size());
}

}  // namespace
