#include <bit>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <string>

#include <gtest/gtest.h>

#include "imsrg/base/constants.hpp"

namespace {

namespace c = imsrg::constants;

// Reference name -> our value, for the constants written as a plain numeric literal.
const std::map<std::string, double>& literal_table() {
  static const std::map<std::string, double> t = {
      {"HBARC", c::hbarc},
      {"M_PROTON", c::m_proton},
      {"M_NEUTRON", c::m_neutron},
      {"M_ELECTRON", c::m_electron},
      {"M_PION_CHARGED", c::m_pion_charged},
      {"M_PION_NEUTRAL", c::m_pion_neutral},
      {"NUCLEON_VECTOR_G", c::nucleon_vector_g},
      {"NUCLEON_AXIAL_G", c::nucleon_axial_g},
      {"PROTON_SPIN_G", c::proton_spin_g},
      {"NEUTRON_SPIN_G", c::neutron_spin_g},
      {"ELECTRON_SPIN_G", c::electron_spin_g},
      {"F_PI", c::f_pi},
      {"HARTREE", c::hartree},
      {"PROTON_RCH2", c::proton_rch2},
      {"NEUTRON_RCH2", c::neutron_rch2},
      {"DARWIN_FOLDY", c::darwin_foldy},
      {"SQRT2", c::sqrt2},
      {"INVSQRT2", c::inv_sqrt2},
      {"LOG2", c::ln2},
      {"PI", c::pi},
      {"SQRTPI", c::sqrt_pi},
  };
  return t;
}

TEST(Constants, EqualReferenceLiterals) {
  EXPECT_EQ(c::hbarc, 197.3269718);
  EXPECT_EQ(c::m_proton, 938.2720813);
  EXPECT_EQ(c::m_neutron, 939.5654133);
  EXPECT_EQ(c::m_electron, 0.5109989461);
  EXPECT_EQ(c::m_nucleon, (938.2720813 + 939.5654133) / 2.0);
  EXPECT_EQ(c::m_pion_charged, 139.57018);
  EXPECT_EQ(c::m_pion_neutral, 134.9766);
  EXPECT_EQ(c::nucleon_vector_g, 1.00);
  EXPECT_EQ(c::nucleon_axial_g, 1.27);
  EXPECT_EQ(c::proton_spin_g, 5.585690569);
  EXPECT_EQ(c::neutron_spin_g, -3.826085);
  EXPECT_EQ(c::electron_spin_g, 2.002319);
  EXPECT_EQ(c::alpha_fs, 1.0 / 137.035999);
  EXPECT_EQ(c::f_pi, 92.2);
  EXPECT_EQ(c::hartree, 27.21138602);
  EXPECT_EQ(c::proton_rch2, 0.707);
  EXPECT_EQ(c::neutron_rch2, -0.1161);
  EXPECT_EQ(c::darwin_foldy, 0.033172);
  EXPECT_EQ(c::sqrt2, 1.41421356237309505);
  EXPECT_EQ(c::inv_sqrt2, 0.7071067811865);
  EXPECT_EQ(c::ln2, 0.69314718055994530942);
  EXPECT_EQ(c::pi, 3.14159265358979324);
  EXPECT_EQ(c::sqrt_pi, 1.7724538509055160);
}

TEST(Constants, InvSqrt2IsTruncatedReferenceValue) {
  EXPECT_NE(c::inv_sqrt2, 1.0 / c::sqrt2);
  EXPECT_NEAR(c::inv_sqrt2, 1.0 / c::sqrt2, 1e-12);
}

TEST(Constants, MatchReferenceFile) {
  std::ifstream in(IMSRG_REFERENCE_PHYSICAL_CONSTANTS);
  if (!in) GTEST_SKIP() << "reference file not available";
  const std::regex line_re(R"(^const double\s+([A-Z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*;)");
  std::size_t matched = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::smatch m;
    if (!std::regex_search(line, m, line_re)) continue;
    const auto it = literal_table().find(m[1].str());
    ASSERT_NE(it, literal_table().end()) << "constant not covered: " << m[1].str();
    const double ref = std::strtod(m[2].str().c_str(), nullptr);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(it->second), std::bit_cast<std::uint64_t>(ref))
        << m[1].str();
    ++matched;
  }
  // Every plain-literal constant of the reference was found (M_NUCLEON and ALPHA_FS are
  // expressions and covered above).
  EXPECT_EQ(matched, literal_table().size());
}

}  // namespace
