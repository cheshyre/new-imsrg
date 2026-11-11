// Wigner symbols and Clebsch-Gordan coefficients against exact sympy values (design/30 section 2).
//
// The reference file is generated at build time by reference/gen_wigner_reference.py (emax 8);
// every line is "<family> <doubled args...> <value>". Tolerance: |delta| <= max(1e-12 |exact|,
// 1e-14). The maximum absolute and relative deviation per family are printed and recorded.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/angmom/wigner.hpp"

namespace {

using imsrg::TwoJ;
using imsrg::TwoM;

struct FamilyStats {
  long count = 0;
  long failures = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

/// Evaluates the symbol of `family` for the doubled arguments `a`; returns false if the family
/// is unknown.
bool evaluate(const std::string& family, const std::vector<int>& a, double* out) {
  auto J = [&](int i) { return TwoJ{a[static_cast<std::size_t>(i)]}; };
  auto M = [&](int i) { return TwoM{a[static_cast<std::size_t>(i)]}; };
  namespace am = imsrg::angmom;
  if (family == "cg_sp" || family == "cg_ls" || family == "cg_l0") {
    *out = am::clebsch_gordan(J(0), M(1), J(2), M(3), J(4), M(5));
  } else if (family == "3j_sp") {
    *out = am::wigner_3j(J(0), J(1), J(2), M(3), M(4), M(5));
  } else if (family == "6j_pandya" || family == "6j_3b") {
    *out = am::wigner_6j(J(0), J(1), J(2), J(3), J(4), J(5));
  } else if (family == "9j_lsjj" || family == "9j_sample") {
    *out = am::wigner_9j(J(0), J(1), J(2), J(3), J(4), J(5), J(6), J(7), J(8));
  } else {
    return false;
  }
  return true;
}

std::size_t arg_count(const std::string& family) { return family.rfind("9j", 0) == 0 ? 9 : 6; }

}  // namespace

TEST(WignerReference, AgainstSympyExact) {
#ifndef IMSRG_WIGNER_REFERENCE_FILE
  GTEST_SKIP() << "sympy was not available at configure time: no exact reference data "
                  "(install sympy in the Python used by CMake to enable this test)";
#else
  std::ifstream in(IMSRG_WIGNER_REFERENCE_FILE);
  ASSERT_TRUE(in) << "cannot open " << IMSRG_WIGNER_REFERENCE_FILE;

  std::map<std::string, FamilyStats> stats;
  std::vector<std::string> failure_lines;
  std::string line;
  long total = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream is(line);
    std::string family;
    is >> family;
    std::vector<int> a(arg_count(family));
    for (int& x : a) is >> x;
    std::string value_text;
    is >> value_text;
    ASSERT_TRUE(static_cast<bool>(is)) << "malformed reference line: " << line;
    const double exact = std::strtod(value_text.c_str(), nullptr);
    double got = 0.0;
    ASSERT_TRUE(evaluate(family, a, &got)) << "unknown family in line: " << line;

    FamilyStats& s = stats[family];
    ++s.count;
    ++total;
    const double delta = std::fabs(got - exact);
    s.max_abs = std::max(s.max_abs, delta);
    if (exact != 0.0) s.max_rel = std::max(s.max_rel, delta / std::fabs(exact));
    if (!(delta <= std::max(1e-12 * std::fabs(exact), 1e-14))) {
      ++s.failures;
      if (failure_lines.size() < 20) {
        std::ostringstream os;
        os.precision(17);
        os << line << "  ->  got " << got << " (delta " << delta << ")";
        failure_lines.push_back(os.str());
      }
    }
  }

  const char* const expected_families[] = {"cg_sp",     "cg_ls", "cg_l0",   "3j_sp",
                                           "6j_pandya", "6j_3b", "9j_lsjj", "9j_sample"};
  long failures = 0;
  std::cout << "Wigner reference comparison (tolerance max(1e-12 rel, 1e-14 abs)):\n";
  for (const char* f : expected_families) {
    const FamilyStats& s = stats[f];
    EXPECT_GT(s.count, 0) << "no reference entries for family " << f;
    failures += s.failures;
    std::cout << "  " << f << ": count=" << s.count << " failures=" << s.failures
              << " max_abs=" << s.max_abs << " max_rel=" << s.max_rel << "\n";
    RecordProperty(std::string(f) + "_count", static_cast<int>(s.count));
    RecordProperty(std::string(f) + "_max_abs", std::to_string(s.max_abs));
    RecordProperty(std::string(f) + "_max_rel", std::to_string(s.max_rel));
  }
  std::cout << "  total entries: " << total << "\n";
  for (const std::string& l : failure_lines) std::cout << "FAIL " << l << "\n";
  EXPECT_EQ(failures, 0) << failures << " reference entries outside tolerance";
#endif
}
