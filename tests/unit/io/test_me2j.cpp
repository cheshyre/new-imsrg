// me2j value counts and the reader on a tiny synthetic file with hand-checked elements.
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/base/types.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/basis/orbit.hpp"
#include "imsrg/io/input_stream.hpp"
#include "imsrg/io/me2j.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::io;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------
// Value counts.

/// Independent brute-force count of the records of a me2j file over explicit (e, l, 2j) orbits.
std::int64_t brute_force_count(int e1max, int e2max, int lmax, int emax_read) {
  struct O {
    int e, l, j2;
  };
  std::vector<O> orbs;
  const int top = std::min(e1max, emax_read);
  for (int e = 0; e <= top; ++e) {
    for (int l = 0; l <= std::min(e, lmax); ++l) {
      if ((e - l) % 2 != 0) continue;
      for (int j2 = 2 * l - 1; j2 <= 2 * l + 1; j2 += 2) {
        if (j2 > 0) orbs.push_back({e, l, j2});
      }
    }
  }
  std::int64_t n = 0;
  const std::size_t N = orbs.size();
  for (std::size_t a = 0; a < N; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      if (orbs[a].e + orbs[b].e > e2max) continue;
      for (std::size_t c = 0; c <= a; ++c) {
        for (std::size_t d = 0; d <= (c == a ? b : c); ++d) {
          if (orbs[c].e + orbs[d].e > e2max) continue;
          if ((orbs[a].l + orbs[b].l + orbs[c].l + orbs[d].l) % 2 != 0) continue;
          const int jmin =
              std::max(std::abs(orbs[a].j2 - orbs[b].j2), std::abs(orbs[c].j2 - orbs[d].j2)) / 2;
          const int jmax = std::min(orbs[a].j2 + orbs[b].j2, orbs[c].j2 + orbs[d].j2) / 2;
          if (jmax >= jmin) n += jmax - jmin + 1;
        }
      }
    }
  }
  return 4 * n;
}

TEST(Me2jCount, ReferenceValueOfTheEmax14File) {
  EXPECT_EQ(me2j_value_count({14, 28, 14}), 168066656);
}

TEST(Me2jCount, SmallCountsFromTheReferenceScript) {
  // Values computed with tools/reference/me2j_value_count.py (imsrg++ loop order).
  EXPECT_EQ(me2j_value_count({1, 2, 1}), 100);
  EXPECT_EQ(me2j_value_count({2, 4, 2}), 904);
  EXPECT_EQ(me2j_value_count({2, 4, 1}), 224);
  EXPECT_EQ(me2j_value_count({3, 6, 3}), 5696);
  EXPECT_EQ(me2j_value_count({4, 8, 4}), 27336);
  EXPECT_EQ(me2j_value_count({4, 4, 4}), 2452);
}

TEST(Me2jCount, AgreesWithBruteForce) {
  for (int e1 = 0; e1 <= 5; ++e1) {
    for (int e2 : {e1, e1 + 1, 2 * e1}) {
      for (int lm : {1, 2, e1}) {
        if (e2 <= 0 || lm <= 0) continue;
        EXPECT_EQ(me2j_value_count({e1, e2, lm}), brute_force_count(e1, e2, lm, 99))
            << e1 << ' ' << e2 << ' ' << lm;
      }
    }
  }
}

TEST(Me2jCount, DefaultsAreTwiceE1maxAndE1max) {
  EXPECT_EQ(me2j_value_count({3, -1, -1}), me2j_value_count({3, 6, 3}));
  EXPECT_EQ(me2j_value_count({3, 0, 0}), me2j_value_count({3, 6, 3}));  // imsrg++ tests > 0
}

TEST(Me2jCount, PrefixIsSmallerAndUsesTheFileE2max) {
  const Me2jTruncation f{4, 8, 4};
  EXPECT_EQ(me2j_value_count(f, 4), me2j_value_count(f));
  EXPECT_EQ(me2j_value_count(f, 99), me2j_value_count(f));
  EXPECT_LT(me2j_value_count(f, 2), me2j_value_count(f));
  EXPECT_EQ(me2j_value_count(f, 2), brute_force_count(4, 8, 4, 2));
  EXPECT_EQ(me2j_value_count(f, 0), brute_force_count(4, 8, 4, 0));
}

// ---------------------------------------------------------------------------------------------
// The synthetic file: e1max = 1 (orbits 0: s1/2, 1: p1/2, 2: p3/2), e2max = 2, lmax = 1.

using Record = std::tuple<int, int, int, int, int>;  // file-order orbit indices and J
using Values = std::array<double, 4>;                // T0 pn, nn, T1 pn, pp

/// File text: one header line, then 4 values per record (zeros unless set in `values`).
std::string synthetic_text(const std::map<Record, Values>& values) {
  struct O {
    int e, l, j2;
  };
  const std::array<O, 3> orbs{{{0, 0, 1}, {1, 1, 1}, {1, 1, 3}}};
  std::ostringstream os;
  os << "synthetic me2j test file, e1max=1\n";
  std::int64_t tokens = 0;
  for (std::size_t a = 0; a < 3; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      if (orbs[a].e + orbs[b].e > 2) continue;
      for (std::size_t c = 0; c <= a; ++c) {
        for (std::size_t d = 0; d <= (c == a ? b : c); ++d) {
          if (orbs[c].e + orbs[d].e > 2) continue;
          if ((orbs[a].l + orbs[b].l + orbs[c].l + orbs[d].l) % 2 != 0) continue;
          const int jmin =
              std::max(std::abs(orbs[a].j2 - orbs[b].j2), std::abs(orbs[c].j2 - orbs[d].j2)) / 2;
          const int jmax = std::min(orbs[a].j2 + orbs[b].j2, orbs[c].j2 + orbs[d].j2) / 2;
          for (int J = jmin; J <= jmax; ++J) {
            Values v{0, 0, 0, 0};
            if (const auto it = values.find({a, b, c, d, J}); it != values.end()) v = it->second;
            os << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << '\n';
            tokens += 4;
          }
        }
      }
    }
  }
  EXPECT_EQ(tokens, me2j_value_count({1, 2, 1}));
  return os.str();
}

fs::path write_synthetic(const std::string& name, const std::string& text, bool gz) {
  const fs::path p = fs::temp_directory_path() / ("imsrg_me2j_test_" + name);
  if (gz) {
    GzipWriter w(p);
    w.write(text);
    w.close();
  } else {
    std::ofstream f(p, std::ios::binary);
    f << text;
  }
  return p;
}

std::shared_ptr<const basis::ModelSpace> model_space(int emax) {
  return basis::ModelSpace::from_names({.emax = emax}, 20.0, "He4", "He4");
}

OrbitIndex orbit(const basis::ModelSpace& ms, int n, int l, int j2, int tz2) {
  const auto o = ms.basis().find(basis::Orbit{n, l, TwoJ{j2}, TwoTz{tz2}});
  EXPECT_TRUE(o.has_value());
  return *o;
}

class Me2jSynthetic : public ::testing::Test {
 protected:
  void SetUp() override {
    ms = model_space(1);
    // s1/2 pair, J = 0: T0 = 0.25, nn = 0.5, T1 = 0.75, pp = 1.0 (all exactly representable).
    values[{0, 0, 0, 0, 0}] = {0.25, 0.5, 0.75, 1.0};
    // s1/2 pair, J = 1: only T = 0 is allowed for identical orbits.
    values[{0, 0, 0, 0, 1}] = {1.25, 1.5, 1.75, 2.0};
    // Distinct orbits p1/2 s1/2, J = 0: T0 = 2, nn = 6, T1 = 4, pp = 8.
    values[{1, 0, 1, 0, 0}] = {2.0, 6.0, 4.0, 8.0};
    path = write_synthetic("synthetic.me2j", synthetic_text(values), false);
  }
  void TearDown() override { fs::remove(path); }

  std::shared_ptr<const basis::ModelSpace> ms;
  std::map<Record, Values> values;
  fs::path path;
};

TEST_F(Me2jSynthetic, IdenticalOrbitJ0HasNormalizationAndPauliGating) {
  Me2jReadReport rep;
  const op::Operator H = read_me2j(path, {.file = {1, 2, 1}}, ms, &rep);
  EXPECT_EQ(rep.values_read, 100);
  EXPECT_GE(rep.seconds, 0.0);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex sp = orbit(*ms, 0, 0, 1, -1);
  const OrbitIndex sn = orbit(*ms, 0, 0, 1, +1);
  // pp, nn: file value / sqrt2 per identical pair (up to the float widening of exact values).
  EXPECT_NEAR(tb.get_normalized(sp, sp, sp, sp, TwoJ{0}), 1.0 / 2.0, 1e-14);
  EXPECT_NEAR(tb.get_normalized(sn, sn, sn, sn, TwoJ{0}), 0.5 / 2.0, 1e-14);
  // pn: (V_T1 + V_T0)/2 with T0 gated off for even J of identical orbits: 0.75 / 2 in the
  // unnormalized-pair convention, times sqrt2 * sqrt2 * 1/2 for the normalized pn state.
  EXPECT_NEAR(tb.get_normalized(sp, sn, sp, sn, TwoJ{0}), 0.75 / 2.0, 1e-14);
}

TEST_F(Me2jSynthetic, IdenticalOrbitJ1UsesOnlyIsospinZero) {
  const op::Operator H = read_me2j(path, {.file = {1, 2, 1}}, ms);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex sp = orbit(*ms, 0, 0, 1, -1);
  const OrbitIndex sn = orbit(*ms, 0, 0, 1, +1);
  // T0 = 1.25 -> 1.25 / 2; the T1 value 1.75 is Pauli-forbidden for odd J of identical orbits.
  EXPECT_NEAR(tb.get_normalized(sp, sn, sp, sn, TwoJ{2}), 1.25 / 2.0, 1e-14);
  EXPECT_EQ(tb.get_normalized(sp, sp, sp, sp, TwoJ{2}), 0.0);  // no such two-proton state
}

TEST_F(Me2jSynthetic, DistinctOrbitsGivePnpnAndPnnpFromT1AndT0) {
  const op::Operator H = read_me2j(path, {.file = {1, 2, 1}}, ms);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex pp1 = orbit(*ms, 0, 1, 1, -1);  // p1/2 proton
  const OrbitIndex pn1 = orbit(*ms, 0, 1, 1, +1);
  const OrbitIndex sp = orbit(*ms, 0, 0, 1, -1);
  const OrbitIndex sn = orbit(*ms, 0, 0, 1, +1);
  EXPECT_NEAR(tb.get_normalized(pp1, sp, pp1, sp, TwoJ{0}), 8.0, 1e-13);  // pp
  EXPECT_NEAR(tb.get_normalized(pn1, sn, pn1, sn, TwoJ{0}), 6.0, 1e-13);  // nn
  // V_pnpn = (T1 + T0) / 2 = 3, V_pnnp = (T1 - T0) / 2 = 1.
  EXPECT_NEAR(tb.get_normalized(pp1, sn, pp1, sn, TwoJ{0}), 3.0, 1e-13);
  EXPECT_NEAR(tb.get_normalized(pp1, sn, pn1, sp, TwoJ{0}), 1.0, 1e-13);
  EXPECT_NEAR(tb.get_normalized(pn1, sp, pn1, sp, TwoJ{0}), 3.0, 1e-13);
  // Hermitian partner.
  EXPECT_NEAR(tb.get_normalized(pn1, sp, pp1, sn, TwoJ{0}), 1.0, 1e-13);
}

TEST_F(Me2jSynthetic, GzipAndPlainGiveTheSameOperator) {
  const auto gz = write_synthetic("synthetic.me2j.gz", synthetic_text(values), true);
  const op::Operator a = read_me2j(path, {.file = {1, 2, 1}}, ms);
  const op::Operator b = read_me2j(gz, {.file = {1, 2, 1}}, ms);
  fs::remove(gz);
  const OrbitIndex pp1 = orbit(*ms, 0, 1, 1, -1);
  const OrbitIndex sn = orbit(*ms, 0, 0, 1, +1);
  EXPECT_EQ(a.two_body().get_normalized(pp1, sn, pp1, sn, TwoJ{0}),
            b.two_body().get_normalized(pp1, sn, pp1, sn, TwoJ{0}));
}

TEST_F(Me2jSynthetic, SmallerModelSpaceReadsOnlyThePrefix) {
  const auto ms0 = model_space(0);  // only the s1/2 orbits
  Me2jReadReport rep;
  const op::Operator H = read_me2j(path, {.file = {1, 2, 1}}, ms0, &rep);
  EXPECT_EQ(rep.values_read, 8);  // (s s s s) with J = 0, 1: two records
  const OrbitIndex sp = orbit(*ms0, 0, 0, 1, -1);
  const OrbitIndex sn = orbit(*ms0, 0, 0, 1, +1);
  EXPECT_NEAR(H.two_body().get_normalized(sp, sn, sp, sn, TwoJ{0}), 0.375, 1e-14);
  EXPECT_NEAR(H.two_body().get_normalized(sp, sn, sp, sn, TwoJ{2}), 0.625, 1e-14);
  // A model space larger than the file: records beyond the file's e1max do not exist.
  const auto ms2 = model_space(2);
  const op::Operator H2 = read_me2j(path, {.file = {1, 2, 1}}, ms2, &rep);
  EXPECT_EQ(rep.values_read, 100);
  (void)H2;
}

TEST_F(Me2jSynthetic, WholeFileModeVerifiesTheValueCount) {
  Me2jReadReport rep;
  const auto ms0 = model_space(0);
  (void)read_me2j(path, {.file = {1, 2, 1}, .read_whole_file = true}, ms0, &rep);
  EXPECT_EQ(rep.values_read, 100);  // the whole file is consumed although ms is small
  // A truncation that describes fewer values than the file holds is detected.
  EXPECT_THROW(read_me2j(path, {.file = {1, 1, 1}, .read_whole_file = true}, ms0), InputError);
}

TEST_F(Me2jSynthetic, TruncatedFileThrowsInputError) {
  const std::string text = synthetic_text(values);
  const auto p = write_synthetic("truncated.me2j", text.substr(0, text.size() / 2), false);
  EXPECT_THROW(read_me2j(p, {.file = {1, 2, 1}}, ms), InputError);
  // Claiming a larger truncation than the file has also ends the data early.
  EXPECT_THROW(read_me2j(path, {.file = {2, 4, 2}}, model_space(2)), InputError);
  fs::remove(p);
}

TEST_F(Me2jSynthetic, MalformedTokenAndMissingFileThrowInputError) {
  const auto p = write_synthetic("malformed.me2j", "header\n0.25 oops 0.75 1.0\n", false);
  EXPECT_THROW(read_me2j(p, {.file = {1, 2, 1}}, ms), InputError);
  fs::remove(p);
  EXPECT_THROW(read_me2j(fs::temp_directory_path() / "imsrg_no_such.me2j", {.file = {1, 2, 1}}, ms),
               InputError);
  EXPECT_THROW(read_me2j(path, {.file = {1, 2, 1}}, nullptr), InputError);
}

}  // namespace
