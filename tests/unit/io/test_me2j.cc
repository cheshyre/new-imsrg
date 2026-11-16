// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// me2j value counts and the reader on a tiny synthetic file with hand-checked
// elements.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/types.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/basis/orbit.h"
#include "imsrg/io/input_stream.h"
#include "imsrg/io/me2j.h"
#include "tests/support/scratch_dir.h"

namespace imsrg::io {
namespace {

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Value counts.

/// Independent brute-force count of the records of a me2j file over explicit
/// (e, l, 2j) orbits.
std::int64_t BruteForceCount(int e1max, int e2max, int lmax, int emax_read) {
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
          if ((orbs[a].l + orbs[b].l + orbs[c].l + orbs[d].l) % 2 != 0)
            continue;
          const int jmin = std::max(std::abs(orbs[a].j2 - orbs[b].j2),
                                    std::abs(orbs[c].j2 - orbs[d].j2)) /
                           2;
          const int jmax =
              std::min(orbs[a].j2 + orbs[b].j2, orbs[c].j2 + orbs[d].j2) / 2;
          if (jmax >= jmin) n += jmax - jmin + 1;
        }
      }
    }
  }
  return 4 * n;
}

TEST(Me2jCount, ReferenceValueOfTheEmax14File) {
  EXPECT_EQ(Me2jValueCount({14, 28, 14}), 168066656);
}

TEST(Me2jCount, SmallCountsFromTheReferenceScript) {
  // Values computed with tools/reference/me2j_value_count.py (imsrg++ loop
  // order).
  EXPECT_EQ(Me2jValueCount({1, 2, 1}), 100);
  EXPECT_EQ(Me2jValueCount({2, 4, 2}), 904);
  EXPECT_EQ(Me2jValueCount({2, 4, 1}), 224);
  EXPECT_EQ(Me2jValueCount({3, 6, 3}), 5696);
  EXPECT_EQ(Me2jValueCount({4, 8, 4}), 27336);
  EXPECT_EQ(Me2jValueCount({4, 4, 4}), 2452);
}

TEST(Me2jCount, AgreesWithBruteForce) {
  for (int e1 = 0; e1 <= 5; ++e1) {
    for (int e2 : {e1, e1 + 1, 2 * e1}) {
      for (int lm : {1, 2, e1}) {
        if (e2 <= 0 || lm <= 0) continue;
        EXPECT_EQ(Me2jValueCount({e1, e2, lm}), BruteForceCount(e1, e2, lm, 99))
            << e1 << ' ' << e2 << ' ' << lm;
      }
    }
  }
}

TEST(Me2jCount, DefaultsAreTwiceE1maxAndE1max) {
  EXPECT_EQ(Me2jValueCount({3, -1, -1}), Me2jValueCount({3, 6, 3}));
  EXPECT_EQ(Me2jValueCount({3, 0, 0}),
            Me2jValueCount({3, 6, 3}));  // imsrg++ tests > 0
}

TEST(Me2jCount, PrefixIsSmallerAndUsesTheFileE2max) {
  const Me2jTruncation f{4, 8, 4};
  EXPECT_EQ(Me2jValueCount(f, 4), Me2jValueCount(f));
  EXPECT_EQ(Me2jValueCount(f, 99), Me2jValueCount(f));
  EXPECT_LT(Me2jValueCount(f, 2), Me2jValueCount(f));
  EXPECT_EQ(Me2jValueCount(f, 2), BruteForceCount(4, 8, 4, 2));
  EXPECT_EQ(Me2jValueCount(f, 0), BruteForceCount(4, 8, 4, 0));
}

// -----------------------------------------------------------------------------
// The synthetic file: e1max = 1 (orbits 0: s1/2, 1: p1/2, 2: p3/2), e2max = 2,
// lmax = 1.

using Record =
    std::tuple<int, int, int, int, int>;  // file-order orbit indices and J
using Values = std::array<double, 4>;     // T0 pn, nn, T1 pn, pp

/// File text: one header line, then 4 values per record (zeros unless set in
/// `values`).
std::string SyntheticText(const std::map<Record, Values>& values) {
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
          if ((orbs[a].l + orbs[b].l + orbs[c].l + orbs[d].l) % 2 != 0)
            continue;
          const int jmin = std::max(std::abs(orbs[a].j2 - orbs[b].j2),
                                    std::abs(orbs[c].j2 - orbs[d].j2)) /
                           2;
          const int jmax =
              std::min(orbs[a].j2 + orbs[b].j2, orbs[c].j2 + orbs[d].j2) / 2;
          for (int J = jmin; J <= jmax; ++J) {
            Values v{0, 0, 0, 0};
            if (const auto it = values.find({a, b, c, d, J});
                it != values.end())
              v = it->second;
            os << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << '\n';
            tokens += 4;
          }
        }
      }
    }
  }
  EXPECT_EQ(tokens, Me2jValueCount({1, 2, 1}));
  return os.str();
}

fs::path WriteSynthetic(const std::string& name, const std::string& text,
                        bool gz) {
  // The process id keeps parallel ctest processes (one per test) from sharing,
  // and deleting, each other's files.
  const fs::path p =
      fs::temp_directory_path() /
      ("imsrg_me2j_test_" + std::to_string(::getpid()) + "_" + name);
  if (gz) {
    GzipWriter w(p);
    w.Write(text);
    w.Close();
  } else {
    std::ofstream f(p, std::ios::binary);
    f << text;
  }
  return p;
}

std::shared_ptr<const basis::ModelSpace> MakeModelSpace(int emax) {
  return basis::ModelSpace::FromNames({.emax = emax}, 20.0, "He4", "He4");
}

OrbitIndex Orbit(const basis::ModelSpace& ms, int n, int l, int j2, int tz2) {
  const auto o = ms.basis().find(basis::Orbit{n, l, TwoJ{j2}, TwoTz{tz2}});
  EXPECT_TRUE(o.has_value());
  return *o;
}

class Me2jSynthetic : public ::testing::Test {
 protected:
  void SetUp() override {
    ms_ = MakeModelSpace(1);
    // s1/2 pair, J = 0: T0 = 0.25, nn = 0.5, T1 = 0.75, pp = 1.0 (all exactly
    // representable).
    values_[{0, 0, 0, 0, 0}] = {0.25, 0.5, 0.75, 1.0};
    // s1/2 pair, J = 1: only T = 0 is allowed for identical orbits.
    values_[{0, 0, 0, 0, 1}] = {1.25, 1.5, 1.75, 2.0};
    // Distinct orbits p1/2 s1/2, J = 0: T0 = 2, nn = 6, T1 = 4, pp = 8.
    values_[{1, 0, 1, 0, 0}] = {2.0, 6.0, 4.0, 8.0};
    path_ = WriteSynthetic("synthetic.me2j", SyntheticText(values_), false);
  }
  void TearDown() override { fs::remove(path_); }

  std::shared_ptr<const basis::ModelSpace> ms_;
  std::map<Record, Values> values_;
  fs::path path_;
};

TEST_F(Me2jSynthetic, IdenticalOrbitJ0HasNormalizationAndPauliGating) {
  Me2jReadReport rep;
  const op::Operator H = ReadMe2j(path_, {.file = {1, 2, 1}}, ms_, &rep);
  EXPECT_EQ(rep.values_read, 100);
  EXPECT_GE(rep.seconds, 0.0);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex sp = Orbit(*ms_, 0, 0, 1, -1);
  const OrbitIndex sn = Orbit(*ms_, 0, 0, 1, +1);
  // pp, nn: file value / sqrt2 per identical pair (up to the float widening of
  // exact values).
  EXPECT_NEAR(tb.get_normalized(sp, sp, sp, sp, TwoJ{0}), 1.0 / 2.0, 1.0e-14);
  EXPECT_NEAR(tb.get_normalized(sn, sn, sn, sn, TwoJ{0}), 0.5 / 2.0, 1.0e-14);
  // pn: (V_T1 + V_T0)/2 with T0 gated off for even J of identical orbits: 0.75
  // / 2 in the unnormalized-pair convention, times sqrt2 * sqrt2 * 1/2 for the
  // normalized pn state.
  EXPECT_NEAR(tb.get_normalized(sp, sn, sp, sn, TwoJ{0}), 0.75 / 2.0, 1.0e-14);
}

TEST_F(Me2jSynthetic, IdenticalOrbitJ1UsesOnlyIsospinZero) {
  const op::Operator H = ReadMe2j(path_, {.file = {1, 2, 1}}, ms_);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex sp = Orbit(*ms_, 0, 0, 1, -1);
  const OrbitIndex sn = Orbit(*ms_, 0, 0, 1, +1);
  // T0 = 1.25 -> 1.25 / 2; the T1 value 1.75 is Pauli-forbidden for odd J of
  // identical orbits.
  EXPECT_NEAR(tb.get_normalized(sp, sn, sp, sn, TwoJ{2}), 1.25 / 2.0, 1.0e-14);
  EXPECT_EQ(tb.get_normalized(sp, sp, sp, sp, TwoJ{2}),
            0.0);  // no such two-proton state
}

TEST_F(Me2jSynthetic, DistinctOrbitsGivePnpnAndPnnpFromT1AndT0) {
  const op::Operator H = ReadMe2j(path_, {.file = {1, 2, 1}}, ms_);
  const op::TwoBody& tb = H.two_body();
  const OrbitIndex pp1 = Orbit(*ms_, 0, 1, 1, -1);  // p1/2 proton
  const OrbitIndex pn1 = Orbit(*ms_, 0, 1, 1, +1);
  const OrbitIndex sp = Orbit(*ms_, 0, 0, 1, -1);
  const OrbitIndex sn = Orbit(*ms_, 0, 0, 1, +1);
  EXPECT_NEAR(tb.get_normalized(pp1, sp, pp1, sp, TwoJ{0}), 8.0,
              1.0e-13);  // pp
  EXPECT_NEAR(tb.get_normalized(pn1, sn, pn1, sn, TwoJ{0}), 6.0,
              1.0e-13);  // nn
  // V_pnpn = (T1 + T0) / 2 = 3, V_pnnp = (T1 - T0) / 2 = 1.
  EXPECT_NEAR(tb.get_normalized(pp1, sn, pp1, sn, TwoJ{0}), 3.0, 1.0e-13);
  EXPECT_NEAR(tb.get_normalized(pp1, sn, pn1, sp, TwoJ{0}), 1.0, 1.0e-13);
  EXPECT_NEAR(tb.get_normalized(pn1, sp, pn1, sp, TwoJ{0}), 3.0, 1.0e-13);
  // Hermitian partner.
  EXPECT_NEAR(tb.get_normalized(pn1, sp, pp1, sn, TwoJ{0}), 1.0, 1.0e-13);
}

TEST_F(Me2jSynthetic, GzipAndPlainGiveTheSameOperator) {
  const auto gz =
      WriteSynthetic("synthetic.me2j.gz", SyntheticText(values_), true);
  const op::Operator a = ReadMe2j(path_, {.file = {1, 2, 1}}, ms_);
  const op::Operator b = ReadMe2j(gz, {.file = {1, 2, 1}}, ms_);
  fs::remove(gz);
  const OrbitIndex pp1 = Orbit(*ms_, 0, 1, 1, -1);
  const OrbitIndex sn = Orbit(*ms_, 0, 0, 1, +1);
  EXPECT_EQ(a.two_body().get_normalized(pp1, sn, pp1, sn, TwoJ{0}),
            b.two_body().get_normalized(pp1, sn, pp1, sn, TwoJ{0}));
}

TEST_F(Me2jSynthetic, SmallerModelSpaceReadsOnlyThePrefix) {
  const auto ms0 = MakeModelSpace(0);  // only the s1/2 orbits
  Me2jReadReport rep;
  const op::Operator H = ReadMe2j(path_, {.file = {1, 2, 1}}, ms0, &rep);
  EXPECT_EQ(rep.values_read, 8);  // (s s s s) with J = 0, 1: two records
  const OrbitIndex sp = Orbit(*ms0, 0, 0, 1, -1);
  const OrbitIndex sn = Orbit(*ms0, 0, 0, 1, +1);
  EXPECT_NEAR(H.two_body().get_normalized(sp, sn, sp, sn, TwoJ{0}), 0.375,
              1.0e-14);
  EXPECT_NEAR(H.two_body().get_normalized(sp, sn, sp, sn, TwoJ{2}), 0.625,
              1.0e-14);
  // A model space larger than the file: records beyond the file's e1max do not
  // exist.
  const auto ms2 = MakeModelSpace(2);
  const op::Operator H2 = ReadMe2j(path_, {.file = {1, 2, 1}}, ms2, &rep);
  EXPECT_EQ(rep.values_read, 100);
  (void)H2;
}

TEST_F(Me2jSynthetic, WholeFileModeVerifiesTheValueCount) {
  Me2jReadReport rep;
  const auto ms0 = MakeModelSpace(0);
  (void)ReadMe2j(path_, {.file = {1, 2, 1}, .read_whole_file = true}, ms0,
                 &rep);
  EXPECT_EQ(rep.values_read,
            100);  // the whole file is consumed although ms is small
  // A truncation that describes fewer values than the file holds is detected.
  EXPECT_THROW(
      ReadMe2j(path_, {.file = {1, 1, 1}, .read_whole_file = true}, ms0),
      InputError);
}

TEST_F(Me2jSynthetic, TruncatedFileThrowsInputError) {
  const std::string text = SyntheticText(values_);
  const auto p =
      WriteSynthetic("truncated.me2j", text.substr(0, text.size() / 2), false);
  EXPECT_THROW(ReadMe2j(p, {.file = {1, 2, 1}}, ms_), InputError);
  // Claiming a larger truncation than the file has also ends the data early.
  EXPECT_THROW(ReadMe2j(path_, {.file = {2, 4, 2}}, MakeModelSpace(2)),
               InputError);
  fs::remove(p);
}

TEST_F(Me2jSynthetic, MalformedTokenAndMissingFileThrowInputError) {
  const auto p =
      WriteSynthetic("malformed.me2j", "header\n0.25 oops 0.75 1.0\n", false);
  EXPECT_THROW(ReadMe2j(p, {.file = {1, 2, 1}}, ms_), InputError);
  fs::remove(p);
  EXPECT_THROW(ReadMe2j(fs::temp_directory_path() / "imsrg_no_such.me2j",
                        {.file = {1, 2, 1}}, ms_),
               InputError);
  EXPECT_THROW(ReadMe2j(path_, {.file = {1, 2, 1}}, nullptr), InputError);
}

// -----------------------------------------------------------------------------
// Binary .me2j.bin files (D57): 255 header bytes, then binary32 values.

/// Writes `header_fill` * 255 bytes and then the float32 values of the
/// whitespace-separated tokens of `text` (after its first line) to `p`;
/// `extra`/`drop` append zero bytes to / remove bytes from the end.
void WriteBinary(const fs::path& p, const std::string& text,
                 unsigned char header_fill, int extra_bytes, int drop_bytes) {
  std::istringstream is(text);
  std::string line;
  std::getline(is, line);  // the text header line is replaced by 255 bytes
  std::vector<float> values;
  double v = 0.0;
  while (is >> v) values.push_back(static_cast<float>(v));
  std::string bytes(255, static_cast<char>(header_fill));
  bytes.append(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(float));
  bytes.append(static_cast<std::size_t>(extra_bytes), '\0');
  bytes.resize(bytes.size() - static_cast<std::size_t>(drop_bytes));
  std::ofstream f(p, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/// The synthetic text with distinct, non-float-representable values in every
/// slot, so that the float rounding is exercised.
std::string RoundingText() {
  std::ostringstream os;
  os << "header\n";
  for (int i = 0; i < 100; ++i) {
    os.precision(17);
    os << (0.1 * (i + 1) + 1.0e-9 * i) << (i % 4 == 3 ? '\n' : ' ');
  }
  return os.str();
}

/// Max over all two-body matrix elements of the difference between a and b
/// (both on the e1max = 1 space).
void ExpectSameTwoBody(const op::Operator& a, const op::Operator& b,
                       const basis::ModelSpace& ms) {
  const std::int32_t n = ms.basis().num_orbits();
  for (std::int32_t i = 0; i < n; ++i) {
    for (std::int32_t j = i; j < n; ++j) {
      for (std::int32_t k = 0; k < n; ++k) {
        for (std::int32_t l = k; l < n; ++l) {
          for (int J = 0; J <= 2; ++J) {
            const OrbitIndex p{i}, q{j}, r{k}, t{l};
            EXPECT_EQ(a.two_body().get_normalized(p, q, r, t, TwoJ{2 * J}),
                      b.two_body().get_normalized(p, q, r, t, TwoJ{2 * J}));
          }
        }
      }
    }
  }
}

TEST(Me2jBinary, RoundTripEqualsFloatRoundedText) {
  test_support::ScratchDir dir;
  const auto ms = MakeModelSpace(1);
  const std::string text = RoundingText();
  // The text file with values pre-rounded to float: reading it as text gives
  // static_cast<double>(static_cast<float>(value)) exactly, as does the .bin.
  const fs::path txt = dir.path() / "a.me2j";
  {
    std::ofstream f(txt);
    f << text;
  }
  const fs::path bin = dir.path() / "a.me2j.bin";
  WriteBinary(bin, text, 0x41, 0, 0);
  Me2jReadReport rep;
  const op::Operator hb = ReadMe2j(bin, {.file = {1, 2, 1}}, ms, &rep);
  EXPECT_EQ(rep.values_read, 100);
  const op::Operator ht = ReadMe2j(txt, {.file = {1, 2, 1}}, ms);
  ExpectSameTwoBody(hb, ht, *ms);
  // Element check against static_cast<double>(static_cast<float>(text value)):
  // the first record is (s s s s, J = 0) with values {v0, v1, v2, v3}; the pp
  // element is v3 / sqrt2 / sqrt2 (identical pairs on both sides).
  std::istringstream is(text);
  std::string line;
  std::getline(is, line);
  double v[4];
  is >> v[0] >> v[1] >> v[2] >> v[3];
  const double pp = static_cast<double>(static_cast<float>(v[3]));
  const OrbitIndex sp = Orbit(*ms, 0, 0, 1, -1);
  EXPECT_NEAR(hb.two_body().get_normalized(sp, sp, sp, sp, TwoJ{0}), pp / 2.0,
              1.0e-15);
  // Whole-file mode on a smaller model space also works.
  EXPECT_NO_THROW(ReadMe2j(bin, {.file = {1, 2, 1}, .read_whole_file = true},
                           MakeModelSpace(0)));
}

TEST(Me2jBinary, PrefixReadOfABinaryFile) {
  test_support::ScratchDir dir;
  const fs::path bin = dir.path() / "p.me2j.bin";
  WriteBinary(bin, RoundingText(), 0, 0, 0);
  Me2jReadReport rep;
  (void)ReadMe2j(bin, {.file = {1, 2, 1}}, MakeModelSpace(0), &rep);
  EXPECT_EQ(rep.values_read, 8);
}

TEST(Me2jBinary, WrongFileSizeThrowsInputError) {
  test_support::ScratchDir dir;
  const auto ms = MakeModelSpace(1);
  const std::string text = RoundingText();
  struct Case {
    const char* name;
    int extra, drop;
  };
  for (const Case c : {Case{"short1", 0, 1}, Case{"long1", 1, 0},
                       Case{"long4", 4, 0}, Case{"short4", 0, 4}}) {
    const fs::path bin = dir.path() / (std::string(c.name) + ".me2j.bin");
    WriteBinary(bin, text, 0, c.extra, c.drop);
    EXPECT_THROW(ReadMe2j(bin, {.file = {1, 2, 1}}, ms), InputError) << c.name;
  }
  // A header-only file.
  const fs::path hdr = dir.path() / "hdr.me2j.bin";
  {
    std::ofstream f(hdr, std::ios::binary);
    f << std::string(255, 'x');
  }
  EXPECT_THROW(ReadMe2j(hdr, {.file = {1, 2, 1}}, ms), InputError);
  // Missing file.
  EXPECT_THROW(ReadMe2j(dir.path() / "none.me2j.bin", {.file = {1, 2, 1}}, ms),
               InputError);
}

TEST(Me2jBinary, HeaderContentIsIgnored) {
  test_support::ScratchDir dir;
  const auto ms = MakeModelSpace(1);
  const std::string text = RoundingText();
  const fs::path zero = dir.path() / "zero.me2j.bin";
  const fs::path junk = dir.path() / "junk.me2j.bin";
  WriteBinary(zero, text, 0x00, 0, 0);
  WriteBinary(junk, text, 0xFF, 0, 0);
  const op::Operator a = ReadMe2j(zero, {.file = {1, 2, 1}}, ms);
  const op::Operator b = ReadMe2j(junk, {.file = {1, 2, 1}}, ms);
  ExpectSameTwoBody(a, b, *ms);
}

}  // namespace
}  // namespace imsrg::io
