// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Tests of the OperatorDump v1 reader and writer: round trips and one test per
// reader error.
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/compare.h"
#include "imsrg/operator/operator_dump.h"

namespace imsrg::op {
namespace {

std::shared_ptr<const basis::ModelSpace> MakeMs(int emax = 1) {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = emax},
                                      16.0, "O16", "O16");
}

/// Deterministic pseudo-value for index pair (i, j) in channel ch, never zero.
double Val(std::size_t ch, std::size_t i, std::size_t j) {
  return 0.25 + 0.125 * static_cast<double>(ch) +
         0.03125 * static_cast<double>(i) - 0.0625 * static_cast<double>(j) +
         1.0 / 3.0;
}

/// Fills `op` with values obeying its hermiticity (H: symmetric, A:
/// antisymmetric with zero diagonal, N: no relation) and the scalar one-body
/// selection rule, written with explicit raw sets.
void Fill(Operator& op) {
  const auto sgn = HermiticitySign(op.spec().hermiticity);
  const auto& ms = op.model_space();
  op.zero_body() = 0.5;
  const auto n = static_cast<std::size_t>(ms.num_orbits());
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      double v = Val(7, a, b);
      if (sgn != 0) v = a <= b ? Val(7, a, b) : sgn * Val(7, b, a);
      if (sgn == -1 && a == b) v = 0.0;
      if (a == 1 && b == 4) v = 0.0;  // exercise the nonzero-only filter
      if (sgn != 0 && a == 4 && b == 1) v = 0.0;
      // Scalar selection rule: only orbits with equal (l, 2j, tz2) are
      // connected.
      const auto& oa =
          ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(a)});
      const auto& ob =
          ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(b)});
      if (oa.l != ob.l || oa.j2 != ob.j2 || oa.tz2 != ob.tz2) v = 0.0;
      op.one_body().set_element(OrbitIndex{static_cast<std::int32_t>(a)},
                                OrbitIndex{static_cast<std::int32_t>(b)}, v);
    }
  }
  if (!op.two_body().allocated()) return;
  const auto& tbc = ms.two_body();
  for (std::int32_t c = 0; c < tbc.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const auto nk = static_cast<std::size_t>(tbc.channel(ch).num_kets);
    for (std::size_t i = 0; i < nk; ++i) {
      for (std::size_t j = 0; j < nk; ++j) {
        double v = Val(static_cast<std::size_t>(c), i, j);
        if (sgn != 0)
          v = i <= j ? Val(static_cast<std::size_t>(c), i, j)
                     : sgn * Val(static_cast<std::size_t>(c), j, i);
        if (sgn == -1 && i == j) v = 0.0;
        op.two_body().set(ch, ch, LocalKetIndex{static_cast<std::int32_t>(i)},
                          LocalKetIndex{static_cast<std::int32_t>(j)}, v);
      }
    }
  }
}

Operator MakeOp(const std::shared_ptr<const basis::ModelSpace>& ms, int rank,
                Hermiticity h) {
  Operator op(ms, OperatorSpec::Scalar(rank, h));
  Fill(op);
  op.SyncToDevice();
  return op;
}

std::string WriteText(const Operator& op) {
  std::ostringstream os;
  WriteOperatorDump(os, op, OperatorDumpWriteOptions{.source = "unit test"});
  return os.str();
}

Operator ReadText(const std::string& text,
                  const std::shared_ptr<const basis::ModelSpace>& ms,
                  const OperatorDumpReadOptions& o = {}) {
  std::istringstream is(text);
  return ReadOperatorDump(is, "mem", ms, o);
}

std::vector<std::string> SplitLines(const std::string& t) {
  std::vector<std::string> out;
  std::istringstream is(t);
  std::string l;
  while (std::getline(is, l)) out.push_back(l);
  return out;
}

std::string JoinLines(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& l : v) s += l + "\n";
  return s;
}

/// Index (0-based) of the first line starting with `prefix`.
std::size_t FindLine(const std::vector<std::string>& v,
                     const std::string& prefix) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i].rfind(prefix, 0) == 0) return i;
  }
  ADD_FAILURE() << "no line starting with " << prefix;
  return 0;
}

/// Expects an InputError whose message starts with "mem:<line>:".
void ExpectErrorAt(const std::string& text, std::size_t line,
                   const std::shared_ptr<const basis::ModelSpace>& ms,
                   const OperatorDumpReadOptions& o = {}) {
  try {
    (void)ReadText(text, ms, o);
    ADD_FAILURE() << "expected InputError at line " << line;
  } catch (const InputError& e) {
    const std::string want = "mem:" + std::to_string(line) + ":";
    EXPECT_EQ(std::string(e.what()).rfind(want, 0), 0U)
        << e.what() << " (wanted " << want << ")";
  }
}

class DumpRoundTrip
    : public ::testing::TestWithParam<std::pair<Hermiticity, int>> {};

TEST_P(DumpRoundTrip, WriterReaderIdentity) {
  const auto ms = MakeMs();
  const auto [h, rank] = GetParam();
  const Operator op = MakeOp(ms, rank, h);
  const std::string text = WriteText(op);
  const Operator back = ReadText(text, ms);
  EXPECT_EQ(MaxAbsDifference(op, back).max(), 0.0);
  EXPECT_EQ(back.spec().hermiticity, h);
  EXPECT_EQ(back.spec().particle_rank, rank);
  EXPECT_EQ(WriteText(back), text);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, DumpRoundTrip,
    ::testing::Values(std::pair{Hermiticity::kHermitian, 2},
                      std::pair{Hermiticity::kAntiHermitian, 2},
                      std::pair{Hermiticity::kNonHermitian, 2},
                      std::pair{Hermiticity::kHermitian, 1}));

TEST(OperatorDump, WriterRecordsAndFormat) {
  const auto ms = MakeMs();
  const auto lines =
      SplitLines(WriteText(MakeOp(ms, 2, Hermiticity::kHermitian)));
  EXPECT_EQ(lines.front(), "# imsrg-operator-dump v1");
  EXPECT_EQ(lines[1], "source unit test");
  EXPECT_EQ(lines[2], "basis emax 1 lmax 1 hw 16");
  EXPECT_EQ(lines[3], "reference O16");
  EXPECT_EQ(lines[4], "norbits 6");
  EXPECT_EQ(lines[5].rfind("orbit 0 ", 0), 0U);
  EXPECT_EQ(lines[FindLine(lines, "spec ")],
            "spec particle_rank 2 rank_J 0 parity 0 delta_Tz 0 hermiticity H");
  EXPECT_EQ(lines[FindLine(lines, "zero_body ")], "zero_body 0.5");
  EXPECT_EQ(lines[lines.size() - 2], "three_body none");
  EXPECT_EQ(lines.back(), "end");
}

TEST(OperatorDump, WriterReferenceOverride) {
  const auto ms = MakeMs();
  std::ostringstream os;
  WriteOperatorDump(
      os, MakeOp(ms, 1, Hermiticity::kHermitian),
      OperatorDumpWriteOptions{.source = "s", .reference = "Custom"});
  EXPECT_NE(os.str().find("\nreference Custom\n"), std::string::npos);
}

TEST(OperatorDump, HeaderAndModelSpaceRoundTrip) {
  const auto ms = MakeMs();
  const std::string text = WriteText(MakeOp(ms, 2, Hermiticity::kHermitian));
  const auto path =
      std::filesystem::temp_directory_path() / "imsrg_dump_header_test.dump";
  {
    std::ofstream f(path);
    f << text;
  }
  const OperatorDumpHeader h = ReadOperatorDumpHeader(path);
  EXPECT_EQ(h.source, "unit test");
  EXPECT_EQ(h.emax, 1);
  EXPECT_EQ(h.reference, "O16");
  EXPECT_EQ(h.orbits.size(), 6U);
  const auto ms2 = ModelSpaceForDump(h);
  EXPECT_EQ(ms2->num_orbits(), ms->num_orbits());
  EXPECT_EQ(ms2->reference().name(), "O16");
  const Operator back = ReadOperatorDump(path, ms2);
  EXPECT_EQ(WriteText(back), text);
  std::filesystem::remove(path);
}

TEST(OperatorDump, ReadsSmallerDumpIntoLargerModelSpace) {
  const auto ms2 = MakeMs(2);
  const auto ms4 = MakeMs(4);
  const Operator small = MakeOp(ms2, 2, Hermiticity::kHermitian);
  const Operator big = ReadText(WriteText(small), ms4);
  const auto& tbc = ms2->two_body();
  auto map = [&](OrbitIndex a) {
    return *ms4->basis().find(ms2->basis().orbit(a));
  };
  int checked = 0;
  for (std::int32_t c = 0; c < tbc.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const auto kets = tbc.channel_kets(ch);
    for (const KetIndex ki : kets) {
      for (const KetIndex kj : kets) {
        const auto bk = tbc.ket(ki);
        const auto kk = tbc.ket(kj);
        const TwoJ J = tbc.channel(ch).two_j;
        EXPECT_EQ(big.two_body().get_normalized(map(bk.p), map(bk.q), map(kk.p),
                                                map(kk.q), J),
                  small.two_body().get_normalized(bk.p, bk.q, kk.p, kk.q, J));
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 100);
  const auto n = ms2->num_orbits();
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < n; ++b) {
      EXPECT_EQ(big.one_body().get(map(OrbitIndex{a}), map(OrbitIndex{b})),
                small.one_body().get(OrbitIndex{a}, OrbitIndex{b}));
    }
  }
  EXPECT_EQ(big.zero_body(), small.zero_body());
}

class DumpErrors : public ::testing::Test {
 protected:
  void SetUp() override {
    ms_ = MakeMs();
    text_ = WriteText(MakeOp(ms_, 2, Hermiticity::kHermitian));
    lines_ = SplitLines(text_);
  }
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::string text_;
  std::vector<std::string> lines_;
};

TEST_F(DumpErrors, BadVersion) {
  lines_[0] = "# imsrg-operator-dump v2";
  ExpectErrorAt(JoinLines(lines_), 1, ms_);
}

TEST_F(DumpErrors, MissingRecord) {
  const auto k = FindLine(lines_, "reference ");
  lines_.erase(lines_.begin() + static_cast<std::ptrdiff_t>(k));
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, MisorderedRecord) {
  const auto k = FindLine(lines_, "basis ");
  std::swap(lines_[k], lines_[k + 1]);
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, WrongCount) {
  const auto k = FindLine(lines_, "one_body ");
  const auto n = std::stoul(lines_[k].substr(9));
  lines_[k] = "one_body " + std::to_string(n - 1);
  // The last one-body entry (the line before "two_body") is read where
  // "two_body" is expected.
  ExpectErrorAt(JoinLines(lines_), FindLine(lines_, "two_body "), ms_);
}

TEST_F(DumpErrors, OrbitIndexOutOfOrder) {
  const auto k = FindLine(lines_, "orbit 1 ");
  lines_[k].replace(6, 1, "2");
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, OrbitNotInBasis) {
  const auto k = FindLine(lines_, "orbit 0 ");
  lines_[k] = "orbit 0 3 0 1 -1 1";
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, HwMismatch) {
  const auto k = FindLine(lines_, "basis ");
  lines_[k] = "basis emax 1 lmax 1 hw 17";
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, OccupationMismatchAndOptOut) {
  const auto k = FindLine(lines_, "orbit 0 ");
  lines_[k] = "orbit 0 0 0 1 -1 0.5";
  const std::string bad = JoinLines(lines_);
  ExpectErrorAt(bad, k + 1, ms_);
  OperatorDumpReadOptions o;
  o.check_occupations = false;
  EXPECT_NO_THROW((void)ReadText(bad, ms_, o));
}

TEST_F(DumpErrors, BadHermiticityLetter) {
  const auto k = FindLine(lines_, "spec ");
  lines_[k].back() = 'X';
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, OneBodyIndexOutOfRange) {
  const auto k = FindLine(lines_, "one_body ");
  lines_[k + 1] = "99 0 1.0";
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, TwoBodyIndexOutOfRange) {
  const auto k = FindLine(lines_, "two_body ");
  lines_[k + 1] = "99 99 0 0 0 0 1.0";
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, PGreaterThanQ) {
  const auto k = FindLine(lines_, "two_body ");
  lines_[k + 1] = "1 0 0 0 0 0 1.0";
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, KetAbsentFromChannel) {
  const auto k = FindLine(lines_, "two_body ");
  lines_[k + 1] = "0 0 0 0 1 1 1.0";  // |a a; J=1> vanishes by antisymmetry
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, BraKetChannelMismatch) {
  const auto k = FindLine(lines_, "two_body ");
  lines_[k + 1] = "0 0 0 1 0 0 1.0";  // Tz = -1 bra against Tz = 0 ket
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, HermiticityViolation) {
  const auto k = FindLine(lines_, "spec ");
  lines_[k].back() = 'A';  // the symmetric data cannot be antihermitian
  ExpectErrorAt(JoinLines(lines_), lines_.size(), ms_);
}

TEST_F(DumpErrors, ThreeBodyEntriesForLowRankAreRejected) {
  const auto k = FindLine(lines_, "three_body");
  lines_[k] = "three_body 1";
  lines_.insert(lines_.begin() + static_cast<std::ptrdiff_t>(k) + 1,
                "0 0 0 1 0 0 0 1 3 1.0");
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST_F(DumpErrors, MissingEnd) {
  lines_.pop_back();
  ExpectErrorAt(JoinLines(lines_), lines_.size() + 1, ms_);
}

TEST_F(DumpErrors, OneBodyEntryBetweenDifferentLjtz) {
  // Orbits 0 (p0s1) and 2 (p0p3): a scalar operator cannot connect them.
  const auto k = FindLine(lines_, "one_body ");
  lines_[k + 1] = "0 2 1.0";
  ExpectErrorAt(JoinLines(lines_), k + 2, ms_);
}

TEST_F(DumpErrors, DuplicateOrbitRecord) {
  const auto k = FindLine(lines_, "orbit 1 ");
  lines_[k] = "orbit 1 0 0 1 -1 1";  // the quantum numbers of orbit 0 again
  ExpectErrorAt(JoinLines(lines_), k + 1, ms_);
}

TEST(OperatorDump, WriterRejectsMalformedSourceAndReference) {
  const auto ms = MakeMs();
  const Operator op = MakeOp(ms, 2, Hermiticity::kHermitian);
  std::ostringstream os;
  OperatorDumpWriteOptions o;
  o.source = "two\nlines";
  EXPECT_THROW(WriteOperatorDump(os, op, o), InputError);
  o.source = "fine";
  o.reference = "two words";
  EXPECT_THROW(WriteOperatorDump(os, op, o), InputError);
}

}  // namespace
}  // namespace imsrg::op
