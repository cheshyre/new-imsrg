// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Tests of the three-body section of the OperatorDump v1 reader and writer:
// bit-identical round trips for H, A and N parts and the reader errors.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/operator/operator.h"
#include "imsrg/operator/operator_dump.h"

namespace imsrg::op {
namespace {

using basis::KeptRow3Index;

std::shared_ptr<const basis::ModelSpace> He4Space() {
  return basis::ModelSpace::FromNames(basis::BasisTruncation{.emax = 2}, 16.0,
                                      "He4", "He4")
      ->WithThreeBody(basis::ThreeBodyCuts{.e3max = 4});
}

OperatorSpec Spec3(Hermiticity h) {
  OperatorSpec s = OperatorSpec::Scalar(3);
  s.hermiticity = h;
  return s;
}

/// Pseudo-random values (a different irrational-looking number per element);
/// every 5th element is zero to exercise the nonzero-only filter.
double Val(std::int32_t ch, std::int64_t i, std::int64_t j) {
  const std::int64_t k = (ch * 7919 + i * 104729 + j * 1299709) % 1000003;
  if (k % 5 == 0) return 0.0;
  return static_cast<double>(k) / 999983.0 - 0.37;
}

Operator RandomThreeBody(const std::shared_ptr<const basis::ModelSpace>& ms,
                         Hermiticity h) {
  Operator op(ms, Spec3(h));
  ThreeBody& t = *op.three_body();
  for (std::int32_t c = 0; c < t.num_channels(); ++c) {
    const std::int64_t n = t.dim(Channel3Index{c});
    for (std::int64_t j = 0; j < n; ++j) {
      for (std::int64_t i = t.packed() ? j : 0; i < n; ++i) {
        const double v =
            (i == j && h == Hermiticity::kAntiHermitian) ? 0.0 : Val(c, i, j);
        t.set(Channel3Index{c}, KeptRow3Index{static_cast<std::int32_t>(i)},
              KeptRow3Index{static_cast<std::int32_t>(j)}, v);
      }
    }
  }
  op.zero_body() = 0.25;
  op.SyncToDevice();
  return op;
}

std::string WriteText(const Operator& op) {
  std::ostringstream os;
  WriteOperatorDump(os, op, OperatorDumpWriteOptions{.source = "unit test"});
  return os.str();
}

Operator ReadText(const std::string& text,
                  const std::shared_ptr<const basis::ModelSpace>& ms) {
  std::istringstream is(text);
  return ReadOperatorDump(is, "mem", ms);
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

std::size_t FindLine(const std::vector<std::string>& v,
                     const std::string& prefix) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i].rfind(prefix, 0) == 0) return i;
  }
  ADD_FAILURE() << "no line starting with " << prefix;
  return 0;
}

void ExpectErrorAt(const std::string& text, std::size_t line,
                   const std::shared_ptr<const basis::ModelSpace>& ms) {
  try {
    (void)ReadText(text, ms);
    ADD_FAILURE() << "expected InputError at line " << line;
  } catch (const InputError& e) {
    EXPECT_EQ(
        std::string(e.what()).rfind("mem:" + std::to_string(line) + ":", 0), 0u)
        << e.what();
  }
}

class ThreeBodyDump : public ::testing::TestWithParam<Hermiticity> {};

TEST_P(ThreeBodyDump, RoundTripIsBitIdentical) {
  const auto ms = He4Space();
  const Operator op = RandomThreeBody(ms, GetParam());
  const std::string text = WriteText(op);
  const Operator back = ReadText(text, ms);
  ASSERT_TRUE(back.three_body().has_value());
  EXPECT_EQ(back.hermiticity(), GetParam());
  EXPECT_EQ(back.three_body()->size(), op.three_body()->size());
  EXPECT_GT(ImsrgppNorm(*back.three_body()), 0.0);
  op.SyncToHost();
  back.SyncToHost();
  const auto a = op.three_body()->host_span();
  const auto b = back.three_body()->host_span();
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t k = 0; k < a.size(); ++k) EXPECT_EQ(a[k], b[k]) << k;
  // Writing again gives the same text.
  EXPECT_EQ(WriteText(back), text);
}

INSTANTIATE_TEST_SUITE_P(Hermiticities, ThreeBodyDump,
                         ::testing::Values(Hermiticity::kHermitian,
                                           Hermiticity::kAntiHermitian,
                                           Hermiticity::kNonHermitian));

TEST(ThreeBodyDumpFormat, SectionLayout) {
  const auto ms = He4Space();
  const Operator op = RandomThreeBody(ms, Hermiticity::kHermitian);
  const auto lines = SplitLines(WriteText(op));
  const auto k = FindLine(lines, "three_body ");
  const std::size_t count = std::stoul(lines[k].substr(11));
  ASSERT_GT(count, 0u);
  EXPECT_EQ(lines[k + count + 1], "end");
  // 10 fields, canonical orbits, odd 2J.
  std::istringstream is(lines[k + 1]);
  std::int64_t f[9];
  for (auto& x : f) is >> x;
  EXPECT_LE(f[0], f[1]);
  EXPECT_LE(f[1], f[2]);
  EXPECT_LE(f[4], f[5]);
  EXPECT_LE(f[5], f[6]);
  EXPECT_EQ(f[8] % 2, 1);
}

TEST(ThreeBodyDumpFormat, NoneForLowRankAndEmptyRank3) {
  const auto ms = He4Space();
  Operator two(ms, OperatorSpec::Scalar(2));
  EXPECT_NE(WriteText(two).find("three_body none\n"), std::string::npos);
  const Operator back = ReadText(WriteText(two), ms);
  EXPECT_FALSE(back.three_body().has_value());
  // A zero rank-3 operator writes an empty section and reads back as rank 3.
  Operator zero(ms, Spec3(Hermiticity::kHermitian));
  const std::string text = WriteText(zero);
  EXPECT_NE(text.find("three_body 0\n"), std::string::npos);
  EXPECT_TRUE(ReadText(text, ms).three_body().has_value());
  // "none" is accepted for a rank-3 spec (zero part).
  auto lines = SplitLines(text);
  lines[FindLine(lines, "three_body")] = "three_body none";
  EXPECT_TRUE(ReadText(JoinLines(lines), ms).three_body().has_value());
}

class ThreeBodyDumpErrors : public ::testing::Test {
 protected:
  void SetUp() override {
    ms_ = He4Space();
    lines_ =
        SplitLines(WriteText(RandomThreeBody(ms_, Hermiticity::kHermitian)));
    k_ = FindLine(lines_, "three_body ");
  }
  std::shared_ptr<const basis::ModelSpace> ms_;
  std::vector<std::string> lines_;
  std::size_t k_ = 0;
};

TEST_F(ThreeBodyDumpErrors, NonCanonicalKet) {
  // Swap the first two orbits of the bra of a record with a != b.
  const std::size_t count = std::stoul(lines_[k_].substr(11));
  for (std::size_t r = 1; r <= count; ++r) {
    std::istringstream is(lines_[k_ + r]);
    std::string a, b, rest, tok;
    is >> a >> b;
    if (a == b) continue;
    while (is >> tok) rest += " " + tok;
    lines_[k_ + r] = b + " " + a + rest;
    ExpectErrorAt(JoinLines(lines_), k_ + r + 1, ms_);
    return;
  }
  FAIL() << "no record with a != b";
}

TEST_F(ThreeBodyDumpErrors, KetNotInSpace) {
  lines_[k_ + 1] = "0 0 0 0 0 0 0 0 1 1.0";  // a a a with Jab = 0: Pauli
  ExpectErrorAt(JoinLines(lines_), k_ + 2, ms_);
}

TEST_F(ThreeBodyDumpErrors, InconsistentChannel) {
  // 2J even.
  lines_[k_ + 1] = "0 0 0 0 0 0 0 0 2 1.0";
  ExpectErrorAt(JoinLines(lines_), k_ + 2, ms_);
}

TEST_F(ThreeBodyDumpErrors, BraKetInDifferentChannels) {
  // Orbits 0 (p proton s1/2) and 1 (neutron): 2Tz differs between bra and ket
  // for these labels (0 0 1 Jab=0 vs 0 1 1 Jab=0).
  lines_[k_ + 1] = "0 0 1 0 0 1 1 0 1 1.0";
  ExpectErrorAt(JoinLines(lines_), k_ + 2, ms_);
}

TEST_F(ThreeBodyDumpErrors, HermiticityViolation) {
  // Make the declared hermiticity antihermitian: the symmetric data fail.
  const auto s = FindLine(lines_, "spec ");
  lines_[s].back() = 'A';
  ExpectErrorAt(JoinLines(lines_), lines_.size(), ms_);
}

TEST_F(ThreeBodyDumpErrors, MissingMirrorRecord) {
  // Drop one off-diagonal record: its mirror remains, violating hermiticity.
  const std::size_t count = std::stoul(lines_[k_].substr(11));
  std::size_t victim = 0;
  for (std::size_t r = 1; r <= count; ++r) {
    std::istringstream is(lines_[k_ + r]);
    std::string f[10];
    for (auto& x : f) is >> x;
    if (f[0] != f[4] || f[1] != f[5] || f[2] != f[6] || f[3] != f[7]) {
      victim = r;
      break;
    }
  }
  ASSERT_GT(victim, 0u);
  lines_.erase(lines_.begin() + static_cast<std::ptrdiff_t>(k_ + victim));
  lines_[k_] = "three_body " + std::to_string(count - 1);
  ExpectErrorAt(JoinLines(lines_), lines_.size(), ms_);
}

TEST_F(ThreeBodyDumpErrors, NoThreeBodySpaceInModelSpace) {
  const auto no3 = basis::ModelSpace::FromNames(
      basis::BasisTruncation{.emax = 2}, 16.0, "He4", "He4");
  const auto s = FindLine(lines_, "spec ");
  ExpectErrorAt(JoinLines(lines_), s + 1, no3);
}

TEST_F(ThreeBodyDumpErrors, EntriesForRank2) {
  const auto s = FindLine(lines_, "spec ");
  lines_[s] = "spec particle_rank 2 rank_J 0 parity 0 delta_Tz 0 hermiticity H";
  ExpectErrorAt(JoinLines(lines_), k_ + 1, ms_);
}

}  // namespace
}  // namespace imsrg::op
