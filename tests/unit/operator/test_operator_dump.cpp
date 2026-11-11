// Tests of the OperatorDump v1 reader and writer: round trips and one test per reader error.
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.hpp"
#include "imsrg/basis/model_space.hpp"
#include "imsrg/operator/compare.hpp"
#include "imsrg/operator/operator_dump.hpp"

namespace {

using namespace imsrg;
using namespace imsrg::op;

std::shared_ptr<const basis::ModelSpace> make_ms(int emax = 1) {
  return basis::ModelSpace::from_names(basis::BasisTruncation{.emax = emax}, 16.0, "O16", "O16");
}

/// Deterministic pseudo-value for index pair (i, j) in channel ch, never zero.
double val(std::size_t ch, std::size_t i, std::size_t j) {
  return 0.25 + 0.125 * static_cast<double>(ch) + 0.03125 * static_cast<double>(i) -
         0.0625 * static_cast<double>(j) + 1.0 / 3.0;
}

/// Fills `op` with values obeying its hermiticity (H: symmetric, A: antisymmetric with zero
/// diagonal, N: no relation) and the scalar one-body selection rule, written with explicit raw
/// sets.
void fill(Operator& op) {
  const auto sgn = hermiticity_sign(op.spec().hermiticity);
  const auto& ms = op.model_space();
  op.zero_body() = 0.5;
  const auto n = static_cast<std::size_t>(ms.num_orbits());
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      double v = val(7, a, b);
      if (sgn != 0) v = a <= b ? val(7, a, b) : sgn * val(7, b, a);
      if (sgn == -1 && a == b) v = 0.0;
      if (a == 1 && b == 4) v = 0.0;  // exercise the nonzero-only filter
      if (sgn != 0 && a == 4 && b == 1) v = 0.0;
      // Scalar selection rule: only orbits with equal (l, 2j, tz2) are connected.
      const auto& oa = ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(a)});
      const auto& ob = ms.basis().orbit(OrbitIndex{static_cast<std::int32_t>(b)});
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
        double v = val(static_cast<std::size_t>(c), i, j);
        if (sgn != 0)
          v = i <= j ? val(static_cast<std::size_t>(c), i, j)
                     : sgn * val(static_cast<std::size_t>(c), j, i);
        if (sgn == -1 && i == j) v = 0.0;
        op.two_body().set(ch, ch, LocalKetIndex{static_cast<std::int32_t>(i)},
                          LocalKetIndex{static_cast<std::int32_t>(j)}, v);
      }
    }
  }
}

Operator make_op(const std::shared_ptr<const basis::ModelSpace>& ms, int rank, Hermiticity h) {
  Operator op(ms, OperatorSpec::scalar(rank, h));
  fill(op);
  op.sync_to_device();
  return op;
}

std::string write_text(const Operator& op) {
  std::ostringstream os;
  write_operator_dump(os, op, OperatorDumpWriteOptions{.source = "unit test"});
  return os.str();
}

Operator read_text(const std::string& text, const std::shared_ptr<const basis::ModelSpace>& ms,
                   const OperatorDumpReadOptions& o = {}) {
  std::istringstream is(text);
  return read_operator_dump(is, "mem", ms, o);
}

std::vector<std::string> split_lines(const std::string& t) {
  std::vector<std::string> out;
  std::istringstream is(t);
  std::string l;
  while (std::getline(is, l)) out.push_back(l);
  return out;
}

std::string join_lines(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& l : v) s += l + "\n";
  return s;
}

/// Index (0-based) of the first line starting with `prefix`.
std::size_t find_line(const std::vector<std::string>& v, const std::string& prefix) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i].rfind(prefix, 0) == 0) return i;
  }
  ADD_FAILURE() << "no line starting with " << prefix;
  return 0;
}

/// Expects an InputError whose message starts with "mem:<line>:".
void expect_error_at(const std::string& text, std::size_t line,
                     const std::shared_ptr<const basis::ModelSpace>& ms,
                     const OperatorDumpReadOptions& o = {}) {
  try {
    (void)read_text(text, ms, o);
    ADD_FAILURE() << "expected InputError at line " << line;
  } catch (const InputError& e) {
    const std::string want = "mem:" + std::to_string(line) + ":";
    EXPECT_EQ(std::string(e.what()).rfind(want, 0), 0U) << e.what() << " (wanted " << want << ")";
  }
}

class DumpRoundTrip : public ::testing::TestWithParam<std::pair<Hermiticity, int>> {};

TEST_P(DumpRoundTrip, WriterReaderIdentity) {
  const auto ms = make_ms();
  const auto [h, rank] = GetParam();
  const Operator op = make_op(ms, rank, h);
  const std::string text = write_text(op);
  const Operator back = read_text(text, ms);
  EXPECT_EQ(max_abs_difference(op, back).max(), 0.0);
  EXPECT_EQ(back.spec().hermiticity, h);
  EXPECT_EQ(back.spec().particle_rank, rank);
  EXPECT_EQ(write_text(back), text);
}

INSTANTIATE_TEST_SUITE_P(Variants, DumpRoundTrip,
                         ::testing::Values(std::pair{Hermiticity::Hermitian, 2},
                                           std::pair{Hermiticity::AntiHermitian, 2},
                                           std::pair{Hermiticity::NonHermitian, 2},
                                           std::pair{Hermiticity::Hermitian, 1}));

TEST(OperatorDump, WriterRecordsAndFormat) {
  const auto ms = make_ms();
  const auto lines = split_lines(write_text(make_op(ms, 2, Hermiticity::Hermitian)));
  EXPECT_EQ(lines.front(), "# imsrg-operator-dump v1");
  EXPECT_EQ(lines[1], "source unit test");
  EXPECT_EQ(lines[2], "basis emax 1 lmax 1 hw 16");
  EXPECT_EQ(lines[3], "reference O16");
  EXPECT_EQ(lines[4], "norbits 6");
  EXPECT_EQ(lines[5].rfind("orbit 0 ", 0), 0U);
  EXPECT_EQ(lines[find_line(lines, "spec ")],
            "spec particle_rank 2 rank_J 0 parity 0 delta_Tz 0 hermiticity H");
  EXPECT_EQ(lines[find_line(lines, "zero_body ")], "zero_body 0.5");
  EXPECT_EQ(lines[lines.size() - 2], "three_body none");
  EXPECT_EQ(lines.back(), "end");
}

TEST(OperatorDump, WriterReferenceOverride) {
  const auto ms = make_ms();
  std::ostringstream os;
  write_operator_dump(os, make_op(ms, 1, Hermiticity::Hermitian),
                      OperatorDumpWriteOptions{.source = "s", .reference = "Custom"});
  EXPECT_NE(os.str().find("\nreference Custom\n"), std::string::npos);
}

TEST(OperatorDump, HeaderAndModelSpaceRoundTrip) {
  const auto ms = make_ms();
  const std::string text = write_text(make_op(ms, 2, Hermiticity::Hermitian));
  const auto path = std::filesystem::temp_directory_path() / "imsrg_dump_header_test.dump";
  {
    std::ofstream f(path);
    f << text;
  }
  const OperatorDumpHeader h = read_operator_dump_header(path);
  EXPECT_EQ(h.source, "unit test");
  EXPECT_EQ(h.emax, 1);
  EXPECT_EQ(h.reference, "O16");
  EXPECT_EQ(h.orbits.size(), 6U);
  const auto ms2 = model_space_for_dump(h);
  EXPECT_EQ(ms2->num_orbits(), ms->num_orbits());
  EXPECT_EQ(ms2->reference().name(), "O16");
  const Operator back = read_operator_dump(path, ms2);
  EXPECT_EQ(write_text(back), text);
  std::filesystem::remove(path);
}

TEST(OperatorDump, ReadsSmallerDumpIntoLargerModelSpace) {
  const auto ms2 = make_ms(2);
  const auto ms4 = make_ms(4);
  const Operator small = make_op(ms2, 2, Hermiticity::Hermitian);
  const Operator big = read_text(write_text(small), ms4);
  const auto& tbc = ms2->two_body();
  auto map = [&](OrbitIndex a) { return *ms4->basis().find(ms2->basis().orbit(a)); };
  int checked = 0;
  for (std::int32_t c = 0; c < tbc.num_channels(); ++c) {
    const ChannelIndex ch{c};
    const auto kets = tbc.channel_kets(ch);
    for (const KetIndex ki : kets) {
      for (const KetIndex kj : kets) {
        const auto bk = tbc.ket(ki);
        const auto kk = tbc.ket(kj);
        const TwoJ J = tbc.channel(ch).two_j;
        EXPECT_EQ(big.two_body().get_normalized(map(bk.p), map(bk.q), map(kk.p), map(kk.q), J),
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
    ms = make_ms();
    text = write_text(make_op(ms, 2, Hermiticity::Hermitian));
    lines = split_lines(text);
  }
  std::shared_ptr<const basis::ModelSpace> ms;
  std::string text;
  std::vector<std::string> lines;
};

TEST_F(DumpErrors, BadVersion) {
  lines[0] = "# imsrg-operator-dump v2";
  expect_error_at(join_lines(lines), 1, ms);
}

TEST_F(DumpErrors, MissingRecord) {
  const auto k = find_line(lines, "reference ");
  lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(k));
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, MisorderedRecord) {
  const auto k = find_line(lines, "basis ");
  std::swap(lines[k], lines[k + 1]);
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, WrongCount) {
  const auto k = find_line(lines, "one_body ");
  const auto n = std::stoul(lines[k].substr(9));
  lines[k] = "one_body " + std::to_string(n - 1);
  // The last one-body entry (the line before "two_body") is read where "two_body" is expected.
  expect_error_at(join_lines(lines), find_line(lines, "two_body "), ms);
}

TEST_F(DumpErrors, OrbitIndexOutOfOrder) {
  const auto k = find_line(lines, "orbit 1 ");
  lines[k].replace(6, 1, "2");
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, OrbitNotInBasis) {
  const auto k = find_line(lines, "orbit 0 ");
  lines[k] = "orbit 0 3 0 1 -1 1";
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, HwMismatch) {
  const auto k = find_line(lines, "basis ");
  lines[k] = "basis emax 1 lmax 1 hw 17";
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, OccupationMismatchAndOptOut) {
  const auto k = find_line(lines, "orbit 0 ");
  lines[k] = "orbit 0 0 0 1 -1 0.5";
  const std::string bad = join_lines(lines);
  expect_error_at(bad, k + 1, ms);
  OperatorDumpReadOptions o;
  o.check_occupations = false;
  EXPECT_NO_THROW((void)read_text(bad, ms, o));
}

TEST_F(DumpErrors, BadHermiticityLetter) {
  const auto k = find_line(lines, "spec ");
  lines[k].back() = 'X';
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST_F(DumpErrors, OneBodyIndexOutOfRange) {
  const auto k = find_line(lines, "one_body ");
  lines[k + 1] = "99 0 1.0";
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, TwoBodyIndexOutOfRange) {
  const auto k = find_line(lines, "two_body ");
  lines[k + 1] = "99 99 0 0 0 0 1.0";
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, PGreaterThanQ) {
  const auto k = find_line(lines, "two_body ");
  lines[k + 1] = "1 0 0 0 0 0 1.0";
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, KetAbsentFromChannel) {
  const auto k = find_line(lines, "two_body ");
  lines[k + 1] = "0 0 0 0 1 1 1.0";  // |a a; J=1> vanishes by antisymmetry
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, BraKetChannelMismatch) {
  const auto k = find_line(lines, "two_body ");
  lines[k + 1] = "0 0 0 1 0 0 1.0";  // Tz = -1 bra against Tz = 0 ket
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, HermiticityViolation) {
  const auto k = find_line(lines, "spec ");
  lines[k].back() = 'A';  // the symmetric data cannot be antihermitian
  expect_error_at(join_lines(lines), lines.size(), ms);
}

TEST_F(DumpErrors, ThreeBodyNotImplemented) {
  lines[find_line(lines, "three_body")] = "three_body 5";
  EXPECT_THROW((void)read_text(join_lines(lines), ms), NotImplementedError);
}

TEST_F(DumpErrors, MissingEnd) {
  lines.pop_back();
  expect_error_at(join_lines(lines), lines.size() + 1, ms);
}

TEST_F(DumpErrors, OneBodyEntryBetweenDifferentLjtz) {
  // Orbits 0 (p0s1) and 2 (p0p3): a scalar operator cannot connect them.
  const auto k = find_line(lines, "one_body ");
  lines[k + 1] = "0 2 1.0";
  expect_error_at(join_lines(lines), k + 2, ms);
}

TEST_F(DumpErrors, DuplicateOrbitRecord) {
  const auto k = find_line(lines, "orbit 1 ");
  lines[k] = "orbit 1 0 0 1 -1 1";  // the quantum numbers of orbit 0 again
  expect_error_at(join_lines(lines), k + 1, ms);
}

TEST(OperatorDump, WriterRejectsMalformedSourceAndReference) {
  const auto ms = make_ms();
  const Operator op = make_op(ms, 2, Hermiticity::Hermitian);
  std::ostringstream os;
  OperatorDumpWriteOptions o;
  o.source = "two\nlines";
  EXPECT_THROW(write_operator_dump(os, op, o), InputError);
  o.source = "fine";
  o.reference = "two words";
  EXPECT_THROW(write_operator_dump(os, op, o), InputError);
}

}  // namespace
