// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// WriteTensorTokyo: (1) round trip of archived 27Al fork `_2b.snt` files (read
// in place, skipped if absent): parsed into a tensor operator on the sd valence
// space and written back in the 79e9330 format, the output must equal the
// archive byte by byte; (2) a small synthetic tensor in both formats against
// hand-derived lines.
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/basis/model_space.h"
#include "imsrg/io/tensor_writers.h"
#include "imsrg/operator/operator.h"

namespace imsrg {
namespace {

namespace fs = std::filesystem;

std::shared_ptr<const basis::ModelSpace> SdSpace() {
  return basis::ModelSpace::FromNames({.emax = 2, .lmax = 2}, 16.0, "O16",
                                      "sd-shell");
}

OrbitIndex FindOrbit(const basis::ModelSpace& ms, int n, int l, int j2,
                     int tz2) {
  for (int a = 0; a < ms.num_orbits(); ++a) {
    const auto& o = ms.basis().orbit(OrbitIndex{a});
    if (o.n == n && o.l == l && o.j2.twice == j2 && o.tz2.twice == tz2)
      return OrbitIndex{a};
  }
  ADD_FAILURE() << "orbit not found " << n << " " << l << " " << j2 << " "
                << tz2;
  return OrbitIndex{0};
}

std::vector<std::string> Lines(const std::string& text) {
  std::vector<std::string> v;
  std::istringstream is(text);
  for (std::string s; std::getline(is, s);) v.push_back(s);
  return v;
}

std::string Write(const op::Operator& op, const io::TensorTokyoOptions& o) {
  std::ostringstream os;
  io::WriteTensorTokyo(os, op, o);
  return os.str();
}

// ---------------------------------------------------------------- archive ---

fs::path FindArchive(const std::string& stem) {
  const char* work = std::getenv("IMSRG_WORK");
  if (work == nullptr) return {};
  const fs::path root =
      fs::path(work) / "inputs/al27/al27_provenance/original/al27_data";
  std::error_code ec;
  if (!fs::exists(root, ec)) return {};
  const std::string suffix = "INTFILE" + stem + "_2b.snt";
  for (fs::recursive_directory_iterator it(root, ec), end; it != end;
       it.increment(ec)) {
    if (ec) break;
    const std::string name = it->path().filename().string();
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return it->path();
  }
  return {};
}

/// Test-local parser of a fork tensor `.snt` into an operator of rank `rank_J`
/// on the sd model space (parity from the first one-body row). The header
/// values are returned through `opts`.
op::Operator ParseSnt(const std::shared_ptr<const basis::ModelSpace>& ms,
                      const std::string& text, int rank_J,
                      io::TensorTokyoOptions& opts) {
  const auto lines = Lines(text);
  auto after = [&](const std::string& s, const std::string& key) {
    return s.substr(s.find(key) + key.size());
  };
  opts.format = io::TokyoFormat::kFork79e9330;
  opts.version = after(lines[0], "IMSRG version ");
  opts.file_2n = after(lines[1], "input 2N: ");
  opts.file_3n = after(lines[2], "input 3N: ");
  {
    std::istringstream is(lines[3]);
    std::string t;
    is >> t >> t >> opts.e1max >> t >> opts.e2max >> t >> opts.e3max >> t >>
        opts.hbar_omega >> t >> opts.a_ref >> t >> opts.z_ref >> t >>
        opts.a_target;
  }
  std::vector<OrbitIndex> file2orb(1, OrbitIndex{0});
  std::size_t i = 7;  // "   np nn Z N"
  int np = 0, nn = 0, zc = 0, nc = 0;
  std::istringstream(lines[i++]) >> np >> nn >> zc >> nc;
  for (int k = 0; k < np + nn; ++k) {
    int idx, n, l, j2, tz2;
    std::istringstream(lines[i++]) >> idx >> n >> l >> j2 >> tz2;
    file2orb.push_back(FindOrbit(*ms, n, l, j2, tz2));
  }
  ++i;  // "! reduced matrix elements"
  op::OperatorSpec spec;
  spec.rank_J = rank_J;
  int num1 = 0, zero = 0;
  double hw = 0.0;
  std::istringstream(lines[i++]) >> num1 >> zero >> hw;
  {
    int a, b;
    std::istringstream(lines[i]) >> a >> b;
    const int la = ms->basis().orbit(file2orb[static_cast<std::size_t>(a)]).l;
    const int lb = ms->basis().orbit(file2orb[static_cast<std::size_t>(b)]).l;
    spec.parity = ((la + lb) % 2 == 0) ? Parity::kEven : Parity::kOdd;
  }
  op::Operator op(ms, spec);
  for (int r = 0; r < num1; ++r) {
    int a, b;
    double v;
    std::istringstream(lines[i++]) >> a >> b >> v;
    op.one_body().set_element(file2orb[static_cast<std::size_t>(a)],
                              file2orb[static_cast<std::size_t>(b)], v);
  }
  int num2 = 0;
  std::istringstream(lines[i++]) >> num2 >> zero >> hw;
  const auto& tb = ms->two_body();
  for (int r = 0; r < num2; ++r) {
    int a, b, c, d, jb, jk;
    double v;
    std::istringstream(lines[i++]) >> a >> b >> c >> d >> jb >> jk >> v;
    const OrbitIndex p = file2orb[static_cast<std::size_t>(a)];
    const OrbitIndex q = file2orb[static_cast<std::size_t>(b)];
    const OrbitIndex s = file2orb[static_cast<std::size_t>(c)];
    const OrbitIndex t = file2orb[static_cast<std::size_t>(d)];
    EXPECT_LE(p.value(), q.value());
    EXPECT_LE(s.value(), t.value());
    const ChannelIndex chb = tb.channel_of(p, q, TwoJ{2 * jb});
    const ChannelIndex chk = tb.channel_of(s, t, TwoJ{2 * jk});
    const auto ib = tb.local_index(chb, tb.ket_index(p, q));
    const auto ik = tb.local_index(chk, tb.ket_index(s, t));
    EXPECT_TRUE(ib && ik);
    if (!ib || !ik) continue;
    op.two_body().set(chb, chk, *ib, *ik, v);
  }
  return op;
}

void CheckArchive(const std::string& stem, int rank_J) {
  const fs::path path = FindArchive(stem);
  if (path.empty()) GTEST_SKIP() << "archived file for " << stem << " absent";
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string want = ss.str();
  const auto ms = SdSpace();
  io::TensorTokyoOptions opts;
  const op::Operator op = ParseSnt(ms, want, rank_J, opts);
  const std::string got = Write(op, opts);
  const auto g = Lines(got);
  const auto w = Lines(want);
  ASSERT_EQ(g.size(), w.size());
  for (std::size_t k = 0; k < g.size(); ++k)
    EXPECT_EQ(g[k], w[k]) << stem << " line " << k + 1;
  EXPECT_EQ(got, want);
}

TEST(TensorTokyoArchive, FSigmap3pRoundTrip) {
  CheckArchive("FSigmap3p_780", 3);
}
TEST(TensorTokyoArchive, FDelta1pRoundTrip) { CheckArchive("FDelta1p_135", 1); }
TEST(TensorTokyoArchive, FM2nRoundTrip) { CheckArchive("FM2n_135", 2); }

// -------------------------------------------------------------- synthetic ---

struct Synth {
  std::shared_ptr<const basis::ModelSpace> ms = SdSpace();
  OrbitIndex pd5 = FindOrbit(*ms, 0, 2, 5, -1);
  OrbitIndex pd3 = FindOrbit(*ms, 0, 2, 3, -1);
  op::Operator op;
  Synth() {
    op::OperatorSpec spec;
    spec.rank_J = 2;
    op = op::Operator(ms, spec);
    // <d5||O||d3> = 0.5 and its Hermitian partner (-1)^((5-3)/2) = -1;
    // d5 d5 = 2.0e-7 (above the fork's absolute threshold only).
    op.one_body().set_element(pd5, pd3, 0.5);
    op.one_body().set_element(pd3, pd5, -0.5);
    op.one_body().set_element(pd5, pd5, 2.0e-7);
    const auto& tb = ms->two_body();
    const ChannelIndex c0 = tb.channel_of(pd5, pd5, TwoJ{0});
    const ChannelIndex c2 = tb.channel_of(pd5, pd5, TwoJ{4});
    const auto i0 = tb.local_index(c0, tb.ket_index(pd5, pd5));
    const auto i2 = tb.local_index(c2, tb.ket_index(pd5, pd5));
    // <d5 d5; J=0||O||d5 d5; J=2> = 0.25 (stored with J_bra < J_ket).
    op.two_body().set(c0, c2, *i0, *i2, 0.25);
    // Below the fork's absolute threshold, above devel's 1e-9.
    const ChannelIndex c3 = tb.channel_of(pd5, pd3, TwoJ{2});
    const ChannelIndex c4 = tb.channel_of(pd5, pd3, TwoJ{4});
    const auto i3 = tb.local_index(c3, tb.ket_index(pd5, pd3));
    const auto i4 = tb.local_index(c4, tb.ket_index(pd5, pd3));
    op.two_body().set(c3, c4, *i3, *i4, 5.0e-8);
    op.SyncToDevice();
  }
};

io::TensorTokyoOptions SynthOptions(io::TokyoFormat f) {
  io::TensorTokyoOptions o;
  o.format = f;
  o.file_2n = "/some/dir/nn.me2j.gz";
  o.file_3n = "none";
  o.e1max = 2;
  o.e2max = 4;
  o.e3max = 6;
  o.hbar_omega = 16.0;
  o.a_ref = 16;
  o.z_ref = 8;
  o.a_target = 16.0;
  o.version = "test";
  return o;
}

TEST(TensorTokyoSynthetic, ForkFormat) {
  Synth s;
  const auto lines =
      Lines(Write(s.op, SynthOptions(io::TokyoFormat::kFork79e9330)));
  ASSERT_EQ(lines.size(), 21u);
  EXPECT_EQ(lines[0],
            "! shell model effective operator generated by IMSRG version test");
  EXPECT_EQ(lines[1], "! input 2N: nn.me2j.gz");
  EXPECT_EQ(lines[3],
            "! e1max: 2  e2max: 4   e3max: 6   hw: 16   Aref: 16  Zref: 8  "
            "A_for_kinetic_energy: 16");
  EXPECT_EQ(lines[4], "! Zero body term: 0");
  EXPECT_EQ(lines[7], "   3   3   8   8");
  EXPECT_EQ(lines[8], "   1   0   2   5  -1");
  EXPECT_EQ(lines[13], "   6   1   0   1   1");
  EXPECT_EQ(lines[14], "! reduced matrix elements");
  // One-body: ordered valence pairs, here (d5,d5), (d5,d3), (d3,d5): a blank
  // of width 12, then the scientific number.
  EXPECT_EQ(lines[15], "3 0 16");
  EXPECT_EQ(lines[16], "   1   1            2.000000e-07");
  EXPECT_EQ(lines[17], "   1   2            5.000000e-01");
  EXPECT_EQ(lines[18], "   2   1            -5.000000e-01");
  // The scientific flag is sticky: hw in the two-body count line; the element
  // of 5e-8 is dropped; only the J_bra <= J_ket element is written.
  EXPECT_EQ(lines[19], "1 0 1.600000e+01");
  EXPECT_EQ(lines[20], "   1    1    1    1      0    2   2.500000e-01");
}

TEST(TensorTokyoSynthetic, DevelFormat) {
  Synth s;
  const auto lines = Lines(Write(s.op, SynthOptions(io::TokyoFormat::kDevel)));
  ASSERT_EQ(lines.size(), 21u);
  // Relative one-body threshold: |2e-7| < 1e-7 ||V1|| drops (d5,d5).
  EXPECT_EQ(lines[15], "2 0 16");
  EXPECT_EQ(lines[16], "   1    2   0.5000000000000000");
  EXPECT_EQ(lines[17], "   2    1   -0.5000000000000000");
  // hw in the sticky fixed format; the 5e-8 element survives (> 1e-9).
  EXPECT_EQ(lines[18], "2 0 16.0000000000000000");
  const std::string x = "   1    1    1    1      0    2   0.2500000000000000";
  const std::string y = "   1    2    1    2      1    2   0.0000000500000000";
  EXPECT_TRUE((lines[19] == x && lines[20] == y) ||
              (lines[19] == y && lines[20] == x))
      << lines[19] << "\n"
      << lines[20];
}

TEST(TensorTokyoSynthetic, ForkNoOneBodyUsesDefaultFormat) {
  // Without one-body rows the sticky scientific flag is never set: the
  // two-body rows use the default format with 6 significant digits.
  Synth s;
  s.op.one_body().set_zero();
  const auto lines =
      Lines(Write(s.op, SynthOptions(io::TokyoFormat::kFork79e9330)));
  ASSERT_EQ(lines.size(), 18u);
  EXPECT_EQ(lines[15], "0 0 16");
  EXPECT_EQ(lines[16], "1 0 16");
  EXPECT_EQ(lines[17], "   1    1    1    1      0    2           0.25");
}

TEST(TensorTokyoSynthetic, ScalarThrows) {
  const auto ms = SdSpace();
  const op::Operator op(ms, op::OperatorSpec::Scalar());
  std::ostringstream os;
  EXPECT_THROW(io::WriteTensorTokyo(os, op, {}), InputError);
}

}  // namespace
}  // namespace imsrg
