// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// NO2B reader: value counts, text/gzip/float/half equivalence on a synthetic
// stream, validation.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/base/half.h"
#include "imsrg/io/input_stream.h"
#include "imsrg/io/no2b.h"
#include "imsrg/operator/no2b.h"

namespace imsrg::io {
namespace {

namespace fs = std::filesystem;

/// A half-exact deterministic value: multiples of 1/64 in [-1.6, 1.6], with
/// about 1% zeros.
float SyntheticValue(std::int64_t k) {
  const std::int64_t m = (k * 37 + k / 7) % 201 - 100;
  return static_cast<float>(m) / 64.0F;
}

fs::path TempDir() {
  // Per-process directory: ctest runs every test in its own process, in
  // parallel.
  const fs::path d = fs::temp_directory_path() /
                     ("imsrg_io_no2b_test_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return d;
}

constexpr No2bFileTruncation kFile{3, 6, 9, -1};
constexpr op::No2bTruncation kContainer{2, 4, 6, 2};

std::string TextBody(std::int64_t count) {
  std::string s = "header line of the text file\n";
  char buf[64];
  for (std::int64_t k = 0; k < count; ++k) {
    std::snprintf(buf, sizeof buf, "%.9g%c",
                  static_cast<double>(SyntheticValue(k)),
                  k % 5 == 4 ? '\n' : ' ');
    s += buf;
  }
  return s;
}

void WriteBinary(const fs::path& p, std::int64_t count, bool half) {
  std::ofstream out(p, std::ios::binary);
  for (std::int64_t k = 0; k < count; ++k) {
    const float v = SyntheticValue(k);
    if (half) {
      const std::uint16_t h = FloatToHalf(v);
      out.write(reinterpret_cast<const char*>(&h), sizeof h);
    } else {
      out.write(reinterpret_cast<const char*>(&v), sizeof v);
    }
  }
}

TEST(No2bCount, ReferenceFile) {
  EXPECT_EQ(No2bValueCount({6, 12, 18, -1}), 32685840);
  EXPECT_EQ(No2bValueCount({6, 12, 18, 6}),
            32685840);  // lmax = e1max explicitly
}

TEST(No2bCount, SmallCases) {
  // e1max = 0: one orbit (s1/2); a=b=c=d=e=f, J = 0 and 1, five isospin values
  // each.
  EXPECT_EQ(No2bValueCount({0, 0, 0, -1}), 10);
  // Monotonic in every truncation.
  EXPECT_LT(No2bValueCount({2, 4, 6, -1}), No2bValueCount({3, 4, 6, -1}));
  EXPECT_LT(No2bValueCount({2, 4, 5, -1}), No2bValueCount({2, 4, 6, -1}));
  EXPECT_LT(No2bValueCount({2, 3, 6, -1}), No2bValueCount({2, 4, 6, -1}));
  EXPECT_LT(No2bValueCount({3, 6, 9, 1}), No2bValueCount({3, 6, 9, -1}));
}

TEST(No2bCount, NegativeTruncationThrows) {
  EXPECT_THROW(No2bValueCount({-1, 0, 0, -1}), InputError);
}

TEST(No2bRead, TextGzipFloatAndHalfAgree) {
  const fs::path dir = TempDir();
  const std::int64_t count = No2bValueCount(kFile);
  ASSERT_GT(count, 1000);

  const fs::path ptxt = dir / "synthetic.txt";
  {
    std::ofstream(ptxt) << TextBody(count);
  }
  const fs::path pgz = dir / "synthetic.txt.gz";
  {
    GzipWriter w(pgz);
    w.Write(TextBody(count));
    w.Close();
  }
  const fs::path pf = dir / "synthetic_f32_stream.bin";
  const fs::path ph = dir / "synthetic_f16_stream.bin";
  WriteBinary(pf, count, false);
  WriteBinary(ph, count, true);

  op::No2bThreeBody a(kContainer), b(kContainer), c(kContainer), d(kContainer);
  No2bReadReport ra, rb, rc, rd;
  ReadNo2b(ptxt, {.file = kFile}, a, &ra);
  ReadNo2b(pgz, {.file = kFile}, b, &rb);
  ReadNo2b(pf, {.file = kFile}, c, &rc);
  ReadNo2b(ph, {.file = kFile, .precision = No2bPrecision::kHalf}, d, &rd);

  ASSERT_GT(a.size(), 0U);
  EXPECT_GT(ra.values_stored, 100);
  EXPECT_LT(ra.values_read,
            count);  // only the prefix needed by the container emax
  EXPECT_GT(ra.values_read, 0);
  for (const auto* r : {&rb, &rc, &rd}) {
    EXPECT_EQ(r->values_read, ra.values_read);
    EXPECT_EQ(r->values_stored, ra.values_stored);
  }
  for (const auto* v : {&b, &c, &d}) {
    EXPECT_EQ(v->size(), a.size());
    EXPECT_EQ(v->norm(), a.norm());
  }
  EXPECT_GT(a.norm(), 0.0);

  // Element-wise: every orbit combination of a few channels must agree.
  const std::int32_t norb =
      static_cast<std::int32_t>(a.isospin_orbits().size());
  for (std::int32_t i = 0; i < norb; ++i) {
    for (std::int32_t k = 0; k <= i; ++k) {
      for (std::int32_t J = 0; J <= 3; ++J) {
        const double x = a.get_isospin(i, k, 0, 1, i, k, 0, 1, J, 1);
        EXPECT_EQ(x, b.get_isospin(i, k, 0, 1, i, k, 0, 1, J, 1));
        EXPECT_EQ(x, c.get_isospin(i, k, 0, 1, i, k, 0, 1, J, 1));
        EXPECT_EQ(x, d.get_isospin(i, k, 0, 1, i, k, 0, 1, J, 1));
      }
    }
  }
}

TEST(No2bRead, WholeFileWhenContainerMatchesFile) {
  const fs::path dir = TempDir();
  const No2bFileTruncation f{2, 4, 6, -1};
  const std::int64_t count = No2bValueCount(f);
  const fs::path pf = dir / "whole_stream.bin";
  WriteBinary(pf, count, false);
  op::No2bThreeBody v(op::No2bTruncation{2, 4, 6, 2});
  No2bReadReport r;
  ReadNo2b(pf, {.file = f}, v, &r);
  EXPECT_EQ(r.values_read, count);
  EXPECT_GT(r.seconds, 0.0);
}

TEST(No2bRead, WrongBinarySizeThrowsWithDiagnostics) {
  const fs::path dir = TempDir();
  const std::int64_t count = No2bValueCount(kFile);
  const fs::path pf = dir / "short_stream.bin";
  WriteBinary(pf, count - 1, false);
  op::No2bThreeBody v(kContainer);
  try {
    ReadNo2b(pf, {.file = kFile}, v);
    FAIL() << "expected InputError";
  } catch (const InputError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find(std::to_string(count * 4)), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string((count - 1) * 4)), std::string::npos)
        << msg;
    EXPECT_NE(msg.find("file3e1max"), std::string::npos) << msg;
    EXPECT_NE(msg.find("file3e3max"), std::string::npos) << msg;
  }
  // A single-precision file read as half (or vice versa) has the wrong size
  // too.
  const fs::path pf2 = dir / "full_stream.bin";
  WriteBinary(pf2, count, false);
  EXPECT_THROW(
      ReadNo2b(pf2, {.file = kFile, .precision = No2bPrecision::kHalf}, v),
      InputError);
  // Wrong truncation.
  EXPECT_THROW(ReadNo2b(pf2, {.file = {3, 6, 8, -1}}, v), InputError);
}

TEST(No2bRead, TextEndingEarlyThrows) {
  const fs::path dir = TempDir();
  const std::int64_t count = No2bValueCount(kFile);
  // Needs at least `values_read` values of the prefix; give fewer than the
  // prefix.
  const fs::path p = dir / "early.txt";
  {
    std::ofstream(p) << TextBody(100);
  }
  op::No2bThreeBody v(kContainer);
  EXPECT_THROW(ReadNo2b(p, {.file = kFile}, v), InputError);
  (void)count;
}

TEST(No2bRead, MalformedTokenThrows) {
  const fs::path dir = TempDir();
  const fs::path p = dir / "bad.txt";
  {
    std::ofstream(p) << "header\n0.5 0.25 oops 1.0\n";
  }
  op::No2bThreeBody v(kContainer);
  EXPECT_THROW(ReadNo2b(p, {.file = kFile}, v), InputError);
}

TEST(No2bRead, MissingFileThrows) {
  op::No2bThreeBody v(kContainer);
  EXPECT_THROW(ReadNo2b(TempDir() / "nope.txt", {.file = kFile}, v),
               InputError);
  EXPECT_THROW(ReadNo2b(TempDir() / "nope_stream.bin", {.file = kFile}, v),
               InputError);
}

}  // namespace
}  // namespace imsrg::io
