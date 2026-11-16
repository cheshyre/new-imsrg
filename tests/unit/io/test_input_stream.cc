// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// TextInput / GzipWriter: plain and gzip tokens, header skip, malformed tokens,
// float parsing.
#include <unistd.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/error.h"
#include "imsrg/io/input_stream.h"

namespace imsrg::io {
namespace {

namespace fs = std::filesystem;

fs::path TempPath(const std::string& name) {
  // Per-process names: ctest runs every test in its own process, in parallel.
  return fs::temp_directory_path() /
         ("imsrg_io_test_" + std::to_string(::getpid()) + "_" + name);
}

void WritePlain(const fs::path& p, const std::string& text) {
  std::ofstream f(p, std::ios::binary);
  f << text;
}

void WriteGz(const fs::path& p, const std::string& text) {
  GzipWriter w(p);
  w.Write(text);
  w.Close();
}

std::vector<std::string> Tokens(const fs::path& p, bool skip_header) {
  TextInput in(p);
  if (skip_header) in.SkipLine();
  std::vector<std::string> out;
  std::string t;
  while (in.NextToken(t)) out.push_back(t);
  return out;
}

const char* const kText =
    "header line with words 1 2 3\n 1.5  -2.25e1\t+3\n\n4.0e-3 5 \r\n6";

TEST(InputStream, PlainAndGzipGiveIdenticalTokens) {
  const auto plain = TempPath("plain.txt");
  const auto gz = TempPath("gz.txt.gz");
  WritePlain(plain, kText);
  WriteGz(gz, kText);
  const auto a = Tokens(plain, true);
  const auto b = Tokens(gz, true);
  EXPECT_EQ(a, b);
  const std::vector<std::string> expected{"1.5",    "-2.25e1", "+3",
                                          "4.0e-3", "5",       "6"};
  EXPECT_EQ(a, expected);
  fs::remove(plain);
  fs::remove(gz);
}

TEST(InputStream, GzipDetectionIsByContentNotName) {
  const auto p = TempPath("misnamed.txt");  // gzip content with a plain name
  WriteGz(p, kText);
  EXPECT_EQ(Tokens(p, true).size(), 6U);
  fs::remove(p);
}

TEST(InputStream, HeaderSkipDiscardsOnlyTheFirstLine) {
  const auto p = TempPath("header.txt");
  WritePlain(p, "a b c\nd e\n");
  EXPECT_EQ(Tokens(p, false).size(), 5U);
  EXPECT_EQ(Tokens(p, true).size(), 2U);
  fs::remove(p);
}

TEST(InputStream, SkipLineOnEmptyFileReportsNothingAvailable) {
  const auto p = TempPath("empty.txt");
  WritePlain(p, "");
  TextInput in(p);
  EXPECT_FALSE(in.SkipLine());
  std::string t;
  EXPECT_FALSE(in.NextToken(t));
  fs::remove(p);
}

TEST(InputStream, TokensLargerThanTheBufferAndManyTokens) {
  const auto p = TempPath("big.txt.gz");
  std::string text = "h\n";
  std::vector<std::string> expected;
  for (int i = 0; i < 40000; ++i) {
    expected.push_back(std::to_string(i) + ".5");
    text += expected.back() + (i % 4 == 3 ? "\n" : " ");
  }
  WriteGz(p, text);
  EXPECT_EQ(Tokens(p, true), expected);
  fs::remove(p);
}

TEST(InputStream, NextFloatParsesAndStopsAtEnd) {
  const auto p = TempPath("floats.txt");
  WritePlain(p, "h\n1.5 -2.25e1 +3 1e-3\n");
  TextInput in(p);
  in.SkipLine();
  float v = 0.0F;
  ASSERT_TRUE(in.NextFloat(v));
  EXPECT_EQ(v, 1.5F);
  ASSERT_TRUE(in.NextFloat(v));
  EXPECT_EQ(v, -22.5F);
  ASSERT_TRUE(in.NextFloat(v));
  EXPECT_EQ(v, 3.0F);
  ASSERT_TRUE(in.NextFloat(v));
  EXPECT_EQ(v, 1.0e-3F);
  v = 7.0F;
  EXPECT_FALSE(in.NextFloat(v));
  EXPECT_EQ(v, 7.0F);  // unchanged at the end of data
  fs::remove(p);
}

TEST(InputStream, MalformedTokenThrowsAndNamesTheToken) {
  for (const char* bad : {"abc", "1.5abc", "--1", "1.2.3", "0x"}) {
    const auto p = TempPath("bad.txt");
    WritePlain(p, std::string("h\n1.0 ") + bad + " 2.0\n");
    TextInput in(p);
    in.SkipLine();
    float v = 0.0F;
    ASSERT_TRUE(in.NextFloat(v));
    try {
      in.NextFloat(v);
      ADD_FAILURE() << "no throw for token " << bad;
    } catch (const InputError& e) {
      EXPECT_NE(std::string(e.what()).find(bad), std::string::npos) << e.what();
    }
    fs::remove(p);
  }
}

TEST(InputStream, MissingFileThrowsInputError) {
  EXPECT_THROW(TextInput(TempPath("does_not_exist.me2j.gz")), InputError);
}

TEST(InputStream, TruncatedGzipStreamThrowsOrEndsEarlyButNeverCrashes) {
  const auto good = TempPath("trunc_src.gz");
  std::string text = "h\n";
  std::mt19937 rng(7);
  for (int i = 0; i < 20000; ++i) text += std::to_string(rng()) + " ";
  WriteGz(good, text);
  std::ifstream f(good, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string bytes = ss.str();
  const auto cut = TempPath("trunc.gz");
  WritePlain(cut, bytes.substr(0, bytes.size() / 2));
  TextInput in(cut);
  in.SkipLine();
  std::string t;
  std::size_t n = 0;
  try {
    while (in.NextToken(t)) ++n;
  } catch (const InputError&) {
  }
  EXPECT_LT(n, 20000U);
  fs::remove(good);
  fs::remove(cut);
}

TEST(InputStream, FloatParsingIsBitIdenticalToIstreamExtraction) {
  std::mt19937_64 rng(20260101);
  std::ostringstream text;
  text << "header\n";
  std::vector<std::string> strs;
  for (int i = 0; i < 5000; ++i) {
    // Seven significant digits, optional sign, optional exponent: the me2j
    // number format.
    std::string s;
    if (rng() % 2 == 0) s += '-';
    s += std::to_string(rng() % 10);
    s += '.';
    for (int d = 0; d < 6; ++d) s += static_cast<char>('0' + rng() % 10);
    if (rng() % 3 == 0)
      s += "e" + std::string(rng() % 2 == 0 ? "-" : "+") + "0" +
           std::to_string(rng() % 9 + 1);
    strs.push_back(s);
    text << s << (i % 4 == 3 ? '\n' : ' ');
  }
  const auto p = TempPath("random_floats.txt.gz");
  WriteGz(p, text.str());
  TextInput in(p);
  in.SkipLine();
  for (const std::string& s : strs) {
    float got = 0.0F;
    ASSERT_TRUE(in.NextFloat(got));
    float want = 0.0F;
    std::istringstream is(s);
    is >> want;
    ASSERT_EQ(std::bit_cast<std::uint32_t>(got),
              std::bit_cast<std::uint32_t>(want))
        << s;
  }
  float extra = 0.0F;
  EXPECT_FALSE(in.NextFloat(extra));
  fs::remove(p);
}

}  // namespace
}  // namespace imsrg::io
