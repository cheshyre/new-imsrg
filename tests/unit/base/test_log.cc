// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/log.h"

namespace {

using imsrg::log::Level;
using imsrg::log::Logger;

struct Capture {
  std::vector<std::pair<Level, std::string>> lines;
  Logger::Sink MakeSink() {
    return [this](Level l, std::string_view s) {
      lines.emplace_back(l, std::string(s));
    };
  }
};

TEST(Log, LevelFiltering) {
  Capture cap;
  Logger log(Level::kWarn, cap.MakeSink());
  log.Trace("t");
  log.Debug("d");
  log.Info("i");
  log.Warn("w {}", 1);
  log.Error("e");
  ASSERT_EQ(cap.lines.size(), std::size_t{2});
  EXPECT_EQ(cap.lines[0].first, Level::kWarn);
  EXPECT_EQ(cap.lines[0].second, "w 1");
  EXPECT_EQ(cap.lines[1].first, Level::kError);
  log.set_level(Level::kTrace);
  EXPECT_EQ(log.level(), Level::kTrace);
  log.Trace("now");
  EXPECT_EQ(cap.lines.size(), std::size_t{3});
  log.set_level(Level::kOff);
  log.Error("silent");
  EXPECT_EQ(cap.lines.size(), std::size_t{3});
  EXPECT_FALSE(log.enabled(Level::kError));
}

TEST(Log, FormattingOnlyWhenEnabled) {
  Capture cap;
  Logger log(Level::kError, cap.MakeSink());
  log.Info("{:>6.2f}|{}", 3.14159, "x");
  EXPECT_TRUE(cap.lines.empty());
  log.Error("{:>6.2f}|{}", 3.14159, "x");
  ASSERT_EQ(cap.lines.size(), std::size_t{1});
  EXPECT_EQ(cap.lines[0].second, "  3.14|x");
}

TEST(Log, RankFiltering) {
  Capture cap;
  Logger log(Level::kInfo, cap.MakeSink());
  log.set_rank(2);
  log.Info("hidden");
  EXPECT_TRUE(cap.lines.empty());
  EXPECT_FALSE(log.enabled(Level::kInfo));
  log.set_rank(0);
  log.Info("shown");
  EXPECT_EQ(cap.lines.size(), std::size_t{1});
  log.set_rank(3, true);
  log.Info("shown too");
  EXPECT_EQ(cap.lines.size(), std::size_t{2});
}

TEST(Log, ToStringAndDefaultLogger) {
  EXPECT_EQ(imsrg::log::ToString(Level::kWarn), "warn");
  EXPECT_EQ(&imsrg::log::DefaultLogger(), &imsrg::log::DefaultLogger());
  imsrg::log::DefaultLogger().set_level(Level::kOff);
  imsrg::log::Info("not printed {}", 1);
  imsrg::log::DefaultLogger().set_level(Level::kInfo);
}

TEST(Log, ThreadSafeCapture) {
  Capture cap;
  Logger log(Level::kInfo,
             cap.MakeSink());  // sink is called under the logger mutex
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&log, t] {
      for (int i = 0; i < 500; ++i) log.Info("{} {}", t, i);
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(cap.lines.size(), std::size_t{2000});
}

}  // namespace
