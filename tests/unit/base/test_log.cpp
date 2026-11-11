#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/log.hpp"

namespace {

using imsrg::log::Level;
using imsrg::log::Logger;

struct Capture {
  std::vector<std::pair<Level, std::string>> lines;
  Logger::Sink sink() {
    return [this](Level l, std::string_view s) { lines.emplace_back(l, std::string(s)); };
  }
};

TEST(Log, LevelFiltering) {
  Capture cap;
  Logger log(Level::Warn, cap.sink());
  log.trace("t");
  log.debug("d");
  log.info("i");
  log.warn("w {}", 1);
  log.error("e");
  ASSERT_EQ(cap.lines.size(), std::size_t{2});
  EXPECT_EQ(cap.lines[0].first, Level::Warn);
  EXPECT_EQ(cap.lines[0].second, "w 1");
  EXPECT_EQ(cap.lines[1].first, Level::Error);
  log.set_level(Level::Trace);
  EXPECT_EQ(log.level(), Level::Trace);
  log.trace("now");
  EXPECT_EQ(cap.lines.size(), std::size_t{3});
  log.set_level(Level::Off);
  log.error("silent");
  EXPECT_EQ(cap.lines.size(), std::size_t{3});
  EXPECT_FALSE(log.enabled(Level::Error));
}

TEST(Log, FormattingOnlyWhenEnabled) {
  Capture cap;
  Logger log(Level::Error, cap.sink());
  log.info("{:>6.2f}|{}", 3.14159, "x");
  EXPECT_TRUE(cap.lines.empty());
  log.error("{:>6.2f}|{}", 3.14159, "x");
  ASSERT_EQ(cap.lines.size(), std::size_t{1});
  EXPECT_EQ(cap.lines[0].second, "  3.14|x");
}

TEST(Log, RankFiltering) {
  Capture cap;
  Logger log(Level::Info, cap.sink());
  log.set_rank(2);
  log.info("hidden");
  EXPECT_TRUE(cap.lines.empty());
  EXPECT_FALSE(log.enabled(Level::Info));
  log.set_rank(0);
  log.info("shown");
  EXPECT_EQ(cap.lines.size(), std::size_t{1});
  log.set_rank(3, true);
  log.info("shown too");
  EXPECT_EQ(cap.lines.size(), std::size_t{2});
}

TEST(Log, ToStringAndDefaultLogger) {
  EXPECT_EQ(imsrg::log::to_string(Level::Warn), "warn");
  EXPECT_EQ(&imsrg::log::default_logger(), &imsrg::log::default_logger());
  imsrg::log::default_logger().set_level(Level::Off);
  imsrg::log::info("not printed {}", 1);
  imsrg::log::default_logger().set_level(Level::Info);
}

TEST(Log, ThreadSafeCapture) {
  Capture cap;
  Logger log(Level::Info, cap.sink());  // sink is called under the logger mutex
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&log, t] {
      for (int i = 0; i < 500; ++i) log.info("{} {}", t, i);
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(cap.lines.size(), std::size_t{2000});
}

}  // namespace
