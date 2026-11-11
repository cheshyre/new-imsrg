#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "imsrg/base/timer.hpp"

namespace {

using namespace imsrg;

std::vector<std::string>& events() {
  static std::vector<std::string> e;
  return e;
}
void push_hook(const char* name) { events().push_back(std::string("push:") + name); }
void pop_hook() { events().push_back("pop"); }

TEST(Timer, AccumulatesCountsAndTimes) {
  TimerRegistry reg;
  for (int i = 0; i < 3; ++i) {
    IMSRG_SCOPED_TIMER(reg, "b_region");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  {
    IMSRG_SCOPED_TIMER(reg, "a_region");
  }
  const auto es = reg.entries();
  ASSERT_EQ(es.size(), std::size_t{2});
  EXPECT_EQ(es[0].name, "a_region");
  EXPECT_EQ(es[1].name, "b_region");
  EXPECT_EQ(es[1].count, 3);
  EXPECT_GE(es[1].seconds, 0.005);
  // An empty region may take less than one steady_clock tick, so only non-negativity holds.
  EXPECT_GE(es[0].seconds, 0.0);
  EXPECT_EQ(es[0].count, 1);
  reg.clear();
  EXPECT_TRUE(reg.entries().empty());
}

TEST(Timer, AddAndReport) {
  TimerRegistry reg;
  reg.add("alpha", 1.5);
  reg.add("alpha", 0.5, 3);
  reg.add("a_much_longer_name", 2.0, 2);
  const auto es = reg.entries();
  ASSERT_EQ(es.size(), std::size_t{2});
  EXPECT_EQ(es[1].name, "alpha");
  EXPECT_DOUBLE_EQ(es[1].seconds, 2.0);
  EXPECT_EQ(es[1].count, 4);
  const std::string r = reg.report();
  EXPECT_NE(r.find("alpha"), std::string::npos);
  EXPECT_NE(r.find("a_much_longer_name"), std::string::npos);
  EXPECT_NE(r.find("mean [ms]"), std::string::npos);
}

TEST(Timer, ThreadSafe) {
  TimerRegistry reg;
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) {
    ts.emplace_back([&reg] {
      for (int i = 0; i < 1000; ++i) reg.add("shared", 1e-6);
    });
  }
  for (auto& th : ts) th.join();
  const auto es = reg.entries();
  ASSERT_EQ(es.size(), std::size_t{1});
  EXPECT_EQ(es[0].count, 8000);
  EXPECT_NEAR(es[0].seconds, 8e-3, 1e-9);
}

TEST(Timer, HooksCalledInOrder) {
  events().clear();
  TimerRegistry reg;
  reg.set_region_hooks(&push_hook, &pop_hook);
  TimerRegistry other;  // hooks are per registry: timers on `other` must not call them
  {
    IMSRG_SCOPED_TIMER(other, "elsewhere");
  }
  {
    IMSRG_SCOPED_TIMER(reg, "outer");
    {
      IMSRG_SCOPED_TIMER(reg, "inner");
    }
  }
  reg.set_region_hooks(nullptr, nullptr);
  {
    IMSRG_SCOPED_TIMER(reg, "unhooked");
  }
  const std::vector<std::string> expected = {"push:outer", "push:inner", "pop", "pop"};
  EXPECT_EQ(events(), expected);
  EXPECT_EQ(reg.entries().size(), std::size_t{3});
}

}  // namespace
