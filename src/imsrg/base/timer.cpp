#include "imsrg/base/timer.hpp"

#include <algorithm>

#include <fmt/format.h>

namespace imsrg {

void TimerRegistry::set_region_hooks(RegionPushHook push, RegionPopHook pop) noexcept {
  push_.store(push, std::memory_order_release);
  pop_.store(pop, std::memory_order_release);
}

void TimerRegistry::add(std::string_view name, double seconds, std::int64_t count) {
  const std::lock_guard<std::mutex> lock(mutex_);
  Acc& a = map_[std::string(name)];
  a.seconds += seconds;
  a.count += count;
}

std::vector<TimerRegistry::Entry> TimerRegistry::entries() const {
  std::vector<Entry> out;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    out.reserve(map_.size());
    for (const auto& [name, a] : map_) out.push_back(Entry{name, a.seconds, a.count});
  }
  std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.name < b.name; });
  return out;
}

void TimerRegistry::clear() {
  const std::lock_guard<std::mutex> lock(mutex_);
  map_.clear();
}

std::string TimerRegistry::report() const {
  const auto es = entries();
  std::size_t w = 4;
  for (const auto& e : es) w = std::max(w, e.name.size());
  std::string out =
      fmt::format("{:<{}}  {:>12}  {:>10}  {:>12}\n", "name", w, "total [s]", "count", "mean [ms]");
  for (const auto& e : es) {
    const double mean_ms = e.count > 0 ? 1e3 * e.seconds / static_cast<double>(e.count) : 0.0;
    out += fmt::format("{:<{}}  {:>12.6f}  {:>10}  {:>12.6f}\n", e.name, w, e.seconds, e.count,
                       mean_ms);
  }
  return out;
}

ScopedTimer::ScopedTimer(TimerRegistry& registry, std::string_view name)
    : registry_(registry), name_(name) {
  if (RegionPushHook push = registry_.push_hook()) push(name_.c_str());
  start_ = std::chrono::steady_clock::now();
}

ScopedTimer::~ScopedTimer() {
  const auto stop = std::chrono::steady_clock::now();
  if (RegionPopHook pop = registry_.pop_hook()) pop();
  registry_.add(name_, std::chrono::duration<double>(stop - start_).count());
}

}  // namespace imsrg
