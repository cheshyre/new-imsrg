#pragma once
/// @file
/// Wall-clock timing of named regions (design/10 section 3.1). `TimerRegistry` is an explicit,
/// thread-safe object (no globals); `ScopedTimer` measures one region with
/// `std::chrono::steady_clock` and optionally notifies a profiler through region hooks.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace imsrg {

/// Called when a timed region starts; receives the region name (valid during the call only).
using RegionPushHook = void (*)(const char*);
/// Called when a timed region ends.
using RegionPopHook = void (*)();

/// Accumulates (total seconds, call count) per region name. Thread-safe; not copyable.
class TimerRegistry {
 public:
  /// One accumulated region.
  struct Entry {
    std::string name;    ///< Region name.
    double seconds;      ///< Total wall time in seconds.
    std::int64_t count;  ///< Number of recorded calls.
  };

  /// Adds `seconds` and `count` calls to region `name` (created on first use). Thread-safe.
  void add(std::string_view name, double seconds, std::int64_t count = 1);
  /// Snapshot of all regions, sorted by name.
  std::vector<Entry> entries() const;
  /// Removes all regions.
  void clear();
  /// Aligned text table with columns name, total [s], count, mean [ms]; one row per region.
  std::string report() const;

  /// Installs profiler hooks that every `ScopedTimer` on this registry calls on entry and exit
  /// (e.g. `exec::push_profiling_region` / `exec::pop_profiling_region`, so NVTX/rocTX tools see
  /// the same regions). `nullptr` means no hook. Thread-safe; per registry, never global.
  void set_region_hooks(RegionPushHook push, RegionPopHook pop) noexcept;
  /// The installed push hook (or nullptr).
  RegionPushHook push_hook() const noexcept { return push_.load(std::memory_order_acquire); }
  /// The installed pop hook (or nullptr).
  RegionPopHook pop_hook() const noexcept { return pop_.load(std::memory_order_acquire); }

 private:
  struct Acc {
    double seconds = 0.0;
    std::int64_t count = 0;
  };
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Acc> map_;
  std::atomic<RegionPushHook> push_{nullptr};
  std::atomic<RegionPopHook> pop_{nullptr};
};

/// RAII timer: on construction calls the registry's push hook (if any) and starts the clock; on
/// destruction calls the pop hook and adds the elapsed time to the registry. Non-copyable. The
/// registry must outlive the timer.
class ScopedTimer {
 public:
  /// Starts timing region `name`, recorded in `registry` on destruction.
  ScopedTimer(TimerRegistry& registry, std::string_view name);
  /// Stops the clock, calls the pop hook and records the elapsed time.
  ~ScopedTimer();
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  TimerRegistry& registry_;
  std::string name_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace imsrg

#define IMSRG_DETAIL_CONCAT_(a, b) a##b
#define IMSRG_DETAIL_CONCAT(a, b) IMSRG_DETAIL_CONCAT_(a, b)
/// Declares a uniquely named `ScopedTimer` for the rest of the enclosing scope.
#define IMSRG_SCOPED_TIMER(registry, name) \
  ::imsrg::ScopedTimer IMSRG_DETAIL_CONCAT(imsrg_scoped_timer_, __LINE__)((registry), (name))
