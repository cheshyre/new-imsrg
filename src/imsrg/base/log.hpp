#pragma once
/// @file
/// Minimal thread-safe logging (design/10 section 3.1). A `Logger` is an explicit object; the one
/// process-wide instance `default_logger()` only configures output and never affects results.
/// Line format: `[imsrg] <level>: <message>` (`[imsrg r<k>] ...` when all ranks print).

#include <atomic>
#include <functional>
#include <mutex>
#include <string_view>
#include <utility>

#include <fmt/format.h>

namespace imsrg::log {

/// Severity, ordered; `Off` disables all output.
enum class Level { Trace, Debug, Info, Warn, Error, Off };

/// Lower-case name of a level ("trace", ..., "error", "off").
std::string_view to_string(Level level) noexcept;

/// Thread-safe logger. The level is atomic; writes are serialized by a mutex. By default only
/// rank 0 prints (see `set_rank`). Not copyable.
class Logger {
 public:
  /// Receives (level, message) where message has no prefix and no trailing newline.
  using Sink = std::function<void(Level, std::string_view)>;

  /// Logger writing formatted lines to stdout.
  explicit Logger(Level level = Level::Info);
  /// Logger delivering each enabled message to `sink` instead of stdout (for capture in tests).
  Logger(Level level, Sink sink);

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  /// Sets the minimum level that is written.
  void set_level(Level level) noexcept { level_.store(level, std::memory_order_relaxed); }
  /// The current minimum level.
  Level level() const noexcept { return level_.load(std::memory_order_relaxed); }

  /// Declares this process's MPI rank. Only rank 0 writes unless `all_ranks`; with `all_ranks`
  /// and rank > 0 the line prefix carries the rank (`[imsrg r3]`).
  void set_rank(int rank, bool all_ranks = false) noexcept;

  /// True if a message of `lvl` would be written (level passes and this rank may print).
  bool enabled(Level lvl) const noexcept;

  /// Writes `message` at `lvl` if enabled. Thread-safe.
  void write(Level lvl, std::string_view message);

  /// Formats and writes at Trace; formatting happens only if enabled.
  template <class... Args>
  void trace(fmt::format_string<Args...> f, Args&&... args) {
    log_at(Level::Trace, f, std::forward<Args>(args)...);
  }
  /// As `trace`, at Debug.
  template <class... Args>
  void debug(fmt::format_string<Args...> f, Args&&... args) {
    log_at(Level::Debug, f, std::forward<Args>(args)...);
  }
  /// As `trace`, at Info.
  template <class... Args>
  void info(fmt::format_string<Args...> f, Args&&... args) {
    log_at(Level::Info, f, std::forward<Args>(args)...);
  }
  /// As `trace`, at Warn.
  template <class... Args>
  void warn(fmt::format_string<Args...> f, Args&&... args) {
    log_at(Level::Warn, f, std::forward<Args>(args)...);
  }
  /// As `trace`, at Error.
  template <class... Args>
  void error(fmt::format_string<Args...> f, Args&&... args) {
    log_at(Level::Error, f, std::forward<Args>(args)...);
  }

 private:
  template <class... Args>
  void log_at(Level lvl, fmt::format_string<Args...> f, Args&&... args) {
    if (enabled(lvl)) write(lvl, fmt::format(f, std::forward<Args>(args)...));
  }

  std::atomic<Level> level_;
  std::atomic<int> rank_{0};
  std::atomic<bool> all_ranks_{false};
  Sink sink_;
  std::mutex mutex_;
};

/// The process-wide logger: the single sanctioned global. It only configures output (level, rank
/// filtering, destination) and never affects numerical results.
Logger& default_logger();

/// Formats and writes to `default_logger()` at Trace.
template <class... Args>
void trace(fmt::format_string<Args...> f, Args&&... args) {
  default_logger().trace(f, std::forward<Args>(args)...);
}
/// As `log::trace`, at Debug.
template <class... Args>
void debug(fmt::format_string<Args...> f, Args&&... args) {
  default_logger().debug(f, std::forward<Args>(args)...);
}
/// As `log::trace`, at Info.
template <class... Args>
void info(fmt::format_string<Args...> f, Args&&... args) {
  default_logger().info(f, std::forward<Args>(args)...);
}
/// As `log::trace`, at Warn.
template <class... Args>
void warn(fmt::format_string<Args...> f, Args&&... args) {
  default_logger().warn(f, std::forward<Args>(args)...);
}
/// As `log::trace`, at Error.
template <class... Args>
void error(fmt::format_string<Args...> f, Args&&... args) {
  default_logger().error(f, std::forward<Args>(args)...);
}

}  // namespace imsrg::log
