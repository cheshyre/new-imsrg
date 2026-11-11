#include "imsrg/base/log.hpp"

#include <cstdio>
#include <string>

namespace imsrg::log {

std::string_view to_string(Level level) noexcept {
  switch (level) {
    case Level::Trace:
      return "trace";
    case Level::Debug:
      return "debug";
    case Level::Info:
      return "info";
    case Level::Warn:
      return "warn";
    case Level::Error:
      return "error";
    case Level::Off:
      return "off";
  }
  return "?";
}

Logger::Logger(Level level) : level_(level) {}

Logger::Logger(Level level, Sink sink) : level_(level), sink_(std::move(sink)) {}

void Logger::set_rank(int rank, bool all_ranks) noexcept {
  rank_.store(rank, std::memory_order_relaxed);
  all_ranks_.store(all_ranks, std::memory_order_relaxed);
}

bool Logger::enabled(Level lvl) const noexcept {
  if (lvl == Level::Off || lvl < level()) return false;
  return all_ranks_.load(std::memory_order_relaxed) || rank_.load(std::memory_order_relaxed) == 0;
}

void Logger::write(Level lvl, std::string_view message) {
  if (!enabled(lvl)) return;
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sink_) {
    sink_(lvl, message);
    return;
  }
  const int rank = rank_.load(std::memory_order_relaxed);
  const std::string line = (all_ranks_.load(std::memory_order_relaxed) && rank > 0)
                               ? fmt::format("[imsrg r{}] {}: {}\n", rank, to_string(lvl), message)
                               : fmt::format("[imsrg] {}: {}\n", to_string(lvl), message);
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fflush(stdout);
}

Logger& default_logger() {
  static Logger logger(Level::Info);
  return logger;
}

}  // namespace imsrg::log
