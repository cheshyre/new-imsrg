// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/log.h"

#include <cstdio>
#include <string>

namespace imsrg::log {

std::string_view ToString(Level level) noexcept {
  switch (level) {
    case Level::kTrace:
      return "trace";
    case Level::kDebug:
      return "debug";
    case Level::kInfo:
      return "info";
    case Level::kWarn:
      return "warn";
    case Level::kError:
      return "error";
    case Level::kOff:
      return "off";
  }
  return "?";
}

Logger::Logger(Level level) : level_(level) {}

Logger::Logger(Level level, Sink sink)
    : level_(level), sink_(std::move(sink)) {}

void Logger::set_rank(int rank, bool all_ranks) noexcept {
  rank_.store(rank, std::memory_order_relaxed);
  all_ranks_.store(all_ranks, std::memory_order_relaxed);
}

bool Logger::enabled(Level lvl) const noexcept {
  if (lvl == Level::kOff || lvl < level()) return false;
  return all_ranks_.load(std::memory_order_relaxed) ||
         rank_.load(std::memory_order_relaxed) == 0;
}

void Logger::Write(Level lvl, std::string_view message) {
  if (!enabled(lvl)) return;
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sink_) {
    sink_(lvl, message);
    return;
  }
  const int rank = rank_.load(std::memory_order_relaxed);
  const std::string line =
      (all_ranks_.load(std::memory_order_relaxed) && rank > 0)
          ? fmt::format("[imsrg r{}] {}: {}\n", rank, ToString(lvl), message)
          : fmt::format("[imsrg] {}: {}\n", ToString(lvl), message);
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fflush(stdout);
}

Logger& DefaultLogger() {
  static Logger logger(Level::kInfo);
  return logger;
}

}  // namespace imsrg::log
