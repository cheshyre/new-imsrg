#pragma once
/// @file
/// Implementation phase of a configuration value, key or operator (design/16 section 2).

#include <string_view>

namespace imsrg::driver {

/// P1 = implemented in the first phase; Planned = parsed, but requesting it fails validation with
/// `E-PLANNED` ("not available in this build"); NotSupported = rejected with a message.
enum class Phase { P1, Planned, NotSupported };

/// "P1", "PL" or "NS" (the abbreviations of design/16).
constexpr std::string_view phase_name(Phase phase) noexcept {
  switch (phase) {
    case Phase::P1:
      return "P1";
    case Phase::Planned:
      return "PL";
    case Phase::NotSupported:
      return "NS";
  }
  return "?";
}

}  // namespace imsrg::driver
