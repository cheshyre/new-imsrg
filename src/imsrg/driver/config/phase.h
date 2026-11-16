// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

#ifndef IMSRG_DRIVER_CONFIG_PHASE_H_
#define IMSRG_DRIVER_CONFIG_PHASE_H_
/// @file
/// Implementation phase of a configuration value, key or operator (design/16
/// section 2).

#include <string_view>

namespace imsrg::driver {

/// P1 = implemented in the first phase; Planned = parsed, but requesting it
/// fails validation with `E-PLANNED` ("not available in this build");
/// NotSupported = rejected with a message.
enum class Phase { kP1, kPlanned, kNotSupported };

/// "P1", "PL" or "NS" (the abbreviations of design/16).
constexpr std::string_view PhaseName(Phase phase) noexcept {
  switch (phase) {
    case Phase::kP1:
      return "P1";
    case Phase::kPlanned:
      return "PL";
    case Phase::kNotSupported:
      return "NS";
  }
  return "?";
}

}  // namespace imsrg::driver

#endif  // IMSRG_DRIVER_CONFIG_PHASE_H_
