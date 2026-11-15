// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/assert.h"

#include <string>

#include "imsrg/base/error.h"

namespace imsrg::detail {

void check_failed(const char* condition, const char* file, int line,
                  std::string_view message) {
  std::string what = "check failed: ";
  what += condition;
  what += " at ";
  what += file;
  what += ':';
  what += std::to_string(line);
  if (!message.empty()) {
    what += ": ";
    what += message;
  }
  throw InvariantError(what);
}

}  // namespace imsrg::detail
