// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/sixj3.h"

#include "imsrg/base/error.h"

namespace imsrg::layout3 {

// Stub (task T1 implements it).
SixJ3Table::SixJ3Table(int emax) : emax_(emax) {
  throw NotImplementedError("SixJ3Table: not implemented yet");
}

std::int64_t SixJ3Table::size_bytes() const noexcept { return 0; }
SixJ3View SixJ3Table::host_view() const noexcept { return {}; }
SixJ3View SixJ3Table::device_view() const noexcept { return {}; }
std::int64_t SixJ3Table::ValueCount(int emax) noexcept {
  const std::int64_t nj = emax + 1;
  const std::int64_t nJ = 2 * emax + 2;
  const std::int64_t nJ3 = 3 * emax + 2;
  return nj * nj * nj * nJ * nJ * nJ3;
}

}  // namespace imsrg::layout3
