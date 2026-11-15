// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/pair_third.h"

#include <vector>

#include "imsrg/base/error.h"

namespace imsrg::layout3 {

// Stubs (task T1 implements them).
CsrView Csr::host_view() const noexcept { return {}; }
CsrView Csr::device_view() const noexcept { return {}; }
std::int64_t Csr::Bytes() const noexcept { return 0; }
Csr Csr::FromHost(const std::vector<std::int64_t>& ptr,
                  const std::vector<std::int32_t>& col,
                  const std::vector<double>& val) {
  (void)ptr;
  (void)col;
  (void)val;
  throw NotImplementedError("Csr::FromHost: not implemented yet");
}

PairThirdSpace::PairThirdSpace(const CanonicalSpace& space,
                               const SixJ3Table& sixj) {
  (void)space;
  (void)sixj;
  throw NotImplementedError("PairThirdSpace: not implemented yet");
}

SpaceCounts PairThirdSpace::Counts() const { return {}; }
std::int64_t PairThirdSpace::StateBytes() const noexcept { return 0; }
PairThirdView PairThirdSpace::host_view() const noexcept { return {}; }
PairThirdView PairThirdSpace::device_view() const noexcept { return {}; }
SpaceCounts PairThirdSpace::CountOnly(const CanonicalSpace& space) {
  (void)space;
  throw NotImplementedError("PairThirdSpace::CountOnly: not implemented yet");
}

}  // namespace imsrg::layout3
