// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/canonical_space.h"

#include <utility>

#include "imsrg/base/error.h"

namespace imsrg::layout3 {

// Stub (task T1 implements it).
CanonicalSpace::CanonicalSpace(std::shared_ptr<const basis::ModelSpace> ms,
                               double de3max, double occnat3_cut)
    : ms_(std::move(ms)) {
  (void)de3max;
  (void)occnat3_cut;
  throw NotImplementedError("CanonicalSpace: not implemented yet");
}

std::int64_t CanonicalSpace::total_size() const noexcept { return 0; }
SpaceCounts CanonicalSpace::Counts() const { return {}; }
std::int64_t CanonicalSpace::MetadataBytes() const noexcept { return 0; }
CanonicalView CanonicalSpace::host_view() const noexcept { return {}; }
CanonicalView CanonicalSpace::device_view() const noexcept { return {}; }

double ImsrgppNorm(const CanonicalSpace& space,
                   std::span<const double> canonical) {
  (void)space;
  (void)canonical;
  throw NotImplementedError("ImsrgppNorm: not implemented yet");
}

}  // namespace imsrg::layout3
