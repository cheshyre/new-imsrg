// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/layouts.h"

#include <vector>

#include "imsrg/base/error.h"

namespace imsrg::layout3 {

// Stubs (task T2 implements them).
LayoutView LayoutTables::device_view() const noexcept { return {}; }
std::int64_t LayoutTables::Bytes() const noexcept { return 0; }

LayoutA::LayoutA(const CanonicalSpace& space, const PairThirdSpace& pt)
    : space_(&space) {
  (void)pt;
  throw NotImplementedError("LayoutA: not implemented yet");
}
SpaceCounts LayoutA::Counts() const { return {}; }
std::int64_t LayoutA::MetadataBytes() const noexcept { return 0; }
std::vector<linalg::BlockShape> LayoutA::Shapes() const { return {}; }
void LayoutA::ToCanonical(std::int32_t, std::span<const double>,
                          std::span<double>) const {}
void LayoutA::FromCanonical(std::int32_t, std::span<const double>,
                            std::span<double>) const {}

LayoutB::LayoutB(const CanonicalSpace& space, const PairThirdSpace& pt,
                 const SixJ3Table& sixj)
    : space_(&space) {
  (void)pt;
  (void)sixj;
  throw NotImplementedError("LayoutB: not implemented yet");
}
SpaceCounts LayoutB::Counts() const { return {}; }
std::int64_t LayoutB::MetadataBytes() const noexcept { return 0; }
std::vector<linalg::BlockShape> LayoutB::Shapes() const { return {}; }
void LayoutB::ToCanonical(std::int32_t, std::span<const double>,
                          std::span<double>) const {}
void LayoutB::FromCanonical(std::int32_t, std::span<const double>,
                            std::span<double>) const {}
SpaceCounts LayoutB::CountOnly(const CanonicalSpace& space) {
  (void)space;
  throw NotImplementedError("LayoutB::CountOnly: not implemented yet");
}

LayoutC::LayoutC(const CanonicalSpace& space, const PairThirdSpace& pt)
    : space_(&space), pt_(&pt) {
  throw NotImplementedError("LayoutC: not implemented yet");
}
SpaceCounts LayoutC::Counts() const { return {}; }
std::int64_t LayoutC::MetadataBytes() const noexcept { return 0; }
std::vector<linalg::BlockShape> LayoutC::Shapes() const { return {}; }
void LayoutC::ToCanonical(std::int32_t, std::span<const double>,
                          std::span<double>) const {}
void LayoutC::FromCanonical(std::int32_t, std::span<const double>,
                            std::span<double>) const {}

}  // namespace imsrg::layout3
