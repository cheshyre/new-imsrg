// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "tools/bench/imsrg3_layout/kernel_context.h"

#include "imsrg/base/error.h"

namespace imsrg::layout3 {

// Stubs (task T3 implements them).
KernelContext::KernelContext(const CanonicalSpace& space,
                             const PairThirdSpace& pt, const SixJ3Table& sixj)
    : space_(&space), pt_(&pt), sixj_(&sixj) {
  throw NotImplementedError("KernelContext: not implemented yet");
}
KernelContext::~KernelContext() = default;

commutator::detail::TwoBodyView KernelContext::TwoBody(
    const op::Operator& op) const {
  (void)op;
  return {};
}
basis::OrbitTableView KernelContext::Orbits() const noexcept { return {}; }
OrbitGroupView KernelContext::Groups() const noexcept { return {}; }
const std::int32_t* KernelContext::WorkOrder() const noexcept {
  return nullptr;
}

int CommutatorHermiticity(const op::Operator& X, const op::Operator& Y) {
  (void)X;
  (void)Y;
  throw NotImplementedError("CommutatorHermiticity: not implemented yet");
}

}  // namespace imsrg::layout3
