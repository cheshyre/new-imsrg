// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/error.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"

namespace imsrg::layout3 {

// Stubs (task T3 implements them).
double TriplesReference(const KernelContext& ctx, const op::Operator& X,
                        const op::Operator& Y,
                        std::span<const double> epsilon) {
  (void)ctx;
  (void)X;
  (void)Y;
  (void)epsilon;
  throw NotImplementedError("TriplesReference: not implemented yet");
}

double TriplesTiled(const KernelContext& ctx, const LayoutView& layout,
                    const op::Operator& X, const op::Operator& Y,
                    std::span<const double> epsilon) {
  (void)ctx;
  (void)layout;
  (void)X;
  (void)Y;
  (void)epsilon;
  throw NotImplementedError("TriplesTiled: not implemented yet");
}

}  // namespace imsrg::layout3
