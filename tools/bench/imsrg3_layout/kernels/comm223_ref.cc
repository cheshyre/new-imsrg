// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/error.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"

namespace imsrg::layout3 {

// Stubs (task T3a implements them).
void Comm223Reference(const KernelContext& ctx, const op::Operator& X,
                      const op::Operator& Y, std::span<double> z,
                      bool both_triangles) {
  (void)ctx;
  (void)X;
  (void)Y;
  (void)z;
  (void)both_triangles;
  throw NotImplementedError("Comm223Reference: not implemented yet");
}

double TriplesReference(const KernelContext& ctx, const op::Operator& X,
                        const op::Operator& Y,
                        std::span<const double> epsilon) {
  (void)ctx;
  (void)X;
  (void)Y;
  (void)epsilon;
  throw NotImplementedError("TriplesReference: not implemented yet");
}

}  // namespace imsrg::layout3
