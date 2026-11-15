// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/error.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"

namespace imsrg::layout3 {

// Stub (task T3 implements it).
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

}  // namespace imsrg::layout3
