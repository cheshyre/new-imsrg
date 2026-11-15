// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/base/error.h"
#include "tools/bench/imsrg3_layout/kernel_context.h"

namespace imsrg::layout3 {

// Stub (task T3 implements it).
void Comm223Tiled(const KernelContext& ctx, const LayoutView& layout,
                  const op::Operator& X, const op::Operator& Y,
                  std::span<double> z, TiledMode mode) {
  (void)ctx;
  (void)layout;
  (void)X;
  (void)Y;
  (void)z;
  (void)mode;
  throw NotImplementedError("Comm223Tiled: not implemented yet");
}

}  // namespace imsrg::layout3
