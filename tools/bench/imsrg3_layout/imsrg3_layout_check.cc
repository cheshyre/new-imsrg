// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// Checks of the IMSRG(3) layout benchmark prototypes (design/14 section 2):
// 6j table, recoupling, counts, layout conversions and cross-layout agreement
// of the kernels. Label `slow` (not run by tools/ci.sh).

#include <gtest/gtest.h>

#include "tools/bench/imsrg3_layout/recoupling.h"

namespace imsrg::layout3 {
namespace {

TEST(Recoupling, SortOrbitsMatchesPermute) {
  // Every ordering of three distinct and of coinciding orbits sorts back.
  const int sets[][3] = {{1, 2, 3}, {3, 1, 2}, {2, 2, 5}, {5, 2, 2}, {4, 4, 4}};
  for (const auto& s : sets) {
    int a = 0;
    int b = 0;
    int c = 0;
    const Perm p = SortOrbits(s[0], s[1], s[2], a, b, c);
    EXPECT_LE(a, b);
    EXPECT_LE(b, c);
    int o1 = 0;
    int o2 = 0;
    int o3 = 0;
    Permute(p, a, b, c, o1, o2, o3);
    EXPECT_EQ(o1, s[0]);
    EXPECT_EQ(o2, s[1]);
    EXPECT_EQ(o3, s[2]);
  }
}

}  // namespace
}  // namespace imsrg::layout3
