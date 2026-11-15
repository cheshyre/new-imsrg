// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

// imsrg3_layout_bench -- IMSRG(3) three-body layout benchmark (design/14
// section 2). Stub (task T4 implements it).

#include <cstdio>

#include "imsrg/exec/runtime.h"

#ifndef IMSRG_BENCH_GIT_HASH
#define IMSRG_BENCH_GIT_HASH "unknown"
#endif

int main(int argc, char** argv) {
  imsrg::exec::Runtime runtime(argc, argv);
  std::printf("imsrg3_layout_bench %s: not implemented yet\n",
              IMSRG_BENCH_GIT_HASH);
  return 0;
}
