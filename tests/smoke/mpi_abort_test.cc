// SPDX-License-Identifier: GPL-3.0-only
// Copyright 2026 The imsrg authors

// Rank 1 calls Environment::Abort(3) at once while rank 0 waits in a barrier:
// the job must terminate promptly with a non-zero status instead of hanging
// (the CTest test expects the failure, WILL_FAIL, and has a short timeout).
#include "imsrg/comm/communicator.h"
#include "imsrg/comm/environment.h"

int main(int argc, char** argv) {
  imsrg::comm::Environment environment(argc, argv);
  const imsrg::comm::Communicator world = environment.world();
  if (world.rank() == 1) imsrg::comm::Environment::Abort(3);
  world.Barrier();  // never completes: the abort ends the job
  // Reached only if the abort was lost; keep the exit code zero so that the
  // WILL_FAIL test then fails loudly after the barrier returns.
  return 0;
}
