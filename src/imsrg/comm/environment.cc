// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2026 The imsrg authors

#include "imsrg/comm/environment.h"

#include "imsrg/base/error.h"

#if IMSRG_COMM_WITH_MPI
#include <mpi.h>
#endif

namespace imsrg::comm {

#if IMSRG_COMM_WITH_MPI

void Environment::Init(int* argc, char*** argv) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized != 0) {
      throw RuntimeError("comm::Environment: MPI was already finalized");
    }
    MPI_Query_thread(&thread_level_);
    return;
  }
  if (MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &thread_level_) !=
      MPI_SUCCESS) {
    throw RuntimeError("comm::Environment: MPI_Init_thread failed");
  }
  owns_ = true;
}

Environment::~Environment() {
  if (!owns_) return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized == 0) MPI_Finalize();
}

bool Environment::MpiEnabled() noexcept { return true; }

bool Environment::IsInitialized() noexcept {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  return initialized != 0 && finalized == 0;
}

#else

void Environment::Init(int*, char***) {}
Environment::~Environment() = default;
bool Environment::MpiEnabled() noexcept { return false; }
bool Environment::IsInitialized() noexcept { return false; }

#endif

Environment::Environment() { Init(nullptr, nullptr); }

Environment::Environment(int& argc, char**& argv) { Init(&argc, &argv); }

Communicator Environment::world() const { return Communicator::World(); }

}  // namespace imsrg::comm
