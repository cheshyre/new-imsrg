#include "imsrg/comm/environment.hpp"

#include "imsrg/base/error.hpp"

#if IMSRG_COMM_WITH_MPI
#include <mpi.h>
#endif

namespace imsrg::comm {

#if IMSRG_COMM_WITH_MPI

void Environment::init(int* argc, char*** argv) {
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
  if (MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &thread_level_) != MPI_SUCCESS) {
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

bool Environment::mpi_enabled() noexcept { return true; }

bool Environment::is_initialized() noexcept {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  return initialized != 0 && finalized == 0;
}

#else

void Environment::init(int*, char***) {}
Environment::~Environment() = default;
bool Environment::mpi_enabled() noexcept { return false; }
bool Environment::is_initialized() noexcept { return false; }

#endif

Environment::Environment() { init(nullptr, nullptr); }

Environment::Environment(int& argc, char**& argv) { init(&argc, &argv); }

Communicator Environment::world() const { return Communicator::world_(); }

}  // namespace imsrg::comm
