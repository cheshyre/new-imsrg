# comm: MPI facade (Communicator, Environment) with a serial stub (design/10 section 2, layer 0).
# mpi.h is included only by the .cpp files of this module; the public headers are MPI-free.
set(_imsrg_comm_sources environment.cpp)
if(IMSRG_ENABLE_MPI)
  list(APPEND _imsrg_comm_sources communicator_mpi.cpp)
  set(_imsrg_comm_private_libs MPI::MPI_CXX)
else()
  list(APPEND _imsrg_comm_sources communicator_serial.cpp)
  set(_imsrg_comm_private_libs)
endif()

imsrg_add_module(NAME comm
  SOURCES ${_imsrg_comm_sources}
  DEPS base
  PRIVATE_LIBS ${_imsrg_comm_private_libs})

if(IMSRG_ENABLE_MPI)
  target_compile_definitions(imsrg_comm PRIVATE IMSRG_COMM_WITH_MPI=1)
else()
  target_compile_definitions(imsrg_comm PRIVATE IMSRG_COMM_WITH_MPI=0)
endif()
