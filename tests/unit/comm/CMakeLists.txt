imsrg_add_test(NAME comm_serial
  SOURCES comm_serial_test.cpp
  DEPS comm
  LABELS unit)

# Two source files: the serial-semantics tests need a gtest main; the MPI test owns main().
imsrg_add_test(NAME comm_mpi_2ranks
  SOURCES comm_mpi_test.cpp
  DEPS comm
  MPI_RANKS 2
  LABELS mpi
  NO_GTEST_MAIN
  TIMEOUT 120)

imsrg_add_test(NAME comm_mpi_4ranks
  SOURCES comm_mpi_test.cpp
  DEPS comm
  MPI_RANKS 4
  LABELS mpi
  NO_GTEST_MAIN
  TIMEOUT 120)
