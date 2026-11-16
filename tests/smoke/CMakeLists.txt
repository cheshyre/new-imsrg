# Toolchain smoke tests (M0). They use Kokkos and MPI directly because the imsrg::exec and
# imsrg::comm facades do not exist yet; module code must go through those facades.

set(_imsrg_space_serial "Serial")
set(_imsrg_space_openmp "OpenMP")
set(_imsrg_space_cuda "Cuda")
set(_imsrg_space_hip "HIP")

imsrg_add_test(NAME smoke_kokkos
  SOURCES kokkos_smoke_test.cc
  LIBS Kokkos::kokkos fmt::fmt
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(smoke_kokkos PRIVATE
  IMSRG_EXPECTED_KOKKOS_SPACE="${_imsrg_space_${IMSRG_KOKKOS_BACKEND}}")

imsrg_add_test(NAME smoke_mpi_hello
  SOURCES mpi_hello_test.cc
  LIBS MPI::MPI_CXX
  MPI_RANKS 2
  ARGS --expected-ranks=2
  LABELS mpi
  NO_GTEST_MAIN
  TIMEOUT 120)

# Environment::Abort (M12b D4): rank 1 aborts while rank 0 waits in a barrier. The job must end at
# once with a non-zero status (WILL_FAIL); a hang would hit the timeout, which also fails the test
# run (the timeout is short, so a hang is reported rather than waited out).
imsrg_add_test(NAME smoke_mpi_abort
  SOURCES mpi_abort_test.cc
  DEPS comm
  MPI_RANKS 2
  LABELS mpi
  NO_GTEST_MAIN
  TIMEOUT 60)
if(TARGET smoke_mpi_abort)
  set_tests_properties(smoke_mpi_abort PROPERTIES WILL_FAIL TRUE)
endif()
