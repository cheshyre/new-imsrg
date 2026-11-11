# Toolchain smoke tests (M0). They use Kokkos and MPI directly because the imsrg::exec and
# imsrg::comm facades do not exist yet; module code must go through those facades.

set(_imsrg_space_serial "Serial")
set(_imsrg_space_openmp "OpenMP")
set(_imsrg_space_cuda "Cuda")
set(_imsrg_space_hip "HIP")

imsrg_add_test(NAME smoke_kokkos
  SOURCES kokkos_smoke_test.cpp
  LIBS Kokkos::kokkos fmt::fmt
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(smoke_kokkos PRIVATE
  IMSRG_EXPECTED_KOKKOS_SPACE="${_imsrg_space_${IMSRG_KOKKOS_BACKEND}}")

imsrg_add_test(NAME smoke_mpi_hello
  SOURCES mpi_hello_test.cpp
  LIBS MPI::MPI_CXX
  MPI_RANKS 2
  ARGS --expected-ranks=2
  LABELS mpi
  NO_GTEST_MAIN
  TIMEOUT 120)
