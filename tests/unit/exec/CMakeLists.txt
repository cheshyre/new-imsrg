# Unit tests of the exec module (Kokkos facade). They run under exec_gtest_main.cpp, which creates
# the exec::Runtime around RUN_ALL_TESTS.

set(_imsrg_space_serial "Serial")
set(_imsrg_space_openmp "OpenMP")
set(_imsrg_space_cuda "Cuda")
set(_imsrg_space_hip "HIP")

imsrg_add_test(NAME exec_runtime
  SOURCES runtime_test.cpp ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS exec
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(exec_runtime PRIVATE
  IMSRG_EXPECTED_KOKKOS_SPACE="${_imsrg_space_${IMSRG_KOKKOS_BACKEND}}")

imsrg_add_test(NAME exec_device_buffer
  SOURCES device_buffer_test.cpp ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS exec
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME exec_kernels
  SOURCES kernels_test.cpp kernels_impl.cpp
          ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS exec
  LABELS unit
  NO_GTEST_MAIN)
