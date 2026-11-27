# ImsrgDependencies.cmake -- third-party dependencies (design/20 section 1).
#
# Defines: Kokkos::kokkos, IMSRG_ENABLE_MPI (+ MPI::MPI_CXX, IMSRG_MPIEXEC_PREFLAGS), fmt::fmt,
# tomlplusplus::tomlplusplus, CLI11::CLI11, GTest::gtest/gmock (tests), Python3_EXECUTABLE (tests).
# Every bundled dependency is added as SYSTEM, so our warning flags never apply to its headers.

include_guard(GLOBAL)

# ---- Kokkos ---------------------------------------------------------------------------------------
# 1. find_package(Kokkos) in Kokkos_ROOT, which defaults to <deps prefix>/kokkos-<backend> (built by
#    tools/bootstrap_deps.sh). Pass -DKokkos_ROOT=... (e.g. a cluster module) to use another install.
# 2. Otherwise build extern/kokkos in-tree with the options of the selected backend (slow, but a fresh
#    clone still configures with one command).
set(_imsrg_backends serial openmp cuda hip)
if(NOT IMSRG_KOKKOS_BACKEND IN_LIST _imsrg_backends)
  message(FATAL_ERROR "IMSRG_KOKKOS_BACKEND='${IMSRG_KOKKOS_BACKEND}' (expected one of ${_imsrg_backends})")
endif()
string(TOUPPER "${IMSRG_KOKKOS_BACKEND}" _imsrg_backend_uc)

# Apple clang needs Homebrew's libomp; Kokkos' config re-runs find_dependency(OpenMP).
if(APPLE AND IMSRG_KOKKOS_BACKEND STREQUAL "openmp" AND NOT OpenMP_ROOT
   AND "$ENV{OpenMP_ROOT}" STREQUAL "")
  foreach(_omp IN ITEMS /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(EXISTS "${_omp}/include/omp.h")
      set(OpenMP_ROOT "${_omp}" CACHE PATH "OpenMP runtime (Homebrew libomp)")
      break()
    endif()
  endforeach()
endif()

if(NOT Kokkos_ROOT AND "$ENV{Kokkos_ROOT}" STREQUAL "" AND NOT Kokkos_DIR)
  # hip: the install with an OpenMP host space (D73; tools/bootstrap_deps.sh kokkos_dir_name).
  if(IMSRG_KOKKOS_BACKEND STREQUAL "hip")
    set(Kokkos_ROOT "${IMSRG_DEPS_PREFIX}/kokkos-hip-openmp")
  else()
    set(Kokkos_ROOT "${IMSRG_DEPS_PREFIX}/kokkos-${IMSRG_KOKKOS_BACKEND}")
  endif()
  file(GLOB _imsrg_kokkos_config "${Kokkos_ROOT}/lib*/cmake/Kokkos/KokkosConfig.cmake")
else()
  set(_imsrg_kokkos_config "user-given")
endif()
if(_imsrg_kokkos_config)
  # A Kokkos the user named, or one that is installed in the deps prefix, must load: falling back
  # to an in-tree build would silently give a different configuration than requested.
  find_package(Kokkos 5 CONFIG REQUIRED)
else()
  find_package(Kokkos 5 CONFIG QUIET)
endif()

if(Kokkos_FOUND)
  set(IMSRG_KOKKOS_SOURCE "installed: ${Kokkos_DIR}")
  # The default execution space must be the selected backend: with serial, no parallel backend
  # may be enabled; with openmp/cuda/hip, that backend must be enabled.
  if(IMSRG_KOKKOS_BACKEND STREQUAL "serial")
    foreach(_dev IN ITEMS OPENMP THREADS CUDA HIP SYCL)
      if(_dev IN_LIST Kokkos_DEVICES)
        message(FATAL_ERROR "Kokkos in ${Kokkos_DIR} enables ${Kokkos_DEVICES}, but "
                            "IMSRG_KOKKOS_BACKEND=serial needs a Serial-only Kokkos")
      endif()
    endforeach()
  elseif(NOT _imsrg_backend_uc IN_LIST Kokkos_DEVICES)
    message(FATAL_ERROR "Kokkos in ${Kokkos_DIR} enables ${Kokkos_DEVICES}, but "
                        "IMSRG_KOKKOS_BACKEND=${IMSRG_KOKKOS_BACKEND}")
  endif()
else()
  message(STATUS "imsrg: no Kokkos in '${Kokkos_ROOT}'; building extern/kokkos in-tree "
                 "(run tools/bootstrap_deps.sh once to avoid this)")
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/kokkos/CMakeLists.txt")
    message(FATAL_ERROR "extern/kokkos is empty: run `git submodule update --init --recursive`")
  endif()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
  set(Kokkos_ENABLE_DEPRECATED_CODE_5 OFF CACHE BOOL "")
  if(NOT IMSRG_KOKKOS_BACKEND STREQUAL "serial")
    set(Kokkos_ENABLE_${_imsrg_backend_uc} ON CACHE BOOL "")
  endif()
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/kokkos" extern/kokkos SYSTEM)
  get_directory_property(Kokkos_VERSION DIRECTORY "${PROJECT_SOURCE_DIR}/extern/kokkos"
                         DEFINITION Kokkos_VERSION)
  set(IMSRG_KOKKOS_SOURCE "in-tree: extern/kokkos")
endif()

# ---- MPI ------------------------------------------------------------------------------------------
if(NOT DEFINED IMSRG_ENABLE_MPI OR IMSRG_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX QUIET)
endif()
option(IMSRG_ENABLE_MPI "Enable MPI (default: ON if MPI is found)" ${MPI_CXX_FOUND})
if(IMSRG_ENABLE_MPI)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "IMSRG_ENABLE_MPI=ON, but no MPI C++ library was found")
  endif()
  # Open MPI refuses to start more ranks than there are cores unless told otherwise, and binds each
  # rank to one core by default, which would serialize its OpenMP threads (MPICH-style launchers do
  # not know these flags, hence the check).
  execute_process(COMMAND "${MPIEXEC_EXECUTABLE}" --version
                  OUTPUT_VARIABLE _imsrg_mpiexec_version ERROR_QUIET TIMEOUT 20)
  if(_imsrg_mpiexec_version MATCHES "Open MPI|OpenRTE")
    set(_imsrg_mpiexec_default "--oversubscribe --bind-to none")
  else()
    set(_imsrg_mpiexec_default "")
  endif()
  set(IMSRG_MPIEXEC_PREFLAGS "${_imsrg_mpiexec_default}" CACHE STRING
      "Extra flags for ${MPIEXEC_EXECUTABLE} in MPI tests (after MPIEXEC_PREFLAGS)")
endif()

# ---- Bundled libraries (git submodules) -----------------------------------------------------------
foreach(_sub IN ITEMS fmt tomlplusplus CLI11)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${_sub}/CMakeLists.txt")
    message(FATAL_ERROR "extern/${_sub} is empty: run `git submodule update --init --recursive`")
  endif()
endforeach()
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/fmt" extern/fmt SYSTEM EXCLUDE_FROM_ALL)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/tomlplusplus" extern/tomlplusplus SYSTEM
                 EXCLUDE_FROM_ALL)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/CLI11" extern/CLI11 SYSTEM EXCLUDE_FROM_ALL)

if(IMSRG_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" extern/googletest SYSTEM
                   EXCLUDE_FROM_ALL)
endif()

# Python: the layering check and pytest need an interpreter (prefer the shared venv); the nanobind
# module needs the Development.Module component too (python/CMakeLists.txt finds nanobind itself).
if(IMSRG_BUILD_TESTS OR IMSRG_BUILD_PYTHON)
  if(NOT Python3_EXECUTABLE AND EXISTS "${IMSRG_VENV}/bin/python3")
    set(Python3_EXECUTABLE "${IMSRG_VENV}/bin/python3")
  endif()
  set(_imsrg_py_components Interpreter)
  if(IMSRG_BUILD_PYTHON)
    list(APPEND _imsrg_py_components Development.Module)
  endif()
  find_package(Python3 3.12 REQUIRED COMPONENTS ${_imsrg_py_components})  # D29: Python >= 3.12
endif()

# ---- Hooks for later milestones -------------------------------------------------------------------
# BLAS/LAPACK (module linalg): Accelerate on macOS (BLA_VENDOR=Apple, LP64 new LAPACK interface)
# unless the user set BLA_VENDOR; elsewhere FindBLAS chooses (OpenBLAS/MKL/...). If Accelerate is not
# usable on macOS, fall back to Homebrew OpenBLAS. Result: INTERFACE target imsrg::blas_lapack.
set(_imsrg_blas_ok FALSE)
if(APPLE AND NOT BLA_VENDOR)
  set(BLA_VENDOR Apple)
  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)
  if(BLAS_FOUND AND LAPACK_FOUND)
    set(_imsrg_blas_ok TRUE)
  else()
    message(STATUS "Accelerate not usable; falling back to OpenBLAS")
    unset(BLA_VENDOR)
    unset(BLAS_FOUND)
    unset(LAPACK_FOUND)
    foreach(_v BLAS_LIBRARIES LAPACK_LIBRARIES LAPACK_LINKER_FLAGS BLAS_LINKER_FLAGS)
      unset(${_v} CACHE)
    endforeach()
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/openblas /usr/local/opt/openblas)
    set(BLA_VENDOR OpenBLAS)
  endif()
endif()
if(NOT _imsrg_blas_ok)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()
set(_imsrg_blas_all "${BLAS_LIBRARIES};${BLAS_LINKER_FLAGS};${LAPACK_LIBRARIES}")
if(_imsrg_blas_all MATCHES "Accelerate")
  set(_imsrg_blas_name "Accelerate")
  set(_imsrg_blas_accel 1)
else()
  set(_imsrg_blas_name "${BLA_VENDOR}")
  if(NOT _imsrg_blas_name)
    set(_imsrg_blas_name "generic")
  endif()
  set(_imsrg_blas_accel 0)
endif()
add_library(imsrg_blas_lapack INTERFACE)
add_library(imsrg::blas_lapack ALIAS imsrg_blas_lapack)
target_link_libraries(imsrg_blas_lapack INTERFACE LAPACK::LAPACK BLAS::BLAS)
target_compile_definitions(imsrg_blas_lapack INTERFACE IMSRG_BLAS_ACCELERATE=${_imsrg_blas_accel})
if(_imsrg_blas_accel)
  target_compile_definitions(imsrg_blas_lapack INTERFACE ACCELERATE_NEW_LAPACK)
endif()
# Serialize BLAS/LAPACK calls made from different host threads (design/12 section 3): the OpenMP
# variant of cray-libsci (libsci_gnu_mp) crashes when two host threads call dgemm at once, even
# with OMP_NUM_THREADS=1 (FP-Q6). Accelerate is reentrant and GroupedGemm relies on concurrent
# calls with it, so the default is OFF there and ON for every other backend (a lock per call costs
# nothing next to a gemm; it only matters when several host threads, e.g. Python threads, solve).
if(_imsrg_blas_accel)
  set(_imsrg_blas_serialize_default OFF)
else()
  set(_imsrg_blas_serialize_default ON)
endif()
option(IMSRG_BLAS_SERIALIZE "Serialize BLAS/LAPACK calls across host threads"
       ${_imsrg_blas_serialize_default})
if(IMSRG_BLAS_SERIALIZE)
  target_compile_definitions(imsrg_blas_lapack INTERFACE IMSRG_BLAS_SERIALIZE=1)
else()
  target_compile_definitions(imsrg_blas_lapack INTERFACE IMSRG_BLAS_SERIALIZE=0)
endif()
# Concurrent GroupedGemm items (one single-threaded BLAS call per host thread, M12a): only for a
# reentrant, single-threaded-per-call BLAS: Accelerate (default ON), or e.g. the serial cray-libsci
# (BLAS_LIBRARIES=.../libsci_gnu.so with IMSRG_BLAS_SERIALIZE=OFF; M14 CPU baseline, FP-Q6).
option(IMSRG_BLAS_CONCURRENT "Run GroupedGemm items concurrently on the host threads"
       ${_imsrg_blas_accel})
if(IMSRG_BLAS_CONCURRENT AND IMSRG_BLAS_SERIALIZE)
  message(FATAL_ERROR "IMSRG_BLAS_CONCURRENT needs IMSRG_BLAS_SERIALIZE=OFF (a reentrant BLAS)")
endif()
if(IMSRG_BLAS_CONCURRENT)
  target_compile_definitions(imsrg_blas_lapack INTERFACE IMSRG_BLAS_CONCURRENT=1)
else()
  target_compile_definitions(imsrg_blas_lapack INTERFACE IMSRG_BLAS_CONCURRENT=0)
endif()
message(STATUS "BLAS/LAPACK backend: ${_imsrg_blas_name} (${BLAS_LIBRARIES}), "
               "calls serialized across host threads: ${IMSRG_BLAS_SERIALIZE}")
# Device BLAS (module linalg, DeviceGemm/DeviceGroupedGemm, design/12 section 3, M14): GEMMs on
# operands in exec device memory. HOST = the host BLAS above (CPU backends, where device memory is
# host memory); ROCBLAS = rocBLAS on the Kokkos HIP stream; KOKKOSKERNELS = KokkosBlas::gemm
# (portable; Kokkos Kernels from $IMSRG_DEPS_PREFIX/kokkoskernels-<backend>, tools/bootstrap_deps.sh).
# AUTO picks ROCBLAS for hip and HOST for serial/openmp. Result: INTERFACE target imsrg::device_blas.
set(IMSRG_DEVICE_BLAS "AUTO" CACHE STRING "Device BLAS backend (AUTO HOST ROCBLAS KOKKOSKERNELS)")
set_property(CACHE IMSRG_DEVICE_BLAS PROPERTY STRINGS AUTO HOST ROCBLAS KOKKOSKERNELS)
set(_imsrg_device_blas "${IMSRG_DEVICE_BLAS}")
if(_imsrg_device_blas STREQUAL "AUTO")
  if(IMSRG_KOKKOS_BACKEND STREQUAL "hip")
    set(_imsrg_device_blas ROCBLAS)
  else()
    set(_imsrg_device_blas HOST)
  endif()
endif()
if(_imsrg_device_blas STREQUAL "HOST" AND IMSRG_KOKKOS_BACKEND MATCHES "^(cuda|hip)$")
  message(FATAL_ERROR "IMSRG_DEVICE_BLAS=HOST cannot run GEMMs on ${IMSRG_KOKKOS_BACKEND} device "
                      "memory; use ROCBLAS or KOKKOSKERNELS")
endif()
if(_imsrg_device_blas STREQUAL "ROCBLAS" AND NOT IMSRG_KOKKOS_BACKEND STREQUAL "hip")
  message(FATAL_ERROR "IMSRG_DEVICE_BLAS=ROCBLAS needs IMSRG_KOKKOS_BACKEND=hip")
endif()
add_library(imsrg_device_blas INTERFACE)
add_library(imsrg::device_blas ALIAS imsrg_device_blas)
if(_imsrg_device_blas STREQUAL "HOST")
  target_compile_definitions(imsrg_device_blas INTERFACE IMSRG_DEVICE_BLAS_HOST=1)
elseif(_imsrg_device_blas STREQUAL "ROCBLAS")
  find_package(rocblas REQUIRED CONFIG)
  target_link_libraries(imsrg_device_blas INTERFACE roc::rocblas)
  target_compile_definitions(imsrg_device_blas INTERFACE IMSRG_DEVICE_BLAS_ROCBLAS=1)
elseif(_imsrg_device_blas STREQUAL "KOKKOSKERNELS")
  if(NOT KokkosKernels_ROOT AND NOT DEFINED ENV{KokkosKernels_ROOT})
    if(IMSRG_KOKKOS_BACKEND STREQUAL "hip")  # built against kokkos-hip-openmp (D73)
      set(KokkosKernels_ROOT "${IMSRG_DEPS_PREFIX}/kokkoskernels-hip-openmp")
    else()
      set(KokkosKernels_ROOT "${IMSRG_DEPS_PREFIX}/kokkoskernels-${IMSRG_KOKKOS_BACKEND}")
    endif()
  endif()
  find_package(KokkosKernels REQUIRED CONFIG)
  target_link_libraries(imsrg_device_blas INTERFACE Kokkos::kokkoskernels)
  target_compile_definitions(imsrg_device_blas INTERFACE IMSRG_DEVICE_BLAS_KOKKOSKERNELS=1)
else()
  message(FATAL_ERROR "IMSRG_DEVICE_BLAS='${IMSRG_DEVICE_BLAS}' (expected AUTO HOST ROCBLAS "
                      "KOKKOSKERNELS)")
endif()
set(IMSRG_DEVICE_BLAS_RESOLVED "${_imsrg_device_blas}" CACHE INTERNAL "")
message(STATUS "Device BLAS backend: ${_imsrg_device_blas}")
# M5 (io): gzip streams over zlib (D4: no Boost.Iostreams).
find_package(ZLIB REQUIRED)
message(STATUS "zlib: ${ZLIB_LIBRARIES} (${ZLIB_VERSION_STRING})")
# M11 (python): nanobind is added from extern/nanobind in the top-level python hook.
