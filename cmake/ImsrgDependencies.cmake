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
  set(Kokkos_ROOT "${IMSRG_DEPS_PREFIX}/kokkos-${IMSRG_KOKKOS_BACKEND}")
endif()
find_package(Kokkos 5 CONFIG QUIET)

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
  # Open MPI refuses to start more ranks than there are cores unless told otherwise (MPICH-style
  # launchers do not know the flag, hence the check).
  execute_process(COMMAND "${MPIEXEC_EXECUTABLE}" --version
                  OUTPUT_VARIABLE _imsrg_mpiexec_version ERROR_QUIET TIMEOUT 20)
  if(_imsrg_mpiexec_version MATCHES "Open MPI|OpenRTE")
    set(_imsrg_mpiexec_default "--oversubscribe")
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
  # Python runs the layering check (stdlib only) and, later, pytest; prefer the shared venv.
  if(NOT Python3_EXECUTABLE AND EXISTS "${IMSRG_VENV}/bin/python3")
    set(Python3_EXECUTABLE "${IMSRG_VENV}/bin/python3")
  endif()
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
endif()

# ---- Hooks for later milestones -------------------------------------------------------------------
# M2 (linalg): find_package(BLAS) and find_package(LAPACK) go here (Accelerate on macOS via
#   BLA_VENDOR=Apple, OpenBLAS/MKL elsewhere; respect a user-set BLA_VENDOR), wrapped in an
#   INTERFACE target that imsrg_linalg links privately.
# M5 (io): find_package(ZLIB REQUIRED) goes here.
# M11 (python): nanobind is added from extern/nanobind in the top-level python hook.
