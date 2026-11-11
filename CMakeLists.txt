# imsrg: In-Medium Similarity Renormalization Group (C++20, Kokkos, MPI, nanobind).
# Build structure: design/20-build-and-tooling.md section 2; module layout: design/10-architecture.md.
# Minimum 3.25 (add_subdirectory SYSTEM); policies up to 3.31 where available, e.g. CMP0156, which
# de-duplicates static libraries on link lines for Apple's linker ("ignoring duplicate libraries").
cmake_minimum_required(VERSION 3.25...3.31)

# Must run before project(): picks a macOS SDK that links if the default one does not.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ImsrgAppleSdk.cmake")

project(imsrg
  VERSION 0.1.0
  DESCRIPTION "In-Medium Similarity Renormalization Group for nuclear structure"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options (design/20 section 2) ----------------------------------------------------------------
option(IMSRG_BUILD_TESTS "Build the tests (GoogleTest, CTest)" ON)
option(IMSRG_BUILD_PYTHON "Build the nanobind module imsrg._core (once python/ exists, M11)" ON)
option(IMSRG_ENABLE_KOKKOS_KERNELS "Use Kokkos Kernels (not supported yet)" OFF)
option(IMSRG_UNITY_BUILD "Unity builds for the imsrg module libraries" OFF)
option(IMSRG_WERROR "Treat warnings in imsrg targets as errors (ON in tools/ci.sh)" OFF)
option(IMSRG_NATIVE_ARCH "Optimize for this machine (-mcpu=native on arm64, else -march=native)" OFF)
option(IMSRG_USE_CCACHE "Use ccache as the compiler launcher if it is found" ON)
set(IMSRG_SANITIZE "none" CACHE STRING
    "Sanitizers: none, or a list of address, undefined, thread (e.g. address;undefined)")
set(IMSRG_KOKKOS_BACKEND "openmp" CACHE STRING
    "Kokkos backend: selects kokkos-<backend> in the deps prefix and the in-tree fallback options")
set_property(CACHE IMSRG_KOKKOS_BACKEND PROPERTY STRINGS serial openmp cuda hip)
set(IMSRG_DEPS_PREFIX "" CACHE PATH
    "tools/bootstrap_deps.sh install prefix (empty: $IMSRG_DEPS_PREFIX, else $HOME/.cache/imsrg-deps)")
set(IMSRG_VENV "" CACHE PATH "Shared Python venv (empty: $IMSRG_VENV, else <deps prefix>/venv)")
set(IMSRG_TEST_DATA_DIR "/Users/cheshyre/Downloads/new-imsrg" CACHE PATH
    "Directory with the big me2j/NO2B inputs; integration tests are skipped if they are absent")
set(IMSRG_REFERENCE_DIR "" CACHE PATH "imsrg-devel build, for golden-data generation (optional)")
# IMSRG_ENABLE_MPI is defined in ImsrgDependencies.cmake (default: ON if MPI is found).

# The same defaulting rule as tools/lib/imsrg_env.sh (see AGENTS.md).
if(IMSRG_DEPS_PREFIX STREQUAL "")
  if(NOT "$ENV{IMSRG_DEPS_PREFIX}" STREQUAL "")
    set(IMSRG_DEPS_PREFIX "$ENV{IMSRG_DEPS_PREFIX}")
  else()
    set(IMSRG_DEPS_PREFIX "$ENV{HOME}/.cache/imsrg-deps")
  endif()
endif()
if(IMSRG_VENV STREQUAL "")
  if(NOT "$ENV{IMSRG_VENV}" STREQUAL "")
    set(IMSRG_VENV "$ENV{IMSRG_VENV}")
  else()
    set(IMSRG_VENV "${IMSRG_DEPS_PREFIX}/venv")
  endif()
endif()

include(ImsrgCompilerFlags)  # ccache, sanitizers, native arch, imsrg::warnings
include(ImsrgDependencies)   # Kokkos, MPI, fmt, toml++, CLI11, GoogleTest, Python
include(ImsrgModule)         # imsrg_add_module()
if(IMSRG_BUILD_TESTS)
  enable_testing()
  include(ImsrgTesting)      # imsrg_add_test()
endif()

add_subdirectory(src)
if(EXISTS "${PROJECT_SOURCE_DIR}/apps/CMakeLists.txt")
  add_subdirectory(apps)
endif()
if(IMSRG_BUILD_PYTHON)
  if(EXISTS "${PROJECT_SOURCE_DIR}/python/CMakeLists.txt")
    # Hook for M11: find Python (Interpreter + Development.Module) and nanobind
    # (add_subdirectory(extern/nanobind)) here, then add python/.
    message(FATAL_ERROR "python/ exists but the nanobind wiring is not written yet (M11)")
  else()
    message(STATUS "imsrg: IMSRG_BUILD_PYTHON=ON, but python/ does not exist yet (M11)")
  endif()
endif()
if(IMSRG_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Written after all modules are defined; read by the layering check (tests/CMakeLists.txt).
imsrg_write_module_deps("${PROJECT_BINARY_DIR}/imsrg_module_deps.json")

message(STATUS "imsrg: backend=${IMSRG_KOKKOS_BACKEND} Kokkos=${Kokkos_VERSION} (${IMSRG_KOKKOS_SOURCE})"
               " MPI=${IMSRG_ENABLE_MPI} build_type=${CMAKE_BUILD_TYPE} werror=${IMSRG_WERROR}"
               " sanitize=${IMSRG_SANITIZE}")
