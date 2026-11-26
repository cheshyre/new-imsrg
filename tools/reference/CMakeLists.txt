# Wrapper build for the golden-data harness `refharness`.
#
# Pulls the OLD imsrg++ sources (imsrg-devel src/, normally the pristine copy made by
# build_imsrg_devel.sh) in via add_subdirectory and links refharness against its IMSRG
# library. This project is never part of the new build. Configure it through
# `tools/reference/build_imsrg_devel.sh harness` (it passes the toolchain settings).
cmake_minimum_required(VERSION 3.25)
project(imsrg_refharness LANGUAGES C CXX)

# build_imsrg_devel.sh always passes -DIMSRG_REFERENCE_SRC; the default is only for manual use.
if(DEFINED ENV{IMSRG_REFERENCE_BUILD})
  set(_imsrg_ref_build "$ENV{IMSRG_REFERENCE_BUILD}")
elseif(DEFINED ENV{IMSRG_WORK})
  set(_imsrg_ref_build "$ENV{IMSRG_WORK}/reference-build")
elseif(DEFINED ENV{IMSRG_REFERENCE_ROOT})
  set(_imsrg_ref_build "$ENV{IMSRG_REFERENCE_ROOT}")
else()
  set(_imsrg_ref_build "/Users/cheshyre/Downloads/new-imsrg-reference")
endif()
set(IMSRG_REFERENCE_SRC "${_imsrg_ref_build}/imsrg-devel-src/src"
    CACHE PATH "imsrg-devel src/ directory (a pristine copy; never the read-only original)")
if(NOT EXISTS "${IMSRG_REFERENCE_SRC}/CMakeLists.txt")
  message(FATAL_ERROR "IMSRG_REFERENCE_SRC=${IMSRG_REFERENCE_SRC} has no CMakeLists.txt")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(IMSRG_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
find_package(OpenMP REQUIRED)  # imported targets are directory-scoped: find it here too

# The old src/CMakeLists.txt redirects CMAKE_CURRENT_BINARY_DIR to "../build" for its generated
# version.cc, so its binary directory must itself be called "build" (then ../build == .).
add_subdirectory("${IMSRG_REFERENCE_SRC}" build)

add_executable(refharness
  refharness.cpp
  rh_basis.cpp
  rh_opdump.cpp
  rh_hamiltonian.cpp
  rh_operator.cpp
  rh_valence.cpp
  rh_commutator.cpp
  rh_bench.cpp
  rh_imsrg3.cpp
  rh_comm3.cpp
  rh_flow.cpp
  rh_factorized.cpp
  rh_tensor.cpp
)
target_compile_features(refharness PRIVATE cxx_std_17)
target_include_directories(refharness PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${IMSRG_REFERENCE_SRC}"
  "${IMSRG_REFERENCE_SRC}/Commutator"
  "${IMSRG_REFERENCE_SRC}/ThreeBodyME"
  "${IMSRG_REFERENCE_SRC}/imsrg_util"
  "${IMSRG_REFERENCE_SRC}/ReferenceImplementations")
# (NO_HDF5 arrives through the IMSRG link interface, as for imsrg++.)
target_link_libraries(refharness PRIVATE IMSRG OpenMP::OpenMP_CXX)
