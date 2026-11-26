# Wrapper build for the fork golden-data harness `forkharness` (tools/reference/fork/forkharness.cpp).
#
# Pulls the sources of the owner's imsrg++ fork at 79e9330 (a patched COPY made by
# tools/reference/build_fork.sh, never the read-only original) in via add_subdirectory and links
# forkharness against its IMSRG interface library. Never part of the new build. Configure it only
# through `tools/reference/build_fork.sh` (it passes the toolchain settings and IMSRG_FORK_SRC).
cmake_minimum_required(VERSION 3.25)
project(imsrg_forkharness LANGUAGES C CXX)

set(IMSRG_FORK_SRC "" CACHE PATH "src/ of a patched copy of the fork (build_fork.sh)")
if(NOT EXISTS "${IMSRG_FORK_SRC}/CMakeLists.txt")
  message(FATAL_ERROR "IMSRG_FORK_SRC=${IMSRG_FORK_SRC} has no CMakeLists.txt")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(IMSRG_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
find_package(OpenMP REQUIRED)
add_subdirectory("${IMSRG_FORK_SRC}" fork)

add_executable(forkharness forkharness.cpp)
target_compile_features(forkharness PRIVATE cxx_std_14)
target_include_directories(forkharness PRIVATE "${IMSRG_FORK_SRC}")
target_link_libraries(forkharness PRIVATE IMSRG OpenMP::OpenMP_CXX)
