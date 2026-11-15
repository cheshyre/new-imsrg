# IMSRG(3) three-body storage layout benchmark (design/14 section 2; results in
# design/benchmarks/imsrg3-layout.md). Prototypes only: built with
# -DIMSRG_BUILD_BENCH=ON, never part of the production library or of the unit
# label. The check executable carries the `slow` label (tools/ci.sh skips it).
add_library(imsrg3_layout_proto STATIC
  sixj3.cc
  canonical_space.cc
  pair_third.cc
  layouts.cc
  kernel_context.cc
  kernels/comm223_ref.cc
  kernels/comm223_tiled.cc
  kernels/triples.cc)
# Headers are included by their path from the repository root
# ("tools/bench/imsrg3_layout/canonical_space.h").
target_include_directories(imsrg3_layout_proto PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(imsrg3_layout_proto
  PUBLIC imsrg::commutator imsrg::operator imsrg::basis imsrg::angmom
         imsrg::linalg imsrg::exec
  PRIVATE imsrg::warnings)

add_executable(imsrg3_layout_bench bench_main.cc)
target_compile_definitions(imsrg3_layout_bench PRIVATE
  IMSRG_BENCH_GIT_HASH="${_bench_git_hash}")
target_link_libraries(imsrg3_layout_bench PRIVATE imsrg3_layout_proto
                                                  imsrg::warnings)

if(IMSRG_BUILD_TESTS)
  imsrg_add_test(NAME imsrg3_layout_check
    SOURCES imsrg3_layout_check.cc
            "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc"
    LIBS imsrg3_layout_proto
    LABELS slow
    NO_GTEST_MAIN
    TIMEOUT 900)
endif()
