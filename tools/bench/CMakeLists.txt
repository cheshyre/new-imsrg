# Micro-benchmarks (design/30 section 5). Built only with -DIMSRG_BUILD_BENCH=ON; release presets.
find_package(Git QUIET)
set(_bench_git_hash "unknown")
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE _bench_git_out OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _bench_git_rc ERROR_QUIET)
  if(_bench_git_rc EQUAL 0)
    set(_bench_git_hash "${_bench_git_out}")
  endif()
endif()

add_executable(commutator_bench commutator_bench.cc)
target_compile_definitions(commutator_bench PRIVATE IMSRG_BENCH_GIT_HASH="${_bench_git_hash}")
target_link_libraries(commutator_bench PRIVATE imsrg::commutator imsrg::comm imsrg::operator imsrg::exec
                                               imsrg::basis imsrg::warnings)

# Scalar-tensor commutator benchmark (M15).
add_executable(tensor_commutator_bench tensor_commutator_bench.cc)
target_link_libraries(tensor_commutator_bench PRIVATE imsrg::commutator imsrg::operator imsrg::exec
                                                      imsrg::basis imsrg::warnings)

# IMSRG(3) three-body storage layout benchmark (design/14 section 2,
# design/benchmarks/imsrg3-layout.md).
add_subdirectory(imsrg3_layout)

# Random particle-rank-3 operators (physical 3B part) as OperatorDumps: the inputs of the
# IMSRG(3)-N7 reference goldens (tools/reference/gen_golden_n7.sh).
add_executable(imsrg_n7_inputs
  n7_inputs.cc
  "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc")
target_include_directories(imsrg_n7_inputs PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(imsrg_n7_inputs PRIVATE imsrg::commutator imsrg::comm imsrg::operator imsrg::exec
                                              imsrg::basis imsrg::warnings)

# IMSRG(3)-N7 timing/checksum benchmark (milestone M10).
add_executable(n7_bench
  n7_bench.cc
  "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc")
target_include_directories(n7_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(n7_bench PRIVATE IMSRG_BENCH_GIT_HASH="${_bench_git_hash}")
target_link_libraries(n7_bench PRIVATE imsrg::commutator imsrg::comm imsrg::operator imsrg::exec
                                       imsrg::basis imsrg::warnings)

# One BCH transform (M14: device residency of a BCH series).
add_executable(bch_bench bch_bench.cc)
target_link_libraries(bch_bench PRIVATE imsrg::flow imsrg::commutator imsrg::operator imsrg::exec
                                        imsrg::basis imsrg::warnings)
