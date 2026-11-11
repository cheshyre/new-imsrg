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

add_executable(commutator_bench commutator_bench.cpp)
target_compile_definitions(commutator_bench PRIVATE IMSRG_BENCH_GIT_HASH="${_bench_git_hash}")
target_link_libraries(commutator_bench PRIVATE imsrg::commutator imsrg::operator imsrg::exec
                                               imsrg::basis imsrg::warnings)
