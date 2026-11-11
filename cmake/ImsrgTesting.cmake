# ImsrgTesting.cmake -- the one way to define a C++ test executable (design/30 section 1).
#
#   imsrg_add_test(NAME <name> SOURCES <file>...
#                  LABELS <label>...            # unit | integration | mpi | slow | reference
#                  [DEPS <module>...]           # imsrg modules, e.g. base exec
#                  [LIBS <target>...]           # other targets, e.g. Kokkos::kokkos MPI::MPI_CXX
#                  [MPI_RANKS <n>]              # run under mpiexec -n <n> (adds the mpi label)
#                  [NO_GTEST_MAIN]              # the test provides main() (Kokkos/MPI init)
#                  [TIMEOUT <seconds>]          # default 300
#                  [ARGS <arg>...])             # extra command-line arguments for the test
#
# Without MPI_RANKS, each GoogleTest case becomes its own CTest test named <name>.<Suite>.<Case>
# (gtest_discover_tests, discovered when ctest runs, so nothing executes at build time). With
# MPI_RANKS, the whole executable is one CTest test run under mpiexec; it is skipped (not defined)
# when IMSRG_ENABLE_MPI is OFF. Select tests with `ctest -L <label>`.

include_guard(GLOBAL)
include(GoogleTest)

set(IMSRG_TEST_LABELS unit integration mpi slow reference)

function(imsrg_add_test)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "NO_GTEST_MAIN" "NAME;MPI_RANKS;TIMEOUT"
                        "SOURCES;LABELS;DEPS;LIBS;ARGS")
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "imsrg_add_test: unexpected arguments '${ARG_UNPARSED_ARGUMENTS}'")
  endif()
  if(NOT ARG_NAME OR NOT ARG_SOURCES)
    message(FATAL_ERROR "imsrg_add_test: NAME and SOURCES are required")
  endif()
  set(_labels ${ARG_LABELS})
  if(ARG_MPI_RANKS AND NOT "mpi" IN_LIST _labels)
    list(APPEND _labels mpi)
  endif()
  if(NOT _labels)
    message(FATAL_ERROR "imsrg_add_test(${ARG_NAME}): LABELS is required (${IMSRG_TEST_LABELS})")
  endif()
  foreach(_l IN LISTS _labels)
    if(NOT _l IN_LIST IMSRG_TEST_LABELS)
      message(FATAL_ERROR "imsrg_add_test(${ARG_NAME}): unknown label '${_l}' (${IMSRG_TEST_LABELS})")
    endif()
  endforeach()
  if(ARG_MPI_RANKS AND NOT IMSRG_ENABLE_MPI)
    message(STATUS "imsrg: skipping MPI test ${ARG_NAME} (IMSRG_ENABLE_MPI=OFF)")
    return()
  endif()
  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 300)
  endif()

  add_executable(${ARG_NAME} ${ARG_SOURCES})
  target_link_libraries(${ARG_NAME} PRIVATE imsrg::warnings GTest::gtest GTest::gmock)
  if(NOT ARG_NO_GTEST_MAIN)
    target_link_libraries(${ARG_NAME} PRIVATE GTest::gtest_main)
  endif()
  foreach(_dep IN LISTS ARG_DEPS)
    target_link_libraries(${ARG_NAME} PRIVATE imsrg::${_dep})
  endforeach()
  if(ARG_LIBS)
    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})
  endif()

  if(ARG_MPI_RANKS)
    separate_arguments(_extra UNIX_COMMAND "${IMSRG_MPIEXEC_PREFLAGS}")
    add_test(NAME ${ARG_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_MPI_RANKS}
                     ${MPIEXEC_PREFLAGS} ${_extra} $<TARGET_FILE:${ARG_NAME}>
                     ${MPIEXEC_POSTFLAGS} ${ARG_ARGS})
    set_tests_properties(${ARG_NAME} PROPERTIES
      LABELS "${_labels}" PROCESSORS ${ARG_MPI_RANKS} TIMEOUT ${ARG_TIMEOUT})
  else()
    gtest_discover_tests(${ARG_NAME}
      TEST_PREFIX "${ARG_NAME}."
      EXTRA_ARGS ${ARG_ARGS}
      DISCOVERY_MODE PRE_TEST
      DISCOVERY_TIMEOUT 60
      TEST_LIST ${ARG_NAME}_TESTS
      PROPERTIES TIMEOUT ${ARG_TIMEOUT})
    # Labels are set by a CTest include file that runs after the discovery file: the PROPERTIES of
    # gtest_discover_tests wraps every value in a bracket argument, which turns a label list such
    # as "integration;slow" into one label.
    set(_labels_file "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}_labels.cmake")
    file(CONFIGURE OUTPUT "${_labels_file}" CONTENT
"if(${ARG_NAME}_TESTS)
  set_tests_properties(\${${ARG_NAME}_TESTS} PROPERTIES LABELS \"${_labels}\")
endif()
" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_labels_file}")
  endif()
endfunction()
