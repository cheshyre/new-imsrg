# Unit tests of the fast commutator module (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cc")
set(_imsrg_comm_cases "${CMAKE_CURRENT_SOURCE_DIR}/commutator_test_cases.cc")

foreach(_t options simple_terms ladder_terms particle_hole dispatch robustness
          factorized general)
  imsrg_add_test(NAME commutator_${_t}
    SOURCES test_${_t}.cc ${_imsrg_exec_main} ${_imsrg_comm_support} ${_imsrg_comm_cases}
    DEPS commutator commutator_ref operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(commutator_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
endforeach()

# The [2,2]->3 term and the perturbative triples (IMSRG(3), design/14 section 4
# item 0) against the imsrg-devel goldens of tests/golden/imsrg3.
imsrg_add_test(NAME commutator_comm223
  SOURCES test_comm223.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_comm223 PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
