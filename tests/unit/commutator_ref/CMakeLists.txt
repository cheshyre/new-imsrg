# Unit tests of the commutator_ref module: golden acceptance of every term against the imsrg-devel
# dumps, and algebraic property tests on random operators (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cc")

imsrg_add_test(NAME commutator_ref_golden
  SOURCES test_golden.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_ref_golden PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

imsrg_add_test(NAME commutator_ref_properties
  SOURCES test_properties.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_ref_properties PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

imsrg_add_test(NAME commutator_ref_factorized
  SOURCES test_factorized.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_ref_factorized PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# The brute-force m-scheme commutator (IMSRG(3) validation oracle, design/14
# section 3) against the J-scheme IMSRG(2) reference and Comm223.
imsrg_add_test(NAME commutator_ref_mscheme
  SOURCES test_mscheme.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
  DEPS commutator_ref commutator operator
  LABELS unit
  NO_GTEST_MAIN)
