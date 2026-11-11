# Unit tests of the commutator_ref module: golden acceptance of every term against the imsrg-devel
# dumps, and algebraic property tests on random operators (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cpp")

imsrg_add_test(NAME commutator_ref_golden
  SOURCES test_golden.cpp ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_include_directories(commutator_ref_golden PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
target_compile_definitions(commutator_ref_golden PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

imsrg_add_test(NAME commutator_ref_properties
  SOURCES test_properties.cpp ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_include_directories(commutator_ref_properties PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
target_compile_definitions(commutator_ref_properties PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
