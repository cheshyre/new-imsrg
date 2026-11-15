# Unit tests of the flow module (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")

foreach(_t generator bch magnus ode)
  imsrg_add_test(NAME flow_${_t}
    SOURCES test_${_t}.cc ${_imsrg_exec_main}
    DEPS flow commutator operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(flow_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
endforeach()

# The IMSRG(3f2) correction of BCH, with the golden and random-operator helpers
# of the commutator tests.
imsrg_add_test(NAME flow_bch_factorized
  SOURCES test_bch_factorized.cc ${_imsrg_exec_main}
          "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cc"
  DEPS flow commutator operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(flow_bch_factorized PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# The three-body part of the generator against tests/golden/n7.
imsrg_add_test(NAME flow_generator3
  SOURCES test_generator3.cc ${_imsrg_exec_main}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
  DEPS flow commutator operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(flow_generator3 PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
