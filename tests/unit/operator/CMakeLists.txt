# Unit tests of the operator module (design/30 sections 1-2). Every test needs the exec runtime
# (DeviceBuffer), hence exec_gtest_main.cpp.
set(_imsrg_golden_dir "${PROJECT_SOURCE_DIR}/tests/golden")
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")

imsrg_add_test(NAME operator_core
  SOURCES test_operator_spec.cpp test_one_body.cpp test_two_body.cpp test_operator.cpp
          test_compare.cpp ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME operator_arithmetic
  SOURCES test_arithmetic.cpp test_norms.cpp ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME operator_dump
  SOURCES test_operator_dump.cpp test_golden_dumps.cpp ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(operator_dump PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

imsrg_add_test(NAME operator_golden_norms
  SOURCES test_golden_norms.cpp ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(operator_golden_norms PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

foreach(_t normal_order no2b basis_transform random_operator)
  imsrg_add_test(NAME operator_${_t}
    SOURCES test_${_t}.cpp ${_imsrg_exec_main}
    DEPS operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(operator_${_t} PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")
endforeach()

# Needs the NO2B input file in IMSRG_TEST_DATA_DIR (outside the repo); skips itself if absent.
imsrg_add_test(NAME operator_no2b_golden
  SOURCES test_no2b_golden.cpp ${_imsrg_exec_main}
  DEPS operator
  LABELS integration
  NO_GTEST_MAIN)
target_compile_definitions(operator_no2b_golden PRIVATE
  IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}" IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")
