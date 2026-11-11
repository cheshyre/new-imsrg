# Tests of the opslib module (kinetic energy, center of mass, radii). They build operators, hence
# the exec runtime main.
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")

imsrg_add_test(NAME opslib_golden
  SOURCES test_golden.cpp ${_imsrg_exec_main}
  DEPS opslib operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(opslib_golden PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}")

imsrg_add_test(NAME opslib_properties
  SOURCES test_properties.cpp ${_imsrg_exec_main}
  DEPS opslib operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(opslib_properties PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# Timings of t_rel and rp2 at emax 6 (printed, not asserted).
imsrg_add_test(NAME opslib_timing
  SOURCES test_timing.cpp ${_imsrg_exec_main}
  DEPS opslib operator
  LABELS unit
  NO_GTEST_MAIN)
