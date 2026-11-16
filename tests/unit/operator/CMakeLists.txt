# Unit tests of the operator module (design/30 sections 1-2). Every test needs the exec runtime
# (DeviceBuffer), hence exec_gtest_main.cc.
set(_imsrg_golden_dir "${PROJECT_SOURCE_DIR}/tests/golden")
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")

imsrg_add_test(NAME operator_core
  SOURCES test_operator_spec.cc test_one_body.cc test_two_body.cc test_three_body.cc test_operator.cc
          test_compare.cc ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME operator_arithmetic
  SOURCES test_arithmetic.cc test_elementwise.cc test_norms.cc ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME operator_dump
  SOURCES test_operator_dump.cc test_operator_dump_3b.cc test_golden_dumps.cc
          test_golden_imsrg3_dump.cc ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(operator_dump PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

imsrg_add_test(NAME operator_golden_norms
  SOURCES test_golden_norms.cc ${_imsrg_exec_main}
  DEPS operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(operator_golden_norms PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

foreach(_t normal_order no2b no2b_get_pn basis_transform random_operator)
  imsrg_add_test(NAME operator_${_t}
    SOURCES test_${_t}.cc ${_imsrg_exec_main}
    DEPS operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(operator_${_t} PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")
endforeach()

# Needs the NO2B input file in IMSRG_TEST_DATA_DIR (outside the repo); skips itself if absent.
imsrg_add_test(NAME operator_no2b_golden
  SOURCES test_no2b_golden.cc ${_imsrg_exec_main}
  DEPS operator
  LABELS integration
  NO_GTEST_MAIN)
target_compile_definitions(operator_no2b_golden PRIVATE
  IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}" IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")

# Pair-third states and the R / E maps (IMSRG(3), design/14 section 4 item 0).
imsrg_add_test(NAME operator_pair_third
  SOURCES test_pair_third.cc pair_third_kernels.cc ${_imsrg_exec_main}
  DEPS operator exec
  LABELS unit
  NO_GTEST_MAIN)

# Masking helpers of the row-range tests (tests/support/row_mask.cc).
imsrg_add_test(NAME operator_three_body_row_mask
  SOURCES test_three_body_row_mask.cc
          ${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc ${_imsrg_exec_main}
  DEPS operator basis
  LABELS unit
  NO_GTEST_MAIN)

# Distributed op::ThreeBody (M12b D7): the same executable serially (unit) and on 2 and 4 ranks.
foreach(_ranks 2 4)
  imsrg_add_test(NAME operator_three_body_distribution_${_ranks}ranks
    SOURCES test_three_body_distribution.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
    DEPS operator basis comm exec
    MPI_RANKS ${_ranks}
    LABELS mpi
    NO_GTEST_MAIN
    TIMEOUT 300)
endforeach()
imsrg_add_test(NAME operator_three_body_distribution_serial
  SOURCES test_three_body_distribution.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
  DEPS operator basis comm exec
  LABELS unit
  NO_GTEST_MAIN)
