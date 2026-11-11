# Tests of the io module (design/13 section 3). The me2j tests build operators, hence the exec
# runtime main.
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")
set(_imsrg_golden_dir "${PROJECT_SOURCE_DIR}/tests/golden")

imsrg_add_test(NAME io_input_stream
  SOURCES test_input_stream.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS unit
  NO_GTEST_MAIN)

imsrg_add_test(NAME io_me2j
  SOURCES test_me2j.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS unit
  NO_GTEST_MAIN)

# Reads of the real emax-14 interaction file against the harness dumps (skipped if absent).
imsrg_add_test(NAME io_me2j_golden
  SOURCES test_me2j_golden.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS integration reference
  NO_GTEST_MAIN)
target_compile_definitions(io_me2j_golden PRIVATE
  IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}")

imsrg_add_test(NAME io_no2b
  SOURCES test_no2b.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS unit
  NO_GTEST_MAIN)

# Reads of the real NO2B 3N file, normal-ordered, against the harness dumps (skipped if absent).
imsrg_add_test(NAME io_no2b_golden
  SOURCES test_no2b_golden.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS integration reference
  NO_GTEST_MAIN)
target_compile_definitions(io_no2b_golden PRIVATE
  IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}")

# Consumes the whole 1.2 GB (uncompressed) file to verify its value count.
imsrg_add_test(NAME io_me2j_full_file
  SOURCES test_me2j_full_file.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS slow integration
  NO_GTEST_MAIN)
target_compile_definitions(io_me2j_full_file PRIVATE
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")

# The tokyo and NuShellX writers against the files of the old code (skipped if absent).
imsrg_add_test(NAME io_valence_writers
  SOURCES test_valence_writers.cpp ${_imsrg_exec_main}
  DEPS io
  LABELS integration reference
  NO_GTEST_MAIN)
target_compile_definitions(io_valence_writers PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")
