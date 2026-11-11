# Tests of the hf module. They build operators, hence the exec runtime main. io is a test-only
# dependency (NO2B reader).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")

foreach(_t golden_2n diis properties)
  imsrg_add_test(NAME hf_${_t}
    SOURCES test_${_t}.cpp ${_imsrg_exec_main}
    DEPS hf operator io
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(hf_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
    IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}"
    IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")
endforeach()

foreach(_t golden_2n3n golden_emax4)
  imsrg_add_test(NAME hf_${_t}
    SOURCES test_${_t}.cpp ${_imsrg_exec_main}
    DEPS hf operator io
    LABELS integration
    NO_GTEST_MAIN)
  target_compile_definitions(hf_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
    IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}"
    IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")
endforeach()

# HNO tests: the emax 2 2N cases are unit tests; the cases that need the NO2B file or the generated
# emax 4 goldens skip when absent, so one binary carries all of them (labels integration).
imsrg_add_test(NAME hf_hno
  SOURCES test_hno.cpp ${_imsrg_exec_main}
  DEPS hf operator io
  LABELS integration reference
  NO_GTEST_MAIN)
target_compile_definitions(hf_hno PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")

# Integration test: Hamiltonian assembled from the me2j / NO2B input files (skips if absent).
imsrg_add_test(NAME hf_integration
  SOURCES test_integration.cpp ${_imsrg_exec_main}
  DEPS hf operator io opslib
  LABELS integration
  NO_GTEST_MAIN)
target_compile_definitions(hf_integration PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")
