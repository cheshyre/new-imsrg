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
