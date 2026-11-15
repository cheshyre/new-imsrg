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
