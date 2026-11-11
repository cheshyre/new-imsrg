# Unit tests of the fast commutator module (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cpp")
set(_imsrg_comm_cases "${CMAKE_CURRENT_SOURCE_DIR}/commutator_test_cases.cpp")

foreach(_t options simple_terms ladder_terms particle_hole dispatch robustness)
  imsrg_add_test(NAME commutator_${_t}
    SOURCES test_${_t}.cpp ${_imsrg_exec_main} ${_imsrg_comm_support} ${_imsrg_comm_cases}
    DEPS commutator commutator_ref operator
    LABELS unit
    NO_GTEST_MAIN)
  target_include_directories(commutator_${_t} PRIVATE
    ${PROJECT_SOURCE_DIR}/tests/support ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(commutator_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
endforeach()
