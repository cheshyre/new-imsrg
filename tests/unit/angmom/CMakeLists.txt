# Unit tests of the angmom module.
#
# angmom_wigner_reference compares against exact sympy values that are generated at build time
# into the build tree (never committed). Without sympy the test skips itself with a warning.

set(_gen "${CMAKE_CURRENT_SOURCE_DIR}/reference/gen_wigner_reference.py")
set(_ref "${CMAKE_CURRENT_BINARY_DIR}/wigner_reference_emax8.txt")
set(_have_sympy FALSE)
if(Python3_EXECUTABLE)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import sympy"
    RESULT_VARIABLE _sympy_rc OUTPUT_QUIET ERROR_QUIET)
  if(_sympy_rc EQUAL 0)
    set(_have_sympy TRUE)
  endif()
endif()

imsrg_add_test(NAME angmom_wigner_reference
  SOURCES wigner_reference_test.cpp
  DEPS angmom
  LABELS unit)
imsrg_add_test(NAME angmom_wigner_identities
  SOURCES wigner_identities_test.cpp
  DEPS angmom
  LABELS unit)

if(_have_sympy)
  add_custom_command(OUTPUT "${_ref}"
    COMMAND "${Python3_EXECUTABLE}" "${_gen}" --emax 8 --out "${_ref}"
    DEPENDS "${_gen}"
    COMMENT "Generating exact sympy Wigner-symbol reference data (once per build tree)"
    VERBATIM)
  add_custom_target(angmom_wigner_reference_data ALL DEPENDS "${_ref}")
  add_dependencies(angmom_wigner_reference angmom_wigner_reference_data)
  target_compile_definitions(angmom_wigner_reference PRIVATE
    IMSRG_WIGNER_REFERENCE_FILE="${_ref}")
else()
  message(WARNING "imsrg: sympy not available for ${Python3_EXECUTABLE}; "
                  "angmom_wigner_reference will be skipped")
endif()

# --- Moshinsky brackets ----------------------------------------------------------------------
# angmom_moshinsky compares against exact rational-arithmetic values generated at build time
# (stdlib-only script, no sympy needed) and against committed imsrg++ samples.
set(_mgen "${CMAKE_CURRENT_SOURCE_DIR}/reference/gen_moshinsky_exact.py")
set(_mref "${CMAKE_CURRENT_BINARY_DIR}/moshinsky_exact_e16.txt")
set(_msamples "${CMAKE_CURRENT_SOURCE_DIR}/data/moshinsky_imsrgpp_samples.txt")

imsrg_add_test(NAME angmom_moshinsky
  SOURCES moshinsky_test.cpp
          ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS angmom exec
  LABELS unit
  NO_GTEST_MAIN)
add_custom_command(OUTPUT "${_mref}"
  COMMAND "${Python3_EXECUTABLE}" "${_mgen}" --emax-e 16 --out "${_mref}"
  DEPENDS "${_mgen}"
  COMMENT "Generating exact Moshinsky reference data (once per build tree)"
  VERBATIM)
add_custom_target(angmom_moshinsky_reference_data ALL DEPENDS "${_mref}")
add_dependencies(angmom_moshinsky angmom_moshinsky_reference_data)
target_compile_definitions(angmom_moshinsky PRIVATE
  IMSRG_MOSHINSKY_EXACT_FILE="${_mref}"
  IMSRG_MOSHINSKY_SAMPLES_FILE="${_msamples}")

# --- Angular-momentum tables (device views; own main initializes the exec runtime) --------------
imsrg_add_test(NAME angmom_tables
  SOURCES tables_test.cpp tables_kernels.cpp
          ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS angmom exec
  LABELS unit
  NO_GTEST_MAIN)
