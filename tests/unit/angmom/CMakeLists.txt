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
