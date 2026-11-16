# Tests of the mbpt module. hf, io and opslib are test-only dependencies (HF solution, input
# readers, Hamiltonian assembly); mbpt itself does not depend on them.
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")
# Out-of-line helpers of hf_test_support.h and mbpt_test_support.h, compiled into every test that
# includes them.
set(_mbpt_support mbpt_test_support.cc ../hf/hf_test_support.cc)

# energies: emax 2 2N cases are unit tests; the 2N3N case skips without the NO2B file.
imsrg_add_test(NAME mbpt_energies
  SOURCES test_energies.cc ${_mbpt_support} ${_imsrg_exec_main}
  DEPS mbpt hf operator io
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(mbpt_energies PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")

imsrg_add_test(NAME mbpt_natural_orbitals
  SOURCES test_natural_orbitals.cc ${_mbpt_support} ${_imsrg_exec_main}
  DEPS mbpt hf operator io
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(mbpt_natural_orbitals PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_GENERATED_GOLDEN_DIR="${IMSRG_GENERATED_GOLDEN_DIR}"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")

# Integration tests IT-1 (EMP2) and IT-5 (NAT-basis E_HF); skip without the input files.
imsrg_add_test(NAME mbpt_integration
  SOURCES test_integration.cc ${_mbpt_support} ${_imsrg_exec_main}
  DEPS mbpt hf operator io opslib
  LABELS integration
  NO_GTEST_MAIN)
target_compile_definitions(mbpt_integration PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden"
  IMSRG_TEST_DATA_DIR="${IMSRG_TEST_DATA_DIR}")
