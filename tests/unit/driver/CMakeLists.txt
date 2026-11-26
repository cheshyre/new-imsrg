# Unit tests of the driver module's configuration layer (design/16-config-and-cli.md section 6.3).
# Test names are global, hence the driver_ prefix. One executable per component so that parallel
# work on different components does not rebuild each other's tests.
set(_driver_test_data "${CMAKE_CURRENT_SOURCE_DIR}/data")
# Out-of-line helpers of the integration-test headers (integration/*.h); every test that includes
# them compiles these sources itself, because they read the IMSRG_*_DIR definitions of the target.
set(_it_support integration/mini_json.cc integration/run_support.cc integration/snt_compare.cc)

# Schema table, enum spellings (design/16 6.3 item 7, enum part).
imsrg_add_test(NAME driver_schema SOURCES test_schema.cc DEPS driver LABELS unit)

# TOML reader, --set, resolved-config writer.
imsrg_add_test(NAME driver_reader SOURCES test_toml_reader.cc test_writer.cc DEPS driver
               LABELS unit)

# Resolution (defaults, derived values) and the validation table (item 8).
imsrg_add_test(NAME driver_resolve SOURCES test_resolve.cc test_validate.cc DEPS driver
               LABELS unit)

# Relative-path rules of load_config (config-file directory vs working directory, D37).
imsrg_add_test(NAME driver_paths SOURCES test_paths.cc DEPS driver LABELS unit)

# Legacy key=value translation (items 1-5); data/legacy_keys.txt lists the 85 old keys.
imsrg_add_test(NAME driver_legacy SOURCES test_legacy_parameters.cc DEPS driver LABELS unit)
target_compile_definitions(driver_legacy PRIVATE
  "IMSRG_DRIVER_TEST_DATA=\"${_driver_test_data}\"")

# Operator-name grammar and catalog (item 9).
imsrg_add_test(NAME driver_operators SOURCES test_operator_request.cc DEPS driver LABELS unit)

# Golden equivalence of the design/16 2.11 examples, TOML vs legacy (item 6), and the
# RunConfig -> TOML -> RunConfig round trip (item 7).
imsrg_add_test(NAME driver_golden SOURCES test_golden_equivalence.cc test_round_trip.cc
               DEPS driver LABELS unit)
target_compile_definitions(driver_golden PRIVATE "IMSRG_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# The imsrg executable (item 10): spawns it and checks output, files and exit codes.
if(TARGET imsrg)
  imsrg_add_test(NAME driver_cli SOURCES test_cli.cc DEPS driver LABELS unit)
  target_compile_definitions(driver_cli PRIVATE
    "IMSRG_EXE_PATH=\"$<TARGET_FILE:imsrg>\"" "IMSRG_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
  add_dependencies(driver_cli imsrg)
endif()

# The point-proton -> charge-radius conversion of the output stage (D35).
imsrg_add_test(NAME driver_charge_radius SOURCES test_charge_radius.cc DEPS driver LABELS unit)

# Pipeline stage 1 (model space, interaction, HF, normal ordering) against the HF goldens; needs
# the input files in IMSRG_TEST_DATA_DIR and skips without them.
imsrg_add_test(NAME driver_pipeline_hamiltonian
  SOURCES test_pipeline_hamiltonian.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_pipeline_hamiltonian PRIVATE
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")

# End-to-end integration tests IT-2, IT-3, IT-4, IT-6, IT-7, IT-10 (emax 2), IT-12 (design/30 section 4): run the
# pipeline in process and compare with tests/golden/e2e; skip without the input files.
imsrg_add_test(NAME driver_integration
  SOURCES ${_it_support} test_integration.cc test_hunter_gatherer.cc test_stage_two_options.cc test_results_schema.cc
  test_imsrg3f2.cc test_checkpoint.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_integration PRIVATE
  "IMSRG_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")

# The ODE flow methods (flow_rk4, flow) against the imsrg++ goldens of tests/golden/e2e/IT-14: He4 and
# the driver level in the regular suite (He4 emax 2, O16 emax 3), O16 emax 4 (7-14 s each in a debug build) under `slow`.
foreach(_variant "" "_slow")
  if(_variant STREQUAL "")
    set(_flow_labels integration)
  else()
    set(_flow_labels integration slow)
  endif()
  imsrg_add_test(NAME driver_flow_methods${_variant}
    SOURCES ${_it_support} test_flow_methods.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
    DEPS driver mbpt
    LABELS ${_flow_labels}
    NO_GTEST_MAIN)
  target_compile_definitions(driver_flow_methods${_variant} PRIVATE
    "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
    "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")
endforeach()
target_compile_definitions(driver_flow_methods_slow PRIVATE IMSRG_FLOW_SLOW=1)
# IT-8 perturbative triples after the Magnus flow (He4 emax 2 and O16 emax 3, about 7 s each in Debug).
imsrg_add_test(NAME driver_triples
  SOURCES ${_it_support} test_triples.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_triples PRIVATE
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")

# IT-9 IMSRG(3)-N7 Magnus flow (He4 and O16 emax 2) against tests/golden/e2e/IT-9; the emax-3 cases
# (about 30 and 21 minutes in imsrg++) are the same source compiled for the slow label.
imsrg_add_test(NAME driver_imsrg3_n7
  SOURCES ${_it_support} test_imsrg3_n7.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_imsrg3_n7 PRIVATE
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")
imsrg_add_test(NAME driver_imsrg3_n7_slow
  SOURCES ${_it_support} test_imsrg3_n7.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS driver LABELS slow integration TIMEOUT 7200 NO_GTEST_MAIN)
target_compile_definitions(driver_imsrg3_n7_slow PRIVATE IMSRG_IT9_SLOW
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")

# driver::Run on several MPI ranks (M12b phase 1, D4): He4 emax 2 gives the serial results, only
# rank 0 writes files. One source, two rank counts; the shared MPI main provides the Kokkos runtime.
foreach(_ranks 2 4)
  imsrg_add_test(NAME driver_mpi_${_ranks}ranks
    SOURCES ${_it_support} test_mpi_driver.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
    DEPS driver comm LABELS mpi MPI_RANKS ${_ranks} NO_GTEST_MAIN TIMEOUT 600)
  if(TARGET driver_mpi_${_ranks}ranks)
    target_compile_definitions(driver_mpi_${_ranks}ranks PRIVATE
      "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
      "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")
  endif()
endforeach()

# CheckReplicatedSetup (M12b review): equal setups pass, rank-dependent occupations or thread counts throw
# on every rank.
foreach(_ranks 2 4)
  imsrg_add_test(NAME driver_mpi_check_${_ranks}ranks
    SOURCES test_mpi_replicated_check.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
    DEPS driver comm LABELS mpi MPI_RANKS ${_ranks} NO_GTEST_MAIN TIMEOUT 300)
endforeach()

# IT-11 (M12b D5): end-to-end MPI invariance. Flows on 2 and 4 ranks reproduce the serial run (results.json
# and .snt to 1e-12 max(1, |ref|), identical step counts), and all ranks hold identical bits of H(s) after every
# step. The slow variant (same source, IMSRG_IT11_SLOW) holds N7 and the perturbative triples; it uses the
# whole machine (ranks x threads = 18 and 16): in phase 1 the N7 terms run on rank 0 only, and rank 0 also
# runs the serial reference, so its thread count sets the wall time (the case list and its coverage
# argument are in test_mpi_it11.cc).
foreach(_variant "" "_slow")
  foreach(_ranks 2 4)
    if(_variant STREQUAL "")
      set(_labels mpi)
      set(_timeout 900)
      math(EXPR _threads "8 / ${_ranks}")
    else()
      set(_labels mpi slow)
      set(_timeout 3600)
      math(EXPR _threads "18 / ${_ranks}")
    endif()
    imsrg_add_test(NAME driver_mpi_it11${_variant}_${_ranks}ranks
      SOURCES ${_it_support} test_mpi_it11.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
      DEPS driver comm LABELS ${_labels} MPI_RANKS ${_ranks} THREADS ${_threads} NO_GTEST_MAIN
      TIMEOUT ${_timeout})
    if(TARGET driver_mpi_it11${_variant}_${_ranks}ranks)
      target_compile_definitions(driver_mpi_it11${_variant}_${_ranks}ranks PRIVATE
        "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
        "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")
      if(_variant STREQUAL "_slow")
        target_compile_definitions(driver_mpi_it11${_variant}_${_ranks}ranks PRIVATE IMSRG_IT11_SLOW=1)
      endif()
    endif()
  endforeach()
endforeach()
