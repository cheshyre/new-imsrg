# Unit tests of the driver module's configuration layer (design/16-config-and-cli.md section 6.3).
# Test names are global, hence the driver_ prefix. One executable per component so that parallel
# work on different components does not rebuild each other's tests.
set(_driver_test_data "${CMAKE_CURRENT_SOURCE_DIR}/data")

# Schema table, enum spellings (design/16 6.3 item 7, enum part).
imsrg_add_test(NAME driver_schema SOURCES test_schema.cpp DEPS driver LABELS unit)

# TOML reader, --set, resolved-config writer.
imsrg_add_test(NAME driver_reader SOURCES test_toml_reader.cpp test_writer.cpp DEPS driver
               LABELS unit)

# Resolution (defaults, derived values) and the validation table (item 8).
imsrg_add_test(NAME driver_resolve SOURCES test_resolve.cpp test_validate.cpp DEPS driver
               LABELS unit)

# Legacy key=value translation (items 1-5); data/legacy_keys.txt lists the 85 old keys.
imsrg_add_test(NAME driver_legacy SOURCES test_legacy_parameters.cpp DEPS driver LABELS unit)
target_compile_definitions(driver_legacy PRIVATE
  "IMSRG_DRIVER_TEST_DATA=\"${_driver_test_data}\"")

# Operator-name grammar and catalog (item 9).
imsrg_add_test(NAME driver_operators SOURCES test_operator_request.cpp DEPS driver LABELS unit)

# Golden equivalence of the design/16 2.11 examples, TOML vs legacy (item 6), and the
# RunConfig -> TOML -> RunConfig round trip (item 7).
imsrg_add_test(NAME driver_golden SOURCES test_golden_equivalence.cpp test_round_trip.cpp
               DEPS driver LABELS unit)
target_compile_definitions(driver_golden PRIVATE "IMSRG_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")

# The imsrg executable (item 10): spawns it and checks output, files and exit codes.
if(TARGET imsrg)
  imsrg_add_test(NAME driver_cli SOURCES test_cli.cpp DEPS driver LABELS unit)
  target_compile_definitions(driver_cli PRIVATE
    "IMSRG_EXE_PATH=\"$<TARGET_FILE:imsrg>\"" "IMSRG_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
  add_dependencies(driver_cli imsrg)
endif()

# Pipeline stage 1 (model space, interaction, HF, normal ordering) against the HF goldens; needs
# the input files in IMSRG_TEST_DATA_DIR and skips without them.
imsrg_add_test(NAME driver_pipeline_hamiltonian
  SOURCES test_pipeline_hamiltonian.cpp ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_pipeline_hamiltonian PRIVATE
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")

# End-to-end integration tests IT-2, IT-3, IT-4, IT-6, IT-7, IT-12 (design/30 section 4): run the
# pipeline in process and compare with tests/golden/e2e; skip without the input files.
imsrg_add_test(NAME driver_integration
  SOURCES test_integration.cpp ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
  DEPS driver LABELS integration NO_GTEST_MAIN)
target_compile_definitions(driver_integration PRIVATE
  "IMSRG_GOLDEN_DIR=\"${PROJECT_SOURCE_DIR}/tests/golden\""
  "IMSRG_TEST_DATA_DIR=\"${IMSRG_TEST_DATA_DIR}\"")
