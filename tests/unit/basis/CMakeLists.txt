# Unit tests of the basis module. Test names are global, hence the basis_ prefix.
# Golden files (imsrg-basis-dump v1, tests/golden/README.md) are read from IMSRG_GOLDEN_DIR.
set(_imsrg_golden_dir "${PROJECT_SOURCE_DIR}/tests/golden")

# Orbits, labels, the single-particle basis, nucleus names and the naive filling.
imsrg_add_test(NAME basis_core
  SOURCES test_orbit.cc test_single_particle_basis.cc test_nucleus.cc
  DEPS basis
  LABELS unit)

# References (occupations), valence-space definitions, partitions.
imsrg_add_test(NAME basis_reference
  SOURCES test_reference.cc test_valence_space.cc test_partition.cc
  DEPS basis
  LABELS unit)

# Two-body kets, channels and per-channel ket subsets.
imsrg_add_test(NAME basis_two_body
  SOURCES test_two_body_channels.cc
  DEPS basis
  LABELS unit)

# Particle-hole (cross-coupled) channels and three-body kets and channels.
imsrg_add_test(NAME basis_channels
  SOURCES test_particle_hole_channels.cc test_three_body_channels.cc
  DEPS basis
  LABELS unit)

# The kept canonical three-body space (ThreeBodySpace) and its ModelSpace integration.
imsrg_add_test(NAME basis_three_body_space
  SOURCES test_three_body_space.cc
  DEPS basis
  LABELS unit)

# The golden-file parser (a test helper), its self-test, and the record-by-record comparison of
# the basis bookkeeping with every golden file (test_golden_basis.cc).
imsrg_add_test(NAME basis_golden_dump
  SOURCES test_golden_basis_dump.cc test_golden_basis.cc golden_basis_dump.cc
  DEPS basis
  LABELS unit)
target_compile_definitions(basis_golden_dump PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

# The ModelSpace bundle (construction paths, sharing, with_reference).
imsrg_add_test(NAME basis_model_space
  SOURCES test_model_space.cc golden_basis_dump.cc
  DEPS basis
  LABELS unit)
target_compile_definitions(basis_model_space PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")

# Device-copyable index tables (DeviceTables): device accessors are evaluated in kernels
# (device_tables_kernels.cc) and compared with the host classes.
imsrg_add_test(NAME basis_device_tables
  SOURCES test_device_tables.cc device_tables_kernels.cc
          ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
  DEPS basis exec
  LABELS unit
  NO_GTEST_MAIN)
