# Unit tests of the basis module. Test names are global, hence the basis_ prefix.
# Golden files (imsrg-basis-dump v1, tests/golden/README.md) are read from IMSRG_GOLDEN_DIR.
set(_imsrg_golden_dir "${PROJECT_SOURCE_DIR}/tests/golden")

# Orbits, labels, the single-particle basis, nucleus names and the naive filling.
imsrg_add_test(NAME basis_core
  SOURCES test_orbit.cpp test_single_particle_basis.cpp test_nucleus.cpp
  DEPS basis
  LABELS unit)

# References (occupations), valence-space definitions, partitions.
imsrg_add_test(NAME basis_reference
  SOURCES test_reference.cpp test_valence_space.cpp test_partition.cpp
  DEPS basis
  LABELS unit)

# Two-body kets, channels and per-channel ket subsets.
imsrg_add_test(NAME basis_two_body
  SOURCES test_two_body_channels.cpp
  DEPS basis
  LABELS unit)

# The golden-file parser (a test helper) and its self-test.
imsrg_add_test(NAME basis_golden_dump
  SOURCES test_golden_basis_dump.cpp golden_basis_dump.cpp
  DEPS basis
  LABELS unit)
target_compile_definitions(basis_golden_dump PRIVATE IMSRG_GOLDEN_DIR="${_imsrg_golden_dir}")
