imsrg_add_test(NAME base_types SOURCES test_types.cpp DEPS base LABELS unit)
imsrg_add_test(NAME base_assert SOURCES test_assert.cpp DEPS base LABELS unit)
imsrg_add_test(NAME base_constants SOURCES test_constants.cpp DEPS base LABELS unit)
target_compile_definitions(base_constants PRIVATE
  IMSRG_REFERENCE_PHYSICAL_CONSTANTS="/Users/cheshyre/Downloads/new-imsrg/imsrg-devel/src/PhysicalConstants.hh")
imsrg_add_test(NAME base_math SOURCES test_math.cpp DEPS base LABELS unit)
imsrg_add_test(NAME base_half SOURCES test_half.cpp DEPS base LABELS unit)
imsrg_add_test(NAME base_log SOURCES test_log.cpp DEPS base LABELS unit)
imsrg_add_test(NAME base_timer SOURCES test_timer.cpp DEPS base LABELS unit)
