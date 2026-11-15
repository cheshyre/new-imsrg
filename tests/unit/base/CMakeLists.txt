imsrg_add_test(NAME base_types SOURCES test_types.cc DEPS base LABELS unit)
imsrg_add_test(NAME base_assert SOURCES test_assert.cc DEPS base LABELS unit)
imsrg_add_test(NAME base_constants SOURCES test_constants.cc DEPS base LABELS unit)
target_compile_definitions(base_constants PRIVATE
  IMSRG_REFERENCE_PHYSICAL_CONSTANTS="/Users/cheshyre/Downloads/new-imsrg/imsrg-devel/src/PhysicalConstants.hh")
imsrg_add_test(NAME base_math SOURCES test_math.cc DEPS base LABELS unit)
imsrg_add_test(NAME base_half SOURCES test_half.cc DEPS base LABELS unit)
imsrg_add_test(NAME base_log SOURCES test_log.cc DEPS base LABELS unit)
imsrg_add_test(NAME base_timer SOURCES test_timer.cc DEPS base LABELS unit)
