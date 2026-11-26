# Unit tests of the linalg module. They run under exec_gtest_main.cc (exec::Runtime around
# RUN_ALL_TESTS) because BlockSparseMatrix owns exec::DeviceBuffer storage.

foreach(_t matrix_view block_sparse_matrix gemm eigh solve device_blas)
  imsrg_add_test(NAME linalg_${_t}
    SOURCES ${_t}_test.cc ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc
    DEPS linalg
    LABELS unit
    NO_GTEST_MAIN)
endforeach()
