# Unit tests of the linalg module. They run under exec_gtest_main.cpp (exec::Runtime around
# RUN_ALL_TESTS) because BlockSparseMatrix owns exec::DeviceBuffer storage.

foreach(_t matrix_view block_sparse_matrix gemm eigh solve)
  imsrg_add_test(NAME linalg_${_t}
    SOURCES ${_t}_test.cpp ${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cpp
    DEPS linalg
    LABELS unit
    NO_GTEST_MAIN)
endforeach()
