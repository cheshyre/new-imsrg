# Unit tests of the fast commutator module (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cc")
set(_imsrg_comm_cases "${CMAKE_CURRENT_SOURCE_DIR}/commutator_test_cases.cc")

foreach(_t options simple_terms ladder_terms particle_hole dispatch robustness
          factorized general)
  imsrg_add_test(NAME commutator_${_t}
    SOURCES test_${_t}.cc ${_imsrg_exec_main} ${_imsrg_comm_support} ${_imsrg_comm_cases}
    DEPS commutator commutator_ref operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(commutator_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
endforeach()

# The [2,2]->3 term and the perturbative triples (IMSRG(3), design/14 section 4
# item 0) against the imsrg-devel goldens of tests/golden/imsrg3.
imsrg_add_test(NAME commutator_comm223
  SOURCES test_comm223.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_comm223 PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# Row-restricted [2,2]->3 (work items from a RowSet): partition invariance,
# bitwise.
imsrg_add_test(NAME commutator_comm223_rows
  SOURCES test_comm223_rows.cc ${_imsrg_exec_main}
          "${CMAKE_CURRENT_SOURCE_DIR}/n7_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          ${_imsrg_comm_support}
  DEPS commutator commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_comm223_rows PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# The batch loop of the row-tiled N7 terms and the partition/masking test
# support (tests/support/n7_partition_support.h), on the [2,2]->3 rows.
imsrg_add_test(NAME commutator_n7_batch_loop
  SOURCES test_n7_batch_loop.cc ${_imsrg_exec_main}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# comm232ss on the row-tiled batch interface: against the element-wise
# reference, partition invariance with masked operands, batch budgets.
imsrg_add_test(NAME commutator_n7_232_batch
  SOURCES test_n7_232_batch.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${CMAKE_CURRENT_SOURCE_DIR}/n7_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator commutator_ref operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_n7_232_batch PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# The row-tiled comm133ss (pair-third batch path) against its element-wise
# reference, the public term's dispatch, and partition invariance with masked
# operands.
imsrg_add_test(NAME commutator_n7_133_batch
  SOURCES test_n7_133_batch.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# The row-tiled comm331ss and comm231ss (batch paths) against their
# element-wise references, partition invariance with masked operands, and
# run-to-run reproducibility.
imsrg_add_test(NAME commutator_n7_331_231_batch
  SOURCES test_n7_331_231_batch.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# The row-tiled comm132ss and comm330ss (kernels/comm132.cc, comm330.cc)
# against their element-wise references, partition invariance with masked
# operands and determinism.
imsrg_add_test(NAME commutator_n7_132_330_batch
  SOURCES test_n7_132_330_batch.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# The fused N7 batch driver (detail::RunN7: all seven terms in one batch loop)
# against the per-term sequence, single terms, partition invariance with
# masked operands, its checks, and the dispatch from Commutator().
imsrg_add_test(NAME commutator_n7_fused
  SOURCES test_n7_fused.cc ${_imsrg_exec_main} ${_imsrg_comm_support}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/n7_partition_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# The IMSRG(3)-N7 terms (design/14 section 4 item 2) against the m-scheme brute
# force (commutator_ref::mscheme, emax 1) and the imsrg-devel goldens of
# tests/golden/n7 (emax 2). One target per group of terms.
set(_imsrg_n7_support
    "${CMAKE_CURRENT_SOURCE_DIR}/n7_test_support.cc"
    "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc")
foreach(_t n7 n7_133_132 n7_231_331 n7_232 n7_full n7_occnat n7_views)
  imsrg_add_test(NAME commutator_${_t}
    SOURCES test_${_t}.cc ${_imsrg_exec_main} ${_imsrg_comm_support} ${_imsrg_n7_support}
    DEPS commutator commutator_ref operator
    LABELS unit
    NO_GTEST_MAIN)
  target_compile_definitions(commutator_${_t} PRIVATE
    IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
endforeach()

# Counting and cost model of the tiled N7 terms (n7_cost.h) against the real
# pair-third space at emax 2-3.
imsrg_add_test(NAME commutator_n7_cost
  SOURCES test_n7_cost.cc ${_imsrg_exec_main} "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# Batch planner and the shared pair-third intermediates H and W of the
# row-tiled N7 terms (detail/n7_batch.h) against dense host references.
imsrg_add_test(NAME commutator_n7_batch
  SOURCES test_n7_batch.cc ${_imsrg_exec_main}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)
