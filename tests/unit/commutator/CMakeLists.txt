# Unit tests of the fast commutator module (design/30 section 2).
set(_imsrg_exec_main "${PROJECT_SOURCE_DIR}/tests/support/exec_gtest_main.cc")
set(_imsrg_comm_support "${PROJECT_SOURCE_DIR}/tests/support/commutator_test_support.cc")
set(_imsrg_comm_cases "${CMAKE_CURRENT_SOURCE_DIR}/commutator_test_cases.cc")

foreach(_t options simple_terms ladder_terms particle_hole dispatch robustness
          factorized general work_items distributed_ladder
          distributed_ph)
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

# The perturbative triples on the reduced TriplesContext against the full
# ThreeBodyContext path (M10-Q46, plan section 3.11).
imsrg_add_test(NAME commutator_triples_context
  SOURCES test_triples_context.cc ${_imsrg_exec_main}
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

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

# The distributed commutator on real MPI ranks (M12b phase 1, D3): one source,
# two rank counts; the shared MPI main provides the Kokkos runtime.
foreach(_ranks 2 4)
  imsrg_add_test(NAME commutator_mpi_${_ranks}ranks
    SOURCES test_mpi_commutator.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
            ${_imsrg_comm_support} ${_imsrg_n7_support}
    DEPS commutator commutator_ref operator flow comm
    LABELS mpi MPI_RANKS ${_ranks} NO_GTEST_MAIN TIMEOUT 600)
endforeach()

# A commutator on a model space with distributed three-body rows is refused on
# every rank (M12b review fix, until D10/D11): 2 and 4 ranks.
foreach(_ranks 2 4)
  imsrg_add_test(NAME commutator_mpi_distributed_3b_${_ranks}ranks
    SOURCES test_mpi_distributed_three_body.cc ${PROJECT_SOURCE_DIR}/tests/support/mpi_test_main.cc
    DEPS commutator operator comm
    LABELS mpi MPI_RANKS ${_ranks} NO_GTEST_MAIN TIMEOUT 300)
endforeach()

# Batch planner and the shared pair-third intermediates H and W of the
# row-tiled N7 terms (detail/n7_batch.h) against dense host references.
imsrg_add_test(NAME commutator_n7_batch
  SOURCES test_n7_batch.cc ${_imsrg_exec_main}
          "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc"
          "${PROJECT_SOURCE_DIR}/tests/support/row_mask.cc"
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)

# Scalar-tensor commutator terms (M15) against the imsrg-devel tensor goldens.
imsrg_add_test(NAME commutator_tensor_golden
  SOURCES test_tensor_golden.cc ${_imsrg_exec_main}
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)
target_compile_definitions(commutator_tensor_golden PRIVATE
  IMSRG_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")

# Fast comm122st (lifted one-body lists) against the element-wise kernel kept
# as a reference implementation (tensor_comm122_ref.cc).
imsrg_add_test(NAME commutator_tensor_comm122
  SOURCES test_tensor_comm122.cc tensor_comm122_ref.cc ${_imsrg_exec_main}
          ${_imsrg_comm_support}
  DEPS commutator operator
  LABELS unit
  NO_GTEST_MAIN)
