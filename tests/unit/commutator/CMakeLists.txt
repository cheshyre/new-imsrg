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

# The IMSRG(3)-N7 terms (design/14 section 4 item 2) against the m-scheme brute
# force (commutator_ref::mscheme, emax 1) and the imsrg-devel goldens of
# tests/golden/n7 (emax 2). One target per group of terms.
set(_imsrg_n7_support
    "${CMAKE_CURRENT_SOURCE_DIR}/n7_test_support.cc"
    "${PROJECT_SOURCE_DIR}/tests/support/three_body_test_support.cc")
foreach(_t n7 n7_133_132 n7_231_331 n7_232 n7_full n7_views)
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
