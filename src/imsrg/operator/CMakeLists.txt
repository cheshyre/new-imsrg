# operator: OperatorSpec, OneBody, TwoBody (on linalg::BlockSparseMatrix), Operator, arithmetic,
# norms, comparison helpers, OperatorDump I/O; later normal ordering, NO2B input container, basis
# transformation and random operators (design/10-architecture.md section 3.3, design/13 section 2,
# design/15-operator-dump-format.md). C++ namespace: imsrg::op (`operator` is a keyword).
# Kernels (Kokkos) live in kernels/*.cc only.
imsrg_add_module(NAME operator
  SOURCES
    operator_spec.cc
    one_body.cc
    two_body.cc
    operator.cc
    compare.cc
    arithmetic.cc
    elementwise.cc
    norms.cc
    operator_dump.cc
    normal_ordering.cc
    normal_ordering_no2b.cc
    no2b.cc
    basis_transform.cc
    random_operator.cc
    kernels/part_kernels.cc
  DEPS basis linalg angmom exec base)
