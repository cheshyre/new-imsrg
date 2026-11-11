# operator: OperatorSpec, OneBody, TwoBody (on linalg::BlockSparseMatrix), Operator, arithmetic,
# norms, comparison helpers, OperatorDump I/O; later normal ordering, NO2B input container, basis
# transformation and random operators (design/10-architecture.md section 3.3, design/13 section 2,
# design/15-operator-dump-format.md). C++ namespace: imsrg::op (`operator` is a keyword).
# Kernels (Kokkos) live in kernels/*.cpp only.
imsrg_add_module(NAME operator
  SOURCES
    operator_spec.cpp
    one_body.cpp
    two_body.cpp
    operator.cpp
    compare.cpp
    arithmetic.cpp
    norms.cpp
    operator_dump.cpp
    kernels/part_kernels.cpp
  DEPS basis linalg angmom exec base)
