# commutator: scalar IMSRG(2) commutator terms, CommutatorOptions/TermMask, Workspace and the
# fast kernels (design/10 section 2, design/12 section 4, design/13 section 6). IMSRG(3) terms
# follow in M10. Kernels (Kokkos) live in kernels/*.cpp only.
imsrg_add_module(NAME commutator
  SOURCES
    options.cpp
  DEPS operator linalg angmom exec comm)
