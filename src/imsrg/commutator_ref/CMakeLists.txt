# commutator_ref: slow reference implementations of the scalar IMSRG(2) commutator terms, mirroring
# imsrg++ ReferenceImplementations. Test-only: no production target links it (design/10 section 2).
imsrg_add_module(NAME commutator_ref
  SOURCES
    commutator_ref.cc
    factorized_ref.cc
    mscheme.cc
  DEPS operator angmom)
