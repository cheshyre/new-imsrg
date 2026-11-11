# hf: spherical Hartree-Fock in the harmonic-oscillator basis (imsrg++ `HartreeFock`): the
# monopole interactions, the self-consistent field iteration (linear density mixing or Pulay DIIS)
# and the HF energy. C++ namespace: imsrg::hf. Host-only (the one-body problem is tiny).
imsrg_add_module(NAME hf
  SOURCES
    monopole.cpp
    diis.cpp
    hartree_fock.cpp
  DEPS operator linalg basis)
