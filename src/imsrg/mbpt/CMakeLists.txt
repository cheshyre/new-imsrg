# mbpt: many-body perturbation theory on the HF-basis Hamiltonian (imsrg++ `GetMP2_Energy`,
# `GetMP3_Energy`, `HFMBPT`): MP2/MP3 energies, the MBPT(2) one-body density matrix and natural
# orbitals. C++ namespace: imsrg::mbpt. Host-only. Must not depend on hf (layering).
imsrg_add_module(NAME mbpt
  SOURCES
    energies.cpp
    natural_orbitals.cpp
  DEPS operator linalg basis)
