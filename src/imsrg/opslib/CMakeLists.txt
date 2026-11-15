# opslib: scalar operators in the harmonic-oscillator basis (kinetic energy, center of mass, radii),
# ported from imsrg++ `imsrg_util.cc`. C++ namespace: imsrg::opslib. Construction is host-only: the
# operators are filled on the host mirror and synchronised to the device at the end.
imsrg_add_module(NAME opslib
  SOURCES
    detail/pair_engine.cc
    kinetic.cc
    center_of_mass.cc
    radius.cc
  DEPS operator basis angmom)
