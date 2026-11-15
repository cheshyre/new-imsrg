# basis: orbits, nucleus names and fillings, references, partitions, valence spaces, two-body
# (and later three-body and particle-hole) channels, and the ModelSpace bundle
# (design/10-architecture.md section 2, layer 2; design/13-physics-modules.md section 1).
# Orderings follow imsrg++ exactly (design/00 D11); tests compare against tests/golden/basis/.
imsrg_add_module(NAME basis
  SOURCES
    orbit.cc
    single_particle_basis.cc
    nucleus.cc
    reference.cc
    valence_space.cc
    partition.cc
    two_body_channels.cc
    particle_hole_channels.cc
    three_body_channels.cc
    three_body_space.cc
    model_space.cc
    device_tables.cc
  DEPS base exec)
