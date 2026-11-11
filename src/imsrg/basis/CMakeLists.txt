# basis: orbits, nucleus names and fillings, references, partitions, valence spaces, two-body
# (and later three-body and particle-hole) channels, and the ModelSpace bundle
# (design/10-architecture.md section 2, layer 2; design/13-physics-modules.md section 1).
# Orderings follow imsrg++ exactly (design/00 D11); tests compare against tests/golden/basis/.
imsrg_add_module(NAME basis
  SOURCES
    orbit.cpp
    single_particle_basis.cpp
    nucleus.cpp
    reference.cpp
    valence_space.cpp
    partition.cpp
    two_body_channels.cpp
    particle_hole_channels.cpp
    three_body_channels.cpp
    model_space.cpp
  DEPS base exec)
