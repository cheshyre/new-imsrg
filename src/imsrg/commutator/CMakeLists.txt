imsrg_add_module(NAME commutator
  SOURCES options.cpp workspace.cpp commutator.cpp ladder_data.cpp lifted_data.cpp pandya_data.cpp
          kernels/simple_terms.cpp kernels/ladder.cpp kernels/particle_hole.cpp
  DEPS operator linalg angmom exec comm)
