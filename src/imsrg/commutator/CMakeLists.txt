imsrg_add_module(NAME commutator
  SOURCES options.cc workspace.cc commutator.cc ladder_data.cc lifted_data.cc pandya_data.cc
          three_body.cc
          kernels/simple_terms.cc kernels/ladder.cc kernels/particle_hole.cc
          kernels/factorized.cc kernels/comm223.cc
  DEPS operator linalg angmom exec comm)
