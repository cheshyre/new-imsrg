imsrg_add_module(NAME commutator
  SOURCES options.cc workspace.cc commutator.cc ladder_data.cc lifted_data.cc pandya_data.cc
          three_body.cc imsrg3.cc n7_cost.cc
          kernels/simple_terms.cc kernels/ladder.cc kernels/particle_hole.cc
          kernels/factorized.cc kernels/comm223.cc
          kernels/comm330.cc kernels/comm331.cc kernels/comm231.cc kernels/comm132.cc
          kernels/comm232.cc kernels/comm133.cc
  DEPS operator linalg angmom exec comm)
