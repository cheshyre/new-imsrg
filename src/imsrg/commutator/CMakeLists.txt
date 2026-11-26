imsrg_add_module(NAME commutator
  SOURCES options.cc workspace.cc commutator.cc ladder_data.cc lifted_data.cc pandya_data.cc work_items.cc
          three_body.cc triples.cc imsrg3.cc n7_cost.cc n7_batches.cc tensor_workspace.cc
          kernels/simple_terms.cc kernels/ladder.cc kernels/particle_hole.cc
          kernels/factorized.cc kernels/comm223.cc
          kernels/comm330.cc kernels/comm331.cc kernels/comm231.cc kernels/comm132.cc
          kernels/comm232.cc kernels/comm133.cc kernels/n7_pair_third.cc
          kernels/tensor_terms.cc kernels/tensor_ph.cc
  DEPS operator linalg angmom exec comm)
