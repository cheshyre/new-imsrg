# flow: generators, BCH transform/product, Magnus solver (design/13 section 7).
imsrg_add_module(NAME flow
  SOURCES generator.cc bch.cc magnus.cc
  DEPS commutator operator)
