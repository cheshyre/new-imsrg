# flow: generators, BCH transform/product, Magnus solver (design/13 section 7).
imsrg_add_module(NAME flow
  SOURCES generator.cpp bch.cpp magnus.cpp
  DEPS commutator operator)
