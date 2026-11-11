# angmom: Wigner symbols and (later) Moshinsky brackets (design/10 section 2, layer 1).
imsrg_add_module(NAME angmom
  SOURCES
    wigner.cpp
    moshinsky.cpp
    tables.cpp
  DEPS base exec)
