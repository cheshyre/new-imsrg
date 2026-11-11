# io: text and binary input readers (M5). Darmstadt me2j two-body files over a buffered zlib byte
# stream (design/13 section 3, design/reference/04-hf-mbpt-io.md section 4). zlib is a private
# dependency: no zlib header is included from a public header. C++ namespace: imsrg::io.
imsrg_add_module(NAME io
  SOURCES
    input_stream.cpp
    me2j.cpp
    no2b.cpp
    valence_writers.cpp
  DEPS operator basis base
  PRIVATE_LIBS ZLIB::ZLIB)
