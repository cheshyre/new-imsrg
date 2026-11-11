# base: strong types, errors and checks, logging, timers, physical constants, small math,
# half->float (design/10-architecture.md section 2, layer 0). No imsrg dependencies; {fmt} is
# the only third-party library (logging).
imsrg_add_module(NAME base
  SOURCES
    assert.cpp
    log.cpp
    timer.cpp
  LIBS fmt::fmt)
