# exec: the Kokkos facade (design/12-compute-backend.md section 2). Kokkos headers appear only in
# this directory and in */kernels/; the public headers runtime.hpp and device_buffer.hpp do not
# include Kokkos. Kokkos::kokkos is linked PUBLIC so kernel TUs of other modules can include
# kokkos_backend.hpp.
imsrg_add_module(NAME exec
  SOURCES
    runtime.cpp
    device_buffer.cpp
  DEPS base
  LIBS Kokkos::kokkos)
