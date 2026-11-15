# exec: the Kokkos facade (design/12-compute-backend.md section 2). Kokkos headers appear only in
# this directory and in */kernels/; the public headers runtime.h and device_buffer.h do not
# include Kokkos. Kokkos::kokkos is linked PUBLIC so kernel TUs of other modules can include
# kokkos_backend.h.
imsrg_add_module(NAME exec
  SOURCES
    runtime.cc
    device_buffer.cc
    host_parallel.cc
  DEPS base
  LIBS Kokkos::kokkos)
