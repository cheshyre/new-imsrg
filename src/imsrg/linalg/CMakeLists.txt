# linalg: matrix views, block-sparse matrix storage, gemm/grouped_gemm/eigh/solve on host memory
# (design/12-compute-backend.md section 3). The BLAS/LAPACK backend (imsrg::blas_lapack, defined in
# cmake/ImsrgDependencies.cmake) is linked privately; only detail/blas.h sees it. The device BLAS
# (imsrg::device_blas: host BLAS, rocBLAS or Kokkos Kernels, IMSRG_DEVICE_BLAS) only in
# kernels/device_blas.cc.
imsrg_add_module(NAME linalg
  SOURCES
    block_sparse_matrix.cc
    blas.cc
    matrix.cc
    kernels/device_blas.cc
  DEPS base exec
  PRIVATE_LIBS imsrg::blas_lapack imsrg::device_blas)
