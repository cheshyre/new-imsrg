# linalg: matrix views, block-sparse matrix storage, gemm/grouped_gemm/eigh/solve on host memory
# (design/12-compute-backend.md section 3). The BLAS/LAPACK backend (imsrg::blas_lapack, defined in
# cmake/ImsrgDependencies.cmake) is linked privately; only detail/blas.h sees it.
imsrg_add_module(NAME linalg
  SOURCES
    block_sparse_matrix.cc
    blas.cc
    matrix.cc
  DEPS base exec
  PRIVATE_LIBS imsrg::blas_lapack)
