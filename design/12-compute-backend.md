# 12 — Compute backend, kernels, and MPI

## 1. Layers

```
algorithms (commutator, flow, hf, ...)
   │ call
   ▼
imsrg::linalg  (gemm, grouped_gemm, eigh)  ── vendor BLAS/LAPACK, cuBLAS/cuSOLVER, hipBLAS/rocSOLVER
imsrg::exec    (DeviceBuffer, parallel_for/reduce, team policies, deep_copy) ── Kokkos
imsrg::comm    (Communicator, allreduce, alltoallv, ownership maps) ── MPI or serial stub
```

## 2. `imsrg::exec`: the Kokkos facade

Kernels use a restricted vocabulary, so a future port has to replace only the facade:

| Facade | Kokkos implementation | Notes |
|--------|----------------------|-------|
| `exec::DefaultSpace`, `exec::HostSpace` | `Kokkos::DefaultExecutionSpace`, `DefaultHostExecutionSpace` | |
| `DeviceBuffer<T>` | pimpl around `Kokkos::View<T*>` | Lives in public headers without Kokkos. `host_mirror()` gives a `std::span<T>` on host after explicit sync. |
| `exec::view(buf)` | returns the raw `Kokkos::View` | Only in kernel TUs. |
| `exec::parallel_for(name, n, f)` | `Kokkos::parallel_for(RangePolicy)` | `f` must be an `IMSRG_LAMBDA` (= `KOKKOS_LAMBDA`). |
| `exec::parallel_for_md(name, {n0,n1}, f)` | `MDRangePolicy<Rank<2>>` | |
| `exec::parallel_reduce(name, n, f, result)` | `Kokkos::parallel_reduce` | Sum/max reducers only. |
| `exec::team_for(name, league, team, scratch_bytes, f)` | `TeamPolicy` with scratch | For ragged channel loops. |
| `exec::deep_copy`, `exec::fence` | Kokkos equivalents | |

The **mandatory backends** are Serial and OpenMP, both tested on the dev machine. CUDA and HIP
must compile and run on clusters later. Threads is optional.

## 3. `imsrg::linalg`

* `MatrixView<T>`: non-owning (ptr, rows, cols, leading dimension, layout). Column-major,
  matching BLAS and imsrg++/Armadillo, which simplifies comparisons.
* `BlockSparseMatrix<T>`: the storage behind `TwoBody` and future 3B blocks. It holds one
  contiguous `DeviceBuffer<T>` and a block table `{row_channel, col_channel, rows, cols, offset}`.
* Operations: `gemm(alpha, A, opA, B, opB, beta, C)`, `grouped_gemm(span<GemmProblem>)` (the
  first implementation loops over `gemm`; later versions may use streams, MAGMA vbatched or
  cuBLAS grouped GEMM), `eigh` (host LAPACK `dsyevd`, used for HF and small problems),
  `solve`/`lstsq` for DIIS.
* Backends are chosen at configure time. A test runs every op against a naive reference loop.

## 4. Kernel strategy

### 4.1 Whole-operator elementwise ops
Use `axpy`, scale, dot/norm, and the generator construction, which is elementwise with
denominators. Because each part is one contiguous buffer, each of these is a single flat
`parallel_for`/`parallel_reduce`. The per-element channel or ket metadata (needed for
denominators) comes from precomputed index arrays on device.

### 4.2 IMSRG(2) terms
Details are in `reference/02-commutators.md`.

| Term | Strategy |
|------|----------|
| comm110, comm111, comm121, comm221-like 1B outputs | Small; flat kernels over orbit pairs with precomputed ket lists. |
| comm122, comm222_pp_hh (ladders) | Per channel: `Z_ch += X_ch P Y_ch - ...`, where `P` is a diagonal occupation weighting. Use grouped GEMM over channels with occupation-scaled copies. |
| comm222_ph | Forward Pandya transform as a **precomputed sparse recoupling map** (6j-weighted gather), then grouped GEMM in the ph channels, then inverse Pandya (scatter-add, or gather from the output side to avoid atomics). The map is built once per ModelSpace. |

Implementation phases for every kernel:
1. Make it correct and simple: a straightforward Kokkos kernel (or host loop) that mirrors the
   reference algorithm.
2. Restructure for throughput: GEMM batching and precomputed maps.
3. Tune for GPU: team policies, scratch memory, avoiding atomics.

Each phase has to pass the same unit tests against `commutator_ref` and the imsrg-devel golden data.

### 4.3 Hermiticity assumptions
imsrg++'s fast kernels silently assume each input is Hermitian or anti-Hermitian
(`reference/02` §8). Our kernels **check** `spec().hermiticity` and either use a general path or
throw, never silently returning wrong results.

### 4.4 Angular-momentum data on device
`angmom::Tables` holds dense 6j tables (and the 9j tables needed for the 3B work) for the ranges
the ModelSpace needs, built once in parallel and then read-only. Device copies live in
`DeviceBuffer`s. Every kernel takes all its recoupling data from the tables or precomputed maps;
none computes a symbol inside a hot loop, and no table is a lazy cache.

## 5. Device-code rules (CUDA/HIP correctness without a GPU to test on)

1. Device lambdas use `IMSRG_LAMBDA` and capture **by value** only. They capture views and
   PODs, never `this`, and never containers (`std::vector`, `std::string`, `std::map`).
2. Functions called from kernels are `IMSRG_FUNCTION` (= `KOKKOS_FUNCTION`), `constexpr`, or
   both, and use no exceptions, no `std::` math other than what Kokkos provides
   (`Kokkos::sqrt`, ...), no virtual calls, no recursion, and no dynamic allocation.
3. Strong types and small structs used in kernels are trivially copyable. A
   `static_assert(std::is_trivially_copyable_v<T>)` sits next to each definition.
4. There is no host-only data access inside kernels: host metadata (channel tables) is copied to
   device buffers first.
5. Reductions use Kokkos reducers. No atomics on `double` in hot loops unless measured to be
   harmless.
6. Every kernel has a name string, used for profiling.
7. Later, a `tools/ci.sh --compile-only-cuda` job (on a machine with nvcc) catches violations;
   until then, reviewers check these rules explicitly.

## 6. MPI design (`imsrg::comm`)

* `Communicator` wraps an `MPI_Comm` (or a serial stub with size 1). Methods:
  `allreduce_sum(span<double>)`, `broadcast`, `alltoallv`, `barrier`, `rank()`, `size()`.
* **Phase 1 (replicated data, distributed work)**: every rank holds the full 1B and 2B
  operators. Expensive terms partition their *outer channel loop* with an `OwnershipMap`
  (greedy by estimated cost, deterministic). Each rank computes its share into Z, and an
  `allreduce_sum` over Z's buffers combines them. This is simple, deterministic, and enough for
  IMSRG(2) up to emax ≈ 12–14 on CPU nodes.
* **Phase 2 (distributed 3B data)**: 3B blocks are owned by ranks (by 3B channel), and
  `ThreeBody` knows its local blocks. Terms that are channel-local in 3B (most of [2,3]→3,
  [3,3]→3 pp/hh, [2,2]→3) run without communication. Terms producing 1B/2B outputs reduce with
  `allreduce`. Terms that need recoupled (ph-type) 3B access use `alltoallv` exchanges of
  recoupled blocks.
* Determinism: reductions use a fixed order so results do not depend on rank count beyond
  floating-point reassociation. Tests run with 1, 2 and 4 ranks at emax ≤ 3 and compare.
* The MPI tests run locally with `mpirun -n 2/4` (Open MPI from Homebrew).

## 7. Precision

* Operators use `double`.
* 3N input storage uses `float` or binary16 (NO2B "half"), converted on read. This matches
  imsrg++ (`reference/01` §5).
* Whether IMSRG(3) 3B operator blocks can use `float` is a question for the IMSRG(3) benchmark
  (`14-imsrg3-plan.md`). The default is `double`.
