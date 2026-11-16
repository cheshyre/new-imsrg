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
| `exec::DefaultExecSpace`, `exec::HostExecSpace`, `exec::DeviceMemorySpace`, `exec::DeviceView<T>` | `Kokkos::DefaultExecutionSpace`, `DefaultHostExecutionSpace`, its memory space, `View<T*, ...>` | Names as implemented in M2 (`exec/kokkos_backend.h`). |
| `DeviceBuffer<T>` | pimpl around `Kokkos::View<T*>` | Lives in public headers without Kokkos. `host_span()` gives a `std::span<T>` on host; `sync_to_host()` / `sync_to_device()` are explicit (fence-only when the mirror aliases the device data). Copy = deep copy (replaces a free `deep_copy`). |
| `exec::view(buf)` | returns the raw `Kokkos::View` | Only in kernel TUs. |
| `exec::ParallelFor(name, n, f)` | `Kokkos::parallel_for(RangePolicy)` | `f` must be an `IMSRG_LAMBDA` (= `KOKKOS_LAMBDA`). |
| `exec::ParallelForDynamic(name, n, f)` | `RangePolicy` with `Schedule<Dynamic>` | Ragged items (one per row/column of the two-body blocks, M12a); device backends ignore the hint. |
| `exec::HostParallelForDynamic(name, n, f)` (`exec/host_parallel.h`, no Kokkos in the header) | host-space `RangePolicy`, `Schedule<Dynamic>`, chunk 1 | Plain host code per item (e.g. one BLAS call); `f` must not throw. |
| `exec::ParallelForMd(name, n0, n1, f)` | `MDRangePolicy<Rank<2>>` | |
| `exec::ParallelReduce(name, n, f, result)`, `exec::ParallelReduceMax(...)` | `Kokkos::parallel_reduce` | Sum and max reducers. |
| `exec::TeamFor(name, league, TeamOptions, f)` | `TeamPolicy` (+ scratch) | Team launch; `f(member)` runs on every team thread. `TeamOptions{schedule (kStatic/kDynamic, chunk 1), team_size, vector_length (<= 0: AUTO), level (kAuto/kLevel0/kLevel1), scratch_bytes}`. Asynchronous. The old `TeamFor(name, league, team, scratch_bytes, f)` overload forwards to it (static schedule). Used by `comm223` (dynamic). |
| `exec::TeamReduce(name, league, TeamOptions, f, double& sum)` | `parallel_reduce` over a `TeamPolicy` | `f(member, double& partial)`; blocking. With `kDynamic` the reduction order, hence the floating-point sum, may vary run to run. |
| `exec::TeamMember`, `TeamScratchMax(level)`, `ResolveScratchLevel(opts)` | `member_type`, `scratch_size_max` | One member type for both schedules (static_assert). `kAuto` uses level 0 if `scratch_bytes` fits `TeamScratchMax(0)`, else level 1; `IMSRG_CHECK` that it fits level 1. Nested `TeamThreadRange`/`ThreadVectorRange`/`single` stay Kokkos. |
| `exec::Fence` | Kokkos equivalent | |

The **mandatory backends** are Serial and OpenMP, both tested on the dev machine. CUDA and HIP
must compile and run on clusters later. Threads is optional.

## 3. `imsrg::linalg`

* `MatrixView<T>`: non-owning (ptr, rows, cols, leading dimension, layout). Column-major,
  matching BLAS and imsrg++/Armadillo, which simplifies comparisons.
* `BlockSparseMatrix<T>`: the storage behind `TwoBody` and future 3B blocks. It holds one
  contiguous `DeviceBuffer<T>` and a block table `{row_channel, col_channel, rows, cols, offset}`.
* Operations: `gemm(alpha, A, opA, B, opB, beta, C)`, `grouped_gemm(span<GemmProblem>)` (checks
  all problems, then, with Accelerate and at least ~1 Mflop, runs them concurrently on the host
  threads, one BLAS call per item, largest first (M12a); otherwise a sequential loop, M12-Q5; later
  versions may use streams, MAGMA vbatched or cuBLAS grouped GEMM), `eigh` (host LAPACK `dsyevd`,
  used for HF and small problems), `solve`/`lstsq` for DIIS.
* Backends are chosen at configure time. A test runs every op against a naive reference loop.
* Threading: BLAS calls made inside parallel regions run single-threaded
  (`VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, or MKL sequential); large standalone GEMMs
  outside parallel regions may use threaded BLAS. MPI tests set `OMP_NUM_THREADS` so that
  ranks × threads ≤ cores (e.g. 4 ranks × 4 threads).

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
| comm222_ph | Forward Pandya transform (6j-weighted gather), then grouped GEMM in the ph channels, then inverse Pandya (gather from the output side, no atomics). **As implemented (M12a):** no stored element-wise recoupling map (for the inverse it would cost several two-body operators of memory); instead the kernels read 6j values from contiguous table rows (`SixJView::row`), resolve the cc kets of orbit pairs from a small per-pair descriptor table (norb^2 entries), fill XK/YK per cc ket with the X and Y transforms fused, store YK for one pair order only (second GEMM after an in-place row-pair flip), combine W in place (no ZB buffer), and run the inverse over pairs of kets with all J at once. |

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
   Exception: the nested lambdas of hierarchical parallelism (`TeamThreadRange`,
   `ThreadVectorRange`, `single`) inside an already running kernel may capture by reference
   (`[&]`), the standard Kokkos idiom; they execute on the device within the outer lambda's
   frame, so the references never cross the host-device boundary (first used by
   `commutator/kernels/comm223.cc`, M10).
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
8. Device lambdas are never defined directly in a GoogleTest `TEST`/`TEST_F` body: its
   `TestBody()` is a private member function, and nvcc/hipcc reject extended lambdas whose
   enclosing function is private or protected. Tests call a free (namespace-scope) function that
   launches the kernel (see `tests/smoke/kokkos_smoke_test.cc`). The same holds for private or
   protected member functions of our own classes.

## 6. MPI design (`imsrg::comm`)

* `Communicator` (`comm/communicator.h`) wraps an `MPI_Comm` (or a serial stub with size 1).
  `comm/ownership_map.h` assigns channels to ranks (`OwnershipMap::Greedy`: LPT by estimated
  cost, deterministic).
* **Phase 1 (implemented; replicated data, distributed work; D59)**: every rank holds the full
  1B and 2B operators and runs the whole pipeline.
  * Work split (`commutator/detail/work_items.h`, `AssignWork`): the ladder terms (pp/hh,
    including 221) by a contiguous cut of the global row sequence whose per-rank capacities leave
    room for the calibrated comm122 and ph loads (`JointLadderItems`); comm122 and the
    particle-hole term by whole channels with LPT; the cheap terms 110/220/111/121 and all N7
    terms on rank 0; 3f2 replicated.
  * Combine (`commutator/commutator.cc`, single exit of `Commutator()`): `ReduceBcastSum` of
    Z0-Z2 (reduce to rank 0, then broadcast; the bits are identical on every rank by
    construction; no `Allreduce`) and a broadcast of Z3. Span collectives are chunked at 2^30
    elements.
  * Driver (`driver/pipeline/mpi_sync.h`): `CheckReplicatedSetup` (occupations, 2B ket counts,
    3B presence, thread count; throws on every rank), then rank 0 broadcasts H and the
    transformed operators. Only rank 0 writes files and info logs. `main` calls
    `Environment::Abort` on an exception.
  * Collective rules (M4): norms and gathers of distributed operators are collectives, so every
    rank calls them. No collective inside rank-0-only code. Compute every reported value before
    any `is_root()` branch.
  * Equal `OMP_NUM_THREADS` on all ranks (checked): bitwise identical results across ranks need
    the same reduction order inside a rank.
  * Results agree with one rank within max|d| <= 1e-12 max(1, max|ref|), with identical step
    counts. Tests run 1, 2 and 4 ranks at small emax.
  * Limit: the particle-hole term goes by whole cc channels, so large imbalance at many ranks
    (M12b-Q3). On one node more ranks at fixed ranks x threads are slower
    (`design/benchmarks/mpi-scaling.md`); MPI is for several nodes.
* **Phase 2 (planned; distributed 3B data)**: rows of the 3B operators are distributed with the
  C1 row partition; halos move in lockstep `Alltoallv` rounds. `ThreeBodyDistribution` lives on
  the model space (new edge `basis -> comm`). Terms that are row-local run without
  communication. Review note M2: a round takes the time of the slowest rank, summed over the
  rounds, so cut batches by cost (not bytes), exchange `TransposeOut` once per commutator, and
  send halos in bulk or prefetch them.
* Determinism: reductions use a fixed order so results do not depend on rank count beyond
  floating-point reassociation. Tests run with 1, 2 and 4 ranks at emax ≤ 3 and compare.
* The MPI tests run locally with `mpirun -n 2/4` (Open MPI from Homebrew).

## 7. Precision

* Operators use `double`.
* 3N input storage uses `float` or binary16 (NO2B "half"), converted on read. This matches
  imsrg++ (`reference/01` §5).
* Whether IMSRG(3) 3B operator blocks can use `float` is a question for the IMSRG(3) benchmark
  (`14-imsrg3-plan.md`). The default is `double`.
