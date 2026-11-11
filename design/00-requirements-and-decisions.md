# 00 — Requirements and decision log

This file records **what** we are building and **which decisions are already made**
(with their rationale). It is the first thing every agent should read after `AGENTS.md`.
Decisions marked *(user)* came from the interview with the project owner on
2026-11-11 and must not be changed without asking. Decisions marked *(design)* are
ours; they may be revised when evidence demands it. Write the change and its reason
here when you revise one.

## 1. Mission

Write a modern, modular, fast, scalable replacement for **imsrg++** (Ragnar Stroberg
et al., reference copy at `../../imsrg-devel` relative to the repo root,
or `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel`). The reference code is used for:

* checking that features are complete (the feature checklists are in `design/reference/*.md`)
* checking implementations: numerical equivalence "within reason" (see `30-testing-and-validation.md`)

We do **not** port line by line. We revisit every design decision.

## 2. Hard requirements *(user)*

| # | Requirement |
|---|-------------|
| R1 | C++ with a modern standard that all major compilers support well → **C++20** *(design, see D1)* |
| R2 | Modular CMake build first. A single plain Makefile for people who avoid CMake comes later (Milestone M13). |
| R3 | Third-party libraries may be used, ideally as **git submodules** in `extern/`. |
| R4 | No circular dependencies; more modular than imsrg++. |
| R5 | Python bindings for the key APIs (quick iteration). |
| R6 | A portable acceleration layer, so hot loops run vectorized on CPUs and on **NVIDIA and AMD GPUs** *(user: NVIDIA + AMD + CPU clusters; Intel GPUs not required)*. |
| R7 | MPI to distribute work and, where needed, data. |
| R8 | Third-party libraries sit behind thin internal facades so they can be replaced. |
| R9 | No ambiguous types ("everything is an int"), no heavy inheritance, no patterns where newcomers make frustrating mistakes (e.g. object slicing). |
| R10 | Results numerically equivalent to imsrg++ within reason, with unit tests, modules, useful docstrings, speed and scalability, fast parallel and incremental builds, and end-to-end integration tests. |
| R11 | Test inputs: 2N `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` (`file2e1max=14 hw=16`); 3N NO2B `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin` (`hw=16`, `file3e1max=6 file3e2max=12 file3e3max=18`, `3bme_type=no2b`). Both live in the main checkout root `/Users/cheshyre/Downloads/new-imsrg/`. **emax ≤ 6 in all tests; IMSRG(3) tests only at emax = 2, 3.** |

## 3. Scope and priorities *(user)*

Implementation order:

1. **Core**: Hartree-Fock, IMSRG(2) with Magnus (also the flow method), single-reference and
   valence-space decoupling (1-step and 2-step), scalar operators (radii, kinetic terms, ...),
   me2j input, Tokyo/NuShellX valence-space output.
2. **3N input and natural orbitals**: NO2B (and later full me3j) 3N files, 3N in HF and
   normal ordering, MBPT(2) natural-orbital basis.
3. **IMSRG(3)** (top priority among the extensions; implement it, test only at emax = 2, 3).
   Priority variants: **IMSRG(3)-N7**, **IMSRG(3f2) factorized**, **perturbative triples**.
   The 3-body storage layout is chosen **by benchmark** (see `14-imsrg3-plan.md`).
4. **Design for, implement later**: tensor and isospin-changing operators, dagger
   (odd-particle-number) operators, 0νββ operators, dark-matter NREFT operators.
5. **Dropped**: atomic systems (`physical_system=atomic`), RPA/TDA operator corrections.

## 4. Process requirements *(user)*

* A feature branch for every task, even small ones. **Git stays local**: feature
  branches merge into the local integration branch `develop`. Never push, never touch `main`.
* Modular commits that explain what was done and, when it is not obvious, why.
* Agent orchestration: Opus agents run the bigger efforts and hand small tasks to Haiku or
  Sonnet agents, depending on how complex they are. Compact at about 60 % context for
  Haiku/Sonnet and about 80 % for Opus (see `41-agent-workflow.md` for how).
* Questions that come up go into `design/QUESTIONS.md` for later review. Do not stop working
  because of them unless the question blocks all progress.
* Installing tools with Homebrew and pip (in a project venv) is allowed *(user)*.
* Building the old imsrg-devel code **outside the new repo**, to make reference data, is allowed *(user)*.
* The unsupervised run is overnight (~8–12 h) *(user)*. Plan for correct, tested infrastructure
  first; breadth second.
* Old command line: the new executable accepts **both** a new TOML config and the old
  `key=value` parameters *(user)*.
* Old-code bugs that change results: **fix them and document the deviation** *(user)*, then
  validate the fixed path some other way (brute force, a patched reference, or a limit case).

## 5. Decision log

| ID | Decision | Rationale | Status |
|----|----------|-----------|--------|
| D1 | **C++20** (`CMAKE_CXX_STANDARD 20`, no extensions). | Concepts, `std::span`, designated initializers, `<bit>`, three-way comparison. Kokkos 4.5+/5.x, CUDA ≥ 12, ROCm hipcc, GCC ≥ 11 and Clang ≥ 15 all support it. We avoid `std::format` and `<ranges>`-heavy code because GCC 11/12 and libc++ support them unevenly; `{fmt}` covers formatting. | design |
| D2 | **Kokkos** (Core) is the performance-portability layer, used only through the `imsrg::exec` facade (see `12-compute-backend.md`). | Kokkos is the only mature option that reaches NVIDIA (CUDA), AMD (HIP), and CPU (OpenMP/Threads/Serial, with SIMD) from one source, with a large DOE user base and Spack/module support on clusters. We considered SYCL/AdaptiveCpp (weaker on AMD/NVIDIA toolchains on clusters), OpenMP offload (uneven compiler maturity and performance), C++ stdpar (no control over memory spaces or scratch, no batched BLAS), RAJA/Umpire (smaller ecosystem for linear algebra), Alpaka (smaller community), and raw CUDA/HIP (two code paths plus a CPU path). | design |
| D3 | Dense linear algebra goes through our own `imsrg::linalg` facade with vendor backends: host BLAS/LAPACK (Accelerate on macOS, OpenBLAS/MKL on Linux), cuBLAS/cuSOLVER, hipBLAS/rocSOLVER. Kokkos Kernels is optional and evaluated later. | GEMM is the main IMSRG(2) cost. Vendor BLAS is fastest. A thin facade keeps the dependency replaceable, and Kokkos Kernels is a heavy build. | design |
| D4 | **No Boost, no GSL, no Armadillo.** We write our own Wigner symbols (validated against sympy), our own ODE integrators (Euler, RK4, Dormand-Prince 5(4)), our own gzip streaming over zlib, and our own half→float conversion. | They were used for only a few functions (see `reference/05` §4). Fewer dependencies mean faster builds, easier GPU porting and easier cluster installs. | design |
| D5 | Python bindings use **nanobind**, packaged with scikit-build-core. | Faster compiles and smaller binaries than pybind11, first-class zero-copy ndarray, stub generation. | design |
| D6 | Unit tests use **GoogleTest** (C++) and **pytest** (Python and integration). | Widely known; `gtest_discover_tests` integrates with CTest. | design |
| D7 | Config uses **toml++**; the CLI uses **CLI11**; formatting and logging use **{fmt}** plus a small in-house logger. All are submodules. | Header-only or small, permissive licenses. | design |
| D8 | **Value-semantic `Operator`** built by composition (zero/one/two/three-body parts plus an `OperatorSpec`). Closed sets of variants use `std::variant` or enums. No virtual inheritance in data types. | Avoids slicing (imsrg++ has a real slicing bug in `TransformToPN`) and keeps copies and moves cheap and correct. | design |
| D9 | **Strong types** for indices and quantum numbers (`OrbitIndex`, `ChannelIndex`, `KetIndex`, `TwoJ`, `TwoTz`, `Parity`, ...). They are trivially copyable and usable in device code. | R9. Mixing up J and 2J, or local and global ket indices, is the most common bug class in this kind of code. | design |
| D10 | **No mutable global state.** Options travel in explicit structs (`CommutatorOptions`, `SolverOptions`, ...). Angular-momentum data comes from immutable precomputed tables, not lazy caches. | imsrg++ uses static maps for flags, profiler state and 6j caches, which forces "serial first pass" hacks and calls `exit()` on cache misses in parallel regions. | design |
| D11 | **Same orbit ordering, ket ordering and normalization conventions as imsrg++**: orbit index formula (shell N ascending, l descending, j descending, proton (tz2 = −1) even and neutron = proton + 1); kets `p ≤ q` in lexicographic order; normalized antisymmetrized J-coupled 2B matrix elements. | Element-by-element comparison with the reference becomes trivial. None of these costs performance. | design |
| D12 | **Same numerics-defining algorithms as imsrg++ by default**: norm conventions (1B weighted by (2j+1), 2B by (2J+1)²), the Magnus step-size rule, the Omega splitting criterion, BCH termination thresholds, HF mixing and convergence. | Reproducible flow trajectories, hence comparable energies. Genuine bugs are fixed per the user policy and logged in `30-testing-and-validation.md` §6. | design |
| D13 | Public headers of the data model **do not include Kokkos**. Device storage goes through a pimpl'd `DeviceBuffer<T>`, and kernels live in `.cpp` files that include the backend. | Incremental build speed (Kokkos_Core.hpp is heavy) and replaceability (R8). | design |
| D14 | MPI phase 1: 1B/2B operator parts replicated; expensive commutator terms split by channel blocks across ranks; results combined with `Allreduce`. Phase 2: 3B blocks distributed by 3B channel with an ownership map. MPI sits behind `imsrg::comm`, which has a serial stub so builds without MPI work. | IMSRG(2) operators fit in node memory up to emax ≈ 14. IMSRG(3) memory is dominated by 3B blocks, which most terms only need channel-locally. | design |
| D15 | 3B input formats (NO2B, me3j) are **separate input types**, converted into the canonical operator layout by explicit functions. The IMSRG(3) operator layout is chosen by benchmark *(user)*. | Separates file-format concerns from the compute layout (imsrg++ mixes them with a slicing-prone class hierarchy). | design + user |
| D16 | Library code never calls `exit()`. User and input errors throw typed exceptions (`imsrg::InputError`, ...). Invariant checks use `IMSRG_ASSERT` (debug) or `IMSRG_CHECK` (always on). | Usable as a library from Python. | design |
| D17 | Executable `imsrg`, C++ namespace `imsrg`, CMake targets `imsrg::<module>`, Python package `imsrg` (extension `imsrg._core`). License **GPL-3.0** (repo LICENSE). | Consistent naming. The project name can change later (see QUESTIONS). | design |
| D18 | Development platform is macOS arm64 (Apple M5 Pro, 18 cores, 48 GB) with Kokkos **OpenMP** and **Serial** backends. GPU builds (CUDA/HIP) cannot be tested here, so we follow strict device-code rules (`12-compute-backend.md` §5) and add compile-only checks later. | No GPU on the dev machine. | design |
| D19 | Tolerance policy *(user)*: tight deterministic unit checks (~1e-10 relative); end-to-end acceptance at 1e-4 MeV (HF, MBPT) and 1e-3 MeV (IMSRG E0, valence-space elements), always logging the actual deviation. | Unit checks catch real bugs; flows involve thresholds and step decisions. | user |
| D20 | Toolchain baseline *(user)*: GCC ≥ 11, CUDA ≥ 12, ROCm ≥ 6, plus Apple clang on the dev machine. Pin the latest stable Kokkos that supports these. | Clusters the owner uses. | user |
| D21 | The generator's denominator floor keeps the sign of the denominator *(user: imsrg++ loses it, which is a bug)*. | Deviation logged in `30` §6. | user |
| D22 | NO2B test file is single precision (float) *(user)*; the me2j test file truncations are e1max=14, e2max=28, lmax=14 (verified by value count). | Input facts. | user + verified |
| D23 | After the IMSRG(3) layout benchmark, the session **chooses the layout itself** and continues *(user)*. | Keeps the overnight run moving. | user |
