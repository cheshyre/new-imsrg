# 00 — Requirements and decision log

This file records **what** we are building and **which decisions are already made**
(with their rationale). It is the first thing every agent should read after `AGENTS.md`.
Decisions marked *(user)* came from the interviews with the project owner
(2026-11-11, 2026-11-13, 2026-11-15) and must not be changed without asking. Decisions marked *(design)* are
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
| R11 | Test inputs: 2N `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` (`file2e1max=14 hw=16`); 3N NO2B `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin` (`hw=16`, `file3e1max=6 file3e2max=12 file3e3max=18`, `3bme_type=no2b`). Both live in the main checkout root `/Users/cheshyre/Downloads/new-imsrg/`. **emax ≤ 6 in all tests; IMSRG(3) tests only at emax = 2, 3.** Exception (D40): occasional emax-8 IMSRG(2) performance benchmarks against imsrg++, never in CI. |

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
  Haiku/Sonnet and about 80 % for Opus (see `41-agent-workflow.md` for how; D31).
* Runs start in the main checkout on `develop` (D31); an owner interview follows every run (D42).
* Code style: the Google C++ Style Guide, with clang-format and cpplint (D38, `17-code-style.md`).
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
| D12 | **Same numerics-defining algorithms as imsrg++ by default**: norm conventions (1B weighted by (2j+1), 2B by (2J+1)², 3B as imsrg++'s Frobenius norm over its canonical pn storage for every layout; zero-body part excluded from the total), the Magnus step-size rule, the Omega splitting criterion, BCH termination thresholds, HF mixing and convergence. | Reproducible flow trajectories, hence comparable energies. Genuine bugs are fixed per the user policy and logged in `30-testing-and-validation.md` §6. | design |
| D13 | Public headers of the data model **do not include Kokkos**. Device storage goes through a pimpl'd `DeviceBuffer<T>`, and kernels live in `.cc` files that include the backend. | Incremental build speed (Kokkos_Core.hpp is heavy) and replaceability (R8). | design |
| D14 | MPI phase 1: 1B/2B operator parts replicated; expensive commutator terms split by channel blocks across ranks; results combined with `Allreduce`. Phase 2: 3B blocks distributed by 3B channel with an ownership map. MPI sits behind `imsrg::comm`, which has a serial stub so builds without MPI work. *Amended by D40:* the phase-2 3B distribution is chosen by evaluation, including the row-wise, cost-balanced block distribution of Hagen and Nam, because whole channels per rank are likely unbalanced. | IMSRG(2) operators fit in node memory up to emax ≈ 14. IMSRG(3) memory is dominated by 3B blocks, which most terms only need channel-locally. | design |
| D15 | 3B input formats (NO2B, me3j) are **separate input types**, converted into the canonical operator layout by explicit functions. The IMSRG(3) operator layout is chosen by benchmark *(user)*. | Separates file-format concerns from the compute layout (imsrg++ mixes them with a slicing-prone class hierarchy). | design + user |
| D16 | Library code never calls `exit()`. User and input errors throw typed exceptions (`imsrg::InputError`, ...). Invariant checks use `IMSRG_ASSERT` (debug) or `IMSRG_CHECK` (always on). Kept as a documented exception to the Google style guide *(user, 2026-11-15; D38)*. | Usable as a library from Python. | design + user |
| D17 | Executable `imsrg`, C++ namespace `imsrg`, CMake targets `imsrg::<module>`, Python package `imsrg` (extension `imsrg._core`). License **GPL-3.0-or-later** (repo LICENSE; per-file SPDX header per D38) *(user, 2026-11-15)*; compatible with imsrg++, whose sources are GPL-2.0-or-later. | Consistent naming. | design + user |
| D18 | Development platform is macOS arm64 (Apple M5 Pro, 18 cores, 48 GB) with Kokkos **OpenMP** and **Serial** backends. GPU builds (CUDA/HIP) cannot be tested here, so we follow strict device-code rules (`12-compute-backend.md` §5) and add compile-only checks later. | No GPU on the dev machine. | design |
| D19 | Tolerance policy *(user)*: tight deterministic unit checks (~1e-10 relative); end-to-end acceptance at 1e-4 MeV (HF, MBPT) and 1e-3 MeV (IMSRG E0, valence-space elements), always logging the actual deviation. The same tolerances apply to IMSRG(3) (D34). | Unit checks catch real bugs; flows involve thresholds and step decisions. | user |
| D20 | Toolchain baseline *(user)*: GCC ≥ 11, CUDA ≥ 12, ROCm ≥ 6, plus Apple clang on the dev machine. Pin the latest stable Kokkos that supports these. *Amended by D30:* CUDA ≥ 12.2, ROCm ≥ 6.2. | Clusters the owner uses. | user |
| D21 | The generator's denominator floor keeps the sign of the denominator *(user: imsrg++ loses it, which is a bug)*. | Deviation logged in `30` §6. | user |
| D22 | NO2B test file is single precision (float) *(user)*; the me2j test file truncations are e1max=14, e2max=28, lmax=14 (verified by value count). | Input facts. | user + verified |
| D23 | After the IMSRG(3) layout benchmark, the session **chooses the layout itself** and continues *(user)*. | Keeps the overnight run moving. | user |
| D24 | Name stays `imsrg` *(user)*. Input formats *(user)*: me2j, NO2B, me3j, plus Navratil/Miyagi operator files. HDF5 3N, Oslo, OakRidge and the other older readers are dropped. HF gets a correct DIIS from the start *(user)*, with plain mixing kept for comparisons. | Owner's actual workflows. | user |
| D25 | Tensor operators store only the `ch_bra ≤ ch_ket` channel blocks, as imsrg++ does *(user, Q5; confirmed 2026-11-15)*. | Half the memory; element-by-element comparison with imsrg++. | user |
| D26 | IMSRG(3) three-body blocks are stored in `double` only; no `float` storage *(user, Q6)*. | Accuracy first; memory savings come from the layout and from distribution. | user |
| D27 | Legacy quirks Q11 (`nsteps=1` with a valence space uses the core generator), Q12 (`imsrg3_at_end` without `IMSRG3` drops the input 3N) and Q14 (`denominator_delta` without an orbit is ignored) are kept as in imsrg++, each with a warning (`W-NSTEPS1`, `W-Q11`, `W-Q3`) *(user)*. Which legacy parameters are actually useful will be re-evaluated later. | Comparability with imsrg++ runs now; cleanup later. | user |
| D28 | The TOML config requires `basis.emax`, `basis.hw` and the interaction files' `e1max` explicitly (no silent defaults); the legacy `key=value` layer keeps the old defaults and warns *(user, Q13)*. | Safer new configs without breaking old command lines. | user |
| D29 | Python: explicit `imsrg.Runtime()` context manager plus a lazy default (Q15); read-only zero-copy host views that raise `StaleHostError` unless `sync_to_host()` was called (Q16); minimum Python **3.12**, so nanobind's stable ABI can be used (Q17) *(user)*. | One wheel per platform; no hidden device-to-host copies. | user |
| D30 | Toolchain baseline raised to **CUDA ≥ 12.2 and ROCm ≥ 6.2** (GCC ≥ 11 unchanged); Kokkos 5.3.0 stays pinned *(user, Q20; amends D20)*. | These are Kokkos 5.3.0's own minimums; clusters with older toolkits load a newer module. | user |
| D31 | Unsupervised runs *(user, O1, O2, Q19)*: start in the **main checkout on `develop`** and merge there (the worktree staging procedure of O1 is only a fallback); keep pace with ≤ 3 leads and Sonnet-heavy delegation, resume stopped agents with `SendMessage` after each usage reset, and launch right after a reset; compaction only through `autoCompactWindow` (600000 for Sonnet/Haiku agents, 800000 for `imsrg-lead`, 800000 for the orchestrator through the settings key), **no** `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`. | The usage budget, not wall time, bounded run 1. The percentage override also applies to subagents on top of their windows (Claude Code docs), which would compact Sonnet/Haiku agents at ~48 %. | user |
| D32 | **IMSRG(3f2)** = IMSRG(2) with hunter-gatherer Magnus, `omega_norm_max` 0.1, and the factorized double-commutator correction in BCH with 1b intermediates on and 2b intermediates off (the owner's pyIMSRG recipe, M1-Q2). `[imsrg3] factorized = true` makes these values the defaults (explicit keys win; `resolved_config.toml` writes them out); new keys `factorized_1b_intermediates` (default true) and `factorized_2b_intermediates` (default false). The IT-10 goldens are regenerated with these settings *(user)*. | The owner's production definition; the old IT-10 goldens used imsrg++ defaults (no hunter-gatherer, 0.25, 2b intermediates on). | user |
| D33 | The 3N test input stays the cE-only NO2B file (IT-4/IT-5 compare HF plus the flow to s = 10); a full N2LO file comes later from the owner and then gets full-flow goldens, while the cE file remains a permanent code-path test *(user, M1-Q1)*. O16 at emax 2 is a known, genuine IMSRG(2) instability; IT-8 keeps He4 emax 2 and O16 emax 3 *(user)*. | No full file is available yet. | user |
| D34 | IMSRG(3) *(user)*: the 3B configuration cuts (dE3max, OccNat3Cut) **drop kets at enumeration**, so the cut defines the 3B space identically in every term (M3-Q9); imsrg++'s IMSRG(3) bugs (`14` §1) are fixed and those terms are validated against the brute-force m-scheme reference; tolerances as in D19 (terms 1e-10 relative, perturbative triples 1e-4 MeV, N7/3f2 E0 1e-3 MeV). Production target: **IMSRG(3)-N7 at emax 10–12** on several nodes. | Memory is the IMSRG(3) bottleneck; consistent cuts are physically cleaner; the target sizes the layout and MPI phase 2. | user |
| D35 | Two imsrg++ quirks are fixed *(user, M6-Q1/Q2)*: Tcm keeps all two-body entries (no 1e-7 threshold; T_int and H change by ≤ 1e-7 MeV per element), and Rp2 excludes the spin-orbit correction, which becomes its own catalog operator (no `include_spin_orbit` option, no compatibility switch). The e2e goldens are regenerated from a **patched imsrg-devel build** (patch kept in `tools/reference/`), so the ITs keep ~1e-10 MeV agreement and identical step counts. | Correct operators without carrying compatibility switches, while keeping tight regression checks. | user |
| D36 | Numerics *(user)*: 1/√2 at full precision (mathematical constants full precision, physical constants copied exactly; M2-Q3); Moshinsky brackets in double-double (M2-Q7); the DIIS design of M6-Q3; the fixes M8-1..M8-3 (the D21 floor keeps the sign, the Hartree-Fock generator leaves η's diagonal zero, a diverging flow throws `NumericalError`); the fast IMSRG(2) commutator gets **general (non-Hermitian) paths** in run 2 (M7-Q2); natural-orbital options `use_NAT_occupations`/`NAT_order` not now (M9-Q3). | Accuracy fixes cost nothing measurable; general operators avoid a hidden restriction. | user |
| D37 | Config and CLI *(user)*: `hartree_fock.mixing` defaults to `diis` in TOML and to linear mixing in the legacy layer (M8-Q3); relative paths in a TOML config resolve against the **config file's directory**, legacy and command-line values against the working directory, and `resolved_config.toml` records absolute paths (M8a-Q5); legacy `valence_space=<A>` with a different `reference=<B>` stays unsupported (M8a-Q4); `results.json` carries a `schema_version`, with the schema documented in `16` and changes in a changelog. | Faster HF for new configs and imsrg++ reproduction for old command lines; portable configs; stable downstream parsing. | user |
| D38 | **Code style: the Google C++ Style Guide** *(user)*, details in `17-code-style.md`: clang-format (Google base, 80 columns, Google include order); cpplint in CI with a `CPPLINT.cfg` that filters its false positives (the linter is not followed blindly); clang-tidy `readability-identifier-naming` for full Google naming (PascalCase functions, where accessors/mutators may stay snake_case; `kName` constants and enumerators); `.h`/`.cc` files; `#define` guards; license boilerplate `// SPDX-License-Identifier: GPL-3.0-or-later` plus `// Copyright 2026 The imsrg authors` (AUTHORS file). Documented exceptions: C++ exceptions (D16), `<filesystem>` (apps, driver, io, tests only), Kokkos `[=]` device lambdas. The conversion is the **first task of run 2** (one branch, one commit per mechanical step, CI and bitwise-identical results after each). | The owner's choice; converting early, while the code base is 41k lines, is cheapest. | user |
| D39 | Priorities *(user)*: run 2 = style conversion → M8 follow-ups and the tasks of the 2026-11-15 interview → **M10**, with 3f2 developed in parallel with the layout benchmark, then perturbative triples → N7; M11 phase 2 and M12a alongside when capacity allows. After IMSRG(3), **tensor operators** come first (the order of me3j input, Navratil/Miyagi files, dagger, 0νββ and DM operators is open). Later: checkpoint/restart (after M13, with scratch storage for Omega) and two new output formats, **me2jp/me3jp** (specification from the owner); no HDF5. Only the tokyo and NuShellX outputs must stay file-compatible with imsrg++. | IMSRG(3) is the top extension; 3f2 needs no 3B storage, so it does not wait for the layout decision. | user |
| D40 | Performance and scale *(user)*: **M12a** single-node CPU optimization first (imsrg++ parity, then explore how far one node goes, judged mainly at larger emax, with occasional **emax-8 IMSRG(2) benchmarks** against imsrg++, never in CI); then **M12b** MPI per D14, evaluating for the 3B blocks the row-wise, cost-balanced block distribution of Hagen and Nam (Prog. Theor. Phys. Suppl. 196, 102 (2012), arXiv:1203.3765, described there for the 2B blocks of coupled cluster) against distribution by channel, which is likely unbalanced. Once scaled, single-node speed may fall back to imsrg++ level. **GPU work starts after M12.** Target hardware: NVIDIA H100/H200/GH200 and A100, AMD MI250X/MI300A, CPU-only clusters. | Understand the single-node limits before scaling; performance matters most in large spaces. | user |
| D41 | Python *(user)*: the main users are the owner's group (pipelines) and operator-algebra work; packaging as **pip wheels** (cp312-abi3 via cibuildwheel, Kokkos OpenMP inside; source builds for GPU/MPI clusters); finalization at interpreter exit stays as is and is revisited for GPU builds (M11-Q1). | Matches the actual users; wheels cover laptops and workstations. | user |
| D42 | Testing and process *(user)*: `tools/ci.sh` stays under 5 minutes (139 s on 2026-11-15); longer tests carry the `slow` label and run before merges that touch numerics; goldens are regenerated only through the scripted `tools/reference` commands, with the reason and the old vs new numbers in the commit message and a reviewer pass; git stays local (no push, no PRs); an owner interview follows every run; the run-1 orchestration and report format are kept. | Fast feedback for agents; traceable golden changes. | user |
| D43 | **IMSRG(3) three-body storage: layout A** (run 2, M10 step 1): the imsrg++ canonical kets \|(ab)Jab, c; J⟩ with a ≤ b ≤ c and every allowed Jab per (2J, parity, 2Tz) channel, kets failing dE3max/OccNat3Cut dropped at enumeration (D34), one block per channel stored as a packed triangle; the [2,2]→3 term and the perturbative triples go through pair-third tiles (one intermediate per orbit-triple pair, contracted with the `comm223ss` recoupling map). Rejected: an orthonormal basis per triple (B) and fully antisymmetrized pair-third storage (C). | Measured (`benchmarks/imsrg3-layout.md`): C needs 8.8× A's memory (702 GB per operator at emax 12 / E3max 14) and 1.7× its [2,2]→3 time; B saves 2 % memory at the target and is within the 2-4 % run-to-run spread of A in every kernel, but needs dense basis maps for every access by orbit labels; A compares element by element with imsrg++. The tiled [2,2]→3 reproduces imsrg++ to 1e-15 and is 54× faster at emax 3 (18 threads). | design (session decision, D23) |
