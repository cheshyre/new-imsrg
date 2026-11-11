# 40 — Roadmap and milestones

The overnight run (~8–12 h) cannot finish everything. Work **top to bottom**, keep `develop`
green, and record progress in `design/STATUS.md`. A milestone is **done** only when its
acceptance criteria pass in `tools/ci.sh` and its branch is merged into `develop`.

Columns: **Lead** = model of the agent leading the milestone; **Workers** = models for its
sub-tasks; **Par** = milestones that may run in parallel (separate worktrees).

| ID | Milestone | Depends | Par | Lead / Workers |
|----|-----------|---------|-----|----------------|
| M0 | Bootstrap: tools, skeleton, CMake, presets, submodules, CI script | — | M1 | opus / sonnet, haiku |
| M1 | Reference build of imsrg-devel + golden data harness | — | M0, M2, M3 | sonnet / haiku |
| M2 | `base` + `exec` + `comm` + `linalg` + `angmom` | M0 | M1 | opus / sonnet |
| M3 | `basis`: orbits, references, partitions, 2B/3B channels | M2 | M1 | sonnet / haiku |
| M4 | `operator`: parts, arithmetic, norms, normal ordering, basis transform, OperatorDump | M3 | — | opus / sonnet |
| M5 | `io`: me2j + NO2B readers, tokyo/nushellx writers | M4, M1 | M6, M7 | sonnet / haiku |
| M6 | `opslib` (scalar) + `hf` (2N + NO2B) | M4 (M5 for tests) | M7 | sonnet / sonnet |
| M7 | `commutator` IMSRG(2) + `commutator_ref` | M4 | M5, M6 | opus / sonnet |
| M8 | `flow` (generators, Magnus, BCH, flow-RK4) + `driver` + `apps/imsrg` + config/legacy CLI | M5, M6, M7 | M11 | opus / sonnet |
| M9 | `mbpt`: MP2/MP3, natural orbitals | M6 | M10 | sonnet / sonnet |
| M10 | IMSRG(3): layout benchmark → perturbative triples → N7 → 3f2 | M8 | M9 | opus / opus+sonnet |
| M11 | Python bindings (nanobind) | M4 (grows with M5–M8) | M8–M10 | sonnet / haiku |
| M12 | MPI work distribution + performance pass | M8 | — | opus / sonnet |
| M13 | Plain Makefile, docs site, polishing | M8 | any | haiku / haiku |

Realistic overnight target: **M0–M8 done**, M9 and M11 started, the M10 layout benchmark
started if time allows. Keep going down the list for as long as the session runs.

---

## M0 — Bootstrap
* `brew install cmake ninja ccache open-mpi libomp` (plus `gsl boost openblas` for M1); `.venv`
  with numpy, pytest, sympy, scikit-build-core.
* Submodules (pinned release tags, recorded in `extern/README.md`): kokkos, googletest, fmt,
  CLI11, tomlplusplus, nanobind.
* Repository skeleton per `10-architecture.md` §1; `imsrg_add_module` helper; presets per
  `20-build-and-tooling.md` §3; `tools/bootstrap_deps.sh` (installs Kokkos once per backend);
  `tools/ci.sh`; `tools/check_layering.py` + its CTest; `.clang-format`; `.gitignore` additions
  (`build*/`, `.venv/`, generated golden data).
* **Accept**: `tools/ci.sh dev-openmp` and `tools/ci.sh dev-serial` both pass with a trivial
  Kokkos kernel test and an MPI hello-world test (`mpirun -n 2`); the layering check passes.

## M1 — Reference build and golden harness
* Build imsrg-devel per `20-build-and-tooling.md` §5 (`imsrg++` and `pyIMSRG`).
* Run the old executable for IT-1 … IT-10 configurations and store the golden outputs.
  IMSRG(3) runs are emax 2 and 3 only.
* Write `tools/reference/refharness.cpp` with the subcommands in `30-testing-and-validation.md`
  §3. The OperatorDump format spec is shared with M4: write the spec first, in
  `docs/operator-dump-format.md`.
* **Accept**: golden files and `tests/golden/README.md` committed (small files only); harness
  builds against the old library and round-trips an OperatorDump.

## M2 — Foundations
* `base`: strong types (with `static_assert`s), errors, assertions, logging, timer registry,
  physical constants (copy imsrg++ values exactly; cite `PhysicalConstants.hh`), factorials,
  half→float.
* `exec`: Runtime (Kokkos + MPI init), DeviceBuffer, parallel patterns. `comm`: Communicator +
  serial stub. `linalg`: MatrixView, BlockSparseMatrix, gemm, grouped_gemm, eigh, with BLAS
  backend detection.
* `angmom`: 3j, 6j, 9j, CG, Moshinsky, Talmi; dense tables.
* **Accept**: unit tests incl. Wigner symbols vs sympy-generated tables (tolerance per
  `30` §2) for all arguments needed up to emax 8; linalg vs naive loops; Kokkos tests pass on
  Serial and OpenMP.

## M3 — basis
* Orbits and index formula, nucleus parsing, reference filling (incl. open shells), named
  valence spaces (copy the imsrg++ table), Partition, TwoBodyChannels with subsets,
  ThreeBodyChannels (layout-agnostic enumeration + cuts), ParticleHoleChannels (Pandya maps).
* **Accept**: orbit/channel/ket counts and orderings identical to imsrg-devel for several
  emax/reference/valence-space combinations (golden from the harness or pyIMSRG).

## M4 — operator
* OperatorSpec, OneBody, TwoBody (BlockSparseMatrix), optional ThreeBody interface; arithmetic;
  norms (D12 conventions); normal ordering (2B; 3B→2B/1B/0B for NO2B input), undo; unitary basis
  transformation; random operator generator; OperatorDump read/write.
* **Accept**: property tests; round trips; norms equal imsrg-devel norms on dumped operators.

## M5 — io
* me2j reader (gz + plain) with validation; NO2B reader (float/half); tokyo and nushellx writers.
* **Accept**: the bare Hamiltonian at emax 2 and 4 read from the real test files equals the
  imsrg-devel dump (tolerance `30` §2); writers reproduce imsrg-devel files numerically.

## M6 — opslib (scalar) and hf
* T_int / Tcm / HCM / Rp2 / Rn2 / Rm2 (as needed by the tests); HF with 2N and NO2B 3N;
  HF-basis transformation and normal ordering.
* **Accept**: IT-1 and the HF part of IT-4 pass.

## M7 — IMSRG(2) commutators
* All scalar IMSRG(2) terms; `commutator_ref`; CommutatorOptions/TermMask; Workspace; first
  Kokkos implementations; grouped GEMM path for pp/hh and ph.
* **Accept**: fast vs reference tests and term-by-term golden tests vs imsrg-devel pass; a
  micro-benchmark at emax 4/6 is recorded.

## M8 — flow, driver, executable
* Generators (phase-1 list in `13` §7), BCH transform/product, Magnus solver with step control
  and Omega stack, flow-RK4/DP45, operator transformation; driver pipeline; TOML config and
  legacy `key=value` translation (`16-config-and-cli.md`); results JSON + flow CSV; tokyo output.
* **Accept**: IT-2, IT-3, IT-4, IT-6, IT-7, IT-12 pass.

## M9 — mbpt
* MP2/MP3 energies, MBPT(2) density matrix, natural orbitals.
* **Accept**: IT-5 passes; MP2/MP3 match imsrg-devel.

## M10 — IMSRG(3)
* Follow `14-imsrg3-plan.md`: layout benchmark and decision → brute-force m-scheme reference →
  perturbative triples → N7 → 3f2.
* **Accept**: IT-8, IT-9, IT-10 pass (emax 2; emax 3 under `slow`); benchmark report written.

## M11 — Python bindings
* Per `18-python-api.md`; grows with each module.
* **Accept**: pytest suite for bindings passes; the example pipeline script runs.

## M12 — MPI and performance
* OwnershipMap-based distribution of the expensive IMSRG(2) terms (D14 phase 1); profiling
  pass; batched GEMM tuning.
* **Accept**: IT-11 passes; benchmark comparison vs imsrg-devel recorded.

## M13 — Makefile and docs
* `Makefile` per `20` §6; Doxygen config; `docs/` user guide (install, run, config reference).
