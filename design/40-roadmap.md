# 40 — Roadmap and milestones

The overnight run (~8–12 h) cannot finish everything. Work **top to bottom**, keep `develop`
green, and record progress in `design/STATUS.md`. A milestone is **done** only when its
acceptance criteria pass in `tools/ci.sh` and its branch is merged into `develop`.

Columns: **Lead** = model of the agent leading the milestone; **Workers** = models for its
sub-tasks; **Par** = milestones that may run in parallel (separate worktrees). Lead `opus` means spawn an
`imsrg-lead` agent; lead `sonnet`/`haiku` means the milestone is small enough to delegate directly
to one `imsrg-implementer`/`imsrg-mechanic`, without a lead agent.

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
| MS | Google C++ style conversion (D38, `17-code-style.md` §5) | M8 | none (first in run 2, alone) | opus / sonnet, haiku |
| M12a | Single-node CPU performance: imsrg++ parity, then explore single-node limits (D40) | M8 | M10, M11 | opus / sonnet |
| M12b | MPI distribution (D14 phase 1; phase 2 for 3B blocks, evaluating the Hagen–Nam scheme, D40) | M12a | M10 | opus / sonnet |
| M13 | Plain Makefile, docs site, polishing | M8 | any | haiku / haiku |
| M14 | GPU builds and tuning (CUDA/HIP; after M12, D40) | M12b | — | opus / sonnet |
| MS2 | Structural Google-style pass (D50: structs with invariants → classes, unsigned → signed, long functions split, long inline header functions out of line); first in run 3, alone, time-boxed | MS | none (alone) | opus / sonnet, haiku |
| M15 | Tensor operators (D53): E2 and other electric multipoles, M1, Gamow–Teller/Fermi (ΔTz = ±1), 0νββ (ΔTz = ±2); scalar–tensor commutators, transformation with the Magnus Omega only; validation against the patched imsrg++ build and an m-scheme brute force for tensors | M10, M12b | — | opus / sonnet |
| M16 | ME1J/ME2JP/ME3JP readers and writers (text, gzip, later binary; `19-mexj-formats.md`), stored Omega with a `transform` task, checkpoint/restart (D54); acceptance includes the J-range safety rules and sentinel tests of `19` §8 (D55) | M15 | — | opus / sonnet |
| AL27 | The 27Al NREFT response task (D67, D76, D77; `50-al27-responses.md`): capabilities C1-C14, then variants v1 (fork compat, archive to 1e-4), v2 (fixed, unsplit), v3 (split Magnus 0.25), farms on GPUs; weekend run until 2026-11-30 10:00 EST | M14, M15, M16 | — | opus / sonnet |
| M17+ | Further extensions: me3j, Navratil/Miyagi files, dagger, DM (order open, D39) | M15 | — | opus / sonnet |

Run 1 (2026-11-11/12) finished M0–M9 and M11 phase 1. **Run-2 order (owner, 2026-11-15, D38/D39):**
(1) MS, alone, before any other branch; (2) the M8 follow-ups and the tasks of the 2026-11-15
interview (below); (3) M10, with the IMSRG(3f2) track in parallel with the layout benchmark, then
perturbative triples → N7; (4) M11 phase 2 and M12a alongside when capacity allows. M12b follows
M12a; GPU work (M14) starts only after M12. Keep going down the list for as long as the session runs.

Run 2 (2026-11-15/16) finished MS, the M8 follow-ups, M10 (3f2, layout A, triples, N7), M11
phase 2 and M12a. **Run-3 order (owner, 2026-11-16, D53):** (1) the small interview tasks (D47–D49,
D51 and their goldens); (2) MS2, alone and time-boxed, with the N7 and MPI work prepared outside the
repository meanwhile; (3) N7 performance with distribution-ready row tiling, triples at emax ≥ 8
and `occ_nat3_cut`, in parallel with M12b phase 1; (4) M12b phase 2 (row-wise 3B distribution,
D45); (5) M15 tensor operators; (6) M16 ME1J/ME2JP/ME3JP with stored Omega and checkpoint/restart.
M12a follow-ups (emax 10–12 commutator benchmarks allowed, D52) and the Python hooks (D51) are
filler; M13 last; M14 after M12b (CUDA and HIP compile checks, NVIDIA tuning first).

## MS — Google C++ style conversion (first task of run 2)
* Follow `17-code-style.md` §5: tooling, `.h`/`.cc` renames, `#define` guards, license boilerplate,
  clang-format at 80 columns, Google naming with clang-tidy, floating-point literals, cpplint fixes,
  structural items as far as time allows.
* **Accept**: `ci.sh` passes with the blocking cpplint step; clang-format and clang-tidy naming
  clean; Python names unchanged; test results bitwise identical to before the conversion.

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
* Build imsrg-devel per `20-build-and-tooling.md` §5 (`imsrg++`; `pyIMSRG` optional).
* Run the old executable for the IT-1 … IT-9 configurations and store the golden outputs
  (IMSRG(3) runs at emax 2 and 3 only). IT-10 (IMSRG(3f2)) and the flow trajectories come from the
  harness (`refharness flow [--factorized]`).
* Write `tools/reference/refharness.cpp` with the subcommands in `30-testing-and-validation.md`
  §3. It writes the OperatorDump format exactly as specified in `15-operator-dump-format.md`,
  which M4 implements too.
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
* T_int / Tcm / HCM / Rp2 / Rn2 / Rm2 (as needed by the tests); HF with 2N and NO2B 3N, plain
  density mixing (the imsrg++ path) **and a correct DIIS** *(owner)*; HF-basis transformation and
  normal ordering.
* **Accept**: IT-1 and the HF part of IT-4 pass; mixing and DIIS converge to the same solution.

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
* **Follow-ups for run 2 (owner, 2026-11-15):** Opus review of the flow numerics; hunter-gatherer
  (prerequisite of 3f2, D32); flow-RK4/DP45; `hartree_fock.mixing` (DIIS default in TOML, D37);
  MP2 column and EMP2; NAT basis and IT-5 in the driver; `Denominators` copying H; config-relative
  TOML paths and absolute paths in `resolved_config.toml` (D37); `results.json` `schema_version`
  (D37); Tcm without threshold and Rp2 without spin-orbit plus `Rso` (D35) with the patched
  reference build and regenerated e2e goldens; general (non-Hermitian) fast-commutator paths
  (D36); full-precision 1/√2 (D36); 3f2 config keys and refharness flags, IT-10 regeneration (D32).

## M9 — mbpt
* MP2/MP3 energies, MBPT(2) density matrix, natural orbitals.
* **Accept**: IT-5 passes; MP2/MP3 match imsrg-devel.

## M10 — IMSRG(3)
* Follow `14-imsrg3-plan.md`: layout benchmark and decision → brute-force m-scheme reference →
  perturbative triples → N7, with **3f2 as a parallel track** (no 3B storage; needs hunter-gatherer
  and the regenerated IT-10 goldens; owner, 2026-11-15, D39). 3B cuts drop kets at enumeration
  (D34). Production target: N7 at emax 10–12 (D34).
* **Accept**: IT-8, IT-9, IT-10 pass (emax 2; emax 3 under `slow`); benchmark report written.

## M11 — Python bindings
* Per `18-python-api.md`; grows with each module.
* **Accept**: pytest suite for bindings passes; the example pipeline script runs.

## M12 — performance (M12a) and MPI (M12b)
*Split by the owner, 2026-11-15 (D40).*
* **M12a, single node first:** reach imsrg++ parity (sparse Pandya maps, smaller Workspace,
  per-channel norm reductions, `No2bThreeBody::get_pn`, `Denominators`), then explore how far
  single-node CPU performance goes; judge mainly at larger emax, with occasional emax-8 IMSRG(2)
  benchmarks against imsrg++ (never in CI). **Accept**: benchmark report in `design/benchmarks/`
  (emax 4, 6, 8) vs imsrg-devel.
* **M12b, then scaling:** OwnershipMap-based distribution of the expensive IMSRG(2) terms (D14
  phase 1), then the 3B-block distribution for IMSRG(3) (phase 2), evaluating the row-wise,
  cost-balanced block distribution of Hagen and Nam (Prog. Theor. Phys. Suppl. 196, 102 (2012),
  arXiv:1203.3765) against distribution by channel. Single-node speed may fall back to imsrg++ level
  once scaled. **Accept**: IT-11 passes; scaling measurements recorded. **Target (owner, 2026-11-16, D45):** N7 at emax 10–12 with E3max 16 (no dE3max) on 22–256 nodes of 100–256 GB (3B operators ≤ 50 % of a node); single channels are split across ranks (row-wise, cost-balanced distribution as the baseline, whole channels only as the comparison); accepted by cost model at ≤ 10 % compute and memory imbalance and communication below compute at ~25 GB/s per node (InfiniBand and Slingshot clusters with Slurm).
* GPU work (M14) starts after M12.

## M13 — Makefile and docs
* `Makefile` per `20` §6; Doxygen config; `docs/` user guide (install, run, config reference).
