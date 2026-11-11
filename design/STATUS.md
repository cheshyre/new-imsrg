# Status

## Current state (keep this section up to date)

* **Phase:** overnight implementation run in progress (started 2026-11-11 23:41 JST).
* **Integration branch:** `develop` (local). It is advanced from the orchestrator's worktree
  branch `claude/imsrg-refactor-overnight-800606` after `tools/ci.sh dev-openmp` passes on the
  merge result (see QUESTIONS O1; the main checkout's index needs a sync in the morning).
* **Merged so far:** M0 (bootstrap: submodules, CMake, presets, `tools/ci.sh`, layering check);
  M1 (reference build script, refharness, golden data for basis/H/HF/MBPT/commutators and
  IT-1..IT-9; missing: `refharness flow`, hence no IT-10 golden and no %.17g IT-2 trajectory);
  M3 (basis: orbits, references, valence spaces, partitions, 2B/ph/3B channels, ModelSpace,
  DeviceTables; all 14 golden basis files match exactly), which also brought M2's early
  `base`/`exec`/`comm` commits (`m2-api-ready`).
* **In progress:** M2 (`feature/m2-foundations`: linalg and angmom committed, finishing CG/
  Moshinsky/Talmi), M4 (`feature/m4-operator`, from `m3-api-ready` + M2). Next: M5+M6 (one lead)
  and M7 from M4's `m4-api-ready` tag.
* **Interruption:** all agents stopped on the 5-hour usage limit from about 01:30 to 02:30 JST and
  were resumed. Usage budget, not wall time, is the binding constraint of the night.
* **Toolchain:** Homebrew cmake 4.4.5, ninja 1.13.2, ccache 4.14.3, Open MPI 5.0.11, libomp
  23.1.5, clang-format 23.1.5, gsl, boost, openblas; Apple clang 21; Python 3.15.2. Kokkos 5.3.0
  (submodule) installed per backend in `~/.cache/imsrg-deps` by `tools/bootstrap_deps.sh`; shared
  venv `~/.cache/imsrg-deps/venv`. Default MacOSX27.0.sdk does not link: `cmake/ImsrgAppleSdk.cmake`
  picks a working SDK (Q24).
* **Verified input facts:** me2j test file = 168,066,656 values (e1max 14, e2max 28, lmax 14);
  NO2B test file = 32,685,840 floats (6, 12, 18; single precision).
* **Blocking questions:** none. Open questions: `QUESTIONS.md` (Q20–Q24 from M0, O1–O2).

## Log (append one entry per merged branch)

| When (JST) | Branch | What landed | Tests | Follow-ups |
|------------|--------|-------------|-------|-----------|
| 2026-11-11 23:30 | feature/design-plan → develop | Design docs, reference analyses, AGENTS.md/CLAUDE.md, agent definitions, OperatorDump spec, me2j count tool, review fixes, next-session prompt | n/a | Start M0/M1 |
| 2026-11-12 00:45 | feature/orchestrator-housekeeping → develop | Agent `autoCompactWindow` 600000 (Sonnet/Haiku) / 800000 (Opus lead) after checking the docs (1M windows); integration mechanics recorded (O1, O2) | n/a (docs/config) | Values apply from the next session |
| 2026-11-12 00:45 | feature/m0-bootstrap → develop | Submodules (Kokkos 5.3.0, googletest 1.18.0, fmt 12.2.1, CLI11 2.7.2, toml++ 3.4.0, nanobind 3.1.0); CMake skeleton with `imsrg_add_module`/test helpers; presets dev-openmp/dev-serial/release-openmp/asan/cuda/hip; `bootstrap_deps.sh` (Kokkos per backend, shared venv); `ci.sh`; `check_layering.py` with fixtures; Apple SDK workaround; Kokkos and MPI smoke tests | `ci.sh` dev-openmp and dev-serial PASS (lead); dev-openmp PASS on the merge (orchestrator) | Q20 (Kokkos 5 needs CUDA 12.2/ROCm 6.2), Q21–Q24 |
| 2026-11-12 01:00 | feature/m1-reference-golden (partial, bf57ae9) → develop | `tools/reference/` (build script for imsrg-devel outside the repo, refharness linked to the old library: dump-basis, dump-hamiltonian, hf, commutator, flow; OperatorDump writer), `tests/golden/` (~1.6 MB: basis emax 2/4, H/HF/MBPT emax 2/4, e2e IT-1..IT-9) with README | CI dev-openmp PASS on the merge (reference code is not part of our build) | M1-Q1: E_HF = -313 MeV with the NO2B file and diverging flows (IT-4/5/8) under investigation; remaining M1 work continues on the branch |
| 2026-11-12 02:40 | feature/m1-reference-golden (final) → develop | Commutator subcommand and emax-2 term goldens, emax-3 e2e goldens, divergence investigation with replacement goldens (IT-4/IT-5 stopped at s = 10; IT-8 He4 emax 2 and O16 emax 3), README index. Key numbers (O16 emax 4, 2N): E_HF −147.893576, EMP2 −15.579114, IMSRG(2) E0 −164.648772308 (256 steps); IT-9 N7 E0 −145.489827 (emax 2), −158.254073 (emax 3) | CI dev-openmp PASS on the merge; 32 dumps round-trip byte-identically (lead) | M1-Q1: 3N file is cE-only (genuine E_HF −313 MeV); O16 emax-2 IMSRG(2) is unstable. TODO: `refharness flow` (IT-10, IT-2 %.17g trajectory) before M10 |
| 2026-11-12 03:00 | feature/m3-basis → develop | `basis`: Orbit/HO basis/nucleus filling, Reference (fractional occupations), named valence spaces, Partition, TwoBodyChannels + 9 subsets, ParticleHoleChannels, ThreeBodyChannels (e3max + Pauli cuts), ModelSpace, DeviceTables; golden parser and record-by-record comparison; 2 extra goldens with active cuts. Brings M2's `m2-api-ready` commits (base, exec, comm) | 310 tests; ci.sh dev-openmp + dev-serial PASS (lead); dev-openmp PASS on the merge | Deviations M3-1..6 in `30` §6; M3-Q1..Q9 (6j Pandya maps left to M7; 3B device tables; dE3max/OccNat cuts per ket) |
