# Status

## Current state (keep this section up to date)

* **Phase:** overnight implementation run in progress (started 2026-11-11 23:41 JST).
* **Integration branch:** `develop` (local). It is advanced from the orchestrator's worktree
  branch `claude/imsrg-refactor-overnight-800606` after `tools/ci.sh dev-openmp` passes on the
  merge result (see QUESTIONS O1; the main checkout's index needs a sync in the morning).
* **Merged so far:** M0 (bootstrap: submodules, CMake, presets, `tools/ci.sh`, layering check);
  M1 partial (reference build script, refharness, golden data for basis/H/HF/MBPT and IT-1..IT-9
  at emax 2/4; IT-4/5/8 goldens are under investigation, see M1-Q1).
* **In progress:** M1 (`feature/m1-reference-golden`: NO2B/divergence investigation, commutator
  goldens, IT-10); M2 (`feature/m2-foundations`). Next: M3 (basis), branched from M2's
  `m2-api-ready` tag.
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
