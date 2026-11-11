# Status

## Current state (keep this section up to date)

* **Phase:** overnight implementation run in progress (started 2026-11-11 23:41 JST).
* **Integration branch:** `develop` (local). It is advanced from the orchestrator's worktree
  branch `claude/imsrg-refactor-overnight-800606` after `tools/ci.sh dev-openmp` passes on the
  merge result (see QUESTIONS O1; the main checkout's index needs a sync in the morning).
* **Merged so far:** M0 (bootstrap: submodules, CMake, presets, `tools/ci.sh`, layering check).
* **In progress:** M1 (`feature/m1-reference-golden`: reference build, refharness, golden data;
  IT-1..IT-7 goldens committed on the branch). Next: M2 (foundations), then M3 (basis) stacked on
  M2's early `base`/`exec` commit.
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
