# Frontier port report — 2026-11-26

Supervised session on OLCF Frontier (project nph123), 2026-11-26 00:00-03:00 EST, orchestrator
Opus 5.5 in the integrator's clone `$IMSRG_WORK/main`. `develop` = **35808b1** (merge of
`feature/frontier-build`), pushed to the hub and to GitHub after `tools/ci.sh frontier-cpu` passed
on the merge. Per-merge details: `design/STATUS.md`; open items: `design/QUESTIONS.md`, section
"Added during the Frontier port".

## 1. Summary

| Task | Result |
|------|--------|
| 1. Build and CI | `tools/ci.sh frontier-cpu` **PASS**: 1241 unit, 69 integration, 18 MPI, 150 Python (1 skipped), 316-344 s. Slow suite 7/7 (2052 s on a node shared with two agent builds). `tools/ci.sh frontier-hip` **PASS** (compile check). `tools/lint.sh`: 290 units, no findings. `frontier-serial`: unit 1241/1241, one integration timeout (FP-Q7). |
| 2. Parallel agents | **Scheme (a) works**: `isolation: worktree` + WorktreeCreate hook cloning the hub, tested end to end, then used for four real tasks. Fallback (b) exists but was not needed. |
| 3. Rules and paths | AGENTS.md, CLAUDE.md, design/20 §9, design/41 §2-3, the agent files, the settings proposal and NEXT_SESSION_PROMPT.md updated; tools/reference and the agent files parameterized (`IMSRG_REFERENCE_ROOT`, `IMSRG_REFERENCE_BUILD`, `IMSRG_WORK`). |
| 4. Reference build | Done: pristine and D35-patched imsrg++ and refharness build on Frontier (`$IMSRG_WORK/reference-build`); regenerated emax-4 goldens match `$IMSRG_WORK/golden-generated` within 1e-15 relative (HNO 2e-13 absolute). |

## 2. Toolchain and CI (design/20 §9)

* `tools/lib/frontier_env.sh`: pinned modules (PrgEnv-gnu 8.7.0, gcc-native 14.2 via the Cray `CC`
  wrapper, cray-mpich 9.1.0, cray-libsci 26.03.0, cmake 4.1.5, ninja 1.13.2, ccache 4.12.1,
  Python 3.14.3; rocm 7.0.2/hipcc for HIP), allocation lookup (job named `imsrg`), `imsrg_compute`
  (`srun --overlap`, 56 cores). `tools/ci.sh frontier-*` runs configure, build and tests on the
  compute node, MPI tests via nested `srun -n`; lint steps on the login node.
* Presets `frontier-cpu` (CI), `frontier-serial`, `frontier-cpu-release`, `frontier-hip`; test data
  `$IMSRG_WORK/inputs`, generated goldens `$IMSRG_WORK/golden-generated` (shared by all clones).
* `tools/bootstrap_deps.sh` without Homebrew: Kokkos openmp, serial, hip in `$IMSRG_DEPS_PREFIX`;
  the venv only on the login node (pip hangs on compute nodes, which have no internet).
  cpplint 2.0.2, clang-format 23.1.5, clang-tidy 23.1.4 pinned in `tools/requirements-dev.txt`.

**What the move exposed (all fixed on the branch, no tolerance changed):**
1. GCC 14 `-Werror`: ~25 warnings Apple clang never gave (dangling-else around gtest macros,
   missing field initializers, a shadowed parameter, `__int128` pedantic, a maybe-uninitialized
   false positive in the legacy translator); fmt built without `-fPIC` broke the Python module.
2. CMake 4 + GCC 14 + Ninja scanned every source for C++20 modules: `CMAKE_CXX_SCAN_FOR_MODULES=OFF`.
3. Two cray-libsci variants in one process: FindMPI interrogated the wrapper and recorded the serial
   libsci as an MPI library while `-fopenmp` links pulled the OpenMP one
   (`CRAYBLAS_WARNING`); fixed with `MPI_CXX_LIB_NAMES=mpi_gnu`.
4. **Threaded cray-libsci is not reentrant across host threads**: two threads in `dgemm` crash it
   (reproduced in 20 lines without Kokkos or our code). `linalg` now serializes BLAS/LAPACK calls
   with one process-wide lock (`IMSRG_BLAS_SERIALIZE`, on except with Accelerate); FP-Q6.
5. IT-14 adaptive flow (`He4Emax2AdaptiveMatchesImsrgFlowFile`): |Δs| 1.27e-5 > 1e-5. Cause: the
   solver-level test used our default HF (DIIS, 8 iterations, rounding-sensitive) while the golden
   used imsrg++'s linear mixing (26 iterations); the adaptive controller amplified the difference.
   With linear mixing every printed s equals imsrg++ run on Frontier (0/572 rows differ); vs the Mac
   golden max |Δs| = 7.6e-6 (the rows where imsrg++ itself differs between Mac and Frontier). FP-Q5.
6. HIP (clang + libstdc++): `vector<pair<string, T>>` with `T` incomplete (`RawValue`, test `Json`)
   and a defaulted `DeviceBuffer` constructor that hipcc made `__host__ __device__`.
7. The clang-tidy wheel lacks `omp.h`; `lint.sh` now also fails when a unit does not parse (before,
   unparsable units silently gave "no findings").

No check that was bitwise on the Mac changed: every unit test passes unchanged, so no Frontier
baselines were needed. The one golden comparison that moved is item 5 (fixed at the cause).

## 3. Agent isolation (design/41 §2-3; proposal P-F1)

* `WorktreeCreate` hook → `tools/agent_clone.sh`: hub clone `$IMSRG_WORK/agents/agent-<id>` on
  `feature/agent-<id>` from `develop` (or `agents/.default-base`, or the spawning lead's pushed
  branch, detected from the hook's `cwd`), submodules borrowed locally, 8 s.
* `PreToolUse` guard `tools/hooks/agent_guard.py` (binds each agent_id to its clone or read-only);
  hub `update` hook (no `main`; `develop` only with `IMSRG_INTEGRATOR=1`); pre-commit and
  pre-merge-commit in the integrator's clone; `tools/agent_retire.sh` (also `WorktreeRemove`).
* Tests: headless sessions (mechanic: own clone, commit, push; denied: edit of main, `git -C main`,
  `cd main`, push to develop; lead → worker: worker started from the lead's pushed commit; removal
  keeps unpushed work; hub and pre-commit rejections verified). Then four production agents
  (GCC warnings, reference port, test fixes, HIP) worked in parallel clones on one allocation; the
  guard denied one stray edit outside a clone. Finding: Claude Code never calls `WorktreeRemove`
  when a subagent finishes, so the integrator retires clones (`agent_retire.sh --all --merged`).

## 4. Usage and environment notes

* Five subagents (one docs fetcher, four implementers incl. one Opus) and three headless test
  sessions. Allocations: 2-hour `batch` jobs named `imsrg`, chained with `--dependency=afterany`
  (FP-Q3). The job 5998241 running at the start belonged to another project of the owner and was
  not used.
* Local git identity set in `$IMSRG_WORK/main` (FP-Q2). Throwaway test clones and a core file were
  moved to `$IMSRG_WORK/trash/` (deleting them was refused by the permission system); the hub
  keeps the test branches `feature/agent-aed5510638e6f6d22`, `feature/agent-ab7ad3f93c28d0a1e`,
  `feature/selftest` (P-F1).
* `.claude/settings.json` (committed) holds only the hooks; `.claude/settings.local.json` was not
  changed (proposal in `design/claude-settings.proposed.json`, FP-Q1).

## 5. For the owner interview

P-F1 (isolation scheme a; hub branch clean-up), P-F2 (GitHub policy; `main`), FP-Q1 (settings
proposal), FP-Q2 (git identity), FP-Q3 (allocation chains for unattended runs), FP-Q4 (which
generated-golden set is current), FP-Q5 (DIIS rounding sensitivity in driver-level tests), FP-Q6
(BLAS lock vs serial libsci), FP-Q7 (frontier-serial warning and timeout). Then run 4 per
`design/NEXT_SESSION_PROMPT.md` (Frontier version).
