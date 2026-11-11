# Open questions for the project owner

Format: `Q<n> (date, source) — question. Options. **Assumed meanwhile:** ...`
Agents append new questions at the end and never block on them: choose the most reversible
option, write it under "Assumed meanwhile", and continue. The owner answers inline (`**Answer:**`).

## From the planning session (2026-11-11)

### Inputs and conventions
* **Q1** — The 2N file `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` has one header
  comment line, `(*** nn_transforms v0.2.0 -- generated on 03/07/2022, 10:56:07 -- me2j-f2 ***)`,
  that does not contain the truncations. **Answer (verified 2026-11-11):** the file holds exactly
  168,066,656 values after the header line, which is the count the imsrg++ loop structure predicts
  for `file2e1max=14 file2e2max=28 file2lmax=14` (e2max=24 would give 115,792,936; lmax=10 would
  give 107,554,520). Use 28 and 14. Counting script: `tools/reference/me2j_value_count.py`.
* **Q2** — NO2B file `..._ms6_12_18.stream.bin`: single precision (float) or half? **Answer
  (owner):** single precision (float).
* **Q3** — Tolerances. **Answer (owner):** keep the deterministic unit-level checks tight (reading,
  Wigner symbols, single commutator terms ~1e-10 relative). Relax end-to-end acceptance to HF
  1e-4 MeV and IMSRG E0 / valence-space elements 1e-3 MeV, and always log the actual deviation.
  Applied in `30-testing-and-validation.md` §2.

### Architecture
* **Q4** — Project and executable name. **Answer (owner):** keep `imsrg` (namespace, executable,
  Python package).
* **Q5** — Tensor operators: store only `ch_bra <= ch_ket` blocks (as imsrg++ does; half the
  memory) or all blocks (simpler kernels)? Decide when tensor operators are implemented.
* **Q6** — May IMSRG(3) 3B blocks use `float` storage to halve memory, if the benchmark shows
  acceptable accuracy (e.g. ΔE0 < 1 keV)? **Assumed meanwhile:** `double` only.
* **Q7** — In imsrg++ the generator's denominator floor (the minimum |denominator|) loses the
  sign of the denominator (`reference/03` §4). Bug (fix) or intended? **Answer (owner):** bug.
  Fix it (keep the sign), list it in the deviation log, and validate it (e.g. a unit test with a
  negative near-zero denominator, and an end-to-end run where the floor never triggers matches
  imsrg++).
* **Q8** — Is HF DIIS needed (imsrg++'s version has off-by-one errors and dead code)? **Answer
  (owner):** DIIS from the start. Implement a correct DIIS in M6, keep plain density mixing as an
  option (it is the imsrg++ default path, needed for comparisons), and test that both converge to
  the same HF solution.
* **Q9** — Oldest toolchains to support? **Answer (owner):** GCC ≥ 11, CUDA ≥ 12, ROCm ≥ 6. Pin
  the latest stable Kokkos release compatible with these (check its requirements at M0).
* **Q10** — Is there a GPU machine for CUDA/HIP tests? **Answer (owner):** not now. Rely on the
  device-code rules (`12-compute-backend.md` §5) and reviews; GPU testing comes later.
* **Q10b** — After the IMSRG(3) layout benchmark, may the session choose the layout itself?
  **Answer (owner):** yes. Decide from the data, document the reasoning and the numbers, and
  continue with the IMSRG(3) terms.

### Config and CLI (from `16-config-and-cli.md`)
* **Q11** — `nsteps=1` with a valence space uses only `core_generator` in imsrg++ (contrary to its
  comment). **Assumed meanwhile:** keep it, with a warning.
* **Q12** — `imsrg3_at_end` without `IMSRG3=true` rebuilds H from the normal-ordered input and
  drops the input 3N. **Assumed meanwhile:** keep it, with a warning.
* **Q13** — The new TOML config requires `emax`, `hw` and the file e1max values explicitly (no
  silent defaults); the legacy `key=value` layer keeps the old defaults and warns. OK?
* **Q14** — `denominator_delta` without `denominator_delta_orbit` is silently ignored in imsrg++.
  **Assumed meanwhile:** keep it, with a warning.

### Python (from `18-python-api.md`)
* **Q15** — Python runtime initialization: lazy on first use, or explicit `imsrg.Runtime()` context
  manager? **Assumed meanwhile:** explicit context manager, plus a lazy default for interactive use.
* **Q16** — Host access to device-resident blocks: read-only zero-copy host views that raise
  `StaleHostError` unless `sync_to_host()` was called? **Assumed meanwhile:** yes.
* **Q17** — Minimum Python version. The stable ABI needs ≥ 3.12. **Assumed meanwhile:** 3.10+,
  without the stable ABI.
* **Q18** — Which legacy input formats are needed? **Answer (owner):** me2j (2N), NO2B and me3j
  (3N), and the Navratil/Miyagi operator formats (`OperatorsFromFile`, two-body currents). Dropped:
  HDF5 3N, Oslo, OakRidge, "Jason", "Takayuki" and the other older readers.

### Process
* **Q19** — Agent compaction: Claude Code exposes a global `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` and a
  per-agent `autoCompactWindow`, not per-model percentages. The proposed settings
  (`design/claude-settings.proposed.json`) approximate "60 % for Sonnet/Haiku, 80 % for Opus".
  Please review them and install them before the run, if you agree.

## Added during implementation

Ids: milestone leads use `M<k>-Q<n>` (e.g. `M2-Q1`) so that parallel branches never collide;
the orchestrator uses `O<n>`. Older entries from the first parallel leads may still read `Q20`,
`Q21`, ...; they are renumbered at merge time if they collide.

* **O1** (2026-11-11, orchestrator) — **How `develop` is advanced in this run.** The desktop app
  started the session in a git worktree (`.claude/worktrees/imsrg-refactor-0e4f11`, branch
  `claude/imsrg-refactor-overnight-800606`), while `develop` stays checked out in the main
  checkout, where this session may not run git (the permission check denied it). Merges therefore
  happen on the worktree branch, which serves as a staging copy of `develop`: `git merge --no-ff
  feature/<x>`, then `tools/ci.sh dev-openmp` on that merge result, and only on PASS is `develop`
  moved to the tested merge commit with `git update-ref refs/heads/develop <new> <old>` (old value
  checked). History is identical to merging on `develop` directly. **Side effect, owner action
  needed:** the main checkout's index and files still show the old `develop` (331b2d4), so `git
  status` there lists the night's work as staged reversals. Do not commit there before syncing.
  If you have no local edits in the main checkout, run
  `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 331b2d4 develop` (safe two-tree
  fast-forward of index and files; it refuses if local edits conflict) or simply
  `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` (HEAD is `develop`). **Assumed
  meanwhile:** this is acceptable, since the owner asked for merges into local `develop`.
* **O2** (2026-11-11, orchestrator) — Q19 follow-up: the Claude Code docs say `autoCompactWindow`
  is an absolute token threshold and that Sonnet 5.5, Haiku 5.5 and Opus 5.x default to 1M-token
  windows, so the planned 150000 meant ~15 %. Changed to 600000 (Sonnet/Haiku agents, ~60 %) and
  800000 (`imsrg-lead`, ~80 %); see `41-agent-workflow.md` §5. The values take effect in the next
  session (definitions load at startup). Combining them with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` is
  undocumented. **Assumed meanwhile:** keep the per-agent windows and recommend dropping the
  percentage override from the proposed settings.
* **Q20** (2026-11-12, M0) — Kokkos 5.3.0 (pinned as instructed: latest 5.x) requires NVCC ≥ 12.2
  and HIPCC ≥ 6.2 (`extern/kokkos/cmake/kokkos_compiler_id.cmake`), slightly above the baseline of
  Q9/D20 (CUDA ≥ 12, ROCm ≥ 6); its GCC ≥ 10.4 meets GCC ≥ 11. Options: raise the baseline to
  CUDA 12.2 and ROCm 6.2; or pin a Kokkos 4.x release for clusters with older toolkits (its
  minimums would need checking). **Assumed meanwhile:** keep Kokkos 5.3.0; CMake asks for
  `find_package(Kokkos 5)`. Moving the pin is a one-line change (`extern/README.md`).
* **Q21** (2026-11-12, M0) — Include style for project headers: `10-architecture.md` §1 said
  `<imsrg/<module>/x.hpp>`, the M0 brief said `"imsrg/<module>/..."`. **Assumed meanwhile:** quotes
  (Google style; `.clang-format` sorts them into the last include group); `10` §1/§2 updated. The
  layering check accepts both forms, so either choice is safe.
* **Q22** (2026-11-12, M0) — The `kokkos-serial` deps build enables `Kokkos_ENABLE_DEBUG` and
  `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`, because dev-serial and asan are the debugging builds. So
  dev-serial differs from dev-openmp in more than the backend, and Serial runs are slower.
  **Assumed meanwhile:** keep it; removing the two options in `tools/bootstrap_deps.sh` rebuilds
  Kokkos automatically (the stamp changes).
* **Q23** (2026-11-12, M0) — The Python venv is shared by all worktrees at `$IMSRG_DEPS_PREFIX/venv`
  (`IMSRG_VENV`), with a gitignored `.venv` symlink per worktree, instead of one `.venv` per
  checkout. **Assumed meanwhile:** shared, as the M0 brief asked; `IMSRG_VENV` can point anywhere.
* **Q24** (2026-11-12, M0) — Owner action, optional: the Command Line Tools on the dev machine make
  `xcrun` pick `MacOSX27.0.sdk`, which `ld-1267` cannot link against (`20` §8). CMake works around
  it automatically; updating or reinstalling the Command Line Tools would make the workaround
  unnecessary. **Assumed meanwhile:** keep the workaround (it switches itself off once the default
  SDK links).
* **M1-Q1** (2026-11-12, M1) — The brief's IT-4/IT-5 command line (`3bme_type=no2b` with the `NO2B_ThBME_..._ms6_12_18.stream.bin` file, emax=4) gives an unphysical old-code result: E_HF = -313.17 MeV (e3hf = -201.9) and a magnus flow that diverges to E0 ~ -1e9 after ~37 time units; IT-8 (emax=2, no 3N) diverges the same way. Is the 3N setup (file/e3max/no2b type) or the shifted-Magnus step control meant to be different for these goldens? **Assumed meanwhile:** the commands are run exactly as in the brief and the diverged runs are committed as-is, documented in tests/golden/e2e/README.md; IT-4/5 comparisons should use the early flow steps and HF numbers only.
