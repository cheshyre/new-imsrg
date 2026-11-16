# Prompt for the next unsupervised implementation session (run 3)

The first run (2026-11-11/12) merged M0–M9 and M11 phase 1 (`design/reports/2026-11-12-overnight.md`).
The second run (2026-11-15/16, `design/reports/2026-11-15-run2.md`) did the Google-style conversion
(MS), the M8 follow-ups, IMSRG(3f2) (IT-10), the three-body layout decision (D43), production
three-body storage with `Comm223` and perturbative triples (IT-8), **IMSRG(3)-N7 (IT-9)**, the charge
radius, M11 phase 2 and its hardening, and M12a. The owner interviews of 2026-11-13 (D25–D29),
2026-11-15 (D30–D42) and **2026-11-16 (D44–D54, `design/reports/2026-11-16-owner-interview.md`)**
answered every open question; the 2026-11-16 interview also switched the license to GPL-3.0-only
(merged, D46) and specified the owner's ME1J/ME2JP/ME3JP formats (`design/19-mexj-formats.md`, D54).
This prompt starts the third run. (The prompts of runs 1 and 2 are in the git history of this file.)

**Owner pre-flight (once, before launching):**

1. Start the session (Opus) as usual: the desktop app opens it in a worktree, which is now the
   standard (D44). Agent definitions load at startup from `.claude/agents/`. Settings as in run 2
   (`.claude/settings.local.json` from `design/claude-settings.proposed.json` without `_comment`; no
   `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; `"autoCompactWindow": 800000` for the orchestrator). Pick a
   permission mode that does not stall.
2. **Launch right after a reset of your 5-hour usage window** (resets at 13:00, 18:00, 23:00, 04:00
   JST). The run follows the usage schedule below (D56): conservative (at most 50 % of each window)
   until 20:00 JST, then normal until the wrap-up at 07:00 JST.
3. After the run (not before): re-sync the main checkout, whose files do not follow `develop` when
   runs advance it from worktrees (`git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` when it
   has no local edits of yours), and regenerate its `build/golden-generated`
   (`tools/reference/gen_golden_generated.sh`) if you run the tests there.
4. Optional: `git config blame.ignoreRevsFile .git-blame-ignore-revs`;
   `git -C /Users/cheshyre/Downloads/new-imsrg remote set-url --push origin DISABLED`.

Then paste everything below the line.

---

You are the **orchestrator** of the third unsupervised implementation run of the imsrg rewrite
(~8–12 h, nobody answers questions until morning). Continue from the state left by run 2 and the
owner's interview of 2026-11-16, milestone by milestone, without breaking what works.

## The owner's instructions (verbatim, from the planning session)

> You will develop a completely refactored version of the imsrg code contained in imsrg-devel.
> This code is provided as a reference for feature completeness and for implementation checks.
> Reconsider all design decisions and develop a modern version that supersedes the imsrg++ code.
>
> Development style: Use feature branches for even small development tasks. Modular commits
> explaining what was done and (in nontrivial cases) the design decision behind it. Manage your
> context. Use opus agents to orchestrate bigger development efforts, which then orchestrate haiku
> or sonnet agents (depending on expected complexity) to achieve small tasks. Compact at 60% for
> haiku and sonnet and 80% for opus.
>
> Collect questions that come up along the way for revision later.

And from the interview of 2026-11-15 (verbatim): "Follow the Google style guide:
https://google.github.io/styleguide/cppguide.html Use clang-format to implement many of the basic
formatting things automatically. Use cpp-lint to lint the code as well (but do not blindly follow the
linter)."

## Usage schedule for this run (owner, 2026-11-16, D56)

Verbatim: "Modify the next session prompt to run conservatively (not consuming more than 50% of usage)
until 8pm and then running as normal until 7am."

* **Until 20:00 JST (conservative phase):** the run uses at most **50 % of each 5-hour usage window**
  (`get_usage`; windows reset at 13:00, 18:00, 23:00, 04:00 JST), leaving the rest to the owner.
  Check `get_usage` before launching any agent and about every 30 minutes. Run at most one lead with
  one or two Sonnet/Haiku workers, or two Sonnet implementers without a lead; start an Opus lead in
  this phase only if the current window is below 25 % and its task is small. At about 45 % start
  nothing new; at 50 % have the running agents stop at their next safe point (a WIP commit), stop
  them, and resume them with `SendMessage` at 20:00 or after the 18:00 reset, whichever comes first
  (the 50 % cap applies again in the new window until 20:00). Work for this phase: the preflight,
  brief A (interview tasks) with Sonnet implementers, and the out-of-repository preparation (N7 and
  M12b plans, the patched-reference changes of brief A).
* **From 20:00 JST (normal phase):** normal pacing as in runs 1 and 2: up to 3 leads, each window
  used fully, stopped agents resumed with `SendMessage` after the 23:00 and 04:00 resets. Start MS2
  (alone) at 20:00 once brief A is merged; then N7 performance in parallel with M12b phase 1. A run
  that starts after 20:00 is in the normal phase from the start.
* **Wrap-up at 07:00 JST:** all merges, CI, `STATUS.md`, `QUESTIONS.md`, the report and the prompts
  are done by 07:00. Leads deliver their final reports by **06:00 JST** and keep working until then;
  write a draft report early as insurance.
* Every brief states the phase it starts in (with the 50 % rule if before 20:00), the 06:00 report
  time and the 07:00 wrap-up.

The owner's binding decisions are in `design/00-requirements-and-decisions.md` §2–5 (**D1–D54**) and
in the answers in `design/QUESTIONS.md`. The most important:

* Hardware targets: NVIDIA GPUs (H100/H200/GH200, A100), AMD GPUs (MI250X/MI300A), CPU clusters →
  Kokkos (D2); baseline GCC ≥ 11, CUDA ≥ 12.2, ROCm ≥ 6.2 (D30). GPU work starts only after M12b
  (D40); then CUDA and HIP compile checks from the start, tuning on NVIDIA first (D53).
* **IMSRG(3)-N7 production target (D45):** emax 10–12 with **E3max 16 and no dE3max** (about 177 GB
  per packed layout-A operator, about 1.06 TB per N7 step), nodes with 256 GB or less (sometimes
  100 GB), 3B operators at most 50 % of a node, many nodes: **single channels are split across
  ranks**; a row-wise, cost-balanced (Hagen–Nam-type) distribution is the baseline; the N7 kernels
  are tiled over row ranges. Acceptance by cost model: ≤ 10 % compute and memory imbalance over
  22–256 ranks, communication per N7 step below compute at ~25 GB/s per node (InfiniBand or
  Slingshot clusters with Slurm).
* **Code style: the Google C++ Style Guide** (D38, `design/17-code-style.md`); the structural pass
  (D50) comes first in this run. **License: GPL-3.0-only** (D46): every new `.h`/`.cc` starts with
  `// SPDX-License-Identifier: GPL-3.0-only` and `// Copyright 2026 The imsrg authors`
  (`tools/ci.sh` checks it).
* **emax ≤ 6 in all tests and runs; IMSRG(3) only at emax 2 and 3.** Benchmark-only exceptions,
  never in CI and without full flows: emax-8 IMSRG(2) benchmarks against imsrg++ (D40), single
  IMSRG(3) kernels at emax 4–6 with one operator under ~8 GB (D52), single IMSRG(2) commutators at
  emax 10–12 (D52).
* Git is **local only**: feature branches → local `develop`. Never push, no PRs. Never touch `main`.
  Runs start in an app worktree and advance `develop` with the staging procedure (D44, below).
* **Golden rule 10:** file searches stay inside the project directories (never `find /`, `find ~`,
  `mdfind`, `grep -r` outside them): a whole-disk search by a helper made macOS ask the owner for
  access to other apps' data. **Put this rule into every brief you write.**
* Homebrew and pip (shared venv) installs are allowed. The old imsrg-devel is built **outside** the
  repo (pristine and D35-patched builds exist; the patch grows with D47/D48 in this run).
* Old-code bugs that change results: fix them, document them in `design/30-testing-and-validation.md`
  §6, and validate the fixed path another way. No compatibility switches (D35).
* Tolerances: unit checks ~1e-10 relative; end to end 1e-4 MeV (HF, MBPT, perturbative triples) and
  1e-3 MeV (IMSRG(2)/(3) E0, valence-space elements). Always log the actual deviation (D19, D34).
* `tools/ci.sh dev-openmp` stays under 5 minutes; longer tests get the `slow` label and run before
  merges that touch numerics (D42). Goldens are regenerated only through the scripted
  `tools/reference` commands, with the reason and old vs new numbers in the commit and a reviewer
  pass (D42).

## Where everything is

* Repository: `/Users/cheshyre/Downloads/new-imsrg` (git), integration branch **`develop`**. The
  session runs in an app worktree (D44). **Preflight:** in the worktree run
  `git merge --ff-only develop` (the app creates the worktree branch from `main` = 71d5917, an
  ancestor of `develop`); check `git merge-base --is-ancestor feature/design-plan develop`,
  `git status --porcelain --untracked-files=no` (empty), then
  `tools/reference/gen_golden_generated.sh` (the emax-4 dump tests **skip silently** without
  `build/golden-generated`) and `tools/ci.sh dev-openmp` (`CI PASS dev-openmp`). If any check fails,
  write `PREFLIGHT_FAILED.md` at the worktree root and stop. If the `imsrg-*` agent types are
  unavailable, spawn `general-purpose` agents with the matching `model` and paste the body of the
  corresponding `.claude/agents/*.md` into their prompt.
* Read first: `AGENTS.md`, `CLAUDE.md`, `design/README.md`, `design/STATUS.md` (current state, log
  of every merge), the run-2 report and the 2026-11-16 interview report,
  `design/00-requirements-and-decisions.md`, `design/40-roadmap.md`, `design/41-agent-workflow.md`.
  Leads read the other design docs and `design/reference/*.md` as their milestone requires; the
  orchestrator delegates big reading.
* Toolchain (installed): Homebrew cmake, ninja, ccache, Open MPI, libomp, clang-format 23.1.5, LLVM
  (clang-tidy 23.1.5 at `/opt/homebrew/opt/llvm/bin/clang-tidy`, `tools/lint.sh`), gsl, boost,
  openblas; Apple clang 21; Kokkos 5.3.0 per backend in `~/.cache/imsrg-deps`; shared venv
  `~/.cache/imsrg-deps/venv` (`.venv` symlink per checkout) with cpplint 2.0.2. A fresh worktree
  needs only `tools/ci.sh dev-openmp` (~40 s submodule fetch on first use; 192 s in all).
* Reference: source `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only); builds, harnesses
  and plans in `/Users/cheshyre/Downloads/new-imsrg-reference/`: the **D35-patched** build
  `build-d35/imsrg++` and `harness-build-d35/refharness` (default of every
  `tools/reference/gen_golden_*.sh`), the pristine `build/`, `harness-build/`, and `plans/`. Golden
  data: `tests/golden/` (README lists every file and command). After any change to the patch or to
  `tools/reference`, rebuild `harness-build-d35` and regenerate `build/golden-generated` in every
  checkout that runs the emax-4 tests.
* Test inputs (never copy them into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  * `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz`: `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`: `3bme_type=no2b file3e1max=6
    file3e2max=12 file3e3max=18 hw=16`. It holds **only the cE contact term** and stays the 3N test
    input (D33; the full N2LO file is still outstanding): E_HF(O16, emax 4) = −313.17 MeV is genuine
    and the full flow diverges, so IT-4/IT-5 compare HF plus the flow stopped at s = 10. O16 at emax 2
    is a known, genuine IMSRG(2) instability (IT-8 uses He4 emax 2 and O16 emax 3).

## State at hand-off (details: `design/STATUS.md`)

* **Merged into `develop`:** everything of runs 1 and 2 (M0–M9, M11 phases 1 and 2 with hardening,
  MS, hunter-gatherer, D35, ODE flow methods, general commutator paths, M10 3f2/layout A/`Comm223`/
  triples/N7 with the comm232 follow-up, charge radius, M12a), the license switch to GPL-3.0-only
  (6bcca88) and the interview documents (the merge after it; use the hash in STATUS).
* **End to end:** `imsrg` reproduces imsrg-devel (D35-patched) for IT-1..IT-10, IT-12, IT-14, IT-15
  and the hunter-gatherer cases (identical step and gather counts, |ΔE0| ≤ 3.8e-10 MeV); numbers in
  the run-2 report §3. CI: 1069 unit, 61 integration, 3 MPI, 149 Python tests (1 skipped);
  `ci.sh dev-openmp` 105 s warm; the `slow` suite (6 tests, IT-9 emax 3 ~39 min) runs before numerics
  merges.
* **Performance:** the full IMSRG(2) commutator is 2.4–4.5x faster than imsrg++ at emax 6–8 (only
  1.1x at emax 4 on 18 threads); O16 emax-8 flow 59.8 s vs 187.8 s (6 threads); IT-9 2–3x faster
  than imsrg++. N7 is correct but not yet scalable (`14` §1.1).
* **Decided but not implemented (this run):** D47 (stage-2 range, double tolerances), D48
  (hunter-gatherer H(s)), D49 (Delta3 floor, triples order and emax ≥ 8, IT-9 Rp2 golden,
  `occ_nat3_cut`), D50 (structural pass, clang-tidy pattern), D51 (strict residency in CI, `18`
  rewrite, Python hooks).
* **Not started:** M12b (MPI), M15 tensor operators, M16 ME1J/ME2JP/ME3JP with stored Omega and
  checkpoint/restart, M13, M14 (GPU), IT-11.

## Plan for this run (owner, D53; in order; at most 3 leads at once)

1. Preflight (above).
2. **Interview tasks** (brief A; small, Sonnet implementers, one Opus review of the numerics, D42
   for the goldens). Runs in the conservative phase (≤ 50 % of each window until 20:00 JST).
3. **MS2 structural style pass** (brief B), **alone** and time-boxed (about 3 h; it touches most
   files); starts at 20:00 JST, when the normal phase begins, once brief A is merged. Meanwhile prepare outside the repository: the N7-performance and M12b implementation plans
   (`new-imsrg-reference/plans/`), read-only reviews, the patched-reference changes.
4. **N7 performance** (brief C) **in parallel with M12b phase 1** (brief D, IMSRG(2) by channel
   blocks): distribution-ready row tiling, triples at emax ≥ 8, `occ_nat3_cut`.
5. **M12b phase 2** (brief D): row-wise 3B distribution (D45), IT-11.
6. **M15 tensor operators** (brief E).
7. Only if all of the above is done: M16 (`19-mexj-formats.md`; brief F), M13, M14. Filler for spare
   capacity: M12a follow-ups (brief G), the Python hooks (D51), the cleanup items of the run-2
   report §7.
8. At the end: the definition of done below.

## Lessons from runs 1 and 2 (follow them)

* **The usage budget is the binding constraint**, not wall time. Keep at most 3 leads running and
  delegate to Sonnet/Haiku; well-specified milestones went well with a single Sonnet
  `imsrg-implementer`. Leads delegate code and read little themselves. The orchestrator reads only
  summaries and small files.
* **Pace with `get_usage`:** check the usage windows before starting a big task; a stopped Opus lead
  that resumes reloads its whole context, so start long leads right after a reset and avoid
  launching new leads in the last minutes of a window. When agents stop on a limit, resume each with
  `SendMessage` after the reset; do not start new agents for the same work. This run follows the
  usage schedule above (D56): conservative until 20:00 JST, then the whole time used until the 07:00
  wrap-up; state it unambiguously in every brief ("keep working; your final report is due by 06:00
  JST").
* **Timing-sensitive tests must tolerate load**: repeat the timed action, allow retries, and run
  candidate tests 100x under `ctest -j 18` before merging them.
* **Prepare outside the repository during exclusive tasks** (MS2 this time): reference builds and
  goldens, refharness flags, read-only Opus reviews, implementation plans; commit afterwards.
* **Stacked branches via API-ready tags** shortened run 1's critical path; the next lead later merges
  the predecessor's **final head**, not the tag.
* **Question ids:** every brief carries an id prefix and a **block of numbers** (`M10-Q70..Q79`,
  `M12b-Q1..Q9`, `MS2-Q1..Q9`, `M15-Q1..Q9`, ...); the orchestrator uses `O4` and up; new questions
  go under "Added during run 3" in `QUESTIONS.md`.
* **Every brief carries the current `develop` hash**, the golden rule 10 text, the emax rule with
  its exceptions, the usage/deadline wording, the license header, and the acceptance command.
* **Merging (D44):** on the worktree branch or a detached HEAD at `develop`: `git merge --no-ff
  --no-commit feature/<x>`, add the `STATUS.md` log entry, commit, then **one chained command**:
  `tools/ci.sh dev-openmp && ctest --preset dev-openmp -L slow && git update-ref refs/heads/develop
  <new> <old>` for numerics merges (`tools/ci.sh dev-openmp && git update-ref ...` otherwise), never
  chained to a grep of the log (run 1 advanced `develop` after a failed CI that way).
* **Reviews:** one `imsrg-reviewer` (Opus) pass on every numerics or kernel merge (D42); they found
  real issues in run 2 (host-only scratch counts breaking CUDA, scratch accounting, exception
  safety).
* **Debug crashes under lldb**, not by guessing (`lldb -b -o run -o bt`); keep Python development
  crashes out of the owner's session (pytest in a subprocess with a timeout).
* **Reference goldens:** after any change to `tools/reference` or the patch, rebuild
  `harness-build-d35` and regenerate `build/golden-generated` (patched harness) in every checkout.
* After an agent's work is merged, stop it (`TaskStop`) if it still waits on background work, and
  check the shared venv for editable installs (`~/.cache/imsrg-deps/venv/bin/pip list | grep imsrg`);
  remove them, because they shadow in-tree builds in other worktrees.
* Tests must not share fixed file names in the system temp directory: use `tests/support/scratch_dir.h`.

## Brief A — interview tasks (ready to use; adjust the hashes)

> You implement the **2026-11-16 interview tasks** on `feature/interview-tasks-r3` (`develop` = <hash>).
> Question ids `IT3-Q1..Q9`. **Golden rule 10** applies (searches only inside the repository, its
> worktrees, `/Users/cheshyre/Downloads/new-imsrg-reference/`, `~/.cache/imsrg-deps/`, temp dirs).
> emax ≤ 6, IMSRG(3) only at emax 2–3. Read `AGENTS.md`, `00` D47–D49 and D51, the QUESTIONS answers
> of M8-Q10, M8-Q40, M8-Q41, M10-Q40, M10-Q60, M10-Q61, M11-Q11, `30` §3 and §6, `tests/golden/README.md`.
> One commit per item; items (1)–(2) touch the same flow code and the D35 patch, so do them in order:
> (1) **D48:** after the final gather `MagnusSolver` sets H(s) to the rebuilt `h_saved_`; add the same
> change to the D35 patch of imsrg-devel (`tools/reference/`), rebuild `harness-build-d35`, regenerate
> IT-10 and the hunter-gatherer goldens through the scripts (expected ΔE0: −13.4 keV IT-10 O16 emax 2,
> +2.8 keV He4 emax 2, −43 / −22 keV IT-2 HG O16 emax 4, −8.1 keV He4 emax 2; steps and gathers
> unchanged); `30` §6 row; replace the unit test `HunterGathererDocumentsImsrgSemantics` by one that
> checks H(s) == Transform(H0). (2) **D47:** stage 2 integrates [s1, s1 + smax] for every method;
> `eta_criterion` and `ode_tolerance` in double; patch the reference the same way where a golden's
> step count changes (check IT-3, IT-7, IT-14, IT-15), regenerate, `30` §6 rows. (3) **D49:** the
> sign-keeping floor for the three-body denominator (unit test with a vanishing Δ3); the triples'
> split-stack order `BchProduct(Omega_n, omega)` with a split-stack consistency test; an IT-9 Rp2
> operator golden (He4 and O16, emax 2) from the patched reference. (4) **D51:** explicit
> `sync_to_host()` in the 23 Python tests and the examples, `tools/ci.sh` runs pytest with
> `IMSRG_RESIDENCY=strict`. (5) **D50:** clang-tidy's variable ignore pattern becomes a curated
> physics-symbol list (`tools/lint.sh` clean; no renames). Acceptance: `tools/ci.sh dev-openmp` PASS,
> `ctest -L slow` PASS, an Opus review of (1)–(3), golden commits with reason and old/new numbers.
> This task starts in the **conservative phase**: at most two Sonnet workers at a time, and the
> orchestrator may stop and later resume you to keep usage at or below 50 % of each 5-hour window
> until 20:00 JST. Report ≤ 250 words when done, at the latest by 06:00 JST.

## Brief B — MS2 structural style pass (ready to use; runs alone)

> You lead **MS2** on `feature/style-structural` (`develop` = <hash>), **alone in the repository**,
> time-boxed to about 3 h. Ids `MS2-Q1..Q9`. **Golden rule 10** applies. Read `AGENTS.md`,
> `17-code-style.md` (§2.D, §3, §6 and the 2026-11-16 note), `00` D38, D46, D50. Items, one commit
> each, `tools/ci.sh dev-openmp` after each, integration outputs bitwise identical to the baseline
> (as MS did): structs with member functions that hold invariants become classes (private data,
> accessors in the allowlist; passive structs stay structs); unsigned types become signed except
> sizes and bit patterns; functions over about 40 lines are split where it helps reading; long
> inline functions in headers move to `.cc` files (device-code lookups and templates excepted, with
> a measured reason when performance argues for inline); imperative function comments become
> descriptive. Delegate mechanical parts to Sonnet/Haiku workers by module. Acceptance: CI PASS, the
> clang-tidy naming check clean, identical outputs, a summary table in `17` §6. Report ≤ 250 words.

## Brief C — N7 performance (ready to use; adjust the hashes)

> You lead **M10 N7 performance** on `feature/m10-n7-perf` (`develop` = <hash>). Ids `M10-Q70..Q79`.
> **Golden rule 10** applies. emax ≤ 6, IMSRG(3) only at emax 2–3, except **timing-only kernel
> runs at emax 4–6** with one operator under ~8 GB (`--count-only` first; never in CI; D52). Read
> `AGENTS.md`, `14` §1.1 (status note of 2026-11-16) and §4, `00` D34, D43, D45, D49, D52, `12` §3–5,
> `benchmarks/imsrg3-layout.md`, QUESTIONS M10-Q41..Q46 and M10-Q60..Q67, `30` §6 M10-N7-1..3. Goal:
> scalability toward the D45 target without changing results: tile every N7 kernel
> (133/132/231/331/232/223) over **row ranges** of channel blocks so that a row-distributed block
> (M12b phase 2) needs no kernel change; `exec::TeamFor`/`TeamReduce` options instead of direct
> Kokkos (M10-Q44/Q65); a lazy `ThreeBodyContext` for IMSRG(2)-only use (M10-Q66); per-tile 6j
> slices and a ket cache for `Comm223` judged by the emax 4–6 timings (M10-Q41/Q42); **perturbative
> triples at emax ≥ 8** (R per tile, no E map, only the ppp/hhh blocks; M10-Q46); **`occ_nat3_cut`**
> with the NAT occupations passed into the three-body space, validated against the brute force with
> the cut (M10-Q62). Coordinate the row-range interface with the M12b lead. Acceptance: all term,
> IT-8, IT-9 and brute-force tests unchanged (bitwise where the arithmetic is unchanged), a
> before/after table in `design/benchmarks/`, cost-model memory and time at emax 12 / E3max 16,
> `tools/ci.sh dev-openmp` PASS, `ctest -L slow` PASS, one Opus review. Report ≤ 250 words.

## Brief D — M12b MPI (phase 1 in parallel with brief C, then phase 2)

> You lead **M12b — MPI distribution** on `feature/m12b-mpi` (`develop` = <hash>). Ids `M12b-Q1..Q19`.
> **Golden rule 10** applies. Read `AGENTS.md`, `14` §1.1 and §4 item 4, `00` D14, D34, D40, D43,
> **D45**, `12`, the `comm` module headers (Communicator, serial stub), the `Workspace` and ownership
> notes of M7. **Phase 1** (D14): 1B/2B operator parts replicated; the expensive IMSRG(2) commutator
> terms split by channel blocks across ranks through an `OwnershipMap`; results combined with
> `Allreduce`; builds without MPI keep working. **Phase 2** (D45): 3B blocks of layout A (packed)
> distributed **row-wise with cost balancing (Hagen and Nam, Prog. Theor. Phys. Suppl. 196, 102
> (2012), arXiv:1203.3765) as the baseline**, splitting single channels across ranks; whole-channel
> ownership is modeled only as the comparison. Target: N7 at emax 10–12, E3max 16 (no dE3max), 22–256
> nodes of 100–256 GB with 3B operators ≤ 50 % of a node; acceptance by cost model: compute and memory
> imbalance ≤ 10 % across 22–256 ranks and communication per N7 step below compute at ~25 GB/s per
> node (InfiniBand or Slingshot, Slurm), plus measurements on 1, 2 and 4 ranks here. Tests: `mpirun -n
> 2/4` unit and integration tests (`mpi` label); **IT-11**: MPI results equal the serial ones within
> 1e-12 relative for IMSRG(2) and N7 at emax 2/3. Record the numbers in `design/benchmarks/`, decide
> the details in a `00` row. Acceptance: IT-11, scaling table, `tools/ci.sh dev-openmp` PASS, the
> existing MPI tests PASS, one Opus review. Report ≤ 250 words.

## Brief E — M15 tensor operators (when C and D are merged)

> You lead **M15 — tensor operators** on `feature/m15-tensor` (`develop` = <hash>). Ids `M15-Q1..Q19`.
> **Golden rule 10** applies. Read `AGENTS.md`, `00` D25, D39, D53, `13` §2 and §6, `15`,
> `reference/02` (tensor commutators) and `reference/05` (operator catalog). Scope (owner, D53): E2
> and other electric multipoles, M1, Gamow–Teller/Fermi (ΔTz = ±1) and 0νββ (ΔTz = ±2) operators;
> only `ch_bra ≤ ch_ket` blocks stored (D25); scalar–tensor IMSRG(2) commutators; transformation with
> the Magnus Omega only (no ODE flow of tensors); the isospin-changing channel blocks for GT/Fermi and
> 0νββ; the tensor valence-space outputs that imsrg++ writes (tokyo/NuShellX tensor files).
> Validation: term by term against the patched imsrg++ build (new `refharness` subcommands) and an
> m-scheme brute force for tensors (extend `commutator_ref::mscheme`), end to end for one E2 and one
> GT case at emax ≤ 4. Stage it: ΔTz = 0 tensors first, then ΔTz = ±1, then ±2. Acceptance: CI PASS,
> `ctest -L slow` PASS, one Opus review. Report ≤ 250 words.

## Brief F — M16 ME1J/ME2JP/ME3JP, stored Omega, checkpoint/restart (only if time remains)

> Specification: `design/19-mexj-formats.md` (D54). Readers and writers (text, gzip for ME2JP and
> ME3JP; the binary `.bin` variant waits for the owner's header specification, so keep the record
> enumeration separate from the value I/O); the evolved H by default; storing the Omega stack and a
> `run.task = "transform"`; checkpoint/restart from these files plus a solver-state file. Write a
> full brief from §7 of the specification when the time comes. **Non-negotiable (owner, D55):** the
> J-range rules and tests of `19-mexj-formats.md` §8 go into that brief verbatim: per-record J ranges
> from one function and stored in the lookup table, exactly 0.0 outside them before any offset is
> computed, `IMSRG_CHECK` on every offset, sentinel-value tests against an independent oracle for
> all J in the union of both sides' ranges and all orbit orderings (with the owner's emax-2 example
> and the last record of a file), the same tests under the `asan` preset, and an Opus review that
> reports on these rules.

## Brief G — M12a follow-ups (filler)

> On `feature/m12a-perf2` (`develop` = <hash>), ids `M12-Q10..Q19`, **golden rule 10** applies.
> Known items: `dsyr2k` for the Hermitian ladder, BCH temporaries kept across steps, launch overhead
> at small emax (emax 4 at 18 threads is only 1.1x imsrg++), the `Denominators` copy of H, cached
> norm buffers (M12-Q2), `get_pn` hash lookups (M12-Q1). Benchmarks: emax 4, 6, 8 against imsrg++
> and **single IMSRG(2) commutators at emax 10–12** (D52; never in CI, no full flows), results in
> `design/benchmarks/imsrg2-performance.md`. Acceptance: no golden regression, before/after table,
> CI PASS. Report ≤ 200 words.

## Definition of done for the run

At the end of the run (or when the budget runs low; write a draft report early as insurance):
* all finished work is merged into `develop`, and `tools/ci.sh dev-openmp` passes on `develop`
  (and `ctest -L slow` for numerics merges);
* unfinished work sits on clearly named branches, with its state described in `STATUS.md`;
* `design/STATUS.md` has an up-to-date "Current state" and a log entry for every merge;
* `design/QUESTIONS.md` holds every open question under "Added during run 3", with your assumptions;
* a report `design/reports/<date>-run3.md` in the format of the run-1 and run-2 reports (summary,
  owner actions, validation table with actual deviations vs imsrg-devel, performance, old-code bugs
  fixed, deviations from the plan, what is left and recommended next steps);
* `design/NEXT_SESSION_PROMPT.md` is updated for the run after this one, and the agenda of
  `design/INTERVIEW_PROMPT.md` is replaced for the owner interview that follows this run (D42).
