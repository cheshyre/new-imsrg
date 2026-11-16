# Prompt for the next unsupervised implementation session (run 4)

The first run (2026-11-11/12) merged M0-M9 and M11 phase 1 (`design/reports/2026-11-12-overnight.md`).
The second run (2026-11-15/16, `design/reports/2026-11-15-run2.md`) did the Google-style conversion
(MS), the M8 follow-ups, IMSRG(3f2), the three-body layout decision (D43), production three-body
storage with `Comm223`, perturbative triples, **IMSRG(3)-N7**, the charge radius, M11 phase 2 and
M12a. The owner interview of 2026-11-16 (D44-D54) answered every run-2 question. The **third run**
(2026-11-16/17, `design/reports/2026-11-16-run3.md`) did the interview tasks (D47-D49, D50, D51), the
MS2 structural style pass, the M15 reference preparation, **N7 performance** (2.4x / 5.5x / 7.5x
faster at emax 3 / 4 / 5; row-range interface; fused driver) and **M12b phase 1 plus the D7/D8 part
of phase 2** (MPI IMSRG(2) commutator, distributed three-body storage). `develop` = **7a78f4d**.
After run 3 comes the **owner interview** (`design/INTERVIEW_PROMPT.md`), whose answers (D61 and up;
D59 and D60 are the M12b and N7 decision rows of run 3) this prompt must reflect before the run
starts. This prompt starts the
fourth run. (The prompts of earlier runs are in the git history of this file.)

**Owner pre-flight (once, before launching):**

1. Start the session (Opus) as usual: the desktop app opens it in a worktree (D44). Agent
   definitions load at startup from `.claude/agents/`. Settings as in run 3
   (`.claude/settings.local.json` from `design/claude-settings.proposed.json` without `_comment`; no
   `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; `"autoCompactWindow": 800000` for the orchestrator). Pick a
   permission mode that does not stall. Disable connectors the run does not need.
2. **Launch right after a reset of your 5-hour usage window** (run 3 measured the actual resets at
   14:10, 19:10, 00:10, 05:10 JST; check yours). The usage schedule of the run is in the section
   below (to be set in the interview).
3. After the run (not before): re-sync the main checkout, whose files do not follow `develop` when
   runs advance it from worktrees (`git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` when it
   has no local edits of yours), and regenerate its `build/golden-generated`
   (`tools/reference/gen_golden_generated.sh`) if you run the tests there.
4. Optional: `git config blame.ignoreRevsFile .git-blame-ignore-revs`;
   `git -C /Users/cheshyre/Downloads/new-imsrg remote set-url --push origin DISABLED`.

Then paste everything below the line.

---

You are the **orchestrator** of the fourth unsupervised implementation run of the imsrg rewrite
(~8-12 h, nobody answers questions until morning). Continue from the state left by run 3 and the
owner's interview after it, milestone by milestone, without breaking what works.

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


## Usage schedule for this run

(owner to set in the interview)

Run 3's rules (D56, relaxed by D58: at most 50 % of each 5-hour window until 20:00, at most 90 %
until 22:00, then full windows; final reports by 06:00, wrap-up 07:00) are in the git history of
this file and in `design/00-requirements-and-decisions.md`. Whatever schedule is set, state it
unambiguously in every brief, give leads a **pause time well before the cap**, and keep a `TaskStop`
backstop (lessons below).

The owner's binding decisions are in `design/00-requirements-and-decisions.md` §2–5 (**D1–D60**) and
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
  (D50) is done. **License: GPL-3.0-only** (D46): every new `.h`/`.cc` starts with
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
  repo (pristine and D35-patched builds exist; the patch carries D47-D49).
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
  of every merge), the run-3 report (`design/reports/2026-11-16-run3.md`), the run-2 report and the interview reports; the hand-off files `new-imsrg-reference/plans/handoff-n7.md` and `handoff-m12b.md`,
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

* **`develop` = 7a78f4d** at the last code merge (run 3 merges: 7fc0457, f40aa47, e6a59c0, cb07036,
  e739622, e9f156d, 538a82c, 3309163, 93e2d9f, 7bfba4e, fe3181b, 7a78f4d; table in the run-3 report §1). Update this hash from
  `STATUS.md` after the interview merge. Preflight: `git merge --ff-only develop` in the worktree.
* **Complete:** M0-M9, M10 (layout A, 3f2, triples, N7 and its performance work except C15), M11,
  M12a, MS and MS2, the interview tasks D47-D51. **Partial:** M12b: phase 1 (IMSRG(2) distributed,
  IT-11, collective driver with rank-0 output) and D7/D8 (distributed `op::ThreeBody`, generator 3B
  kernel; distributed 3B parts are refused by `Commutator()` until D10). **Prepared:** the M15
  tensor reference goldens (`tests/golden/tensor/`, 91 files). **Not started:** M15 library code,
  M16, M13, M14 (GPU).
* **End to end:** IT-1..IT-10, IT-12, IT-14, IT-15, hunter-gatherer cases and IT-11 (MPI 2/4 ranks,
  max relative deviation 1.2e-16) pass; D48/D49 moved IT-2 HG, IT-10 and IT-8 by design (goldens
  regenerated, run-3 report §3). **CI:** on 93e2d9f 1231 unit, 69 integration, 18 MPI, 150 Python
  (1 skipped), slow suite 7/7 in 1027 s (17 min on a quiet machine); `ci.sh dev-openmp` about 2 min.
  On 7bfba4e: 1235 unit, 69 integration, 18 MPI, 150 Python (1 skipped); slow suite 7/7 in 940 s.
* **Performance:** N7 commutator (O16, full, release) 255 -> 105 ms, 6.3 -> 1.15 s, 46.6 -> 6.2 s at
  emax 3 / 4 / 5 (about 1.7e10 multiply-adds/s); cost at emax 12 / E3max 16: 3.37e13 multiply-adds per
  commutator, 177 GB per packed operator, 673 node-seconds at 5e10/s. MPI: compute imbalance of the
  cost model 0-0.1 % at 22 ranks, 3 % at 64 ranks at emax 12 only, 310-880 % at 256 ranks
  (whole comm122/ph channels); one node, 16 threads: more ranks are slower.
* **Open questions:** run 3's (IT3, O4, MS2, M15-Q1..Q5, M12b-Q1..Q19, M10-Q70..Q79) were taken to the
  interview; their answers are in `QUESTIONS.md`. M15-Q1 (phase convention) gates M15.

## Plan for this run (in order; at most 3 leads at once; the owner's interview may change it)

1. **M12b phase 2 completion** (brief "D-phase2"): row panels for comm122 and the particle-hole
   terms, D9 halo exchange, D10 distributed N7 driver and lifting the D7 guard, D11, D12, weight
   recalibration on an idle machine.
2. **N7 follow-ups** (brief "C-followup"): tiled `BuildH`, C15, partition timings at emax 5,
   `CommutatorOptions::imsrg3_batch_bytes`, factoring the duplicated triples table builders.
   Items 1 and 2 touch different files except `ThreeBodyTerms()` and the batch interface (D9/D10 vs
   `imsrg3_batch_bytes`): merge shared interfaces early.
3. **M15 tensor operators** (brief E) after the owner answered M15-Q1..Q5; the reference goldens
   exist.
4. **M16 formats** (brief F).
5. **Fillers** (brief G and others): M12a follow-ups, Python hooks (D51), the cleanup items of run-2
   report §7, a check that imsrg++ reads our `.me2j.bin` (converter script and refharness), a
   deterministic reduction for the perturbative triples (MS2-Q3).
6. At the end: the definition of done below. M13 and M14 only if everything above is done.

## Lessons from runs 1, 2 and 3 (follow them)

* **The usage budget is the binding constraint**, not wall time. Keep at most 3 leads running and
  delegate to Sonnet/Haiku; well-specified milestones went well with a single Sonnet
  `imsrg-implementer`. Leads delegate code and read little themselves. The orchestrator reads only
  summaries and small files. Measured in run 3: about 60 % of the cost is agents re-reading their
  own long contexts, 20 % cache writes, 15 % output; two leads at once cost 0.5-0.7 % of a window
  per minute, one lead 0.2-0.25 %.
* **Fresh leads at each usage reset, with hand-off notes after every wave** (run 3): a resumed lead
  reloads its whole long context, which costs more than briefing a new lead. Ask each lead for a
  hand-off file (`new-imsrg-reference/plans/handoff-<task>.md`: where things are, what is merged,
  what is left, in order) at every wave and before every pause; start the next lead from it.
* **Leads may ignore pause messages** (run 3: an N7 lead kept working through the 21:35 pause and
  the window reached 97 % against a 90 % cap). **Pause early** (about 5 points of the cap below
  it), give the lead a hard time to end its turn, and use `TaskStop` as the backstop 15 minutes later.
  Check `get_usage` after the pause time, not only before launching.
* **Schedule timing runs while other teams pause, and record the load average** at the start of every
  run (the benchmark scripts wait for a 1-minute load below 6; an 18-thread run itself drives the
  load to 9-20). Timings under load are only usable as ratios; weights (w_122, w_ph) must be
  recalibrated on an idle machine.
* **Merge shared interfaces early** (run 3: the M12b P0 interface and the N7 C1+C2 row-range
  interface were merged within minutes so that both leads built on them); then the conflicts of the
  final merge were few and the orchestrator resolved them during a pause.
* **The orchestrator runs the slow suites on merges while the teams are idle**; a combined merge of two
  leads gets one chain (`ci.sh`, lint, slow), bisect on failure.
* **Keep the slow suite short** (run 3: IT-11 slow was 2406 s until the lead cut it to 385 s / 191 s;
  the whole suite was 54-89 min under load and 17 min after the N7 speed-up). Target under about
  30 minutes on a quiet machine.
* **Timing-sensitive tests must tolerate load**: repeat the timed action, allow retries, and run
  candidate tests 100x under `ctest -j 18` before merging them.
* **Prepare outside the repository during exclusive tasks**: reference builds and goldens, refharness
  flags, read-only Opus reviews, implementation plans (run 3: planner plus a plan review before the
  N7 and M12b leads started; the review found one blocker and four majors in advance).
* **Reviews:** one `imsrg-reviewer` (Opus) pass on every numerics or kernel merge (D42): read the
  diff while a Sonnet worker runs the tests; do not let the reviewer rebuild and run tests itself.
* **Stacked branches via API-ready tags** shortened run 1's critical path; the next lead later merges
  the predecessor's **final head**, not the tag.
* **Question ids:** every brief carries an id prefix and a **block of numbers**; the orchestrator
  uses `O5` and up; new questions go under a new "Added during run 4" heading in `QUESTIONS.md`.
* **Every brief carries the current `develop` hash**, the golden rule 10 text, the emax rule with
  its exceptions, the usage/deadline wording, the license header, and the acceptance command.
* **Merging (D44):** on the worktree branch or a detached HEAD at `develop`: `git merge --no-ff
  --no-commit feature/<x>`, add the `STATUS.md` log entry, commit, then **one chained command**:
  `tools/ci.sh dev-openmp && ctest --preset dev-openmp -L slow && git update-ref refs/heads/develop
  <new> <old>` for numerics merges (`tools/ci.sh dev-openmp && git update-ref ...` otherwise), never
  chained to a grep of the log (run 1 advanced `develop` after a failed CI that way). Run
  `tools/lint.sh` on merges that add names (run 3: the allowlist needed `comm`, `ranges`,
  `num_parts`, `part`).
* **Debug crashes under lldb**, not by guessing (`lldb -b -o run -o bt`); keep Python development
  crashes out of the owner's session (pytest in a subprocess with a timeout).
* **Reference goldens:** after any change to `tools/reference` or the patch, rebuild
  `harness-build-d35` and regenerate `build/golden-generated` (patched harness) in every checkout.
* After an agent's work is merged, stop it (`TaskStop`) if it still waits on background work, and
  check the shared venv for editable installs (`~/.cache/imsrg-deps/venv/bin/pip list | grep imsrg`);
  remove them, because they shadow in-tree builds in other worktrees. A fresh worktree has empty
  `extern/`: workers copy it from the lead's worktree.
* Tests must not share fixed file names in the system temp directory: use `tests/support/scratch_dir.h`.

## Briefs of run 3 (A, B, C, D phase 1): done

Briefs A (interview tasks), B (MS2), C (N7 performance) and D phase 1 (M12b phase 1) are merged
(`design/reports/2026-11-16-run3.md`); their text is in the git history of this file. The briefs below
continue them. All hashes say `develop` = **7a78f4d**: replace by the hash in `STATUS.md` after the
interview.

## Brief D-phase2 — M12b phase 2, rest (starts first)

> You lead **M12b phase 2, rest** on `feature/m12b-mpi-3` (`develop` = 7a78f4d). Ids `M12b-Q20..Q39`.
> **Golden rule 10** applies. emax <= 6 in tests, IMSRG(3) only at emax 2-3; cost-model and timing
> runs as in D52. Read `AGENTS.md`, `00` D14, D45, D59, `12` §6, `14` §1.1 and §4 item 4,
> `design/benchmarks/mpi-scaling.md` (section "Hand-off": the row-panel sketch and the acceptance
> numbers), `new-imsrg-reference/plans/m12b-mpi-plan.md` (D1-D12, "Review notes" M2 round schedule and
> M4 hidden collectives) and `new-imsrg-reference/plans/handoff-m12b.md`. State: phase 1 and D7/D8 are
> merged (ownership map, contiguous cost-balanced ladder cut joint with comm122/ph, `CombinePartials`,
> collective driver with rank-0 output, `ThreeBodyDistribution`, distributed `op::ThreeBody`,
> generator 3B kernel; `Commutator()` refuses distributed 3B parts on every rank). Do, in order:
> (1) **row panels for comm122 and the particle-hole terms** (with a per-(J', panel) inverse mask,
> plan review note m6; M12b-Q3/Q18) so that the cost-model imbalance at 256 ranks falls from
> 310-880 % to the D45 target <= 10 %, then **recalibrate w_122 = 59 and w_ph = 2.9 on an idle
> machine** (check the load before each run; the weights were measured at load 12-50);
> (2) decide the M2 round schedule in a `00` row, then **D9** halo exchange (lockstep `Alltoallv`
> rounds, M12b-Q9) on the C6 batches; (3) **D10** distributed N7 driver on `detail::RunN7` batches
> (the kernels pass the masked-operand test), then **lift the D7 guard**; (4) D11, D12 as in the
> plan; (5) NAT occupations broadcast once (C13 is merged). Coordinate any change of
> `ThreeBodyTerms()` or the batch interface with the N7 follow-up lead (brief C-followup). Acceptance:
> IT-11 on 2 and 4 ranks (rows distribution) with identical step counts and max relative deviation
> <= 1e-12 (run 3 reached 1.2e-16), cost-model imbalance and communication tables for 22/64/256
> ranks at emax 8/10/12 in `mpi-scaling.md`, the slow IT-11 cases under about 10 minutes in total,
> `tools/ci.sh dev-openmp` + lint + dev-serial PASS, `ctest -L slow` PASS, one Opus review (read the
> diff while a Sonnet worker runs the tests). Write `handoff-m12b.md` at every wave and before every
> pause. Report <= 250 words.

## Brief C-followup — N7 follow-ups (in parallel with D-phase2, a different lead)

> You lead the **N7 follow-ups** on `feature/m10-n7-perf-3` (`develop` = 7a78f4d). Ids
> `M10-Q80..Q89`. **Golden rule 10** applies. emax <= 6 in tests, IMSRG(3) at emax 2-3; **timing-only
> kernel runs at emax 4-6** with one operator under ~8 GB (D52; `n7_bench --count-only` first; never
> in CI). Read `AGENTS.md`, `design/benchmarks/imsrg3-n7-performance.md` (§3 timings, §6 partitions, §7
> open items), `new-imsrg-reference/plans/m10-n7-perf-plan.md` (C15, review notes) and
> `new-imsrg-reference/plans/handoff-n7.md`. Do, in order: (1) a **tiled `BuildH`** (load the tile's
> rows into scratch first; H runs at ~4e9 multiply-adds/s against ~1.7e10 for W at emax 4, so it is
> the largest single-node gain left besides comm232); (2) **C15** Comm223 locality (per-tile 6j
> slices, ket cache; M10-Q41/Q42), kept only if quiet-machine timings at emax 4-6 show a gain;
> (3) **a row-cost model that matches the measured partition timings**: the quiet-machine run at
> the end of run 3 (O16 emax 5, `n7_bench --partition`) gave balanced max/mean 1.16 / 1.31 / 1.46
> at P = 4 / 16 / 64 against 1.02-1.04 predicted by the R5 weights, because a part's time grows
> with its row count (`design/benchmarks/imsrg3-n7-performance.md`, "Quiet-machine
> measurements"); fit per-row terms, re-run the partition timings, target max/mean <= 1.10 at
> P = 16, and do this **first**, before the D-phase2 lead relies on the weights; (4) **`CommutatorOptions::imsrg3_batch_bytes`** (M10-Q74), after agreeing with the
> D-phase2 lead who edits `ThreeBodyTerms()`; (5) factor the duplicated group/leg table builders of
> `commutator/triples.cc` and `three_body.cc` (M10-Q75; the `TriplesContext` build at emax 12 takes
> ~150 s and 4.4 GB), reuse the unused rank-3 output of `n7_bench --partition`. Timing runs: schedule
> them while the other lead pauses, and record the 1-minute load at the start of each. Acceptance: all
> term, IT-8, IT-9 and brute-force tests unchanged (bitwise where the arithmetic is unchanged, else
> max|dZ| <= 1e-12 max(1, max|Z|)), before/after tables appended to the N7 report,
> `tools/ci.sh dev-openmp` + lint PASS, `ctest -L slow` PASS (the orchestrator may run it), one short
> Opus review. Write `handoff-n7.md` at every wave. Report <= 250 words.

## Brief E — M15 tensor operators (after the owner answered M15-Q1..Q5)

> You lead **M15 — tensor operators** on `feature/m15-tensor` (`develop` = 7a78f4d). Ids `M15-Q6..Q29`
> (Q1-Q5 exist, with the owner's answers). **Golden rule 10** applies. Read `AGENTS.md`, `00` D25,
> D39, D53, `13` §2 and §6, `15` ("Tensor reference dumps"), `reference/02` (tensor commutators),
> `reference/05` (operator catalog), QUESTIONS M15-Q1..Q5 and the tensor goldens
> `tests/golden/tensor/` with `tools/reference/gen_golden_tensor.sh` (91 files: dump-tensor for E1,
> E2, E3, M1, Fermi, GT, the 0nubb operators and DGT in HO and HF bases; commutator-tensor term by
> term and `all`; bch-tensor with a random Omega; produced by run 3's phase 0, **not yet reviewed:
> review them first**). Scope (owner, D53): E2 and other electric multipoles, M1, Gamow-Teller/Fermi
> (Delta Tz = +-1) and 0nubb (Delta Tz = +-2) operators; only `ch_bra <= ch_ket` blocks stored (D25);
> scalar-tensor IMSRG(2) commutators; transformation with the Magnus Omega only (no ODE flow of
> tensors); the isospin-changing channel blocks; the tensor valence-space outputs that imsrg++ writes
> (tokyo/NuShellX tensor files). Follow the owner's answer to M15-Q1 for the phase convention (the
> goldens use imsrg++'s gauge). Validation: term by term against the goldens, an m-scheme brute force
> for tensors (extend `commutator_ref::mscheme`), end to end for one E2 and one GT case at emax <= 4.
> Stage it: Delta Tz = 0 tensors first, then +-1, then +-2. Acceptance: CI PASS, `ctest -L slow` PASS,
> one Opus review. Report <= 250 words.

## Brief F — M16 ME1J/ME2JP/ME3JP, stored Omega, checkpoint/restart

> Specification: `design/19-mexj-formats.md` (D54; the binary layout is fixed by D57: 255-byte header,
> float32 records; checkpoints and stored Omegas stay text/gzip unless the owner's answer to O4 says
> otherwise). Readers and writers (text, gzip, binary for ME2JP and ME3JP); the evolved H by default;
> storing the Omega stack and a `run.task = "transform"`; checkpoint/restart from these files plus a
> solver-state file. Write a full brief from §7 of the specification when the time comes (`develop` =
> 7a78f4d). **Non-negotiable (owner, D55):** the J-range rules and tests of `19-mexj-formats.md` §8
> go into that brief verbatim: per-record J ranges from one function and stored in the lookup table,
> exactly 0.0 outside them before any offset is computed, `IMSRG_CHECK` on every offset,
> sentinel-value tests against an independent oracle for all J in the union of both sides' ranges and
> all orbit orderings (with the owner's emax-2 example and the last record of a file), the same tests
> under the `asan` preset, and an Opus review that reports on these rules.

## Brief G — fillers (M12a follow-ups, Python hooks, cleanup)

> On `feature/m12a-perf2` (`develop` = 7a78f4d), ids `M12-Q10..Q19`, **golden rule 10** applies.
> Known items: `dsyr2k` for the Hermitian ladder, BCH temporaries kept across steps, launch overhead
> at small emax (emax 4 at 18 threads is only 1.1x imsrg++), the `Denominators` copy of H, cached
> norm buffers (M12-Q2), `get_pn` hash lookups (M12-Q1). Benchmarks: emax 4, 6, 8 against imsrg++
> and **single IMSRG(2) commutators at emax 10-12** (D52; never in CI, no full flows), results in
> `design/benchmarks/imsrg2-performance.md`. Other fillers, one small Sonnet worker each: the Python
> hooks of D51; the cleanup items of run-2 report §7; the `.me2j.bin` cross-check with imsrg++ (the reader
> is merged; `tools/reference/me2j_text_to_bin.py` plus the refharness `2bme=<path>.bin`); a deterministic reduction for the perturbative
> triples (MS2-Q3); `PrintTo` for the parameterized basis tests (MS2-Q4). Acceptance: no golden
> regression, before/after table where timed, CI PASS. Report <= 200 words.

## Definition of done for the run

At the end of the run (or when the budget runs low; write a draft report early as insurance):
* all finished work is merged into `develop`, and `tools/ci.sh dev-openmp` passes on `develop`
  (and `ctest -L slow` for numerics merges);
* unfinished work sits on clearly named branches, with its state described in `STATUS.md` and a
  hand-off file for every lead;
* `design/STATUS.md` has an up-to-date "Current state" and a log entry for every merge;
* `design/QUESTIONS.md` holds every open question under "Added during run 4", with your assumptions;
* a report `design/reports/<date>-run4.md` in the format of the run-2 and run-3 reports (summary,
  owner actions, validation table with actual deviations vs imsrg-devel, performance, old-code bugs
  fixed, deviations from the plan including usage-cap compliance, what is left and recommended next
  steps, a usage timeline);
* `design/NEXT_SESSION_PROMPT.md` is updated for the run after this one, and the agenda of
  `design/INTERVIEW_PROMPT.md` is replaced for the owner interview that follows this run (D42).
