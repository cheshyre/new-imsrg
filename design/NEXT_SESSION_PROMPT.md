# Prompt for the next unsupervised implementation session (run 3)

The first run (2026-11-11/12) merged M0–M9 and M11 phase 1 (`design/reports/2026-11-12-overnight.md`).
The second run (2026-11-15/16, `design/reports/2026-11-15-run2.md`) did the Google-style conversion
(MS), the M8 follow-ups, IMSRG(3f2) (IT-10), the three-body layout decision (D43), production
three-body storage with `Comm223` and perturbative triples (IT-8), **IMSRG(3)-N7 (IT-9)**, the charge
radius, M11 phase 2 and its hardening, and started M12a. The owner interviews of 2026-11-13 (D25–D29), 2026-11-15 (D30–D42) and
the one after run 2 (D44 and later; `design/reports/<date>-owner-interview.md`, written by that
session) answered the open questions. This prompt starts the third run. (The prompts of runs 1 and 2
are in the git history of this file.)

**Owner pre-flight (once, before launching):**

1. **Sync the main checkout** (QUESTIONS O1, O3): runs 1 and 2 and the interviews advanced `develop`
   from worktrees, so the main checkout's files may still show an old `develop`. Check `git status`
   there: it should list only staged reversals of the newer commits. With no local edits of your own:
   `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` (HEAD is `develop`), or
   `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u <old develop hash> develop`.
2. Regenerate the gitignored emax-4 dumps in that checkout: `tools/reference/gen_golden_generated.sh`
   (patched harness by default; stale dumps make `Emax4TrelAndRp2` fail, M8-Q21).
3. Start the session (Opus) **in the main checkout on `develop`** (D31); in run 2 the desktop app
   opened a worktree instead (O3), which works with the staging procedure below but leaves the main
   checkout stale. Agent definitions load at startup from `.claude/agents/`. Settings are as in run 2
   (`.claude/settings.local.json` from `design/claude-settings.proposed.json` without `_comment`; no
   `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; `"autoCompactWindow": 800000` for the orchestrator). Pick a
   permission mode that does not stall.
4. **Launch right after a reset of your 5-hour usage window** (resets at 13:00, 18:00, 23:00, 04:00
   JST; runs 1 and 2 each hit the limit two times).
5. Optional: `git config blame.ignoreRevsFile .git-blame-ignore-revs`;
   `git -C /Users/cheshyre/Downloads/new-imsrg remote set-url --push origin DISABLED`.

Then paste everything below the line.

---

You are the **orchestrator** of the third unsupervised implementation run of the imsrg rewrite
(~8–12 h, nobody answers questions until morning). Continue from the state left by run 2 and the
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

The owner's binding decisions are in `design/00-requirements-and-decisions.md` §2–5 (D1–D43 and the
rows of the interview after run 2) and in the answers in `design/QUESTIONS.md`. The most important:

* Hardware targets: NVIDIA GPUs (H100/H200/GH200, A100), AMD GPUs (MI250X/MI300A), CPU clusters →
  Kokkos (D2); baseline GCC ≥ 11, CUDA ≥ 12.2, ROCm ≥ 6.2 (D30). GPU work starts only after M12 (D40).
* Order: core → 3N input + natural orbitals → **IMSRG(3)** (top-priority extension: N7, 3f2,
  perturbative triples; layout A decided, D43). After IMSRG(3): tensor operators first (D39).
  Atomic systems and RPA/TDA are dropped.
* **Code style: the Google C++ Style Guide** (D38, `design/17-code-style.md`), converted in run 2.
* **emax ≤ 6 in all tests and runs; IMSRG(3) only at emax 2 and 3.** Exception: occasional emax-8
  IMSRG(2) performance benchmarks against imsrg++, never in CI (D40).
* Git is **local only**: feature branches → local `develop`. Never push, no PRs. Never touch `main`.
* **Golden rule 10:** file searches stay inside the project directories (never `find /`, `find ~`,
  `mdfind`, `grep -r` outside them): a whole-disk search by a helper made macOS ask the owner for
  access to other apps' data. **Put this rule into every brief you write.**
* Homebrew and pip (shared venv) installs are allowed. The old imsrg-devel is built **outside** the
  repo (pristine and D35-patched builds exist).
* Old-code bugs that change results: fix them, document them in `design/30-testing-and-validation.md`
  §6, and validate the fixed path another way. No compatibility switches (D35).
* Tolerances: unit checks ~1e-10 relative; end to end 1e-4 MeV (HF, MBPT, perturbative triples) and
  1e-3 MeV (IMSRG(2)/(3) E0, valence-space elements). Always log the actual deviation (D19, D34).
* `tools/ci.sh dev-openmp` stays under 5 minutes; longer tests get the `slow` label and run before
  merges that touch numerics (D42). Goldens are regenerated only through the scripted
  `tools/reference` commands, with the reason and old vs new numbers in the commit and a reviewer
  pass (D42).

## Where everything is

* Repository: `/Users/cheshyre/Downloads/new-imsrg` (git), integration branch **`develop`**.
  Preflight: `git branch --show-current` prints `develop` (in a worktree: the staging branch is at
  `develop`'s tip, QUESTIONS O3); `git merge-base --is-ancestor feature/design-plan develop`
  succeeds; `git status --porcelain --untracked-files=no` prints nothing; `tools/ci.sh dev-openmp`
  prints `CI PASS dev-openmp`. If any check fails, write `PREFLIGHT_FAILED.md` at the repo root and
  stop. If the `imsrg-*` agent types are unavailable, spawn `general-purpose` agents with the
  matching `model` and paste the body of the corresponding `.claude/agents/*.md` into their prompt.
* Read first: `AGENTS.md`, `CLAUDE.md`, `design/README.md`, `design/STATUS.md` (current state, log
  of every merge), the report of run 2 and the interview report after it,
  `design/00-requirements-and-decisions.md`, `design/40-roadmap.md`, `design/41-agent-workflow.md`.
  Leads read the other design docs and `design/reference/*.md` as their milestone requires; the
  orchestrator delegates big reading.
* Toolchain (installed): Homebrew cmake, ninja, ccache, Open MPI, libomp, clang-format 23.1.5, LLVM
  (clang-tidy 23.1.5 at `/opt/homebrew/opt/llvm/bin/clang-tidy`, `tools/lint.sh`), gsl, boost,
  openblas; Apple clang 21; Kokkos 5.3.0 per backend in `~/.cache/imsrg-deps`; shared venv
  `~/.cache/imsrg-deps/venv` (`.venv` symlink per checkout) with cpplint 2.0.2. A fresh worktree
  needs only `tools/ci.sh dev-openmp` (~40 s submodule fetch on first use).
* Reference: source `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only); builds, harnesses
  and plans in `/Users/cheshyre/Downloads/new-imsrg-reference/`: the **D35-patched** build
  `build-d35/imsrg++` and `harness-build-d35/refharness` (default of every
  `tools/reference/gen_golden_*.sh`), the pristine `build/`, `harness-build/`, and `plans/`
  (implementation plans of run 2). Golden data: `tests/golden/` (README lists every file and command).
  **After syncing, regenerate `build/golden-generated` with the patched harness** (`tools/reference/
  gen_golden_generated.sh`) in every checkout and worktree that runs the emax-4 tests.
* Test inputs (never copy them into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  * `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz`: `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`: `3bme_type=no2b file3e1max=6
    file3e2max=12 file3e3max=18 hw=16`. It holds **only the cE contact term** and stays the 3N test
    input (D33): E_HF(O16, emax 4) = −313.17 MeV is genuine and the full flow diverges, so IT-4/IT-5
    compare HF plus the flow stopped at s = 10. O16 at emax 2 is a known, genuine IMSRG(2)
    instability (IT-8 uses He4 emax 2 and O16 emax 3).

## State at hand-off (details: `design/STATUS.md`)

* **Merged into `develop`:** M0–M9; M11 phases 1 and 2 (118 Python tests); MS (Google style);
  hunter-gatherer Magnus; D35 (Tcm/Rp2/Rso, patched reference); flow-review fixes; D37 config/driver
  tasks (DIIS default, config-relative paths, `schema_version`, EMP2, NAT basis, IT-5); ODE flow
  methods (`flow_rk4`, `flow`; IT-14); general (non-Hermitian) fast commutator paths and
  full-precision 1/√2 (D36); M10 3f2 (IT-10, 1b intermediates); M10 layout benchmark and decision
  (D43, layout A, packed); M10 production three-body pieces (`ThreeBodySpace`, `SixJ3Table`,
  `op::ThreeBody`, `PairThirdSpace`, `commutator::Comm223`, `PerturbativeTriplesEnergy`) and the
  perturbative triples in the driver (IT-8, `results.json` schema 2); **IMSRG(3)-N7** (m-scheme brute
  force, all N7 terms, three-body generator, rank-3 Magnus/BCH, `[imsrg3] variant = "n7"`, IT-9);
  charge radius with Rso (`results.json` schema 3) and the two-stage ODE test (IT-15); Python hardening
  (149 tests).
* **End to end:** `imsrg` reproduces imsrg-devel (D35-patched) for IT-1..IT-8, IT-10, IT-12, IT-14 and
  the hunter-gatherer cases (identical step and gather counts, |ΔE0| ≤ 3e-10 MeV); numbers in the
  run-2 report §3. CI on `develop`: 1069 unit, 61 integration, 3 MPI, 149 Python tests (1 skipped), `ci.sh dev-openmp` 129 s (`develop` 9bef685 at the time of writing; use the hash in STATUS); the `slow` suite has 6 tests (IT-9 emax 3 took 39 min on the merge), run it before numerics merges.
* **IMSRG(3)-N7 (IT-9): merged.** Every N7 term vs the m-scheme brute force <= 6.1e-15 and vs imsrg-devel
  <= 6.1e-16 (full commutator 5.7e-16); IT-9 He4/O16 at emax 2 and 3 with identical step counts
  (250/147/248/173), |dE0| <= 3.8e-10 MeV; ours 12.5 s vs 24 s (O16 emax 2), 644 s vs 21 min (O16 emax 3),
  720 s vs 30 min (He4 emax 3). imsrg++ bug fixed: the `comm232` hermX*hermY shortcut is wrong for
  non-Hermitian operands (`30` section 6 M10-N7-1..3). Open: N7 is correct but not yet scalable (comm232
  tiling, tiled 133/132/231/331, `14` section 1.1; M10-Q60..Q67).
* **M12a single-node performance: merged (9bef685; CI and the slow suite passed).** Full
  IMSRG(2) commutator, ms, before M12a -> now / imsrg++: emax 4: 1 thread 6.71 -> 1.16 / 5.21, 6 threads
  2.18 -> 0.65 / 1.53; emax 6: 100 -> 18.2 / 70.6, 24.9 -> 5.1 / 15.2 (18 threads 3.9 / 9.5); emax 8:
  866 -> 169 / 671, 199 -> 49 / 155 (18 threads 31 / 95). Full O16 flow (same steps and E0): emax 8
  59.8 s vs 187.8 s (6 threads), 40.5 s vs 148.8 s (18); emax 6 7.9 s vs 24.3 s. Workspace scratch 10.4x
  -> 5.5x a 2B operator; peak RSS at emax 8 1.1 GB vs 2.3-2.6 GB. Weak spot: emax 4 at 18 threads
  only 1.1x (launch overhead). Not built: stored sparse Pandya maps. Report
  `design/benchmarks/imsrg2-performance.md`; M12-Q1..Q5.
* **Not started:** M12b (MPI), M13 (Makefile, docs), M14 (GPU), tensor operators (D39), IT-11, 3f2
  2b intermediates (E-PLANNED, M10-Q20), NAT options (M9-Q3), checkpoint/restart.
* **Open questions** with assumptions: `design/QUESTIONS.md`, "Added during run 2" and the answers
  of the interview after run 2 (apply them first: read the interview report and the new D-rows).

## Plan for this run (in order; at most 3 leads at once)

1. Preflight (above); apply the interview answers that change code or goldens (small tasks, one
   Sonnet implementer each, review per D42 where goldens change).
2. **IMSRG(3)-N7 performance** (brief below): tile `comm232ss` and the 133/132/231/331 kernels so N7
   scales toward the production target (emax 10-12, D34), without changing any result (IT-9 and the
   term goldens stay). Then the 3f2 2b intermediates if the owner asked for them (M10-Q20).
3. **M12a follow-ups** (brief below): parity is reached; continue to explore single-node limits; occasional
   emax-8 benchmarks against imsrg++, never in CI.
4. **M12b MPI** (brief below), after M12a's data structures settle: D14 phase 1 (IMSRG(2) by channel
   blocks), phase 2 (3B blocks, Hagen–Nam evaluation, D40), IT-11. N7 has merged, so phase 2 can
   distribute its blocks.
5. **Tensor operators** (D39: the first extension after IMSRG(3); tensor `OperatorSpec` exists, only
   `ch_bra ≤ ch_ket` blocks stored, D25): design check against `13`/`15`, tensor commutators for the
   IMSRG(2) terms, transformation of tensor observables (not number-conserving operators cannot flow
   today, M8-Q40). Plan in `40-roadmap.md` M15+; write a brief when the time comes and keep it small.
6. **M13** (Makefile, docs) and M14 (GPU) only if everything above is done; M11 follow-ups (generator
   callbacks, stop flag, M11-Q13) and the cleanup items of the run-2 report §7 are filler tasks for
   spare capacity.
7. At the end: the definition of done below.

## Lessons from runs 1 and 2 (follow them)

* **The usage budget is the binding constraint**, not wall time. Keep at most 3 leads running and
  delegate to Sonnet/Haiku; well-specified milestones went well with a single Sonnet
  `imsrg-implementer`. Leads delegate code and read little themselves. The orchestrator reads only
  summaries and small files.
* **Pace with `get_usage`:** check the usage windows (5-hour windows reset at 13:00, 18:00, 23:00,
  04:00 JST) before starting a big task; a stopped Opus lead that resumes reloads its whole context,
  so start long leads right after a reset and avoid launching new leads in the last minutes of a
  window. When agents stop on a limit, resume each with `SendMessage` after the reset (transcripts
  are kept); do not start new agents for the same work. Run 2 hit the limit at ~16:35 and ~21:15 and
  also ~01:30; ~30 % of the weekly budget was used by 05:30 after 16 h). The owner sets a wrap-up
  deadline and does not mind finishing early or late: state it unambiguously in every brief.
* **Timing-sensitive tests must tolerate load** (the Ctrl-C Python tests failed once in a loaded CI
  run and held back a merge): repeat the timed action, allow retries, and run candidate tests 100x under
  `ctest -j 18` before merging them.
* **Prepare outside the repository during exclusive tasks.** When one task has to run alone
  (a rename, a regeneration of goldens), do the work that cannot conflict in a scratch tree: reference
  builds and goldens, refharness flags, read-only Opus reviews, and implementation plans
  (`new-imsrg-reference/plans/`); commit it afterwards. This shortened the critical path in run 2.
* **Stacked branches via API-ready tags** shortened run 1's critical path: a lead tags
  `m<k>-api-ready` once its public API is stable and tested, and the next milestone branches from the
  tag. The next lead must later merge the predecessor's **final head**, not the tag.
* **Question ids:** every brief carries the id prefix (`M<k>-Q<n>`, `M12a-Q<n>`, `M12b-Q<n>`, ...) and
  a **block of numbers** for the lead (e.g. `M10-Q50..Q59`), the orchestrator uses `O<n>`; new
  questions go under "Added during run 3". Parallel branches then never collide in `QUESTIONS.md`.
* **Every brief carries the current `develop` hash** (and the hash of its own base branch if
  stacked), the golden rule 10 text, the emax rule, the usage/deadline wording, and the acceptance
  command.
* **Merging:** in the main checkout, `git merge --no-ff --no-commit feature/<x>`, add the `STATUS.md`
  log entry, commit, run `tools/ci.sh dev-openmp`, and revert at once if it fails. In a worktree
  (staging procedure of O1/O3) **chain the slow tests and the ref update to the exit codes
  themselves**: `tools/ci.sh dev-openmp && ctest --preset dev-openmp -L slow && git update-ref
  refs/heads/develop <new> <old>` for numerics merges (`tools/ci.sh dev-openmp && git update-ref ...`
  otherwise), never to a grep of the log (run 1 advanced `develop` after a failed CI that way).
* **Reviews:** one `imsrg-reviewer` (Opus) pass on every numerics or kernel merge (D42). In run 2
  the Opus reviews found real issues (host-only scratch counts breaking CUDA, scratch accounting,
  exception safety) and were cheap relative to their value.
* **Debug crashes under lldb**, not by guessing: a segfault in the Python exit hook (a solver
  constructor that threw left a dangling registry entry) popped macOS "Python quit unexpectedly"
  dialogs for the owner. Agents run crashing tests under `lldb -b -o run -o bt` and keep Python
  development crashes out of the owner's session (run pytest in a subprocess with a timeout).
* **Reference goldens:** after any change to `tools/reference` or the patch, rebuild
  `harness-build-d35` from the merged tree and regenerate `build/golden-generated` (patched harness)
  in every checkout; the unit tests `Emax4TrelAndRp2` etc. fail on stale dumps.
* **Phrase deadlines unambiguously** ("keep working; your final report is due by 09:30"): one lead
  read "report by 09:00" as "stop now".
* After an agent's work is merged, stop it (`TaskStop`) if it is still waiting on background work,
  and check the shared venv for editable installs (`~/.cache/imsrg-deps/venv/bin/pip list | grep
  imsrg`); remove them, because they shadow in-tree builds in other worktrees.
* Tests must not share fixed file names in the system temp directory (ctest runs tests in parallel
  processes): use `tests/support/scratch_dir.h`.
* Reading inputs is fast (me2j emax-4 prefix 3.6 ms, whole file 9 s): no caches are needed.

## Brief for the N7 performance task (ready to use; adjust the hashes)

> You lead **M10 N7 performance** on `feature/m10-n7-perf` (`develop` = 9bef685, update it). Question
> ids `M10-Q<block>` (the last used is Q67). **Golden rule 10:** searches stay inside the repository,
> its worktrees, `/Users/cheshyre/Downloads/new-imsrg-reference/`, `~/.cache/imsrg-deps/` and temp
> directories; never `find /`, `find ~`, `mdfind`. emax <= 6 everywhere, IMSRG(3) only at emax 2 and 3.
> Delegate code to Sonnet workers, one Opus review of the kernels; final report due by the owner's
> wrap-up deadline (run 2: 08:00 JST; ask at launch if none is given; leads report about 1 h before
> it; keep working until then). Read `AGENTS.md`, `design/14-imsrg3-plan.md` section 1.1 and 4, `00` (D34,
> D43), `12` sections 3-5, `benchmarks/imsrg3-layout.md`, QUESTIONS M10-Q41..Q46 and M10-Q60..Q67, the
> N7 kernels (`commutator/kernels/`), `30` section 6 M10-N7-1..3. State: all N7 terms are correct (<= 6.1e-15
> vs the m-scheme brute force) and IT-9 passes (O16 emax 3: 644 s vs 21 min for imsrg++). Goal:
> scalability toward emax 10-12 without changing results: tile `comm232ss` by column ranges and clamp
> J3 per batch (the one block set is ~33 GB at emax 12 / E3max 14, M10-Q63), tiled 133/132/231/331
> kernels in the style of `Comm223`, `exec::TeamFor`/`TeamReduce` options instead of direct Kokkos
> (M10-Q44/Q65), per-tile 6j slices and a ket cache (M10-Q41/Q42), a lazy `ThreeBodyContext` for
> IMSRG(2)-only use (M10-Q66). Judge scaling by counting and cost models, not by running beyond
> emax 3. Acceptance: all existing term, IT-9 and brute-force tests unchanged (bitwise where the
> arithmetic is unchanged), benchmark table before/after in `design/benchmarks/`, `tools/ci.sh
> dev-openmp` PASS and `ctest -L slow` PASS. Report <= 250 words.

## Brief for the M12a follow-ups (ready to use; adjust the hashes)

> You lead **M12a — single-node CPU performance** on `feature/m12a-perf` (`develop` = 9bef685, update it; base: the M12a
> merge). Question ids `M12-Q<block>` (Q1..Q5 used). **Golden rule 10 applies** (searches only inside the project
> directories). Delegate to Sonnet workers; one Opus review per kernel merge. Goal (D40): imsrg++
> parity is reached (1.5-3x faster); explore how far one node goes, judged mainly at larger emax. Known
> follow-ups: `dsyr2k` for the Hermitian ladder, BCH temporaries kept across steps, launch overhead at
> small emax (emax 4 at 18 threads is only 1.1x imsrg++), stored sparse Pandya maps (about six
> operators of memory; only if the memory is affordable), per-channel norm reductions, `op::No2bThreeBody::get_pn` linear scan and CG recomputation (M6-Q4),
> the `Denominators` copy of H (F2b), the general commutator paths' cost (M7-Q10: ≈ 2× ladder and 4×
> ph flops, only when flagged NonHermitian), `Comm223` gathers (M10-Q41/Q42: per-tile transposed 6j
> slices, a ket cache). Today: the full IMSRG(2) commutator takes 1.16 ms (emax 4, 1 thread; imsrg++ 5.21), 18.2 ms
> (emax 6; 70.6) and 169 ms (emax 8; 671); the Workspace is 5.5x a 2B operator.
> Benchmarks: `tools/bench`, `refharness bench-commutator`, results in
> `design/benchmarks/imsrg2-performance.md` (emax 4, 6, and **occasional emax-8** runs against
> imsrg++, never in CI: the one exception to the emax rule, D40). Acceptance: no regression in any
> golden test (bitwise where the arithmetic is unchanged); benchmark table with before/after and
> imsrg++; `tools/ci.sh dev-openmp` PASS. Report ≤ 250 words.

## Brief for M12b — MPI (ready to use once M12a's data structures have settled)

> You lead **M12b — MPI distribution** on `feature/m12b-mpi` (`develop` = 9bef685, update it). Question ids
> `M12b-Q<block>`. **Golden rule 10 applies.** Read `AGENTS.md`, `14-imsrg3-plan.md` §4 item 4, `00`
> (D14, D34, D40, D43), `12`, `comm` module headers (Communicator, serial stub), the `Workspace`
> and ownership notes of M7. Phase 1 (D14): 1B/2B operator parts replicated; the expensive IMSRG(2)
> commutator terms split by channel blocks across ranks through an `OwnershipMap`; results combined
> with `Allreduce`; builds without MPI keep working (serial stub). Phase 2: 3B blocks distributed by
> 3B channel (D43 layout A, packed blocks) with an ownership map; **evaluate** (as measurement, with
> a cost model and a small implementation) the row-wise, cost-balanced block distribution of Hagen
> and Nam (Prog. Theor. Phys. Suppl. 196, 102 (2012), arXiv:1203.3765) against distribution by whole
> channels, which is likely unbalanced; record the numbers in `design/benchmarks/` and decide in a
> `00` row. Targets: NVIDIA H100/H200/GH200, A100; AMD MI250X/MI300A; CPU clusters; N7 at emax 10–12
> on several nodes (D34), so distribution decisions are judged at those sizes by counting, not by
> running (emax rule). Tests: `mpirun -n 2/4` unit and integration tests under the `mpi` label;
> **IT-11**: MPI result identical to the serial result within 1e-12 relative for IMSRG(2) and N7 at
> emax 2/3 (reduction order may differ at round-off). Single-node speed may fall back to imsrg++
> level once scaled (D40). Acceptance: IT-11, scaling measurements (ranks 1, 2, 4 on the dev machine),
> `tools/ci.sh dev-openmp` PASS, the 3 existing MPI tests still PASS. Report ≤ 250 words.

## Definition of done for the run

At the end of the run (or when the budget runs low; write a draft report early as insurance):
* all finished work is merged into `develop`, and `tools/ci.sh dev-openmp` passes on `develop`
  (and `ctest -L slow` for numerics merges);
* unfinished work sits on clearly named branches, with its state described in `STATUS.md`;
* `design/STATUS.md` has an up-to-date "Current state" and a log entry for every merge;
* `design/QUESTIONS.md` holds every open question, with your assumptions;
* a report `design/reports/<date>-run3.md` in the format of the run-1 and run-2 reports (summary,
  owner actions, validation table with actual deviations vs imsrg-devel, performance, old-code bugs
  fixed, deviations from the plan, what is left and recommended next steps);
* `design/NEXT_SESSION_PROMPT.md` is updated for the run after this one, and the agenda of
  `design/INTERVIEW_PROMPT.md` is refreshed for the owner interview that follows this run (D42).
