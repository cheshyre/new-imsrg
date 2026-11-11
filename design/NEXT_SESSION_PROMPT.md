# Prompt for the next unsupervised implementation session (run 2)

The first unsupervised run (2026-11-11 23:41 → 2026-11-12 ~08:45 JST) merged M0–M7, M8a, M9 and
M11 phase 1; see `design/reports/2026-11-12-overnight.md`. This prompt starts the second run.
(The prompt of the first run is in the git history of this file.)

**Owner pre-flight (once, before launching):**

1. **Sync the main checkout** (QUESTIONS O1): the first run advanced `develop` from a worktree, so
   the main checkout's index still shows 331b2d4. With no local edits there:
   `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 331b2d4 develop`
   (or `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard`).
2. Recommended: run the Q&A/interview session first (`design/INTERVIEW_PROMPT.md`), so the open
   questions are answered and the priorities below reflect your decisions.
3. Start the session (Opus) **in the main checkout on `develop`** if possible: then merges happen
   on `develop` directly. If the app starts it in a worktree, the staging procedure of O1 applies
   (below). Agent definitions load at startup from `.claude/agents/` on `develop`.
4. Settings: `design/claude-settings.proposed.json` (pick a permission mode that does not stall).
   The first run stopped twice on the 5-hour usage limit (01:30–02:30, 04:15–07:30); plan the run
   around your usage windows if you can.
5. Optional hard stop against pushing:
   `git -C /Users/cheshyre/Downloads/new-imsrg remote set-url --push origin DISABLED`.

Then paste everything below the line.

---

You are the **orchestrator** of the second unsupervised implementation run of the imsrg rewrite
(~8–12 h, nobody answers questions until morning). Continue from the state left by the first run,
milestone by milestone, without breaking what works.

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

The owner's binding decisions are in `design/00-requirements-and-decisions.md` §2–5 (and any
answers recorded in `design/QUESTIONS.md` since the first run). The most important ones:

* Hardware targets: NVIDIA GPUs, AMD GPUs, CPU clusters → Kokkos (D2).
* Order: core (HF + IMSRG(2), single-reference and valence-space) → 3N input (NO2B) + natural
  orbitals → **IMSRG(3)** (top-priority extension; variants IMSRG(3)-N7, IMSRG(3f2) factorized,
  perturbative triples; the 3B storage layout is chosen **by benchmark**, and the session
  **chooses the layout itself** and continues, D23). Tensor, dagger, 0νββ and DM operators: design
  for them, implement later. Atomic systems and RPA/TDA are dropped.
* **emax ≤ 6 in all tests and runs; IMSRG(3) only at emax 2 and 3.**
* Git is **local only**: feature branches → local `develop`. Never push. Never touch `main`.
* Homebrew and pip (shared venv) installs are allowed. The old imsrg-devel is built **outside** the
  repo (already done, see below).
* Old-code bugs that change results: fix them, document them in `design/30-testing-and-validation.md`
  §6, and validate the fixed path another way.
* The CLI accepts both a TOML config and the old `key=value` parameters (implemented in M8a).
* Tolerances: deterministic unit checks ~1e-10 relative; end to end 1e-4 MeV (HF, MBPT) and
  1e-3 MeV (IMSRG E0, valence-space elements). Always log the actual deviation.
* HF has a correct DIIS (done); inputs: me2j, NO2B (done), me3j and Navratil/Miyagi operator files
  (later).

## Where everything is

* Repository: `/Users/cheshyre/Downloads/new-imsrg` (git), integration branch **`develop`**.
  Preflight: `git branch --show-current`; `git merge-base --is-ancestor feature/design-plan develop`
  must succeed; then `tools/ci.sh dev-openmp` must print `CI PASS dev-openmp` on `develop`. If
  either fails, write `PREFLIGHT_FAILED.md` at the repo root with the details and stop. If the
  `imsrg-*` agent types are unavailable, spawn `general-purpose` agents with the matching `model`
  and paste the body of the corresponding `.claude/agents/*.md` into their prompt.
* Read first: `AGENTS.md`, `CLAUDE.md`, `design/README.md`, `design/STATUS.md` (current state, log
  of every merge), `design/reports/2026-11-12-overnight.md`, `design/00-requirements-and-decisions.md`,
  `design/40-roadmap.md`, `design/41-agent-workflow.md`. Leads read the other design docs and
  `design/reference/*.md` as their milestone requires; the orchestrator delegates big reading.
* Toolchain (installed): Homebrew cmake, ninja, ccache, Open MPI, libomp, clang-format, gsl, boost,
  openblas; Apple clang 21; Kokkos 5.3.0 per backend in `~/.cache/imsrg-deps` (via
  `tools/bootstrap_deps.sh`); shared venv `~/.cache/imsrg-deps/venv` (`.venv` symlink per checkout).
  A fresh worktree needs only `tools/ci.sh dev-openmp` (~40 s submodule fetch on first use).
* Reference: source `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only); build, harness
  and old executable in `/Users/cheshyre/Downloads/new-imsrg-reference/` (`build/imsrg++`,
  `harness-build/refharness`; rebuild with `tools/reference/build_imsrg_devel.sh` if missing).
  Golden data: `tests/golden/` (README lists every file and command).
* Test inputs (never copy them into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  * `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz`: `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`: `3bme_type=no2b file3e1max=6
    file3e2max=12 file3e3max=18 hw=16`. It holds **only the cE contact term** (M1-Q1): E_HF(O16,
    emax 4) = −313.17 MeV is genuine and the full flow diverges, so IT-4/IT-5 compare HF plus the
    flow stopped at s = 10. O16 at emax 2 is a genuine IMSRG(2) instability (IT-8 uses He4 emax 2
    and O16 emax 3). Use a different 3N file only if the owner provided one (check QUESTIONS).

## State at hand-off (details: `design/STATUS.md`)

* **Merged into `develop`:** M0 bootstrap; M1 reference build, refharness, golden data (incl.
  IT-1..IT-10 goldens); M2 base/exec/comm/linalg/angmom; M3 basis; M4 operator (`imsrg::op`);
  M5+M6 io/opslib/hf (HF matches imsrg++ to 2e-13, DIIS); M7 IMSRG(2) commutators (all scalar terms,
  ≤ 1.5e-15 vs goldens, 1.1–1.2× imsrg++ time); M8a config + legacy CLI + `apps/imsrg`; M9 mbpt;
  M11 phase 1 Python bindings.
* **M8 (flow, BCH, Magnus, driver pipeline, integration tests):** see the M8 entry in
  `design/STATUS.md` for what was merged or what remains on `feature/m8-flow-driver`.
* **Not started:** M10 (IMSRG(3)), M11 phase 2, M12 (MPI + performance), M13 (Makefile, docs).

## Plan for this run (in order; at most 3 leads at once)

1. Preflight (above).
2. **Finish M8** from `feature/m8-flow-driver` (or from `develop` if it was merged): the remaining
   integration tests among IT-2, IT-3, IT-4, IT-6, IT-7, IT-12 (acceptance of M8 in
   `40-roadmap.md`), flow-RK4/DP45, 2-step valence-space decoupling, more generators; resolve
   M8a-Q1 (generator/method enums now live in `flow`). Merge when CI passes.
3. **M10 IMSRG(3)**, following `design/14-imsrg3-plan.md`: the layout benchmark (brief below) and
   the decision row in `00-requirements-and-decisions.md`, then the brute-force m-scheme reference,
   perturbative triples (IT-8), IMSRG(3)-N7 (IT-9), IMSRG(3f2) (IT-10; check the definition in
   M1-Q2 against the owner's answer). This is the top priority after M8.
4. In parallel with M10 (one lead or one Sonnet implementer each): **M11 phase 2** (bind io, hf,
   mbpt, commutators, flow, pipeline; `18-python-api.md`), **M12** (MPI distribution, IT-11;
   performance follow-ups: sparse Pandya maps and Workspace memory (M7-Q4), per-channel norm
   reductions, `op::No2bThreeBody::get_pn` (M6-Q4)), then **M13** if time remains.
5. At the end: the definition of done below.

## Lessons from the first run (follow them)

* **The usage budget is the binding constraint**, not wall time. Keep at most 3 Opus leads running;
  well-specified milestones went well with a single Sonnet `imsrg-implementer` (M9, M11 phase 1,
  the refharness flow subcommand). Leads delegate code to Sonnet/Haiku and read little themselves.
  The orchestrator reads only summaries and small files.
* When agents stop on a usage limit, resume each with `SendMessage` after the reset (their
  transcripts are kept); do not start new agents for the same work.
* **Stacked branches via API-ready tags** shortened the critical path: a lead tags `m<k>-api-ready`
  once its public API is stable and tested, and the next milestone branches from the tag (use a
  background `until git rev-parse -q --verify refs/tags/<tag>` loop to get notified). The next lead
  must later merge the predecessor's **final head**, not the tag (M7's comm121ss fix landed after
  its tag). Tell leads when a predecessor lands in `develop`.
* **Question ids:** leads use `M<k>-Q<n>`, the orchestrator `O<n>`, so parallel branches never
  collide in `QUESTIONS.md`.
* **Merging when the session runs in a worktree** (O1): merge on the worktree branch with
  `git merge --no-ff --no-commit feature/<x>`, add the `STATUS.md` log entry, commit, run
  `tools/ci.sh dev-openmp`, and only on PASS move `develop`:
  `git update-ref refs/heads/develop <new> <old>`. The session may not run git in the main
  checkout. Putting the STATUS entry into the merge commit keeps `develop`'s first-parent history
  to merges only.
* **Phrase deadlines unambiguously** ("keep working; your final report is due by 09:30"): one lead
  read "report by 09:00" as "stop now".
* After an agent's work is merged, stop it (`TaskStop`) if it is still waiting on background work,
  and check the shared venv for editable installs (`~/.cache/imsrg-deps/venv/bin/pip list | grep
  imsrg`); remove them, because they shadow in-tree builds in other worktrees.
* Reading inputs is fast (me2j emax-4 prefix 3.6 ms, whole file 9 s; NO2B 0.05 s at emax 4): no
  caches are needed.

## Brief for the M10 layout benchmark (ready to use; adjust the develop hash)

> You lead **M10 step 1 — the IMSRG(3) three-body storage layout benchmark and the layout
> decision**. The owner told the session to decide itself and continue (D23). Usage budget is
> tight: delegate to Sonnet/Haiku workers, one `imsrg-reviewer` pass (opus) at most. Never block;
> questions go to `design/QUESTIONS.md` as `M10-Q<n>`. Problem sizes: emax 2 and 3 only.
> Branch `feature/m10-layout-benchmark` from `develop`; prototypes in `tools/bench/imsrg3_layout/`
> (a benchmark target outside `ci.sh`'s unit label); production modules only get small additive
> accessors (separate commits). Read `AGENTS.md`, `design/STATUS.md`, `design/14-imsrg3-plan.md`
> (all), `00` (D12 3B norm, D15, D23), `12` §3–§5, `reference/02-commutators.md` §1, §4, §8,
> `reference/01-core-data-model.md` (3B storage), the `basis` ThreeBodyChannels header.
> Priorities: (1) the `ThreeBodyLayout` interface and layouts A (canonical a ≤ b ≤ c × Jab),
> B (orthonormal antisymmetric basis per channel), C (partially antisymmetric |(ab)Jab c⟩), with
> memory and metadata build time at emax 2/3; (2) the [2,2]→3 kernel (`comm223`-type, the hotspot)
> in each layout, cross-layout agreement to 1e-10 on random 2B inputs, release timings with 18
> OpenMP threads; (3) the perturbative-triples energy per layout, then [2,3]→3 and [3,3]→2 if time
> remains; (4) decide: best speed at acceptable memory, written up in
> `design/benchmarks/imsrg3-layout.md`, a decision row in `00`, next steps in `14`.
> Acceptance: report and decision committed; prototypes build; numbers are real measurements;
> `tools/ci.sh dev-openmp` PASS. Report ≤ 250 words.

## Definition of done for the run

At the end of the run (or when the budget runs low; write a draft report early as insurance):
* all finished work is merged into `develop`, and `tools/ci.sh dev-openmp` passes on `develop`;
* unfinished work sits on clearly named branches, with its state described in `STATUS.md`;
* `design/STATUS.md` has an up-to-date "Current state" and a log entry for every merge;
* `design/QUESTIONS.md` holds every open question, with your assumptions;
* a report `design/reports/<date>-run2.md` summarizes milestones completed, test and validation
  status (with the comparison numbers against imsrg-devel), performance numbers, deviations from
  the design, open questions, and the recommended next steps;
* `design/NEXT_SESSION_PROMPT.md` is updated for the run after this one.
