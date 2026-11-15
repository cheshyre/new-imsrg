# Prompt for the next unsupervised implementation session (run 2)

The first unsupervised run (2026-11-11 23:41 → 2026-11-12 ~08:45 JST) merged M0–M9 (M8 in two
parts, M8a config/CLI and M8 flow/pipeline) and M11 phase 1; see `design/reports/2026-11-12-overnight.md`.
The owner then answered every open question and settled the next design decisions in two interviews
(2026-11-13: D25–D29; 2026-11-15: D30–D42, report `design/reports/2026-11-15-owner-interview.md`).
This prompt starts the second run. (The prompt of the first run is in the git history of this file.)

**Owner pre-flight (once, before launching):**

1. **Sync the main checkout** (QUESTIONS O1): the interview sessions advanced `develop` from worktrees
   again, so the main checkout's files still show 68c8927. Check `git status` there: it should list
   only staged reversals of the newer commits. With no local edits of your own:
   `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 68c8927 develop`
   (a two-tree fast-forward of index and files that refuses if local edits conflict), or simply
   `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` (HEAD is `develop`).
2. Start the session (Opus) **in the main checkout on `develop`** (D31): merges then happen on
   `develop` directly. Agent definitions load at startup from `.claude/agents/`.
3. Settings (D31: no `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; `"autoCompactWindow": 800000` for the
   orchestrator). After step 1, in the main checkout: install the proposal as
   `.claude/settings.local.json` (gitignored), without its `_comment` key, and rename the run-1 copy
   `.claude/settings.json` (untracked, still setting `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80`, which would
   otherwise keep applying because settings files are merged) to `.claude/settings.json.bak`
   (gitignored by `*.bak`):
   `python3 -c "import json; d=json.load(open('design/claude-settings.proposed.json')); d.pop('_comment', None); json.dump(d, open('.claude/settings.local.json', 'w'), indent=2)"`
   and `mv .claude/settings.json .claude/settings.json.bak`. Pick a permission mode that does not
   stall in the app's permission-mode picker. **Launch right after a reset of your 5-hour usage
   window** (run 1 stopped 01:30–02:30 and 04:15–07:30).
4. Optional hard stop against pushing:
   `git -C /Users/cheshyre/Downloads/new-imsrg remote set-url --push origin DISABLED`.

Then paste everything below the line.

---

You are the **orchestrator** of the second unsupervised implementation run of the imsrg rewrite
(~8–12 h, nobody answers questions until morning). Continue from the state left by the first run and
the owner's interview of 2026-11-15, milestone by milestone, without breaking what works.

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

The owner's binding decisions are in `design/00-requirements-and-decisions.md` §2–5 (D1–D42) and in
the answers in `design/QUESTIONS.md`. The most important ones:

* Hardware targets: NVIDIA GPUs, AMD GPUs, CPU clusters → Kokkos (D2); baseline GCC ≥ 11, CUDA ≥ 12.2,
  ROCm ≥ 6.2 (D30). GPU work starts only after M12 (D40).
* Order: core → 3N input + natural orbitals → **IMSRG(3)** (top-priority extension: N7, 3f2,
  perturbative triples; the 3B layout is chosen by benchmark and the session decides itself, D23).
  After IMSRG(3): tensor operators first (D39). Atomic systems and RPA/TDA are dropped.
* **Code style: the Google C++ Style Guide** (D38, `design/17-code-style.md`); the conversion of the
  existing code is the **first task of this run**, done alone before any other branch starts.
* **emax ≤ 6 in all tests and runs; IMSRG(3) only at emax 2 and 3.** Exception: occasional emax-8
  IMSRG(2) performance benchmarks against imsrg++, never in CI (D40).
* Git is **local only**: feature branches → local `develop`. Never push, no PRs. Never touch `main`.
* Homebrew and pip (shared venv) installs are allowed. The old imsrg-devel is built **outside** the
  repo (already done, see below; a patched second build is a task of this run).
* Old-code bugs that change results: fix them, document them in `design/30-testing-and-validation.md`
  §6, and validate the fixed path another way.
* Tolerances: unit checks ~1e-10 relative; end to end 1e-4 MeV (HF, MBPT, perturbative triples) and
  1e-3 MeV (IMSRG(2)/(3) E0, valence-space elements). Always log the actual deviation (D19, D34).
* `tools/ci.sh dev-openmp` stays under 5 minutes; longer tests get the `slow` label and run before
  merges that touch numerics (D42). Goldens are regenerated only through the scripted
  `tools/reference` commands, with the reason and old vs new numbers in the commit and a reviewer
  pass (D42).

## Where everything is

* Repository: `/Users/cheshyre/Downloads/new-imsrg` (git), integration branch **`develop`**.
  Preflight: `git branch --show-current` prints `develop`; `git merge-base --is-ancestor
  feature/design-plan develop` succeeds; `git status --porcelain --untracked-files=no` prints nothing
  (untracked ignored files such as the settings are fine); then `tools/ci.sh dev-openmp` prints
  `CI PASS dev-openmp`. If any check fails, write `PREFLIGHT_FAILED.md` at the repo root with the
  details and stop. If the `imsrg-*` agent types are unavailable, spawn `general-purpose` agents
  with the matching `model` and paste the body of the corresponding `.claude/agents/*.md` into their
  prompt.
* Read first: `AGENTS.md`, `CLAUDE.md`, `design/README.md`, `design/STATUS.md` (current state, log
  of every merge), `design/reports/2026-11-15-owner-interview.md`, `design/00-requirements-and-decisions.md`,
  `design/40-roadmap.md`, `design/41-agent-workflow.md`. Leads read the other design docs and
  `design/reference/*.md` as their milestone requires; the orchestrator delegates big reading.
* Toolchain (installed): Homebrew cmake, ninja, ccache, Open MPI, libomp, clang-format 23.1.5, gsl,
  boost, openblas; Apple clang 21; Kokkos 5.3.0 per backend in `~/.cache/imsrg-deps` (via
  `tools/bootstrap_deps.sh`); shared venv `~/.cache/imsrg-deps/venv` (`.venv` symlink per checkout)
  with cpplint 2.0.2. **clang-tidy is not installed** (`brew install llvm` provides it). A fresh
  worktree needs only `tools/ci.sh dev-openmp` (~40 s submodule fetch on first use).
* Reference: source `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only); build, harness
  and old executable in `/Users/cheshyre/Downloads/new-imsrg-reference/` (`build/imsrg++`,
  `harness-build/refharness`; rebuild with `tools/reference/build_imsrg_devel.sh` if missing).
  Golden data: `tests/golden/` (README lists every file and command).
* Test inputs (never copy them into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  * `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz`: `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`: `3bme_type=no2b file3e1max=6
    file3e2max=12 file3e3max=18 hw=16`. It holds **only the cE contact term** and **stays the 3N test
    input** (D33): E_HF(O16, emax 4) = −313.17 MeV is genuine and the full flow diverges, so IT-4/IT-5
    compare HF plus the flow stopped at s = 10. A full N2LO file comes later from the owner. O16 at
    emax 2 is a known, genuine IMSRG(2) instability (IT-8 uses He4 emax 2 and O16 emax 3).

## State at hand-off (details: `design/STATUS.md`)

* **Merged into `develop`:** M0 bootstrap; M1 reference build, refharness, golden data (incl.
  IT-1..IT-10 goldens); M2 base/exec/comm/linalg/angmom; M3 basis; M4 operator (`imsrg::op`);
  M5+M6 io/opslib/hf (HF matches imsrg++ to 2e-13, DIIS); M7 IMSRG(2) commutators (all scalar terms,
  ≤ 1.5e-15 vs goldens, 1.1–1.2× imsrg++ time); M8a config + legacy CLI + `apps/imsrg`; M8 flow
  (generators, BCH, Magnus with the imsrg++ step rule) + driver pipeline: IT-2, IT-3, IT-4, IT-6,
  IT-7, IT-12 pass with identical step counts and |ΔE0| ≤ 3e-10 MeV; M9 mbpt; M11 phase 1 Python
  bindings (Python ≥ 3.12, stable ABI). CI on `develop`: 851 unit, 34 integration, 3 MPI, 23 Python
  tests; `tools/ci.sh dev-openmp` takes 139 s.
* **Not started:** MS (style conversion), M10 (IMSRG(3)), M11 phase 2, M12a/M12b, M13, M14 (GPU).
* **Owner decisions of 2026-11-15** (D30–D42; details in the report): the IMSRG(3f2) definition
  (D32: hunter-gatherer, `omega_norm_max` 0.1, factorized correction with 1b intermediates on and
  2b off; the IT-10 goldens must be regenerated); Tcm without the 1e-7 threshold and Rp2 without the
  spin-orbit term, validated with goldens from a patched imsrg-devel build (D35); general
  (non-Hermitian) fast-commutator paths and full-precision 1/√2 (D36); `hartree_fock.mixing` default
  DIIS in TOML, TOML paths relative to the config file, `results.json` `schema_version` (D37);
  Google style (D38); 3B cuts drop kets at enumeration, N7 production target emax 10–12 (D34);
  M12 split into single-node performance (M12a) before MPI (M12b) (D40).

## Plan for this run (in order; at most 3 leads at once)

1. Preflight (above).
2. **MS — Google style conversion** (one Opus lead with Haiku/Sonnet workers; brief below). It runs
   **alone**: no other branch starts until it is merged, because it renames every file and most
   functions. Merge when CI passes and the test results are bitwise identical.
3. **M8 follow-ups and the interview tasks** (from `develop` after MS; one lead, or two or three
   Sonnet implementers with an Opus review; `40-roadmap.md` M8 lists them):
   a. Opus review of the flow numerics (first).
   b. Hunter-gatherer Magnus (`imsrg.omega.hunter_gatherer`; imsrg++ `IMSRGSolver.cc:85-115`), a
      prerequisite of 3f2.
   c. Tcm without threshold, Rp2 without spin-orbit, `Rso` as the separate correction, charge radius
      adds `Rso` (D35); a **patched imsrg-devel build** (patch file in `tools/reference/`, separate
      build directory, `30` §3) and the e2e goldens containing T_int or Rp2 regenerated from it, so
      the ITs keep ~1e-10 MeV and identical step counts. Golden policy D42.
   d. Config and driver: `hartree_fock.mixing` (`diis` in TOML, linear in legacy); TOML paths relative
      to the config file and absolute paths in `resolved_config.toml` (adapt `examples/config/` and
      the TOML/legacy equivalence test); `results.json` `schema_version` with the schema table and a
      changelog in `16` §2.10; MP2 column and EMP2; NAT basis and IT-5 in the driver; `Denominators`
      copying H every step; flow-RK4/DP45.
   e. General (non-Hermitian) paths in the fast IMSRG(2) commutator, tested against `commutator_ref`.
   f. Full-precision 1/√2 (rename the 13-digit copy `imsrgpp_inv_sqrt2`, tests only).
4. **M10 IMSRG(3)**, following `design/14-imsrg3-plan.md`, the top priority after step 3, in two
   tracks:
   * **3f2 track** (needs hunter-gatherer from 3b; brief below): config keys (D32), refharness flags
     and the regenerated IT-10 goldens, the factorized double commutator in BCH, IT-10.
   * **Layout track:** the layout benchmark (brief below) and the decision row in `00`, then the
     brute-force m-scheme reference, perturbative triples (IT-8), IMSRG(3)-N7 (IT-9). imsrg++'s
     IMSRG(3) bugs are fixed and validated against brute force (D34).
5. In parallel with M10 when capacity allows (one lead or one Sonnet implementer each):
   **M11 phase 2** (the `StaleHostError` state machine of `18` §2.2 first; then bind io, hf, mbpt,
   commutators, flow, pipeline; users are pipelines and operator algebra, D41) and **M12a**
   (single-node CPU performance: sparse Pandya maps and Workspace memory (M7-Q4), per-channel norm
   reductions, `op::No2bThreeBody::get_pn` (M6-Q4); reach imsrg++ parity, then explore; occasional
   emax-8 benchmarks against imsrg++, never in CI; results in `design/benchmarks/`). M12b, M13 and
   M14 only if everything above is done.
6. At the end: the definition of done below.

## Lessons from the first run (follow them)

* **The usage budget is the binding constraint**, not wall time. Keep at most 3 leads running and
  delegate to Sonnet/Haiku; well-specified milestones went well with a single Sonnet
  `imsrg-implementer` (M9, M11 phase 1, the refharness flow subcommand). Leads delegate code and read
  little themselves. The orchestrator reads only summaries and small files.
* When agents stop on a usage limit, resume each with `SendMessage` after the reset (their
  transcripts are kept); do not start new agents for the same work.
* **Stacked branches via API-ready tags** shortened the critical path: a lead tags `m<k>-api-ready`
  once its public API is stable and tested, and the next milestone branches from the tag (use a
  background `until git rev-parse -q --verify refs/tags/<tag>` loop to get notified). The next lead
  must later merge the predecessor's **final head**, not the tag (M7's comm121ss fix landed after
  its tag). Tell leads when a predecessor lands in `develop`. (MS is the exception: nothing stacks on
  it before it is merged.)
* **Question ids:** leads use `M<k>-Q<n>` (`MS-Q<n>` for the style conversion), the orchestrator
  `O<n>`, so parallel branches never collide in `QUESTIONS.md`; new questions go under "Added during
  run 2".
* **Merging:** in the main checkout, merge on `develop` directly: `git merge --no-ff --no-commit
  feature/<x>`, add the `STATUS.md` log entry, commit, run `tools/ci.sh dev-openmp`, and revert at
  once if it fails. If the app forces a worktree after all, use the staging procedure of QUESTIONS
  O1 and chain the ref update to the CI exit status itself (`tools/ci.sh dev-openmp && git
  update-ref refs/heads/develop <new> <old>`), never to a grep of its log: run 1 once advanced
  `develop` after a failed CI that way.
* **Phrase deadlines unambiguously** ("keep working; your final report is due by 09:30"): one lead
  read "report by 09:00" as "stop now".
* After an agent's work is merged, stop it (`TaskStop`) if it is still waiting on background work,
  and check the shared venv for editable installs (`~/.cache/imsrg-deps/venv/bin/pip list | grep
  imsrg`); remove them, because they shadow in-tree builds in other worktrees.
* Tests must not share fixed file names in the system temp directory: ctest runs every test in
  its own process in parallel (run 1 hit such a race in the io tests). Use per-process names (the
  pattern in `tests/unit/driver/integration/run_support.hpp`) or, better, a shared helper; the
  remaining fixed names are listed in the `STATUS.md` log entry of 2026-11-12 08:55.
* Reading inputs is fast (me2j emax-4 prefix 3.6 ms, whole file 9 s; NO2B 0.05 s at emax 4): no
  caches are needed.

## Brief for MS — the Google style conversion (ready to use; adjust the develop hash)

> You lead **MS — the conversion of the imsrg rewrite to the Google C++ Style Guide** (owner
> decision D38: "Follow the Google style guide ... Use clang-format ... Use cpp-lint ... but do not
> blindly follow the linter"). Usage budget is tight: delegate the mechanical steps to Haiku/Sonnet
> workers and review their diffs; one `imsrg-reviewer` pass at the end. Never block; questions go to
> `design/QUESTIONS.md` as `MS-Q<n>`. Branch `feature/style-google` from `develop`. Read
> `AGENTS.md`, `design/17-code-style.md` (all; §5 is your plan, §2 the measured scale, §3 the
> exceptions, §4 the tool configuration), `10-architecture.md` §1–2 and `20-build-and-tooling.md`
> §2–4. Tools: clang-format 23.1.5 (installed), cpplint 2.0.2 (`~/.cache/imsrg-deps/venv/bin/cpplint`),
> clang-tidy (`brew install llvm`; use `compile_commands.json` from the dev-openmp build). Before step
> 1, record the IT `results.json` numbers and the test counts; after **every** step: build,
> `tools/ci.sh dev-openmp` PASS, and identical numbers. One commit per step of `17` §5, each with a
> body explaining the step. Never touch `tools/reference/`, `extern/` or `tests/golden/`. Python names
> must not change (the bindings name every function explicitly). Exceptions stay: C++ exceptions,
> `<filesystem>` outside kernels, Kokkos `[=]` device lambdas. Finally update `AGENTS.md` (code
> conventions), `10` §1, `20` and the status line of `17`. Acceptance: `17` §5. Report ≤ 250 words
> with before/after counts (cpplint findings, files, renamed names) and every rule you could not apply.

## Brief for the M10 layout benchmark (ready to use; adjust the develop hash)

> You lead **M10 step 1 — the IMSRG(3) three-body storage layout benchmark and the layout
> decision**. The owner told the session to decide itself and continue (D23). Usage budget is
> tight: delegate to Sonnet/Haiku workers, one `imsrg-reviewer` pass (opus) at most. Never block;
> questions go to `design/QUESTIONS.md` as `M10-Q<n>`. Problem sizes: emax 2 and 3 only, but judge
> memory and scaling against the owner's production target, **IMSRG(3)-N7 at emax 10–12** on several
> nodes (D34). Kets failing dE3max/OccNat3Cut are dropped at enumeration (D34). IMSRG(3f2) is a
> separate track, not part of this brief. Branch `feature/m10-layout-benchmark` from `develop` (after
> MS; the old empty branch of that name can be reset); prototypes in `tools/bench/imsrg3_layout/`
> (a benchmark target outside `ci.sh`'s unit label); production modules only get small additive
> accessors (separate commits). Read `AGENTS.md`, `design/STATUS.md`, `design/14-imsrg3-plan.md`
> (all), `00` (D12 3B norm, D15, D23, D34), `12` §3–§5, `reference/02-commutators.md` §1, §4, §8,
> `reference/01-core-data-model.md` (3B storage), the `basis` ThreeBodyChannels header.
> Priorities: (1) the `ThreeBodyLayout` interface and layouts A (canonical a ≤ b ≤ c × Jab),
> B (orthonormal antisymmetric basis per channel), C (partially antisymmetric |(ab)Jab c⟩), with
> memory and metadata build time at emax 2/3 and an extrapolation to emax 10–12; (2) the [2,2]→3
> kernel (`comm223`-type, the hotspot) in each layout, cross-layout agreement to 1e-10 on random 2B
> inputs, release timings with 18 OpenMP threads; (3) the perturbative-triples energy per layout,
> then [2,3]→3 and [3,3]→2 if time remains; (4) decide: best speed at acceptable memory, written up in
> `design/benchmarks/imsrg3-layout.md`, a decision row in `00`, next steps in `14`.
> Acceptance: report and decision committed; prototypes build; numbers are real measurements;
> `tools/ci.sh dev-openmp` PASS. Report ≤ 250 words.

## Brief for the M10 3f2 track (ready to use once hunter-gatherer is merged)

> You lead **M10 track 3f2 — IMSRG(3f2)**, the owner's factorized approximation (D32): plain IMSRG(2)
> plus hunter-gatherer Magnus, `omega_norm_max` 0.1, and the factorized double-commutator correction
> in BCH with 1b intermediates on and 2b intermediates off (pyIMSRG: `SetHunterGatherer(True)`,
> `SetOmegaNormMax(0.1)`, `BCH.SetUseFactorizedCorrection(True)`,
> `FactorizedDoubleCommutator.SetUse_1b_Intermediates(True)`, `SetUse_2b_Intermediates(False)`).
> Delegate to Sonnet workers; one Opus review of the kernels. Questions as `M10-Q<n>`. Branch
> `feature/m10-3f2` from `develop`. Read `14` §1 and §4 item 3, `13` §6–7, `16` §2.8,
> `reference/02-commutators.md` (factorized terms), imsrg++ `Commutator/FactorizedDoubleCommutator.cc`
> and `BCH.cc:182-191`, `tools/reference/rh_flow.cpp`. Steps: (1) refharness `flow` flags for
> hunter-gatherer, `omega_norm_max` and the 1b/2b intermediates; regenerate the IT-10 goldens with
> the recipe (golden policy D42; old vs new numbers in the commit); (2) `BchOptions::factorized` with
> both intermediate switches, term tests against imsrg-devel (its `UnitTestFactorized` cases) at
> 1e-10; (3) config keys `factorized_1b_intermediates`/`factorized_2b_intermediates` and the recipe
> defaults of `[imsrg3] factorized = true` (resolved config writes them out); (4) IT-10 at emax 2
> (emax 3 under `slow`). Acceptance: IT-10 E0 within 1e-3 MeV (log the deviation and step counts);
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
* `design/NEXT_SESSION_PROMPT.md` is updated for the run after this one, and the agenda of
  `design/INTERVIEW_PROMPT.md` is refreshed for the owner interview that follows this run (D42).
