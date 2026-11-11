# Prompt for the owner Q&A and design-interview session

**How to use:** after syncing the main checkout (QUESTIONS O1), start an **interactive** Claude
Code session (Opus) in `/Users/cheshyre/Downloads/new-imsrg` on `develop`, and paste everything
below the line. Plan 60–90 minutes. The session only edits documentation; the next unsupervised
run (`design/NEXT_SESSION_PROMPT.md`) builds on what it records.

---

You will interview the owner of the imsrg rewrite (a modern C++20/Kokkos/MPI replacement of
imsrg++). Two goals:

1. **Resolve the open questions** recorded during the planning session and the first unsupervised
   run (2026-11-11/12) in `design/QUESTIONS.md`. Each one says which assumption the run made.
2. **Interview the owner about further design decisions** needed for the next milestones, so the
   next unsupervised run can work without blocking.

## Preparation (before your first question; ≤ 10 minutes, keep it to yourself)

Read `AGENTS.md`, `design/STATUS.md`, `design/reports/2026-11-12-overnight.md`,
`design/00-requirements-and-decisions.md`, `design/QUESTIONS.md`, and skim `design/40-roadmap.md`
and `design/14-imsrg3-plan.md`. For every question on the agenda, note in 2–3 sentences the
context (which file/section or code it concerns), the options, the assumption the run made, and
what would change under each answer. Look things up in the code only when needed to explain a
consequence.

## How to run the interview

* Use `AskUserQuestion` for decisions with clear options: at most 4 options, your recommendation
  first and marked as such, each option with its concrete consequence. Batch related low-impact
  items (up to 4 questions per call). Use plain conversation for open topics.
* Go from high to low impact. Be brief: one or two sentences of context per question. Do not
  re-ask anything already answered in `QUESTIONS.md`.
* If the owner wants to defer an item, record "deferred" with the reason and keep the assumption.
* Edit documentation only, never code. Work on a branch `feature/owner-answers-<date>` cut from
  `develop`; at the end, `tools/ci.sh dev-openmp` must pass, then merge into `develop` with
  `--no-ff`. Git stays local: never push.

## Agenda, part 1: open questions (ids from `design/QUESTIONS.md`)

**A. Owner facts and actions (ask first)**
* **O1**: was the main checkout synced? Should future unsupervised runs start in the main checkout
  on `develop` (simplest) or in an app worktree (merge via staging branch + `update-ref`)?
* **M1-Q1**: the 3N test file holds only the cE contact term (E_HF(O16, emax 4) = −313.17 MeV, the
  flow diverges). Is a full N2LO 3N file (all LECs) available, and which file should IT-4, IT-5 and
  the IMSRG(3) tests use? Keep the cE-only file as a code-path test?
* **Q20**: Kokkos 5.3.0 needs NVCC ≥ 12.2 and ROCm ≥ 6.2, slightly above the baseline (CUDA ≥ 12,
  ROCm ≥ 6). Raise the baseline, or pin Kokkos 4.x for older clusters? Which clusters and toolkits
  matter?
* **Q19/O2**: compaction settings (now `autoCompactWindow` 600000 for Sonnet/Haiku agents and
  800000 for Opus leads; drop `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`?) and how to plan unsupervised runs
  around the 5-hour usage limit (the first run stopped 01:30–02:30 and 04:15–07:30).
* **Q24**: optional update of the Command Line Tools (the default macOS SDK does not link; CMake
  works around it).

**B. Physics and numerics (high impact)**
* **M1-Q2**: is IMSRG(3f2) ("factorized") what the run assumed, i.e. IMSRG(2) plus
  `BCH::SetUseFactorizedCorrection(true)`? The IT-10 goldens were made that way.
* **Q6**: may IMSRG(3) 3B blocks use `float` storage when the benchmark shows ΔE0 < ~1 keV?
* **M3-Q9**: the dE3max/OccNat 3B cuts are applied per ket, not when enumerating kets (memory vs
  imsrg++ behaviour).
* **Q5**: tensor operators: store only `ch_bra ≤ ch_ket` blocks (half memory) or all blocks?
* **M6-Q1, M6-Q2**: imsrg++ quirks kept on purpose: Rp2 includes the spin-orbit correction; Tcm's
  two-body part drops entries with |v| ≤ 1e-7. Keep (comparability) or fix?
* **M2-Q3**: imsrg++'s `INVSQRT2` has 13 digits (6.7e-14 relative error). Copy exactly or use full
  precision?
* **M2-Q7** (and `30` §6): Moshinsky brackets in double-double (imsrg++'s double version misses
  1e-12 on 22 brackets). Confirm.
* **M6-Q3**: the DIIS design (Pulay on the Fock matrix, error Fρ − ρF). Confirm.
* **M7-Q2**: the fast commutator path rejects operators without a definite hermiticity. Acceptable,
  or must general operators work?
* **M9-Q3**: natural-orbital options not implemented (`use_NAT_occupations`, `NAT_order`
  "energy"/"mp2"). Needed?
* **D21** in practice: the sign-keeping denominator floor had no effect in any IT (no denominator
  fell below 1e-6; M8-1). Also M8-2 (the Hartree-Fock generator no longer fills the diagonal) and
  M8-3 (a diverging flow throws `NumericalError`). Confirm.

**C. Configuration and CLI**
* **Q13**: must the TOML config state `basis.emax`, `basis.hw` and the file e1max values
  explicitly (no silent defaults), while the legacy CLI keeps the old defaults?
* **M8a-Q1..Q7**: generator/method enum placement; `flow`/`flow_rk4` as P1; `steps = 1` with a
  valence space; legacy `valence_space=<nucleus>` with a different `reference`; exit codes;
  particle ranks and orbit spellings of the operator catalog; `dE3max` absent instead of 99.
* **M8-Q1..Q4**: mapping of the config enums to `flow`; no MP2 column yet; Omega always kept in
  memory and HF mixing fixed to linear in the driver; log and output-file-name differences from
  imsrg++.
* **Q11, Q12, Q14**: legacy quirks (`nsteps=1` with a valence space, `imsrg3_at_end` without
  `IMSRG3`, `denominator_delta` without an orbit).

**D. Python**
* **Q15, Q16, Q17, M11-Q1**: runtime initialization (explicit `imsrg.Runtime()` vs lazy), host
  views of device data, minimum Python version (3.10 without stable ABI?), finalization with live
  operators at interpreter exit.

**E. Low-impact confirmations (summarize one line each and ask the owner to flag any to revisit)**
Q21, Q22, Q23, M2-Q1, M2-Q2, M2-Q4, M2-Q5, M2-Q6, M3-Q1..M3-Q8, M4-Q1, M4-Q2, M5-Q1, M6-Q4, M7-Q1,
M7-Q3, M7-Q4, M9-Q1, M9-Q2, M9-Q4, M11-Q2, plus any later ids in `QUESTIONS.md` not listed here.

## Agenda, part 2: further design decisions (interview)

1. **Priorities after M8**: order of M10 (IMSRG(3)), M12 (MPI and performance), M11 phase 2,
   tensor/dagger/0νββ/dark-matter operators, me3j input, Navratil/Miyagi operator files.
2. **IMSRG(3) targets**: production sizes beyond the emax ≤ 3 tests (memory per node, emax/E3max),
   which variants matter most, how to treat the imsrg++ IMSRG(3) bugs listed in `14` §1, and
   accuracy expectations.
3. **Hardware and scale**: target clusters (GPU types, node memory, interconnect, schedulers), the
   MPI distribution plan (D14 phases 1/2), performance goals relative to imsrg++ (single-node CPU is
   now 1.1–1.2× slower for the IMSRG(2) commutator at emax 4/6), when GPU work should start.
4. **Outputs and workflows**: results JSON schema, checkpoint/restart, which old output formats must
   stay compatible (tokyo, NuShellX, operator files), HDF5 or not.
5. **Python API**: main users and workflows (pipelines, operator algebra, notebooks), packaging
   (pip wheels, conda).
6. **Testing policy**: integration-test runtime budget, golden-data regeneration, tolerances for
   IMSRG(3), the O16 emax-2 IMSRG(2) instability.
7. **Process**: feedback on the agent orchestration and the overnight report; git policy (local
   only for now? when to push or open PRs); review cadence; licensing and naming.

## Recording the answers

* In `design/QUESTIONS.md`: under each question, `**Answer (owner, <date>):** ...` (or
  "deferred: <reason>").
* Decisions become rows in `design/00-requirements-and-decisions.md` with the next free D-number,
  marked *(user)*.
* Update the affected design docs (`13`, `14`, `16`, `18`, `20`, `30`, `40`) and the plan in
  `design/NEXT_SESSION_PROMPT.md` (priorities, inputs, the 3N file).
* Write `design/reports/<date>-owner-interview.md`: decisions taken, changed priorities, follow-up
  tasks for the next unsupervised run.
* Finish with a short summary to the owner: what was decided, what was deferred, and whether the
  next unsupervised run is ready to start.
