# Prompt for the owner Q&A and design-interview session (after run 2)

> **Status:** refreshed at the end of run 2 (2026-11-15/16) for the interview that follows it (D42).
> The interviews so far: 2026-11-13 (D25–D29) and 2026-11-15 (D30–D42, report
> `design/reports/2026-11-15-owner-interview.md`). The next free decision number is D44 (D43 is the
> layout decision of run 2; check `00` for later rows). Every run-2 question below carries the
> assumption the run made, so the owner can answer with one word.

**How to use:** after syncing the main checkout (QUESTIONS O1/O3), start an **interactive** Claude
Code session (Opus) in `/Users/cheshyre/Downloads/new-imsrg` on `develop`, and paste everything
below the line. Plan 60–90 minutes. The session only edits documentation; the next unsupervised
run (`design/NEXT_SESSION_PROMPT.md`) builds on what it records.

---

You will interview the owner of the imsrg rewrite (a modern C++20/Kokkos/MPI replacement of
imsrg++). Two goals:

1. **Resolve the open questions** recorded during run 2 (2026-11-15/16) in `design/QUESTIONS.md`
   ("Added during run 2"). Each one says which assumption the run made.
2. **Interview the owner about the design decisions** needed for the next milestones (N7 at scale,
   MPI, tensor operators), so run 3 can work without blocking.

## Preparation (before your first question; ≤ 10 minutes, keep it to yourself)

Read `AGENTS.md`, `design/STATUS.md`, `design/reports/2026-11-15-run2.md` (and the run-1 report if
needed), `design/00-requirements-and-decisions.md` (D30–D43), `design/QUESTIONS.md` (section "Added
during run 2" only; everything before it is answered), and skim `design/40-roadmap.md`,
`design/14-imsrg3-plan.md` and `design/benchmarks/imsrg3-layout.md`. For every question on the
agenda, note in 2–3 sentences the context (file/section), the options, the assumption the run made,
and what would change under each answer. Look things up in the code only when needed to explain a
consequence. Run 2 did not finish IMSRG(3)-N7 and M12a at the time this agenda was written: read the
run-2 report for the final state and any question ids it adds (M10-Q50 and later, M12a-Q...), and
add them to the matching group below.

## How to run the interview (the method that worked on 2026-11-15)

* Use `AskUserQuestion` for decisions with clear options: **batches of up to 4 questions**, at most 4
  options each, your **recommendation first** and marked as such, the **concrete consequence of each
  option** in its description. Use plain conversation for open topics.
* **Process and tooling recommendations are accepted quickly; physics and code-quality points are
  where the owner overrides.** On 2026-11-15 the owner chose to fix imsrg++ quirks instead of
  reproducing them, refused compatibility switches, and required the Google style. So: batch the
  low-impact confirmations, and give the physics and code-quality items a focused question each.
  Present **physics trade-offs with concrete numeric consequences** (deviations, step counts,
  memory, timings: the numbers are in the report and in each question below).
* When no option fits, the owner **answers in precise free text: record those answers verbatim**
  (quoted) in `QUESTIONS.md`, then restate the decision in your words and confirm.
* **Do not pad questions.** Go from high to low impact, one or two sentences of context each. Do not
  re-ask anything already answered in `QUESTIONS.md`.
* **The owner may interject new requests mid-interview** (a new feature, a change of priority):
  treat them as part of the interview, record them, and fold them into the plan.
* Phrase every question so that a one-word answer is unambiguous. If an answer is ambiguous anyway
  (a plain "Yes" to an either/or question, as happened with Q5), ask back before recording it.
* If the owner wants to defer an item, record "deferred" with the reason and keep the assumption.
* Edit documentation only. Exception: if the owner explicitly asks for a small code change during
  the interview, make it on its own `feature/` branch with tests and merge it only after
  `tools/ci.sh dev-openmp` passes; larger work goes into the next-session plan.
* Work on a branch `feature/owner-answers-<date>` cut from `develop`; at the end,
  `tools/ci.sh dev-openmp` must pass, then merge into `develop` with `--no-ff`. If the session runs
  in an app worktree while `develop` is checked out in the main checkout, follow QUESTIONS O1/O3:
  merge on your worktree branch, then `tools/ci.sh dev-openmp && git update-ref refs/heads/develop
  <new> <old>` (never chain the update to a grep of the CI log). Git stays local: never push.

## Agenda, part 1: open questions of run 2 (ids from `design/QUESTIONS.md`)

Format: **id** — summary; *assumption taken*; what an answer changes.

**A. Owner facts and actions (ask first)**
* **O3** — run 2 started in an app worktree again (D31 asked for the main checkout), so the main
  checkout's files are stale. *Accepted as in run 1.* Ask: was the main checkout synced and
  `build/golden-generated` regenerated (`tools/reference/gen_golden_generated.sh`, M8-Q21)? Is the
  app setting that opens sessions in the main checkout (no worktree) available? Optional:
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`.
* **M10-Q2** — which E3max/dE3max does the owner use at emax 10–12, and how much memory per node?
  *Assumed: 4 nodes × 256 GB, at most 50 % for the 3B operators of an N7 step.* At E3max 14 layout A
  needs 38–40 GB per packed operator, at E3max 16 without a dE3max cut 157–177 GB, with dE3max 8
  4.4 GB. The layout choice holds in every case; only the node count and the MPI phase-2 design change.
* **Full N2LO 3N file** (D33): available yet? It would give full-flow goldens for IT-4/IT-5/IT-9.
* **MS-Q4** — `AUTHORS` names the owner as holder of "Copyright 2026 The imsrg authors"; SPDX says
  `GPL-3.0-or-later`. *Assumed as written.* Confirm holder list and "or later".

**B. Physics and numerics (high impact; focused questions)**
* **M8-Q10** — hunter-gatherer `H(s)` semantics: imsrg++ `GatherOmega` rebuilds the saved H from H0
  through the gathered Omega but leaves `H(s)` untouched, so `H(s)` and `Transform(H0)` differ by up
  to 2e-3 after a solve (O16 emax 2 test); stage 2's first eta uses the pre-gather `H(s)`. *Copied
  from imsrg++* (the IT-10 and hunter-gatherer goldens depend on it). Alternative: re-evaluate
  `H(s)` from the gatherer or gather before the next stage (a cleaner definition, new goldens, a
  deviation entry). Given D35 (fix quirks, no switches) this is the question most likely to be overridden.
* **M8-Q40** — flow `s` continuation: imsrg++'s adaptive `flow` integrates a copy of the solver, so s
  restarts at 0 in stage 2. *Implemented: s continues for `flow` and `flow_rk4`* (same results when
  stage 1 converged). Also (2) the new divergence guard (norm or E0 > 1e12/1e9 ends the flow), (4)
  hunter-gatherer with a flow method is a ConfigError, (7) no `shell-model-atan` cutoff switch in the
  Magnus path (a no-op at the default cutoff). Confirm all.
* **M8-Q20** — no charge-radius computation exists yet; D35 wants Rso added "wherever the code
  computes a charge radius". *Assumed: the output stage task implements charge radius = Rp2 + Rso/Z
  + nucleon-size terms (`PROTON_RCH2`, `NEUTRON_RCH2`, `DARWIN_FOLDY`, `16` §6).* Schedule it in run 3?
* **M10-Q20** — 3f2 2b intermediates are E-PLANNED (about 25 two-body operators of memory;
  imsrg++'s own default has both on, the owner's recipe D32 has 2b off). *Not scheduled.* Needed?
* **M10-Q21 / Q22 / Q23** — 3f2 details. Q21: `factorized` with `in_flow`/`at_end` is an error
  (V-57) instead of double counting. Q22: the correction needs an anti-Hermitian eta (imsrg++'s sign
  is "a guess" for the Hermitian case). Q23: the correction applies to every BCH call, including
  transformations of Rp2 etc. and the hunter-gatherer rebuild, as in imsrg++ (the IT-10 goldens
  contain it); should observables be transformed without it? *All assumed fine.*
* **M10-Q40** — Omega combination order in the perturbative triples for a split Omega stack:
  imsrg++ uses e^{Omega_0} e^{Omega_n}, the H transformation applies the oldest piece first.
  *Reproduced imsrg++* (identical for a single Omega, as in IT-8). A fix would be an old-code bug
  entry in `30` §6; under D35 the owner probably prefers the fix: ask, with the consequence (no
  change for the tests, which have a single Omega).
* **M10-Q43** — imsrg++ `comm223ss` returns early when ||X2||·||Y2|| < 1e-6; `Comm223` always
  computes. *Not reproduced* (differs only for a tiny nonzero product). Confirm.
* **M10-Q46** — perturbative triples use E3max = 3 emax as imsrg++; at emax 10–12 without a dE3max
  cut the triples' tables alone need tens of GB. *As imsrg++; triples at emax ≥ 8 would need R built
  per tile.* Are triples at emax ≥ 8 wanted (decides a follow-up task)?
* **M8-Q41** — `ode_tolerance` is rounded to float as in imsrg++ (1e-6 → 9.99999997e-7), contrary to
  the "fix: double" decision of Q14 for other parameters. *Assumed: kept for step-count identity.*
  One line to change.
* **M8-Q32** — `hartree_fock.mixing = "diis"` (the TOML default) changes E_HF at 1e-8 MeV and may
  change flow step counts versus imsrg++; step-count tests set `mixing = "linear"`. *Assumed.*
* **M8-Q12** — flow fixes: ds starts at min(ds_0, dsmax) without `magnus_adaptive`; NaN/inf is
  `Diverged` and leaves H as it is (imsrg++ multiplies by 1/0); `denominator_cutoff <= 0` is an
  error; `omega.storage = scratch|discard` E-PLANNED. *Implemented.* Confirm.
* **M7-Q10 / M7-Q11** — general fast paths follow the hermiticity flag, not the data (about 2× ladder
  and 4× particle-hole flops when flagged NonHermitian); the reference's general support rests on
  bilinearity, the m-scheme brute force stays on the M10 list. *Accepted.*

**C. Code style**
* **MS-Q1** — physics capitals kept in about 900 variable/parameter/member names (`X`, `Y`, `Z`, `H`,
  `J`, `Jp`, `Tz`, `hw_MeV`, `twoJ`), documented exception 6 in `17` §3; lower-casing would merge
  `j` and `J` in the same scope. *Kept.* Strict snake_case is a clang-tidy run away.
* **MS-Q2** — accessors (const methods returning a property or part of the object, possibly by index
  or key, e.g. `num_orbits()`, `orbit(i)`, `spec()`) and `set_*` mutators stay snake_case, everything
  else PascalCase (199-name allowlist in `.clang-tidy`). *Assumed.* Alternative: PascalCase for all
  methods (collides with type names such as `Orbit`, `Channel`, `Reference`).
* **MS-Q3** — header guards outside `src/` use cpplint's fallback without the `IMSRG_` prefix
  (`TESTS_UNIT_HF_HF_TEST_SUPPORT_H_`). *Fine.*
* **MS-Q5** — structural items left unconverted: long inline functions in headers, 36 structs with
  member functions, functions over ~40 lines, unsigned types. *Guidelines for new code; a cleanup
  task if wanted.*

**D. Configuration and outputs**
* **M8-Q30** — a default `output.directory` (no key in the file) and `--set` values resolve against
  the working directory, not the config directory. *Assumed.*
* **M8-Q31** — `results.json` `schema_version` 1 covers the run-2 table; bump only after a release.
  The perturbative-triples merge moved it to schema 2: confirm the policy.
* **M8-Q33** — NAT basis covers `natural_orbitals` defaults; `use_occupations` and `order = "energy"`
  stay E-PLANNED (M9-Q3, D36). *Assumed.*
* **M8-Q11** — gather counting in `SolveResult` and the `results.json` keys. *Assumed.*

**E. Python**
* **M11-Q11** — stale-view policy: `Auto` raises `StaleHostError` only where memory is not host
  accessible (GPU builds); `Strict` (`imsrg.strict_residency()` or `IMSRG_RESIDENCY=strict`) raises
  everywhere; `Implicit` never. *Assumed Auto default* (23 old tests read library views directly).
  Should the default be `Strict`?
* **M11-Q12** — `Runtime.finalize(force=True)` and the exit hook free device data of live operators
  and solvers but refuse while numpy views are alive. *Assumed.*
* **M11-Q13** — no stop request from a C++ solver status callback (an exception aborts at a step
  boundary); Python generator callbacks are not bound. *Assumed enough; a `flow` API addition if not.*
* **M11-Q14** — the bound API follows the C++ modules (`imsrg.io.read_me2j`, `hf.solve`,
  `flow.MagnusSolver`, `pipeline.run`) rather than the dataclass tree proposed in `18`; `opslib`
  bound too. *Assumed.* Update `18` or rework?

**F. IMSRG(3) engineering (batch; flag any to revisit)**
* **M10-Q1** — layout counts at emax 10–12 come from `--count-only` (enumeration only), covered by the
  emax rule as bookkeeping. **M10-Q3** — layout A with packed blocks (halves memory; unpacking cost
  4 % of a [2,2]→3 at emax 3). **M10-Q41/Q42** — `Comm223` per-tile gathers cost 1.7× the prototype
  at emax 3; locality at emax 12 is unmeasured (6j table 451 MB, strided reads); a timing-only run at
  emax 4–6 would settle it: *allowed?* **M10-Q44** — the kernel calls Kokkos directly until
  `exec::TeamFor` gets schedule/scratch/reducer options. **M10-Q45** — extra accessors
  (`ScratchBytesPerTeam()`, `impl()`) kept; the E map is built but unused by these kernels.
* Add the questions the N7 and M12a leads recorded at the end of run 2 (see the run-2 report).

**G. Low-impact confirmations (summarize one line each and ask the owner to flag any to revisit)**
M8-Q22 (merge resolution), M8-Q21 (resolved: gen scripts default to the patched harness), M11-Q10
(resolved at the merge), MS-Q3, M10-Q45, plus any later ids in `QUESTIONS.md` not listed above.

## Agenda, part 2: further design decisions (interview)

1. **Priorities for run 3**: the proposal is N7/IT-9 first, then M12a, M12b MPI, tensor operators,
   M13 only if all done (D39/D40). Is the order right now that 3f2 and the triples are done?
   Which IMSRG(3)-N7 terms and presets matter most, and what accuracy is expected at emax 2/3 (D34)?
2. **Tensor operators (D39, first extension)**: which operators and ranks first (E1/E2 transitions,
   M1, Gamow-Teller, 0νββ, dagger, DM), transformation only or also flow (M8-Q40: tensors cannot
   flow today), `ch_bra ≤ ch_ket` storage (D25), validation against imsrg++ (goldens needed from
   the patched build).
3. **N7 at scale and MPI (M12b)**: target E3max/dE3max, nodes and memory (M10-Q2), the schedulers
   and interconnect of the clusters, when the Hagen–Nam block-distribution evaluation should run
   and what scaling is "good enough"; when GPU work (M14, after M12) should start and on which
   machine first (H100/GH200, A100, MI250X/MI300A).
4. **Performance goals (M12a)**: imsrg++ parity is the first target (today 1.10–1.20× the time at
   emax 4/6); how much effort beyond parity; is the occasional emax-8 benchmark against imsrg++
   (one run, never in CI) enough?
5. **Outputs and workflows**: charge radius and other observables in the output stage (M8-Q20),
   me2jp/me3jp specifications (the owner supplies them), checkpoint/restart and Omega scratch
   storage (after M13), 3N inputs (me3j, Navratil/Miyagi operator files) and their order.
6. **Testing policy**: the `slow` label and its run before numerics merges worked; golden
   regeneration with a patched reference worked; tolerances for IMSRG(3) as D19/D34; how long may
   `ci.sh` get (75–90 s now, limit 5 min).
7. **Process**: feedback on run 2's orchestration (worktree start instead of the main checkout,
   out-of-repository preparation during exclusive tasks, usage pacing with the 5-hour windows,
   Opus reviews on numerics merges), the incidents (the whole-disk `find` and the macOS privacy
   prompt, golden rule 10; "Python quit unexpectedly" dialogs, lldb for crashes), the length and
   format of the report, review cadence, licensing and naming.

## Recording the answers

* In `design/QUESTIONS.md`: under each question, `**Answer (owner, <date>):** ...` (verbatim for
  free-text answers; or "deferred: <reason>").
* Decisions become rows in `design/00-requirements-and-decisions.md` with the next free D-number,
  marked *(user)*.
* Update the affected design docs (`13`, `14`, `16`, `17`, `18`, `20`, `30`, `40`) and the plan in
  `design/NEXT_SESSION_PROMPT.md` (priorities, inputs, the 3N file, the briefs).
* Write `design/reports/<date>-owner-interview.md`: decisions taken, changed priorities, follow-up
  tasks for the next unsupervised run.
* Finish with a short summary to the owner: what was decided, what was deferred, and whether the
  next unsupervised run is ready to start.
