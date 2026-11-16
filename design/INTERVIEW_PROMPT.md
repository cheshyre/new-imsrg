# Prompt for the owner Q&A and design-interview session (after run 3)

> **Status:** written at the end of run 3 (2026-11-16/17) for the interview that follows it. The
> interviews so far: 2026-11-13 (D25-D29), 2026-11-15 (D30-D42), 2026-11-16 (D44-D54, report
> `design/reports/2026-11-16-owner-interview.md`). Run 3 added D55-D60 (check `00` for the next free
> number). Every run-3 question below carries the assumption the run made, so the owner can
> answer with one word. Run 3's report: `design/reports/2026-11-16-run3.md`; `develop` = 7a78f4d.
> Method additions kept from the last interview: show the owner a progress bar of the agenda after
> each round (owner request), and measure a numeric consequence live (a worker in a throwaway
> worktree) when a decision hinges on it.

**How to use:** after syncing the main checkout (QUESTIONS O1/O3), start an **interactive** Claude
Code session (Opus) in `/Users/cheshyre/Downloads/new-imsrg` on `develop`, and paste everything
below the line. Plan 60-90 minutes. The session only edits documentation; the next unsupervised
run (`design/NEXT_SESSION_PROMPT.md`, run 4) builds on what it records.

---

You will interview the owner of the imsrg rewrite (a modern C++20/Kokkos/MPI replacement of
imsrg++). Two goals:

1. **Resolve the open questions** recorded during run 3 (2026-11-16/17) in `design/QUESTIONS.md`
   ("Added during run 3"). Each one says which assumption the run made.
2. **Decide the process and design points** needed for run 4 (the order of the work, the usage
   rules after the cap breach, the MPI imbalance at 256 ranks), so run 4 can work without blocking.

## Preparation (before your first question; <= 10 minutes, keep it to yourself)

Read `AGENTS.md`, `design/STATUS.md`, `design/reports/2026-11-16-run3.md`,
`design/00-requirements-and-decisions.md` (D55-D60), `design/QUESTIONS.md` (section "Added during
run 3" only; everything before it is answered), and skim `design/40-roadmap.md`,
`design/benchmarks/imsrg3-n7-performance.md` and `design/benchmarks/mpi-scaling.md`. For every
question on the agenda, note in 2-3 sentences the context (file/section), the options, the
assumption the run made, and what would change under each answer. Look things up in the code only
when needed to explain a consequence.

## How to run the interview (the method that worked on 2026-11-15 and 2026-11-16)

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

## Agenda, part 1: open questions of run 3 (ids from `design/QUESTIONS.md`)

Format: **id** - summary; *assumption taken*; what an answer changes.

**A. Owner actions and facts (ask first)**
* **Main checkout:** was it re-synced (`git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` when it
  has no local edits) and `tools/reference/gen_golden_generated.sh` run there? (QUESTIONS O1/O3.)
* **Optional end-to-end check of D57:** does imsrg++ read a `.me2j.bin` written by our rule (255-byte
  header, float32)? A. Tichai's reader was the source of the rule. Our reader is merged; the
  converter `tools/reference/me2j_text_to_bin.py` and the refharness make the check easy.
* **Full N2LO 3N file** (D33): available yet? It would give full-flow goldens for IT-4/IT-5/IT-9.

**B. Flow and IMSRG(3) (IT3; high impact, focused questions)**
* **IT3-Q2** - after the final hunter-gatherer gather the rebuilt H(s) has ||eta|| 2000-5000 times
  `eta_criterion` (E0 8-43 keV from the last hunter line). *Assumed: the flow does not continue; a
  warning is logged, eta and `norm_eta` describe the rebuilt H, one extra flow row.* Alternative:
  continue until the rebuilt H converges (changes step counts and goldens).
* **IT3-Q7 / Q8 / Q9** - the three-body denominator floor (D21 form, same cutoff as 1B/2B) and the
  split Omega stack order `BchProduct(Omega_n, omega)` in the perturbative triples. *Assumed: one
  cutoff for 1B/2B/3B; Magnus applies Omega_0 first; BCH keeps imsrg++'s truncation.* The premise
  "IT-8 has a single Omega" was wrong: IT-8 triples dE moved by -3.9e-4 (He4 emax 2) and -2.0e-3 MeV
  (O16 emax 3), beyond the 1e-4 tolerance; goldens regenerated from the patched reference. Is a
  split-flow end-to-end triples test (split vs single Omega) wanted? Separate 3B cutoff wanted?
* **IT3-Q3** - the patched imsrg++ adaptive `flow` restarts s at 0 in stage 2 (ours continues).
  *Assumed: no change; only final quantities are compared.*
* **IT3-Q1** - goldens whose numbers did not change keep older reference-build stamps
  (`cb9d11ba30f579d4`, `5fdd2b73265417f2`). *Assumed: fine (listed as equivalent in
  `tests/golden/README.md`).* Alternative: regenerate every golden once.
* **IT3-Q4** - strict Python residency in CI. *Assumed: nothing else; a GPU build still needs a real GPU
  run.* **IT3-Q5** - physics-symbol list (159 names; `iJ`, `nA`, `Hh`, `Lam` counted as physics
  notation). *Assumed: kept;* alternative: reject the prefixed forms (about 20 names shrink, renames).
  **IT3-Q6** - `design/18` rewritten to the bound API, old pyIMSRG mapping table dropped.
  *Assumed: dropped;* keep it for the term-by-term milestone?
* **IT3-Q10, Q11, Q12** - confirmations: `ode_tolerance` double-ness not testable separately (covered
  by construction); IT-8/emax2 stamp note; Ctrl-C test children restore the default SIGINT. *Assumed: fine.*

**C. Formats (O4)**
* **O4** - the binary ME2JP/ME3JP variant (D57) stores float32 after a 255-byte header, so it cannot
  carry bit-exact checkpoints or stored Omegas (D54 wants restart bit for bit). *Assumed: `.bin` is
  for interactions and other inputs; checkpoints and Omegas are text or gzip.* Alternative: an extra
  double-precision binary variant (`.bin64` or `prec=64` in the header; jIMSRG would not read it).

**D. Style (MS2)**
* **MS2-Q1** - 12 structs with member functions became classes; the device views and kernel data
  bundles (`MatrixView`, `TwoBodyView`, `PandyaData`, `LadderData`, ...) stay structs (about 150
  sites). *Assumed: stay structs;* alternative: convert them now that the N7/M12b restructuring is
  merged.
* **MS2-Q2** - scoped enums use `std::int8_t` (14 of them), NO2B monopole orbit indices
  `std::int16_t`; unsigned only for bit patterns and modular arithmetic. *Assumed: yes;* a one-line
  revert per enum if unsigned underlying types should be allowed.
* **MS2-Q3 / Q4** - the perturbative-triples energy differs 1-2 ulp run to run (reduction order);
  parameterized test names embed heap pointers; the reference harness emax-4 HF/HNO output is not
  bitwise reproducible. *Assumed: harmless;* a deterministic triples reduction could be added
  with the tiled triples.

**E. Tensor operators (M15), needed before M15 starts**
* **M15-Q1** (decide this one) - phase/gauge of reduced matrix elements: imsrg++ stores
  `(-1)^{j_bra-j_ket}` times the Edmonds element (`(-1)^{j_bra-j_ket+lambda}` for E-multipoles).
  *Assumed: keep imsrg++'s gauge internally* (the 6j/9j phases of `TensorCommutators.cc` were
  derived for it); Edmonds only with conversion at the edges. Needed before the tensor
  `OperatorDump` v2.
* **M15-Q2** - dump v2 stores only `ch_bra <= ch_ket` blocks plus the Hermiticity rule; whether
  Delta Tz > 0 operators are stored as `O + O^dagger` (imsrg++) or one-directional. *Assumed: as
  imsrg++.* **M15-Q3** - J=0 isospin-changing scalars (Fermi, 0nubb, DGT) stored reduced like J>0.
  *Assumed: yes;* the scalar-scalar commutator with them needs its own reference.
* **M15-Q4** - one-body Y exercises only comm111st/121st/122st; the 222/221 tensor terms are
  goldened only with seeded random tensors at emax 1 (size limit). *Assumed: enough for the first
  milestone.* **M15-Q5** - the 0nubb operators and the Magnus golden (one BCH step with a random Omega
  of norm 0.3). *Assumed: acceptable;* a flow-produced Omega needs a harness hook.

**F. MPI (M12b)**
* **M12b-Q1** - the communicator lives in `WorkspaceOptions::comm` (phase 1) and the model space's
  `ThreeBodyDistribution` (edge `basis -> comm`). *Assumed: yes* (alternative `operator -> comm`).
* **M12b-Q2** - combine by reduce + broadcast (identical bits), not `Allreduce`. *Assumed: yes.*
* **M12b-Q3 / Q18** - the ladder is a contiguous cost-balanced row cut; comm122 and the
  particle-hole terms go by whole channels (ph inverse transform only partly distributed).
  *Assumed: acceptable now; row panels are the first task of run 4* (imbalance 310-880 % at 256 ranks).
* **M12b-Q4, Q5** - cheap terms, the 3f2 correction and N7 on rank 0 until the distributed N7
  driver (D10). *Assumed: yes.* **M12b-Q6** - every rank reads the interaction files. *Assumed: yes;*
  root read + broadcast if I/O contention shows on a cluster.
* **M12b-Q7, Q16** - output only on rank 0 (warnings of other ranks are lost); `results.json`
  records no rank/thread count. *Assumed: fine for now.*
* **M12b-Q8** - distributed 3B parts are packed only. *Assumed: yes.* **M12b-Q9** - lockstep
  `Alltoallv` rounds instead of one-sided `MPI_Get`. *Assumed: rounds* (deterministic, portable to
  InfiniBand and Slingshot). **M12b-Q10** - partition weights from the cost model with measured
  w_122 = 59, w_ph = 2.9 (measured under load). *Assumed: recalibrate on an idle machine.*
* **M12b-Q11, Q17** - one rank per node with OpenMP threads; on one node more ranks are slower (emax 8:
  30.5 / 43.5 / 61.8 ms for 1 / 2 / 4 ranks at 16 threads in total). *Assumed: MPI across nodes
  only, no warning;* one rank per GPU in M14. **M12b-Q12** - device-aware MPI only in M14.
  **M12b-Q13** - Python bindings stay serial. **M12b-Q14** - config key `[parallel]
  three_body_distribution` ("rows" default with several ranks, "replicated" for tests), in phase 2.
* **M12b-Q15** - IT-11 tolerance max|d| <= 1e-12 max(1, max|ref|), identical step counts.
  **M12b-Q19** - `CheckReplicatedSetup` requires equal thread counts on all ranks, so heterogeneous
  nodes fail. *Assumed: keep* (bitwise identity needs the same reduction order).

**G. N7 performance (M10-Q70..Q79; the rest of the Q70-Q79 are taken as proposed, confirm in one go)**
* **M10-Q74** (open) - batch budget `kN7BatchBytes` = 512 MiB bounds H + W of one batch;
  `CommutatorOptions::imsrg3_batch_bytes` not added yet (it changes the `ThreeBodyTerms()`
  signature, which collided with M12b). *Assumed: add after the M12b merge (one-line plumbing).*
* **M10-Q75** (open) - `TriplesResult::kept_rows` / `total_kets` now count the kets of the triples
  the context holds (hhh/ppp of channels with an hhh triple), before and after the dE3max/OccNat3
  cuts; the driver log text still reads "out of". *Assumed: fine;* the duplicated table builders
  can be factored later.
* **M10-Q70..Q73, Q76..Q79** (answered in `QUESTIONS.md`; confirm): a row is one stored segment
  z(m >= k, k), storage unchanged (Q70); line attribution, the zipper `RowPartition::Balanced`
  (Q71); 133/331/231/132 on the intermediates H and W, element-wise `Comm<term>ssReference` kept for
  tests (Q72); one fused `RunN7` loop, `terms::` functions rebuild H/W per call (Q73);
  `imsrg3.occ_nat3_cut` uses the MBPT(2) natural occupations (Q76); the `exec::TeamFor`/`TeamReduce`
  facade (Q77); `WorkspaceOptions::three_body_context` kAuto/kSkip (Q78); the fused driver differs
  from the per-term sequence by 8e-17 at emax 3 (Q79).

## Agenda, part 2: decisions for the owner

1. **Order of run 4.** Proposal: (1) M12b phase 2 completion (row panels for comm122/ph, D9, D10 and
   lifting the D7 guard, D11, D12, weight recalibration); (2) N7 follow-ups (tiled `BuildH`, C15,
   partition timings, `imsrg3_batch_bytes`); (3) M15 tensor operators after M15-Q1..Q5 (goldens
   exist); (4) M16 formats; (5) fillers (Python hooks D51, run-2 section 7 cleanup). Is the order right? Should M15 go before M12b phase 2?
2. **Pause and cap rules after the breach.** In run 3 an N7 lead ignored the 21:35 pause message and
   the window reached 97 % at 22:00 against the 90 % cap (D58); the lead was stopped at 22:01.
   Should the rules be stricter (pause message earlier, a mandatory TaskStop backstop for leads,
   a lower cap for the last hour before a deadline)? Which cap should apply in run 4?
3. **Weekly usage budget per run.** Run 3 used about 17 points of the weekly limit (44 % -> 61 %).
   What weekly ceiling should run 4 stay under, and should the run stop when it is reached?
4. **The D45 imbalance at 256 ranks**, and the measured N7 row partition: on a quiet machine the
   balanced N7 partition reached max/mean 1.16 / 1.31 / 1.46 at P = 4 / 16 / 64 (the R5 model
   predicted 1.02-1.04), so the row-cost model needs per-row terms before phase 2 relies on it.
   The IMSRG(2) side: The cost model says the compute imbalance is 0-0.1 % at 22
   ranks, 3 % at 64 ranks only at emax 12 (58 % at emax 10, 146 % at emax 8) and 310-880 % at 256
   ranks, because comm122/ph go by whole channels. Is "row panels for comm122/ph first" the right
   priority, and are 22/64/256 ranks the right targets (which emax and E3max matter at 256 ranks)?
5. **Process feedback on run 3:** fresh leads at each reset with hand-off notes, early merges of
   shared interfaces, planner and plan review outside the repository, orchestrator-run slow suites.
   Anything to change? (Details: run-3 report section 6.)

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
