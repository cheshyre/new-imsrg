# Interview prompt: plan the 27Al weekend session (part 2 of the 2026-11-26 interview)

You will interview the owner of the imsrg rewrite (C++20/Kokkos/MPI replacement of imsrg++, on OLCF
Frontier, project nph123) to plan the **unsupervised weekend session that runs until Monday
2026-11-30, 10:00 Knoxville time (EST)**. The owner is in Wako (JST = EST + 14 h). Part 1 of the
interview (2026-11-26) decided everything else (D61-D71); the night of 2026-11-26 built all the
capabilities. Your job: settle the remaining decisions, develop the weekend plan with the owner, get
an explicit OK, and write it as `design/NEXT_SESSION_PROMPT.md`.

## Preparation (<= 10 minutes, keep it to yourself; delegate big reading to a read-only subagent)

Read `AGENTS.md`, `CLAUDE.md`, `design/STATUS.md`, `design/reports/2026-11-26-owner-interview.md`
(decisions and the state after the night: §2-§4), `design/50-al27-responses.md` (the task verbatim,
variants, capabilities C1-C14, compute), D61-D71 in `design/00-requirements-and-decisions.md`,
`design/benchmarks/al27-production.md` and `design/benchmarks/gpu-m14.md`, the section "Added during
the Al27 night (2026-11-26)" of `design/QUESTIONS.md` (AL-A*, AL-B*, AL-C*, AL-D*, AL-E*, AL-F*), the
current `design/NEXT_SESSION_PROMPT.md` (run-4 plan written before the task; keep its Frontier
sections), `design/41-agent-workflow.md` §2-3, `design/20-build-and-tooling.md` §9.

State (develop f2c5fae, 2026-11-26 20:00 EST): both 27Al flows (N2LOsat, ΔN2LOGO; variant 1,
compat profile) reproduce README §3.3 to every digit, H `.snt` within 5e-7; Omegas of both flows in
both formats in `$IMSRG_WORK/runs/al27/v1/`. Merged: HF-reference NAT, truncation, Omega I/O
(native float64 + ME2JP float32), checkpoints, tensor operators (M15), fork reference build and
goldens, NREFT operators incl. FPhitp (√(4π), J = 2, 4), fork compat profile (three fork bugs
emulated: tensor normal ordering, odd-L ph phases, plus writer details), the `transform` farm
(Omegas in MPI shared memory), the GPU path (rocBLAS; Kokkos Kernels built and measured).
Not yet: GPU wired end to end into the flow driver and the farm; any farm run at emax 10; variants
2 and 3; the isospin-averaging example (C14); the DIIS change (D64); the hub-branch clean-up script
(D61); AL-D1 (each restart re-runs the 17-min setup).

## Method (worked in earlier interviews)

* `AskUserQuestion` in batches of up to 4, at most 4 options, **recommendation first and marked**,
  the concrete consequence of each option in its description (numbers: deviations, node-hours, wall
  time). Process questions are accepted quickly; physics and code quality get a focused question each.
* When no option fits, the owner answers in precise free text: record it **verbatim** in
  `QUESTIONS.md`, restate it, confirm. Ask back when an answer is ambiguous. Do not re-ask decided items.
* Show a progress bar after every round (owner preference).
* Edit documentation only; a small code change the owner asks for goes to an `imsrg-implementer`
  (own clone, `isolation: worktree`) and is merged after `tools/ci.sh frontier-cpu` passes.
* Git: branch `feature/owner-answers-2026-11-27` from develop in `$IMSRG_WORK/main`, commits with
  `IMSRG_INTEGRATOR=1`; at the end merge into develop (`--no-ff`), `tools/ci.sh frontier-cpu` on the
  compute node of the allocation named `imsrg` (`squeue -u $USER -n imsrg`; if none:
  `sbatch -A nph123 -N 1 -t 02:00:00 -J imsrg --wrap "sleep 7190"`), and only on `CI PASS`:
  `IMSRG_INTEGRATOR=1 git push origin develop`, then `git push github develop`.

## Agenda

### A. Open decisions (from the night; ask first, high impact first)

1. **AL-A-Q6, imsrg++'s tensor one-body norm**: the integer degeneracy can be negative at emax 10 for
   rank 5 (2j_p = 21, 2j_q = 11, L = 5 → −22), the norm becomes NaN and imsrg++'s BCH runs to 40
   nested commutators (possibly the ~210 "BCH not converged after 12" warnings per archived log).
   Recommendation: emulate in variant 1 (compat), a physically sensible norm in variants 2/3; check
   which archived J=5 operators hit it.
2. **GPU BLAS backend (D70 rule "5x or more")**: rocBLAS vs Kokkos Kernels on one GCD: scalar emax 8
   4.6x, scalar emax 10 6.3x, tensor rank 5 emax 10 3.6x (the farm is dominated by tensors).
   Borderline; owner decides the default.
3. **AL-E-Q2**: five bitwise 3B-generator tests fail on the GPU (device vs host atan, 4e-16): a
   tolerance for the device build, or one shared atan implementation?
4. **Fork tensor goldens** add 1.7 MB to `tests/golden/` (each file < 1 MB): keep or trim?
5. The other AL-* questions in one confirmation batch (AL-C1..C5, AL-A-Q1..Q5, AL-B2..B4,
   AL-D1..D3, AL-E-Q1/Q3..Q5, AL-F2..F4), with AL-D1 (store H_0 in checkpoints to skip the 17-min
   setup on restart) and AL-E-Q3 (serial libsci + `IMSRG_BLAS_CONCURRENT` 6-7x faster on the CPU,
   relates to FP-Q6/D63) as the two with consequences.

### B. The weekend plan (develop it with the owner)

Propose, then revise:

1. **Phase 0, shake-down (1-2 h, one lead)**: GPU in the flow driver and the farm end to end; one
   farm item at emax 10 timed; the J=0 and J=1 operators of variant 1 (owner: the fastest benchmarks)
   compared against the archive (`$IMSRG_WORK/inputs/al27/al27_provenance/original/...`, 1e-4).
2. **Phase 1, variant 1 farm**: all 34 operators × 71 q for both Hamiltonians (~4,800 items, one GCD
   each, ~600 nodes) from the stored Omegas, native and ME2JP Omegas compared (D69); acceptance:
   every archived `.snt` to 1e-4; FPhitp2/4 computed; the six ΔN2LOGO FM2p placeholders (q ≥ 975)
   filled; the isospin-averaging example (C14).
3. **Phase 2, variant 2** (fixed solver, unsplit; the compat profile counts as "significant solver
   changes"): flows on GPU, farm, a difference report per column.
4. **Phase 3, variant 3** (split Magnus, omega_norm_max 0.25; ~30 Omegas per flow): flows, farm,
   difference report.
5. **Fillers while farm jobs queue**: the run-4 order (D66: M12b → M14 → M16 → M15 → N7 → fillers;
   much of M14/M15/M16 is done), D64 DIIS change, hub-branch clean-up script, AL-D1.

Ask about: the order and what may stay half-done; acceptance numbers per phase; whether results
(`.snt`, comparison tables) are committed (small summaries only) or stay in `$IMSRG_WORK/runs/al27/`;
anything the run must never do.

### C. Logistics (recommend from the night)

* **Usage**: three Opus leads hit the 5-hour limit after ~2.5-3 h, three times on 2026-11-26. Cap
  rule for the weekend (D58's 90 %, or no cap as on the night), the number of parallel leads, and
  what the orchestrator does at a limit stop (leads commit/push; resume after the reset).
* **Allocations**: the upkeep tick does not fire while the orchestrator is blocked, so chains must
  sustain themselves (several successors queued ahead, or a self-resubmitting job script);
  `extended` allows one running job per user; farm jobs of 184+ nodes get 12 h; budget 50,000
  node-hours (D65): estimate per phase.
* Reporting (STATUS.md, an interim report per phase), pushes of develop to GitHub during the run
  (after CI), what to do when stuck (park the item, continue with the next phase or fillers).

### D. Write it down

* Rewrite `design/NEXT_SESSION_PROMPT.md` as the weekend prompt: title, owner pre-flight, usage and
  allocation rules, the phases as lead briefs (goal, acceptance, docs, files, branch names,
  dependencies, at most N leads), definition of done, keep the Frontier sections.
* Update `design/50` (plan, compute per phase), `design/40-roadmap.md` (the task as a milestone),
  D-rows (next free number after D71), answers in `QUESTIONS.md`, and a short part-2 section in
  `design/reports/2026-11-26-owner-interview.md`.
* Merge and push as above. Finish with a short summary: decisions, the plan in a few lines, and
  what the owner must do before launching the weekend session.
