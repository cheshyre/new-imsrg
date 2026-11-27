# Prompt for the 27Al weekend run (until Mon 2026-11-30 10:00 EST)

Planned in the owner interview of 2026-11-26/27 (part 1: D61-D71; part 2: D72-D78, report
`design/reports/2026-11-26-owner-interview.md` §5). The task, its variants and the compute estimate
are in `design/50-al27-responses.md` (§1 verbatim task, §5 compat profile, §7 plan and node-hours).
The run-4 prompt that preceded this one is in the git history of this file.

**Owner pre-flight (once, before launching):**

1. Start the session (Opus) on a Frontier login node in `$IMSRG_WORK/main`, on `develop`
   (`git pull --ff-only origin develop`). Permissions from `.claude/settings.local.json`; no
   `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; a permission mode that does not stall; connectors off.
2. No usage cap this weekend (D77): launch whenever ready.
3. Then paste everything below the line.

---

You are the **orchestrator** of the unsupervised **27Al weekend run** of the imsrg rewrite. It runs
from now until **Monday 2026-11-30 10:00 EST** (Knoxville; the owner is in Wako, JST = EST + 14 h,
and does not answer questions before then). Goal: reproduce and extend the owner's 2024
VS-IMSRG(2) 27Al dark-matter NREFT responses with the rewrite, in three variants, on Frontier GPUs
(D67, D76, D77). Use the whole window; do not finish early.

## Read first (delegate big reading; keep your own context small)

`AGENTS.md`, `CLAUDE.md`, `design/STATUS.md`, `design/50-al27-responses.md` (all of it),
`design/reports/2026-11-26-owner-interview.md` (§2-§5), D61-D78 in
`design/00-requirements-and-decisions.md`, the part-2 answers at the end of `design/QUESTIONS.md`,
`design/benchmarks/al27-production.md`, `design/benchmarks/gpu-m14.md`, `design/41-agent-workflow.md`
§2-3, `design/20-build-and-tooling.md` §9.

## The task (owner, verbatim; full text in design/50 §1)

> Reproduce the output .snt matrix elements to at least 1e-4 using the given code. If it requires
> undoing some modifications to the solver then implement these modifications. [...] In the case
> that significant solver changes were required to reproduce the given results, rerun the all the
> responses with the updated solver, still using the approach where the Magnus operator is not
> split. Document the differences. Finally rerun the full set of calculations with a split Magnus
> operator using a reasonable default value for the splitting (typically 0.25 for the
> omega_norm_threshold is sufficient). [...] We are running on Frontier so the calculations should
> use GPUs for expensive operations.

Tilde Phi': J = 2, 4 with √(4π) (design/50 §4). Scale (owner, verbatim, D77): "Not a must this
weekend. Run each variant as its own ~600-1,200-node job; document how 19,320 ranks would be
reached. Use the GPU resources efficiently with what you have available."

## Variants and acceptance (D76)

| Variant | Behaviour | Magnus | Acceptance |
|---|---|---|---|
| v1 | `run.compat = "imsrg_fork_79e9330"` (fork bugs emulated, domega 0.2, imsrg++ tensor norm) | unsplit | every archived `.snt` element to **1e-4 absolute** (flows already match README §3.3) |
| v2 | default behaviour: no fork emulation, domega 0.1, tensor norm (2j_p+1)(2j_q+1)/(2L+1) | unsplit | converged flows; difference report vs v1 |
| v3 | as v2 | split, `omega_norm_max = 0.25` | converged flows; difference report vs v2; elements above **1e-3 relative** investigated |

* **NAT orbitals are ordered by descending |occupation| in every variant** (owner, D72). Until now
  only the compat profile did this; Phase 0 makes it the default before any v2/v3 flow.
* **Difference reports** (owner, verbatim): "note both absolute and relative differences (ignoring
  matrix elements in the .snt that are very small, smaller than 1e-6)": per element |Δ| and
  |Δ|/|ref|, the relative one only for |ref| ≥ 1e-6, summarized per operator column and q, plus E0
  and the H `.snt`. File: `design/benchmarks/al27-variants.md`.
* A v1 column that misses 1e-4: one lead finds the cause (fork reference build at emax 4-6,
  per-term goldens) while everything else goes on; unresolved by Monday = recorded with evidence.
  Never accept "close enough"; never loosen a tolerance.
* Priorities: **v1 must be complete** by Monday; v2 next; v3 may be partial (the farm's failure
  manifest says what is missing).

## Rules for this run

* **27Al work only, no fillers** (D77). Decided items marked "later" in QUESTIONS.md (atan device
  tolerance, CPU preset switch, KK compile check, AL-B3, AL-A-Q1) wait until after the weekend.
* **No usage cap** (D77). At most **3 Opus leads** at once, each delegating code to Sonnet
  implementers (Haiku for mechanics). Every lead commits and pushes WIP to its hub branch at least
  every 30 minutes and before it ends a turn, and keeps a hand-off file
  `$IMSRG_REFERENCE_ROOT/plans/handoff-<task>.md`. After a usage-limit stop, start **fresh** leads
  from the hand-off files (cheaper than resuming long contexts).
* **emax rule:** tests stay at emax ≤ 6 (IMSRG(3) at 2-3). The 27Al production (D67) is the explicit
  exception: HF/NAT at emax 14, IMSRG(2) flows and the farm at emax 10, in Slurm jobs only.
* **Golden rule 10 (put it into every brief):** searches stay inside the checkout, `$IMSRG_WORK`,
  `$IMSRG_REFERENCE_ROOT`, `$IMSRG_DEPS_PREFIX` and scratch dirs; the login node only edits, runs git
  and short commands; builds, tests and runs go to compute nodes.
* **Never:** write, move or chmod anything in `/lustre/orion/nph123/proj-shared/nuccor_interactions`
  (read-only inputs); push anything but `develop` to GitHub; touch `main`; force-push; loosen a
  tolerance; commit `.snt` sets, Omegas or other generated data (> 1 MB).
* **Results:** `$IMSRG_WORK/runs/al27/<v1|v2|v3>/<Hamiltonian>/` (`.snt`, Omegas, logs, manifests;
  one tarball per variant at the end). Git gets only summaries (`design/benchmarks/al27-*.md`,
  < 100 kB each), comparison tools and Slurm scripts (`tools/al27/`).
* Budget: 50,000 node-hours (D65); estimate ~1,500-3,300 for the whole run (design/50 §7.1). Record
  every production job (id, nodes, elapsed, node-hours) in `design/benchmarks/al27-jobs.md`.

## Allocations (D77)

* **Development:** each active lead gets one node from a **self-resubmitting chain**: write first
  (orchestrator, before briefing leads) `tools/al27/alloc_chain.sh NAME`: an sbatch script
  (`-A nph123 -N 1 -t 02:00:00 -J NAME`) that sleeps, and ~10 min before its end submits its own
  successor with `--dependency=afterany:$SLURM_JOB_ID` under the same name, unless
  `$IMSRG_WORK/alloc/NAME.stop` exists. One chain `imsrg` for CI and the orchestrator, one per lead
  (`imsrg-a`, `imsrg-b`, ...). Chains then survive usage-limit stops. Stop chains that are no longer
  needed (stop file + `scancel`).
* **Production:** flows on one node each (one flow per GCD, up to 4 flows per node: 66 GB host peak
  each); farm jobs in the `batch` partition, one job per variant (v1: both Omega formats), walltime
  = 2x the Phase-0 estimate (≥ 184 nodes: 12 h limit). Pack several items per rank with the refit
  cost model (longest first) so GCDs do not idle; ~600-1,200 nodes per job. `extended` allows one
  running job per user.

## Where everything is

* Repositories (design/41 §2): hub `$IMSRG_HUB`; the orchestrator works in `$IMSRG_WORK/main` on
  `develop` (integration branch); code-writing agents get clones under `$IMSRG_WORK/agents/` from
  the WorktreeCreate hook. **Preflight:** `git status --porcelain --untracked-files=no` (empty),
  `git fetch origin && git merge --ff-only origin/develop`, `squeue -u $USER -n imsrg` (an
  allocation), `ls $IMSRG_WORK/golden-generated` (the emax-4 dump tests **skip silently** without
  it), then `tools/ci.sh frontier-cpu` (`CI PASS frontier-cpu`). If any check fails, write
  `PREFLIGHT_FAILED.md` and stop. If the `imsrg-*` agent types are unavailable, spawn
  `general-purpose` agents with the matching `model` and `isolation: "worktree"` and paste the
  body of the corresponding `.claude/agents/*.md` into their prompt.
* Read first: `AGENTS.md`, `CLAUDE.md`, `design/README.md`, `design/STATUS.md` (current state, log
  of every merge), the Frontier-port report (`design/reports/2026-11-26-frontier-port.md`), the
  run-3 report (`design/reports/2026-11-16-run3.md`) and the interview reports; the hand-off files
  `$IMSRG_REFERENCE_ROOT/plans/handoff-n7.md` and `handoff-m12b.md`,
  `design/00-requirements-and-decisions.md`, `design/40-roadmap.md`, `design/41-agent-workflow.md`.
  Leads read the other design docs and `design/reference/*.md` as their milestone requires; the
  orchestrator delegates big reading.
* Toolchain (design/20 §9): modules via `tools/lib/frontier_env.sh` (PrgEnv-gnu, gcc-native 14.2
  through the Cray `CC` wrapper, cray-mpich, cray-libsci, cmake 4.1.5, ninja, ccache, Python 3.14;
  rocm 7.0.2 for HIP); Kokkos 5.3.0 (openmp, serial, hip) and the shared venv (cpplint 2.0.2,
  clang-format 23.1.5, clang-tidy 23.1.4) in `$IMSRG_DEPS_PREFIX`; ccache in `$CCACHE_DIR`.
* Reference: source `$IMSRG_REFERENCE_ROOT/imsrg-devel` (read-only), SHA files and
  `golden-generated-d35` there too; builds (pristine and **D35-patched** imsrg++ and
  `harness-build-d35/refharness`, the default of every `tools/reference/gen_golden_*.sh`) under
  `$IMSRG_REFERENCE_BUILD` (`tools/reference/ref_env.sh`). Golden data: `tests/golden/` (README
  lists every file and command). After any change to the patch or to `tools/reference`, rebuild
  `harness-build-d35` and regenerate `$IMSRG_WORK/golden-generated`.
* Test inputs (never copy them into the repo), in `$IMSRG_WORK/inputs/`:
  * `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz`: `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`: `3bme_type=no2b file3e1max=6
    file3e2max=12 file3e3max=18 hw=16`. It holds **only the cE contact term** and stays the 3N test
    input (D33; the full N2LO file is still outstanding): E_HF(O16, emax 4) = −313.17 MeV is genuine
    and the full flow diverges, so IT-4/IT-5 compare HF plus the flow stopped at s = 10. O16 at emax 2
    is a known, genuine IMSRG(2) instability (IT-8 uses He4 emax 2 and O16 emax 3).

## State at hand-off (develop after the part-2 merge; details `design/STATUS.md`)

* Both v1 flows (N2LOsat, ΔN2LOGO(394)) reproduce README §3.3 to every printed digit; H `.snt`
  within 5e-7. Their Omegas (native float64 and ME2JP float32) and checkpoints are in
  `$IMSRG_WORK/runs/al27/v1/{N2LOsat,DN2LOGO}/`. Configs `tools/al27/{n2losat,dn2logo}_v1.toml`,
  scripts `tools/al27/` (`run_flow.sh`, `farm_sweep.sh`, `make_work_list.py`, `compare_snt.py`,
  `build_release.sh`).
* Merged: tensor operators (M15), NREFT library incl. FPhitp, fork compat profile, Omega I/O,
  checkpoints, the `transform` farm (Omegas in MPI shared memory, failure manifest), GPU path
  (rocBLAS device GEMMs and IMSRG(2) scalar and tensor kernels; Kokkos Kernels measured).
* Archive: `$IMSRG_WORK/inputs/al27/` (`README_al27_provenance.md`; `al27_provenance/original/`
  holds the archived `.snt` and logs). The six ΔN2LOGO `FM2p` values at q = 975-1050 MeV are
  placeholders in the archive (job J00011500 died): our `.snt` fill them.
* Not yet: GPU in the flow driver and the farm end to end; kokkos-hip has only a Serial host space;
  NAT |occupation| is compat-only; restarts redo the 16-min preparation; no farm run at emax 10;
  variants 2 and 3; C14 isospin example.

## Plan

Phase 0 starts at once with leads A and B in parallel; the orchestrator does its own Phase-0 tasks
meanwhile. Phases 2/3 flows start as soon as A and B are merged, overlapping the v1 farm.

### Brief A — GPU production path (Phase 0; branch `feature/al27-gpu-production`; lead, node `imsrg-a`)

Goal: flows and farm items run their expensive work on one GCD each.
1. `tools/bootstrap_deps.sh`: kokkos-hip with `Kokkos_ENABLE_OPENMP` as host space (AL-E-Q5, D73);
   `frontier-hip` uses it; host-parallel preparation then uses the rank's cores.
2. Flow driver: device path for the IMSRG(2) flow (scalar commutators, BCH, generator) with one
   flow per GCD; checkpoints and Omega files unchanged.
3. Farm: the device path inside `transform::TransformItem` (tensor and scalar BCH with stored
   Omegas); Omegas from the node's shared window, materialized one at a time on the device.
4. Acceptance: GPU vs CPU equality on an emax-4 flow (E0 and `.snt` ≤ 1e-10 relative) and on
   farm items of rank 0, 1, 5 (≤ 1e-12 relative); `tools/ci.sh frontier-cpu` and
   `tools/ci.sh frontier-hip-run` (the 5 known atan Bitwise failures excepted, AL-E-Q2).
5. Timing at emax 10 on one GCD: one item per rank L = 0..5 and TCM, from the v1 Omegas, both
   Omega formats; refit the farm cost model (AL-F3) and give the walltime per farm job; check
   which v1 J = 5 items hit imsrg++'s NaN norm (40-term BCH). Results in
   `design/benchmarks/al27-farm.md`.

### Brief B — defaults, restarts, farm log (Phase 0; branch `feature/al27-defaults-restart`; lead, node `imsrg-b`)

1. NAT order: `mbpt::NatSorting::kAbsDescending` is the default for every run; signed order only as
   an explicit option for imsrg-devel comparisons; test with a channel with a negative occupation;
   design/30 §6 row AL-NAT "validated by" filled in (D72).
2. Tensor one-body norm: default (2j_p+1)(2j_q+1)/(2L+1) per element; imsrg++'s integer formula only
   under `run.compat` (D72, row AL-A-Q6); unit tests of both; v1 results unchanged.
3. Checkpoints store H_0 (IMSRG basis) and the NAT coefficients; a restart skips readers, HF and NAT;
   checkpoint on SIGTERM (`--signal=B:TERM@300` in `tools/al27/run_flow.sh`); a killed-and-restarted
   flow is bit-identical (native) (D75).
4. Farm: per-item log line per rank (name, start, end, seconds, BCH terms, max Omega norm) (AL-F4);
   fork file names `<stem><name>.snt` under `run.compat` in driver and farm, `<stem>_<name>.snt`
   otherwise (AL-F2).
5. v2/v3 configs `tools/al27/{n2losat,dn2logo}_{v2,v3}.toml` (no compat, domega default 0.1; v3
   `omega_norm_max = 0.25`).
Acceptance: `tools/ci.sh frontier-cpu` (+ `ctest -L slow`); v1 emax-4 fork golden still passes.

### Orchestrator, Phase 0 (while A and B work)

* Write `tools/al27/alloc_chain.sh` and start the chains (see Allocations).
* After A and B are merged: farm the J = 0 and J = 1 v1 operators (FM0, FPhipp0, FSigmap1,
  FSigmapp1, FDelta1, p and n, all 71 q, both Hamiltonians: 1,420 items) on GPUs against the archive
  at 1e-4 ("the fastest benchmarks", owner); re-check the TCM `.snt` of both v1 flows (all 158
  archived two-body elements, fork tokyo format, AL-D3).
* Write the first interim report section (Phase 0) in `design/reports/2026-11-27-weekend.md`.

### Brief C — v1 farm (Phase 1; branch `feature/al27-v1-farm`; lead)

All 34 operators × 71 q + TCM for both Hamiltonians from the stored v1 Omegas, in both Omega
formats (native float64 and ME2JP float32, D69): one job of ~1,208 nodes or several smaller ones.
Acceptance: every archived `.snt` element to 1e-4 absolute (`tools/al27/compare_snt.py`), per column
max deviation in `design/benchmarks/al27-v1-farm.md`; native vs ME2JP differences per column;
FPhitp2/4 computed (no archive); the six ΔN2LOGO FM2p values at q ≥ 975 MeV produced; the C14
example (one nontrivial operator written with and without isospin averaging, difference
documented). Failures resubmitted from the manifest. A column that misses 1e-4 gets an
investigation lead (brief E).

### Brief D — variants 2 and 3 (Phases 2/3; branch `feature/al27-variants`; lead; starts after A + B merge)

1. Flows v2 and v3 for both Hamiltonians on GPUs (4 flows; one GCD each; checkpoints every 30 min,
   restarts from H_0). v2 and v3 share H_0 per Hamiltonian. Record stage ends (steps, s, E0, norms)
   and v3's number of Omegas per flow (first measurement: design/50 §6.1 expects 30-100).
2. Farms v2 and v3 (native Omegas), ~604 nodes each, items packed by the refit cost model.
3. A difference tool `tools/al27/diff_variants.py` (absolute and relative, |ref| ≥ 1e-6, per column
   and q) and the report `design/benchmarks/al27-variants.md`: v2 vs v1, v3 vs v2; for v3 vs v2
   elements above 1e-3 relative, find the cause in the worst columns or record it with evidence.

### Brief E — investigation (on demand; branch `feature/al27-investigate-<topic>`; lead)

For any v1 column that misses 1e-4 or any unexplained v3-vs-v2 deviation: reproduce at emax 4-6
against the fork reference build (`$IMSRG_REFERENCE_ROOT/imsrg-fork-79e9330`, harness
`tools/reference/gen_golden_*fork*.sh`), find the term, fix or emulate, document (design/30 §6 for
fork or devel bugs).

## Merging, CI, reporting

* Merge in `$IMSRG_WORK/main` on `develop` (design/41 §2): `git fetch origin`,
  `IMSRG_INTEGRATOR=1 git merge --no-ff --no-commit origin/feature/<x>`, `STATUS.md` log entry,
  commit, then one chained command: `tools/ci.sh frontier-cpu && <ctest -L slow for numerics> &&
  IMSRG_INTEGRATOR=1 git push origin develop && git push github develop`. GPU merges also run
  `tools/ci.sh frontier-hip-run`. CI FAIL: `git reset --keep` to the previous develop, fix on the
  branch. Retire merged clones by name (`tools/agent_retire.sh <name>`), never `--all` while agents run.
* One `imsrg-reviewer` (Opus) pass on every numerics merge (A, B, fixes from E).
* `design/STATUS.md` at every merge; an **interim report section per finished phase** in
  `design/reports/2026-11-27-weekend.md`; `develop` to the hub and GitHub after every CI PASS.
* Questions: append under "Added during the 27Al weekend" in `QUESTIONS.md`, ids `WE-<lead>-<n>`;
  choose the most reversible option, note it, continue. When stuck: park the item with a hand-off
  note and continue with the next phase item.

## Lessons (from runs 1-3 and the Al27 night)

* Three Opus leads exhaust a 5-hour window in ~2.5-3 h; leads delegate and read little themselves.
* The upkeep tick does not fire while the orchestrator is blocked: hence self-resubmitting chains.
* Agents cannot download from GitHub; compute nodes have no internet.
* `agent_retire.sh --all --merged` once deleted a fresh clone (fixed in 9fb645e): retire by name.
* Every brief carries the current `develop` hash, golden rule 10, the emax rule with the 27Al
  exception, the deadline, the license header rule, the branch name and the acceptance command.
* Merge shared interfaces early; debug crashes under gdb on the compute node; timings only on idle
  nodes, with the load recorded.
* When a subagent reports a numerical mismatch, do not accept "close enough": cause, or a
  question with evidence.

## Timeline and definition of done

* Leads report and push by **Monday 09:00 EST**; the orchestrator then merges, runs CI, writes the
  final report and pushes by **10:00 EST**. Write a draft report early as insurance.
* Done means: all finished work merged into `develop` with `CI PASS frontier-cpu`, pushed to the hub
  and GitHub; unfinished work on named branches with hand-off files; v1 `.snt` sets complete and
  compared (`design/benchmarks/al27-v1-farm.md`); v2/v3 as far as they got with difference reports
  (`design/benchmarks/al27-variants.md`); job and node-hour table (`al27-jobs.md`); one tarball per
  variant under `$IMSRG_WORK/runs/al27/`; the final report `design/reports/2026-11-27-weekend.md`
  (summary, validation table with actual deviations, timings, node-hours per phase, fork/devel bugs
  found, deviations from the plan, the path to 19,320 ranks, what is left); `STATUS.md` current;
  open questions in `QUESTIONS.md`; this file rewritten for the next session.
