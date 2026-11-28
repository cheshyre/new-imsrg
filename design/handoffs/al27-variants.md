# Hand-off: Lead D, 27Al weekend variants v2 and v3

(Kept in the branch: agents cannot write to `$IMSRG_REFERENCE_ROOT/plans/`.)

Branch: feature/al27-variants (hub), base develop 0024e6c. Node chain: imsrg-a.
Brief: design/NEXT_SESSION_PROMPT.md "Brief D" plus the orchestrator's brief of 2026-11-27 01:10.
Runs (never in git): `$IMSRG_WORK/runs/al27/{v2,v3}/{N2LOsat,DN2LOGO}/`, farms in `farm-native/`.

## Decisions
* All four flows (v2, v3 x N2LOsat, DN2LOGO) in ONE single-node chain job, one GCD each. v3 uses
  B's `checkpoint.prepared_from` = v2's checkpoint: it waits for v2's `prepared/` (same H_0 bit for
  bit, so v3 vs v2 isolates the split Magnus; no extra wall time, since v3 would otherwise redo the
  ~22 min preparation in parallel anyway).
* Device memory (checked 2026-11-27): `MagnusSolver` keeps every Omega as a device operator
  (`omegas_`, src/imsrg/flow/magnus.cc PushOmega; closed ones are only read again for the
  checkpoint/Omega files and the final operator transforms), 0.29 GB each at emax 10. Estimate of
  the v3 Omega count from the v1 GPU flow (N2LOsat): ||Omega|| grows monotonically to 2.5
  (stage 1, 37 steps) and 8.0 (stage 2, 272 steps); with domega 0.1 and omega_norm_max 0.25 a new
  Omega starts every ~3 domega-limited steps, so ~35-45 Omegas per flow, ~10-15 GB of the GCD's
  64 GB (plus ~3 GB for H, eta, h_saved, BCH temporaries). Spilling closed Omegas to the host is
  therefore not needed; the real v3 count and the device memory are measured in the chain (VRAM log)
  and the decision is revisited above ~120 Omegas.

## Tasks
- [x] (a) tools/al27/diff_variants.py + tests (merged from feature/al27-variants-diff 6318a22)
- [x] (b) tools/al27/flows_chain.sh + flow_summary.py (merged from feature/al27-variants-chain bc154cf; mock chain t4 OK)
- [x] B (origin/feature/al27-defaults-restart 90f59c6) merged into this branch (orchestrator: develop 4583b44 = B, review running); frontier-hip built in this clone (build/frontier-hip, 2026-11-27 18:2x); the 4 v2/v3 configs pass `--validate`; chain fixed (empty argument) and re-tested (t5)
- [x] orchestrator "go" (18:32 EST Fri): chain submitted, link 1 job 6015996 (frontier00102, started 18:33), link 2 6015997 pending (afterany); chain script runs/al27/v2/N2LOsat/chain-20261127-183256.sh; binary = this clone's build/frontier-hip at 42a46f7 (contains B 90f59c6); develop to be merged later
- [x] v2 + v3 flows (job 6015996, 47 min, all converged in link 1; flow_summary.md in each run dir)

| flow | stage 1 steps / s / E0 | stage 2 steps / s / E0 | Omegas | wall |
|---|---|---|---:|---|
| v1 N2LOsat (ref) | 37 / 13.322 / -131.499115 | 272 / 57.655 / -213.961460 | 2 | |
| v2 N2LOsat | 54 / 13.315 / -131.498815 | 489 / 57.914 / -213.956872 | 2 | 46:52 |
| v3 N2LOsat | 44 / 13.049 / -131.608497 | 146 / 48.854 / -214.428763 | 42 | 11:35 (+prep of v2) |
| v1 DN2LOGO (ref) | 30 / 11.306 / -157.357310 | 253 / 47.146 / -217.066365 | 2 | |
| v2 DN2LOGO | 43 / 11.170 / -157.356574 | 465 / 47.305 / -217.070296 | 2 | 46:14 |
| v3 DN2LOGO | 36 / 10.431 / -157.252591 | 135 / 41.727 / -216.987333 | 40 | 11:17 |

  (steps = last step index = flow-file rows - 1 per stage; the first draft of this table
  listed the row counts, one more.) v3 - v2: -0.472 MeV (N2LOsat), +0.083 MeV (DN2LOGO).
  Note: AdaptiveStep ds = min(domega/eta, domega/(eta |Omega|), ...): v3's small |Omega| gives
  up to |Omega| (~8) times larger steps than v2 -> scan 6016711/6016712 separates the step-size
  effect (domega 0.05, 0.025 at split 0.25), the split threshold (1.0) and v2's own step
  convergence (unsplit, domega 0.05). Configs: tools/al27/make_scan_configs.py, copies in
  runs/al27/scan/configs.
- [x] v2 farms (job 6016703, 30 nodes, 19.6 min, 0 failed); v2 vs v1 in design/benchmarks/al27-variants.md section 4,
  decomposition farm 6016893 (v2 Omegas + fork operator behaviour, runs/al27/v2/<H>-compatop/)
- [x] v3-vs-v2 E0 investigation: scan + ODE flows (runs/al27/scan/), report section 3: split converges to the ODE
  IMSRG(2) result; the v3-v2 gap is the unsplit Magnus(2) truncation
- Report sources: generated sections in runs/al27/variants-md/*.md (diff_variants.py output); the document is
  assembled from a template by a scratch builder (placeholders SCANTABLE, V2V1TABLES, DECOMPTABLES, V3V2TABLES);
  if the builder is lost, edit design/benchmarks/al27-variants.md directly
- [x] v3 farms (job 6016892, 80 nodes, 40 min, 0 failed); v3 vs v2 in al27-variants.md section 5 with the operator
  investigation (split-threshold/step checks 6017305 + 6017358, ODE flow with FM0p_150/FPhipp0p_0): cause = the
  Magnus(2)-vs-IMSRG(2) truncation of v2; v3 converged in threshold (few 1e-4), time-step difference (domega 0.1 vs 0.025) 0.09-1.2 % (median)
- [x] design/benchmarks/al27-variants.md (78 kB)

## Exit codes (checked in B's branch)
* SIGTERM -> flag; after the next Magnus step `Run` writes a checkpoint and throws
  `InterruptedError`; apps/imsrg/main.cc returns `kInterruptedExitStatus` = 75
  (src/imsrg/driver/termination.h). The chain's `st == 75` branch is right; no change to B.
* TERM during the preparation or the final writers/transforms is not acted on (the flag is read
  only between flow steps): the step is killed at the walltime; the chain counts no error
  (`stop` flag) and the next link restarts from the last checkpoint / prepared/.
* `checkpoint.restart = true` with only `prepared/` (no state.txt) starts the flow from the
  prepared H_0 (run.cc ~1058); prepared.toml is written last (readiness marker).

## Launch (after B is merged and develop merged here; frontier-hip build)
```
tools/al27/flows_chain.sh tools/al27/n2losat_v2.toml:$IMSRG_WORK/runs/al27/v2/N2LOsat \
  tools/al27/dn2logo_v2.toml:$IMSRG_WORK/runs/al27/v2/DN2LOGO \
  tools/al27/n2losat_v3.toml:$IMSRG_WORK/runs/al27/v3/N2LOsat \
  tools/al27/dn2logo_v3.toml:$IMSRG_WORK/runs/al27/v3/DN2LOGO
```

## Jobs
See design/benchmarks/al27-jobs.md, section Lead D.

## Review fixes (feature/al27-variants-fixes)
* `tools/al27/diff_variants.py`: the per-column header (`\|ref\|`) was unescaped, so no per-column
  table rendered; fixed, tables now also list element counts, rel-eligible counts and the denominator
  of the |ref| >= 1e-3 count, per column, per q and per part; a test checks cell counts of every table.
* `tools/al27/flow_summary.py`: steps = last istep for Magnus (not the row count), ODE stages (istep
  stays 0) are kept row by row and reported as "N rows (ODE)".
* `tools/al27/dominant_rel.py` (was the untested scratch `variants-md/domrel*.py`): dominant elements
  now need |ref| >= 1e-6 as well; operators vanishing in the reference are reported as "vanishes".
* `tools/al27/splice_tables.py` re-inserts regenerated tables into the report (headings stay).
* Regeneration: `diff_variants.py` for the four comparisons and the two decomposition farms,
  `dominant_rel.py --part all|1b` for 5.1; the "Flow (results.json, flow.txt)" parts of the report
  were kept (v3 N2LOsat flow.txt was overwritten on 2026-11-27 21:09; its numbers are from results.json).
