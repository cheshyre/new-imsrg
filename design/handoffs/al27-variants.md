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
- [ ] orchestrator "go" -> merge develop -> submit the chain (command below)
- [ ] v2 + v3 flows (chain job)
- [ ] v2 farms (both H), compare v2 vs v1
- [ ] v3 farms, compare v3 vs v2, investigate > 1e-3 relative
- [ ] design/benchmarks/al27-variants.md

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
