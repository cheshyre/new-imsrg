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
* Device memory risk for v3: every Omega stays a device operator in `MagnusSolver::omegas_`
  (0.29 GB at emax 10); 100 Omegas ~ 29 GB of the GCD's 64 GB.

## Tasks
- [ ] (a) tools/al27/diff_variants.py + tests (worker, branch feature/al27-variants-diff)
- [ ] (b) tools/al27/flows_chain.sh + flow_summary.py (worker, branch feature/al27-variants-chain)
- [ ] wait for B merged into develop; merge develop; build frontier-hip release binary
- [ ] v2 + v3 flows (chain job)
- [ ] v2 farms (both H), compare v2 vs v1
- [ ] v3 farms, compare v3 vs v2, investigate > 1e-3 relative
- [ ] design/benchmarks/al27-variants.md

## Jobs
See design/benchmarks/al27-jobs.md, section Lead D.
