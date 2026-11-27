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
- [ ] (a) tools/al27/diff_variants.py + tests (worker, branch feature/al27-variants-diff)
- [ ] (b) tools/al27/flows_chain.sh + flow_summary.py (worker, branch feature/al27-variants-chain)
- [ ] wait for B merged into develop; merge develop; build frontier-hip release binary
- [ ] v2 + v3 flows (chain job)
- [ ] v2 farms (both H), compare v2 vs v1
- [ ] v3 farms, compare v3 vs v2, investigate > 1e-3 relative
- [ ] design/benchmarks/al27-variants.md

## Jobs
See design/benchmarks/al27-jobs.md, section Lead D.
