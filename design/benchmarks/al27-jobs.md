# 27Al weekend: Slurm jobs and node-hours

One row per production or validation job (development-chain nodes `imsrg-*` excluded).
Node-hours = nodes x elapsed (Frontier bills whole nodes for the job length).

## Lead C (v1 farm, branch feature/al27-v1-farm)

| Job id | Purpose | Nodes | Partition | Elapsed | Node-hours | Result |
|---|---|---|---|---|---|---|
| 6006462 | v1 J=0/1 + TCM validation, native, both H (4 steps, CPU) | 32 | batch | 00:30:54 | 16.5 | 1,422 items, 0 failed; all columns PASS vs archive (max 1e-9) |
| 6006648 | v1 CPU timing, one item per L = 2..5 (8 items, N2LOsat native) | 1 | batch | 00:28:17 | 0.5 | L2 ~740 s, L3 ~1160 s, L4 ~1370 s, L5 ~1690 s per item (7 threads) |
| 6006851 | v1 native, L = 2..5 operators, both H (8 steps, CPU) | 108 | batch | 01:29:41 | 161.4 | 3,408 items, 0 failed; v1 native complete, all archived columns PASS (max 1e-9) |
| 6007374 | GPU farm test (frontier-hip, A's branch): 13 items (L0-L5 + TCM) x native/ME2JP, N2LOsat | 2 | batch | 00:13:23 | 0.45 | 0 failed; GPU native = CPU native at every printed digit; ME2JP vs native <= 1e-8 |
| 6007467 | v1 ME2JP farm + native GPU rerun, both H (4 steps x 2415 items, GPU) | 60 | batch | 00:19:34 | 19.6 | 9,660 items, 0 failed; ME2JP all archived columns PASS (max 1e-6); GPU native = CPU native (6e-18) |

## Lead A (GPU production path, branch feature/al27-gpu-production)

| Job id | Purpose | Nodes | Partition | Elapsed | Node-hours | Result |
|---|---|---|---|---|---|---|
| 6006593 | al27-timing: emax-10 farm item timings (v1 Omegas, both formats) + GPU flow restart (Brief A item 5) | 1 | batch | 0:22:46 | 0.38 | |
| 6006646 | al27-farm-j5check-native: FAILED at once (farm_sweep.sh heredoc joined the srun lines; fixed) | 3 | batch | 0:03 | 0.0 | |
| 6006647 | al27-farm-j5check-native: FAILED, same; resubmitted as 6006670 / 6006671 | 3 | batch | 0:03 | 0.0 | |
| 6006670 | al27-farm-j5check-native: v1 N2LOsat farm, all 426 J = 5 items (native): NaN-norm / 40-term check, timings | 3 | batch | 0:16:54 | 0.85 | |
| 6006671 | al27-farm-j5check-native: same for DN2LOGO | 3 | batch | 0:16:45 | 0.84 | |
| 6006799 | al27-flows: full v1 N2LOsat flow on one GCD (14 cores): emax-10 flow step timing and GPU-vs-CPU check against the v1 CPU flow; output runs/al27/v1-gpu/N2LOsat | 1 | batch | 0:40:00 | 0.67 | |

## Lead D (variants, branch feature/al27-variants)

| Job id | Purpose | Nodes | Partition | Elapsed | Node-hours | Result |
|---|---|---|---|---|---|---|
| 6007408, 6007411, 6007430, 6007431, 6007492, 6007493, 6007524, 6007525 | al27-chaintest: flows_chain.sh with a mock binary (4 two-link chains t1-t4; t2/t3 TIMEOUT while fixing the TERM delivery) | 1 each | batch | 0:19:48 total | 0.33 | t4: TERM -> checkpoint -> exit 75 -> link 2 restart -> FLOW_DONE; v3-style wait for prepared/ OK; VRAM log OK |
| 6015953, 6015954 | al27-chaintest t5: mock chain after the empty-argument fix (mock rejects empty words), configs copied | 1 each | batch | 0:03:02 total | 0.05 | link 1 TERM -> 75, link 2 FLOW_DONE for both; B waited for A's prepared/ |
| 6015996 (+6015997, 0:00:03) | al27-v23-flows: v2 + v3 flows, N2LOsat + DN2LOGO, one GCD each (flows_chain.sh link 1; v3 from v2's prepared H_0) | 1 | batch | 0:47:00 | 0.78 | all 4 converged in link 1: v2 46-47 min (incl. ~22 min preparation, 66.1 GB RSS), v3 11-12 min after v2's prepared/ (17.4 GB RSS, 40-42 Omegas) |
| 6016703 | al27-v2-farm: v2 native farms, both H (2 steps x 2415 items, GPU, 15 nodes each) | 30 | batch | 0:19:36 | 9.8 | 4,830 items, 0 failed; GPU work 27.0 GCD-h (N2LOsat), TCM 792 s longest item, ~28 BCH terms per item |
| 6016727 | al27-v3-farmtime: v3 N2LOsat, 13 items (one per family x L + TCM), GPU | 2 | batch | 0:15:08 | 0.50 | v3/v2 item time 3.5-8x (42 Omegas, ~170 BCH terms in total vs ~28); TCM 806 s (host-bound, 1x) |
| 6016892 | al27-v3-farm: v3 native farms, both H (2 x 2415 items, 40 nodes each) | 80 | batch | 0:39:57 | 53.3 | 4,830 items, 0 failed; item work 153 / 145 GCD-h (N2LOsat / DN2LOGO), ~170 BCH terms per item, longest 808 s; farm TCM = flow TCM exactly |
| 6016893 | al27-v2-compatop: v2 Omegas transformed with run.compat = fork (operator-side fork behaviours), 136 items per H (all 34 columns at q = 0, 150, 450, 900) | 4 | batch | 0:07:40 | 0.51 | 272 items, 0 failed (al27-variants.md section 4.2) |
| 6016814 | al27-flows (flows_gpu.sh): scan points split 0.1 (domega 0.1) and ODE flow (imsrg.method = flow), both H, from v2's prepared H_0 | 1 | batch | 0:23:41 | 0.39 | all 4 converged (ODE 21-24 min on a GCD) |
| 6016711, 6016712 (+6016713, 6016714) | al27-scan-N2LOsat / -DN2LOGO: v3-vs-v2 investigation flows (domega 0.025/0.05 split 0.25, split 1.0, unsplit domega 0.05; from v2's prepared H_0) | 1 each | batch | 0:30:03 + 0:28:08 (+0:01:36) | 0.99 | all 8 converged in link 1 |
| 6017305 | al27-flows: N2LOsat operator checks: split 1.0 / 0.1 / 0.25 (domega 0.025) with Omegas, ODE flow carrying FM0p_150 + FPhipp0p_0 | 1 | batch | 0:36:12 | 0.60 | all 4 converged; ODE with 2 operators 36 min |
| 6017358 | al27-opcheck: 12 items (worst v3-vs-v2 columns) on the three split-check Omega stacks | 6 | batch | 0:15:50 | 1.58 | 36 items, 0 failed (al27-variants.md section 5.2) |

Lead D total (2026-11-27 18:30 - 21:05 EST): ~68.8 node-hours (v3 farm 53.3, v2 farm 9.8, flows and scans 3.4, checks 2.3).

## Development chains (`tools/al27/alloc_chain.sh`, one node each)

| Chain | First job | Last job | Jobs | Node-hours |
|---|---|---|---|---|

## Restart check

v3 N2LOsat, binary `imsrg.gpu-20261127-183256` of the reference run (frontier-hip, 42a46f7), one node each; details in `design/benchmarks/al27-restart.md`.

| Job | What | Nodes | Partition | Elapsed | Node-hours | Notes |
|---|---|---:|---|---|---:|---|
| 6017687 | straight re-run, attempt 1 (flow_file not yet overridden) | 1 | batch | ~0:12 | 0.20 | results bit-identical to the reference; its flow.txt overwrote the production one |
| 6017688 + 6017694 | killed run, attempt 1: link 1 (TERM at 21:08:02, stage 2 step 11) + link 2 (restart) | 1 + 1 | batch | ~0:01 + ~0:11 | 0.20 | bit-identical; flow.txt also went to the production path |
| 6017812 | straight re-run, attempt 2 (flow_file overridden into the run directory) | 1 | batch | 0:11:25 | 0.19 | |
| 6017813 + 6017827 | killed run, attempt 2: link 1 (TERM at 21:22:04, stage 2 step 3, exit 75, 0:39) + link 2 (restart, 11:03) | 1 + 1 | batch | 0:00:39 + 0:11:03 | 0.20 | |

Restart check total: about 0.8 node-hours (a reservation of 6 min for link 1 and 30 min for link 2 and the straight job, billed by use).

## Lead E (investigation)

Branch feature/al27-investigate-variants; development chain `imsrg-e`; runs in
`$IMSRG_WORK/runs/al27/investigate/` (configs in `configs/`). Budget 40 node-hours.

| Job | What | Nodes | Partition | Elapsed | Node-hours | Notes |
|---|---|---:|---|---|---:|---|
| 6018030 | al27-flows (flows_gpu.sh): DN2LOGO ODE IMSRG(2) flow (dopri5, tol 1e-6) carrying FM0p_150 + FPhipp0p_0, from v2's prepared H_0; binary `imsrg.gpu-20261127-202543` (that of the N2LOsat ODE-ops run 6017305) | 1 | batch | 0:33:51 | 0.56 | task 2; converged, al27-variants.md 5.2 |
