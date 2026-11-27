# 27Al weekend: Slurm jobs and node-hours

One row per production or validation job (development-chain nodes `imsrg-*` excluded).
Node-hours = nodes x elapsed (Frontier bills whole nodes for the job length).

## Lead C (v1 farm, branch feature/al27-v1-farm)

| Job id | Purpose | Nodes | Partition | Elapsed | Node-hours | Result |
|---|---|---|---|---|---|---|
| 6006462 | v1 J=0/1 + TCM validation, native, both H (4 steps, CPU) | 32 | batch | 00:30:54 | 16.5 | 1,422 items, 0 failed; all columns PASS vs archive (max 1e-9) |

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

## Development chains (`tools/al27/alloc_chain.sh`, one node each)

| Chain | First job | Last job | Jobs | Node-hours |
|---|---|---|---|---|
