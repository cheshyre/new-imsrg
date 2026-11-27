# 27Al weekend: Slurm jobs and node-hours

One row per production or validation job (development-chain nodes `imsrg-*` excluded).
Node-hours = nodes x elapsed (Frontier bills whole nodes for the job length).

## Lead C (v1 farm, branch feature/al27-v1-farm)

| Job id | Purpose | Nodes | Partition | Elapsed | Node-hours | Result |
|---|---|---|---|---|---|---|
| 6006462 | v1 J=0/1 + TCM validation, native, both H (4 steps, CPU) | 32 | batch | 00:30:54 | 16.5 | 1,422 items, 0 failed; all columns PASS vs archive (max 1e-9) |
| 6006648 | v1 CPU timing, one item per L = 2..5 (8 items, N2LOsat native) | 1 | batch | 00:28:17 | 0.5 | L2 ~740 s, L3 ~1160 s, L4 ~1370 s, L5 ~1690 s per item (7 threads) |
| 6006851 | v1 native, L = 2..5 operators, both H (8 steps, CPU) | 108 | batch | running | | |
