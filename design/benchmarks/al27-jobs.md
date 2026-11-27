# 27Al weekend: Slurm jobs (node-hours)

Every production or benchmark job of the 27Al weekend run (D65 budget 50,000 node-hours). The
self-resubmitting development chains (`imsrg`, `imsrg-a`, `imsrg-b`, ...) are not listed.

| job id | name | who | nodes | elapsed | node-hours | purpose |
|---|---|---|---:|---|---:|---|
| 6006593 | al27-timing | lead A | 1 | 0:22:46 | 0.38 | emax-10 farm item timings (v1 Omegas, both formats) + GPU flow restart (Brief A item 5) |
| 6006646 | al27-farm-j5check-native | lead A | 3 | 0:03 | 0.0 | FAILED at once (farm_sweep.sh heredoc joined the srun lines; fixed) |
| 6006647 | al27-farm-j5check-native | lead A | 3 | 0:03 | 0.0 | FAILED, same; resubmitted as 6006670 / 6006671 |
| 6006670 | al27-farm-j5check-native | lead A | 3 | 0:16:54 | 0.85 | v1 N2LOsat farm, all 426 J = 5 items (native): NaN-norm / 40-term check, timings |
| 6006671 | al27-farm-j5check-native | lead A | 3 | 0:16:45 | 0.84 | same for DN2LOGO |
| 6006799 | al27-flows | lead A | 1 | | | full v1 N2LOsat flow on one GCD (14 cores): emax-10 flow step timing and GPU-vs-CPU check against the v1 CPU flow; output runs/al27/v1-gpu/N2LOsat |
