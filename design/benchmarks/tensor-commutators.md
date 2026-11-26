# Scalar-tensor commutator timings (M15, 2026-11-26)

`tools/bench/tensor_commutator_bench --emax N --rank L --reps R` (frontier-cpu-release, one Frontier
CPU node shared with other agents, OMP_NUM_THREADS=16, O16 reference, random anti-Hermitian scalar X
and Hermitian rank-L tensor Y with 1B + 2B parts). Times in ms, best of R. Code of commit c4fb10f
(before the c3 <= c4 Zbar storage and the parallel 9j build).

| emax | L | norb | first call (tables) | 122 | 221 | 222 pp/hh | 222 ph | all | peak RSS |
|-----:|--:|-----:|--------------------:|----:|----:|----------:|-------:|----:|---------:|
| 4 | 2 | 30 | 172 | 5 | 29 | 28 | 43 | 78 | 0.06 GB |
| 4 | 5 | 30 | 349 | 4 | 59 | 57 | 58 | 119 | 0.06 GB |
| 6 | 2 | 56 | 1186 | 113 | 62 | 63 | 331 | 518 | 0.35 GB |
| 6 | 5 | 56 | 3782 | 146 | 110 | 111 | 641 | 907 | 0.44 GB |
| 8 | 5 | 90 | 26107 | 2377 | 551 | 564 | 5951 | 8951 | 3.3 GB |
| 10 | 5 | 132 | 130669 | 21614 | 3508 | 3617 | 38056 | 63096 | 18.1 GB |

Notes: the first call is dominated by the serial 9j table build (parallel since; the table is 1.1 GB
at emax 10). comm122st (element-wise, getter lookups) and comm222_phst (Pandya with 9j sums, host
BLAS) dominate; both are GPU/tuning targets for the farm (C10/C11). Zbar storage was halved after
these runs (c3 <= c4 via the tensor hermiticity relation).

**After the c3 <= c4 Zbar storage and the parallel 9j build** (same setup, emax 8, rank 5): first
call 11.0 s (was 26.1 s), comm222_phst 6.7 s, peak RSS 2.19 GB (was 3.29 GB). Phases of comm222_phst
(temporary timers): Xbar 18 ms, Ybar pieces 0.91 s, gemms 0.11 s, **inverse transform 5.5 s**
(about 5e9 9j x Zbar gathers: N_Z elements x J3 x J4 x 2; embarrassingly parallel, a GPU target).
