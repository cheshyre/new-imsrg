# MPI scaling of the IMSRG(2) commutator (M12b phase 1)

## Purpose

Measure what the phase-1 distribution (design/12 section 6: replicated 1B/2B operators, the
expensive terms split by channel blocks over ranks, combined by `ReduceBcastSum`) buys on one
node, at 1, 2 and 4 ranks with the same total thread count (16). Tool:
`tools/bench/mpi_scaling.sh` driving `commutator_bench` (MPI usage in its header comment).

## Machine, build, method

* Apple M5 Pro (`sysctl -n machdep.cpu.brand_string`), 6 P + 12 E cores (18 cores), 48 GB, macOS.
* `cmake --preset release-openmp -B build/release-openmp -DIMSRG_BUILD_BENCH=ON` (configure
  line: `MPI=ON build_type=Release`), AppleClang 21, Open MPI 5.0.11, Kokkos 5.3.0 OpenMP;
  bench commit `05fe5f1` (branch `task/m12b-d5b`).
* `mpiexec -n P --oversubscribe --bind-to none` (the build's `IMSRG_MPIEXEC_PREFLAGS`),
  `OMP_NUM_THREADS=T`, `OMP_PROC_BIND=false`, (P,T) = (1,16), (2,8), (4,4); P x T = 16.
* O16 reference, random scalar X (anti-Hermitian) and Y (Hermitian), same seed on all ranks
  (verified by hash). Per invocation 5 repetitions after one warm-up, each between barriers;
  a repetition's time is the maximum over ranks; the invocation reports the minimum over
  repetitions. Best of 5 invocations (the one with the smallest full commutator; its term
  rows are shown). "Full" includes the combine. Terms show each rank's own share (no
  combine). "Combine" is `ReduceBcastSum` alone over Z's 0+1+2-body buffer size.
* The script waits while the 1-minute load average is above 6 (poll 30 s, give up after
  40 min). Note that the loadavg is updated every 5 s and a whole configuration takes only a
  few seconds, so the recorded load is the value before the runs; it was never above 6, no
  run was repeated or marked. The machine was otherwise idle apart from this session.

## Result

| emax | P x T | full commutator ms | slowest terms (ms) | combine ms | speedup vs 1 rank | load before runs |
|---|---|---|---|---|---|---|
| 6 | 1 x 16 | 3.6298 | comm222_phss 1.8338, comm222_pp_hh_221ss 0.8204, comm221ss 0.7826 | - | 1.00 | 4.95 (5 runs) |
| 6 | 2 x 8 | 4.8071 | comm222_phss 2.6119, comm222_pp_hh_221ss 0.8370, comm221ss 0.8164 | 0.5336 | 0.76 | 4.95 (5 runs) |
| 6 | 4 x 4 | 7.4751 | comm222_phss 4.1177, comm222_pp_hh_221ss 1.0369, comm221ss 0.9399 | 1.7595 | 0.49 | 4.95 (5 runs) |
| 8 | 1 x 16 | 30.5059 | comm222_phss 15.0246, comm222_pp_hh_221ss 10.7243, comm222_pp_hhss 10.3067 | - | 1.00 | 4.95 (5 runs) |
| 8 | 2 x 8 | 43.5355 | comm222_phss 21.9171, comm222_pp_hh_221ss 11.5742, comm222_pp_hhss 11.1382 | 3.6850 | 0.70 | 4.95, 4.27 x4 |
| 8 | 4 x 4 | 61.7522 | comm222_phss 30.9949, comm222_pp_hh_221ss 13.4547, comm222_pp_hhss 13.1503 | 12.3220 | 0.49 | 4.27, 5.21 x4 |

Checksums at emax 6 (`--checksum`): serial `[A,H] 5c8e22bcbaeb2bb5`, `[N,H] 934583118d7651a1`
(identical at the base commit a0eb28b and now); 2 ranks `9f463c8517d91ac2` / `1770d8cb347c7370`;
4 ranks `8635d3540033ce3f` / `791544d1ee9c7577`. All ranks agree within a run; the hash
differs between rank counts by floating-point reassociation (expected, design/12 section 6).

## Reading

* On one node with all 16 threads already in use by a single rank, splitting the work over
  ranks cannot add compute: the 2 N^3-class ladder terms drop to about the same time per rank
  (10.3 -> 11.1 -> 13.2 ms at emax 8: half the work on half the threads), and the combine
  (3.7 ms at 2 ranks, 12.3 ms at 4 ranks at emax 8, a 0.06 GB buffer) plus the replicated
  work are pure overhead. The full commutator is therefore slower with more ranks (0.70 and
  0.49 at emax 8). Phase 1 is meant for several nodes, where it adds cores; this measurement
  does not show multi-node scaling.
* The particle-hole term does not shrink with ranks (15.0 -> 21.9 -> 31.0 ms at emax 8; 1.8
  -> 2.6 -> 4.1 ms at emax 6): per-rank time grows roughly like 1/threads, which is what
  the ph inverse transform running on every rank (`kernels/particle_hole.cc` ~636-700, a known
  limit) would give, so the per-term times do show it. The measurement cannot separate it from
  the loss of threads per rank; a run with P ranks x 16 threads on a larger machine would.
* The combine grows faster than linearly with ranks (3.4x from 2 to 4 ranks at emax 8):
  `ReduceBcastSum` is a linear reduce to rank 0 plus a broadcast.

## Joint cut (task m12b-joint-cut, 2026-11-17)

`WorkAssignment::Build` (P > 1) builds the comm122 and ph maps (LPT) first, then cuts the ladder
row sequence at the cumulative capacities max(0, t - other_r), with other_r = w_122 load_122(r) +
w_ph load_ph(r) and t = (T_ladder + sum other) / P (`JointLadderItems`). Weights, in ladder cost
units, from the one-rank release bench (`OMP_NUM_THREADS=16 OMP_PROC_BIND=false commutator_bench
--emax 8 --reps 5`, best of 3 runs, minimum times 10.59 / 4.76 / 15.52 ms for `comm222_pp_hhss` /
`comm122ss` / `comm222_phss`, units 5.841e9 / 4.452e7 / 2.913e9): **w_ph = 2.9, w_122 = 59**
(the earlier estimate of w_ph was 2.9 as well). The machine was shared with other builds (load
average 12-50, above the target of 6, so the weights are good to about 15 %); the emax 6 check
(`--reps 20`) gives w_ph 3.0 and w_122 44. comm122 is thus about 17 % of the IMSRG(2) time at
emax 8 and had been left out of the old estimate.

Total compute imbalance (max / mean - 1, calibrated weights), old = the table of the Hand-off
(ph share x ph imbalance, comm122 ignored), equal = the same metric with the old equal ladder cut:

| emax | ranks | old (Hand-off) | equal cut, w_122 included | joint cut |
|---|---|---|---|---|
| 8 | 22 | 16 % | 19.1 % | 0.1 % |
| 8 | 64 | 160 % | 180.5 % | 145.9 % |
| 8 | 256 | 820 % | 918.1 % | 883.6 % |
| 10 | 22 | 1.7 % | 2.6 % | 0.0 % |
| 10 | 64 | 90 % | 106.5 % | 58.2 % |
| 10 | 256 | 500 % | 581.1 % | 532.8 % |
| 12 | 22 | 1.7 % | 1.5 % | 0.0 % |
| 12 | 64 | 52 % | 63.0 % | 3.0 % |
| 12 | 256 | 310 % | 372.0 % | 311.9 % |

The cut removes the imbalance while some rank has no whole-channel item above the target; where a
single comm122 or ph channel (4-6 % of its term) is larger than t, that rank's other_r exceeds t
and nothing can be taken from it (emax 8 at 64 ranks, all emax at 256 ranks): that remainder
needs splitting the ph channels (phase 2). The serial path (P = 1) is unchanged
(`commutator_bench --checksum --emax 6` hashes 5c8e22bcbaeb2bb5 / 934583118d7651a1 before and after).

## `commutator_bench --model-mpi` (count-only cost model)

Imbalance = max / mean load - 1 per term, from the work items' cost formulas
(`commutator/detail/work_items.h`). "ladder with row blocks" is the map the code uses: a contiguous cut of the global row sequence
(since D2c cost-balanced; since the joint cut its capacities are what comm122 and ph leave free, so
its own imbalance is no longer small by design). "total" rows: max / mean of ladder + w_122
comm122 + w_ph ph loads with the calibrated weights, for the joint cut and for the old equal
ladder cut. Output (joint cut):

```
Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
  In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
  For unit testing set OMP_PROC_BIND=false

emax 8: 108 two-body channels (103 non-empty), max N 648, Z2 0.06 GB
  ladder, plan 2 N^3         sum 6.283e+09  largest   8.7 %  imbalance %: 2:0.0 4:0.0 22:90.5 64:454.3 256:2117.2
  ladder, 2 N^2 (kpp+khh)    sum 5.841e+09  largest   8.7 %  imbalance %: 2:0.0 4:0.0 22:91.7 64:457.6 256:2130.5
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.1 22:25.4 64:72.1 256:20.6
  comm122 (whole channels)   sum 4.452e+07  largest   6.2 %  imbalance %: 2:0.0 4:0.0 22:35.6 64:294.6 256:1478.2
  ph 4 Ncc^2 nk (whole)      sum 2.913e+09  largest   5.8 %  imbalance %: 2:0.0 4:0.1 22:27.1 64:269.7 256:1378.7
  total, calibrated weights  w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:0.1 64:145.9 256:883.6
  total, equal ladder cut    w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:19.1 64:180.5 256:918.1
emax 10: 132 two-body channels (127 non-empty), max N 1341, Z2 0.29 GB
  ladder, plan 2 N^3         sum 6.811e+10  largest   7.1 %  imbalance %: 2:0.0 4:0.0 22:55.8 64:353.2 256:1712.7
  ladder, 2 N^2 (kpp+khh)    sum 6.527e+10  largest   7.1 %  imbalance %: 2:0.0 4:0.0 22:56.5 64:355.3 256:1721.1
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.0 22:1.8 64:71.9 256:20.9
  comm122 (whole channels)   sum 2.737e+08  largest   5.1 %  imbalance %: 2:0.0 4:0.0 22:12.4 64:227.1 256:1208.4
  ph 4 Ncc^2 nk (whole)      sum 1.857e+10  largest   4.7 %  imbalance %: 2:0.0 4:0.0 22:3.8 64:199.3 256:1097.3
  total, calibrated weights  w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:0.0 64:58.2 256:532.8
  total, equal ladder cut    w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:2.6 64:106.5 256:581.1
emax 12: 156 two-body channels (151 non-empty), max N 2483, Z2 1.18 GB
  ladder, plan 2 N^3         sum 5.093e+11  largest   6.0 %  imbalance %: 2:0.0 4:0.0 22:32.3 64:284.7 256:1438.9
  ladder, 2 N^2 (kpp+khh)    sum 4.958e+11  largest   6.0 %  imbalance %: 2:0.0 4:0.0 22:32.7 64:286.0 256:1444.0
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.0 22:1.5 64:58.9 256:22.1
  comm122 (whole channels)   sum 1.271e+09  largest   4.4 %  imbalance %: 2:0.0 4:0.0 22:0.4 64:180.5 256:1022.1
  ph 4 Ncc^2 nk (whole)      sum 8.801e+10  largest   3.9 %  imbalance %: 2:0.0 4:0.0 22:5.0 64:150.9 256:903.4
  total, calibrated weights  w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:0.0 64:3.0 256:311.9
  total, equal ladder cut    w_122 59.0  w_ph 2.9  imbalance %: 2:0.0 4:0.0 22:1.5 64:63.0 256:372.0
```

## Hand-off (2026-11-17 02:45, M12b lead, run 3)

**State** (`feature/m12b-mpi-2`). Phase 1 is complete: merge candidate a2ce18e (ci.sh dev-openmp
and dev-serial PASS, lint clean; slow IT-11 2/4 ranks 385 s / 191 s instead of 2406 s / 341 s, at
load 10-33). After it: D7 (`basis::ThreeBodyDistribution`, distributed `op::ThreeBody`, collective
`ImsrgppNorm`, `GatherToRoot`), D8 (three-body generator as a device kernel over the model space's
rows; gathered distributed eta3 = serial bitwise on 2/4 ranks), the fixes of an Opus review of
D7/D8 (no blocker; the major one: distributed three-body parts are refused on every rank by
`Commutator()` and `CheckReplicatedSetup` until D10/D11), and the joint cut above.

**Acceptance against the cost model (phase 1).** Total compute imbalance with calibrated weights
(section "Joint cut"): 0.0-0.1 % at 22 ranks (emax 8-12) and 3.0 % at 64 ranks (emax 12), so
the 10 % goal holds there; it fails at 64 ranks for emax 8-10 (146 / 58 %) and at 256 ranks for
all emax (310-880 %), where one whole comm122 or ph channel exceeds a rank's share. Memory is
replicated in phase 1. Communication: the combine moves about 2 x Z2 per rank per commutator
(emax 12: 2.4 GB, 0.1 s at 25 GB/s) and does not shrink with P; at this machine's rate (about
1.4 s per emax-12 IMSRG(2) commutator on one 16-thread node, extrapolated from emax 8) it equals
the compute at about 15 nodes, at the plan's 5e10 multiply-adds/s per node at about 100. N7 steps:
phase 1 computes all N7 terms on rank 0, so the N7 criteria need phase 2 (plan §6.2 model: 2.8 s
communication vs 210 s compute per step at 22 ranks, 1.3 s vs 18 s at 256; not yet measured). On
one node more ranks at fixed ranks x threads are slower (section "Result").

**Next steps, in order.**
1. ph and comm122 row panels (M12b-Q3, Q18; the 256-rank limit): each rank computes the W rows
   of its panel and the transposed panel that the combine `W -/+ W^T` needs (about 2x the
   panel's GEMM, no communication), then the inverse mask per (J', row) (review note m6); the
   same cut with capacities then balances all three terms. Recalibrate w_ph and w_122 on an
   idle machine (they were measured at load 12-50).
2. Review note M2 before D9/D10: simulate the lockstep round schedule (round time = slowest
   rank, summed over rounds) in D12 / `n7_bench --model-mpi`; cut batches by cost, not bytes;
   calibrate line weights on measured per-batch times (C14's model); exchange `TransposeOut` once
   per commutator (foreign rows accumulated locally, one `Alltoallv` at the end); send X3/Y3
   halos in bulk when memory allows (2.6 GB per operand at 22 ranks), else prefetch round r+1
   with a non-blocking exchange during round r.
3. D9 (`HaloPlan`, `ExchangeHalo`, reverse plan) once brief C's C6 `PlanBatches` is in develop
   (Opus review). D10 step A (replicated storage, partitioned lines) needs C11 (`RunN7Batch` with
   a RowSet); step B with halos; then lift the D7 guard in `Commutator()`/`CheckReplicatedSetup`.
   D11 (config key `[parallel] three_body_distribution`, driver builds the partition with
   `RowPartition::Balanced` and N7 line weights, triples over an OwnershipMap of bra triples,
   IT-11 N7 with distributed storage), D12 (phase-2 model and measurements).
4. m5: rank 0 broadcasts the NAT occupations once C13 lands (`CheckReplicatedSetup` throws on a
   mismatch until then).

## Particle-hole inverse transform: pair skip and J' trim (2026-11-17)

In the distributed path the inverse transform of the ph term now skips a ket pair when all
masked zbar values are zero and trims the J' loop to the span of nonzero values (the serial
path is unchanged; `commutator_bench --checksum --emax 6` hashes are identical before and
after). Release build, emax 8, best of 3 invocations, `tools/bench/mpi_scaling.sh`
(`EMAXES=8 RUNS=3`), times in ms:

| P x T | ph before | ph after | full before | full after | load before runs (before / after) |
|---|---|---|---|---|---|
| 1 x 16 | 15.80 | 14.81 | 34.64 | 30.02 | 5.97 / 4.51 |
| 2 x 8 | 24.09 | 21.85 | 44.72 | 43.47 | 5.97 / 4.51-5.51 |
| 4 x 4 | 32.72 | 30.10 | 63.18 | 60.29 | 5.31 / 5.51 |

The 1 x 16 change is run-to-run noise (same code path). The gain is about 2 ms (7-9 %) of the
ph term at 2 and 4 ranks; the term still grows with the rank count, so the rest of the
inverse transform (pair loop, 6j row lookups, the masked gather itself) and the cost of the
whole-channel split remain (M12b-Q18). Loads were below 6 for all runs, but other jobs were
running on the machine.
