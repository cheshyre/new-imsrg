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

## `commutator_bench --model-mpi` (count-only cost model)

Imbalance = max / mean load - 1 per term, from the work items' cost formulas
(`commutator/detail/work_items.h`). "ladder with row blocks" is the map the code uses: since D2c
a contiguous cost-balanced cut of the global row sequence (it was LPT over row blocks: 2.5 / 9.0 /
15.3 % at emax 8 and 1.6 / 8.2 / 16.7 % at emax 12 for 22 / 64 / 256 ranks). Output at a2ce18e:

```
  In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
  For unit testing set OMP_PROC_BIND=false
emax 8: 108 two-body channels (103 non-empty), max N 648, Z2 0.06 GB
  ladder, plan 2 N^3         sum 6.283e+09  largest   8.7 %  imbalance %: 2:0.0 4:0.0 22:90.5 64:454.3 256:2117.2
  ladder, 2 N^2 (kpp+khh)    sum 5.841e+09  largest   8.7 %  imbalance %: 2:0.0 4:0.0 22:91.7 64:457.6 256:2130.5
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.0 22:0.2 64:0.7 256:3.3
  comm122 (whole channels)   sum 4.452e+07  largest   6.2 %  imbalance %: 2:0.0 4:0.0 22:35.6 64:294.6 256:1478.2
  ph 4 Ncc^2 nk (whole)      sum 2.913e+09  largest   5.8 %  imbalance %: 2:0.0 4:0.1 22:27.1 64:269.7 256:1378.7
emax 10: 132 two-body channels (127 non-empty), max N 1341, Z2 0.29 GB
  ladder, plan 2 N^3         sum 6.811e+10  largest   7.1 %  imbalance %: 2:0.0 4:0.0 22:55.8 64:353.2 256:1712.7
  ladder, 2 N^2 (kpp+khh)    sum 6.527e+10  largest   7.1 %  imbalance %: 2:0.0 4:0.0 22:56.5 64:355.3 256:1721.1
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.0 22:0.1 64:0.3 256:1.1
  comm122 (whole channels)   sum 2.737e+08  largest   5.1 %  imbalance %: 2:0.0 4:0.0 22:12.4 64:227.1 256:1208.4
  ph 4 Ncc^2 nk (whole)      sum 1.857e+10  largest   4.7 %  imbalance %: 2:0.0 4:0.0 22:3.8 64:199.3 256:1097.3
emax 12: 156 two-body channels (151 non-empty), max N 2483, Z2 1.18 GB
  ladder, plan 2 N^3         sum 5.093e+11  largest   6.0 %  imbalance %: 2:0.0 4:0.0 22:32.3 64:284.7 256:1438.9
  ladder, 2 N^2 (kpp+khh)    sum 4.958e+11  largest   6.0 %  imbalance %: 2:0.0 4:0.0 22:32.7 64:286.0 256:1444.0
  ladder with row blocks (implementation weight), imbalance %: 2:0.0 4:0.0 22:0.0 64:0.1 256:0.6
  comm122 (whole channels)   sum 1.271e+09  largest   4.4 %  imbalance %: 2:0.0 4:0.0 22:0.4 64:180.5 256:1022.1
  ph 4 Ncc^2 nk (whole)      sum 8.801e+10  largest   3.9 %  imbalance %: 2:0.0 4:0.0 22:5.0 64:150.9 256:903.4
```

## Hand-off (2026-11-17, M12b lead, run 3; draft before the 02:45 pause)

**State.** Phase 1 is complete on `feature/m12b-mpi-2` (merge candidate a2ce18e: ci.sh
dev-openmp and dev-serial PASS, lint clean, slow IT-11 2/4 ranks 385 s / 191 s instead of
2406 s / 341 s at load 10-33). On top (6ed3c14): D7 (`basis::ThreeBodyDistribution`, distributed
`op::ThreeBody`, collective `ImsrgppNorm`, `GatherToRoot`) and D8 (three-body generator as a device
kernel over the model space's rows; gathered distributed eta3 = serial bitwise on 2/4 ranks).
An Opus review of D7/D8 found no blocker; its fixes (below) are the next task.

**Acceptance against the cost model (phase 1).** Term weights calibrated on the 1 x 16 run at
emax 8: a ph cost unit takes 2.9x the time of a ladder unit (15.0 ms / 2.91e9 vs 10.3 ms /
5.84e9), so the ph term is 59 % (emax 8), 45 % (emax 10) and 34 % (emax 12) of the IMSRG(2) time.
Since the ladder is balanced to <= 3.3 %, the total compute imbalance is about ph share x ph
imbalance:

| emax | 22 ranks | 64 ranks | 256 ranks |
|---|---|---|---|
| 8 | 16 % | 160 % | 820 % |
| 10 | 1.7 % | 90 % | 500 % |
| 12 | 1.7 % | 52 % | 310 % |

So the 10 % goal holds at 22 ranks for emax 10-12 only. Memory is replicated in phase 1 (no
memory distribution). Communication: the combine moves about 2 x Z2 per rank per commutator
(emax 12: 2.4 GB, 0.1 s at 25 GB/s) and does not shrink with P; at this machine's measured rate
(about 1.4 s per emax-12 IMSRG(2) commutator on one 16-thread node, extrapolated from emax 8) it
equals the compute at about 15 nodes, at the plan's 5e10 multiply-adds/s per node at about 100.
N7 steps: phase 1 computes all N7 terms on rank 0, so the N7 criteria need phase 2 (plan §6.2
model: 2.8 s communication vs 210 s compute per step at 22 ranks, 1.3 s vs 18 s at 256; not yet
measured). On one node more ranks at fixed ranks x threads are slower (above).

**Next steps, in order.**
1. Opus review fixes of D7/D8: reject distributed three-body parts on every rank in
   `Commutator()` and in `CheckReplicatedSetup`/`BroadcastOperator` until D10/D11 (today the
   rank-0 N7 path would throw on rank 0 only and the others hang in the combine); check the
   `GatherToRoot` int counts and the root's displacement total on every rank before the
   collective; validate the partition against the kept space in `WithThreeBodyDistribution`;
   tests for i < j get/set values, a rank without rows, the distribution drop in
   `WithReference*` and the dump guard; a D8 case whose core orbits are not the lowest indices
   (the -v branch and the negative floor); `SyncToHost` before host reads in the D8 test; docs of
   `generator.h` (Hermitian H3, device kernel, complexity); `OneBodyDiagonal` -> snake_case.
2. Phase-1 balance (M12b-Q3, Q18): (a) joint cut: assign ph and comm122 first, then cut the
   ladder rows with per-rank capacities (target minus the other loads) using calibrated term
   weights; this absorbs the ph imbalance while the largest ph channel (3.9 % of the ph work at
   emax 12) stays below the mean per-rank load, i.e. up to about 75 ranks at emax 12; (b) ph row
   panels for more ranks: each rank computes the W rows of its panel and the transposed panel the
   combine `W -/+ W^T` needs (about 2x the panel's GEMM, no communication), then the inverse mask
   per (J', row) (review note m6).
3. Review note M2 before D9/D10: simulate the lockstep round schedule (round time = slowest
   rank, summed over rounds) in D12 / `n7_bench --model-mpi`; cut batches by cost, not bytes;
   calibrate line weights on measured per-batch times (C14's model); exchange `TransposeOut` once
   per commutator (foreign rows accumulated locally, one `Alltoallv` at the end); send X3/Y3
   halos in bulk when memory allows (2.6 GB per operand at 22 ranks), else prefetch round r+1
   with a non-blocking exchange during round r.
4. D9 (`HaloPlan`, `ExchangeHalo`, reverse plan) once brief C's C6 `PlanBatches` is in develop;
   Opus review. D10 step A (replicated storage, partitioned lines) needs C11 (`RunN7Batch` with a
   RowSet); step B with halos. D11 (config key `[parallel] three_body_distribution`, triples
   over an OwnershipMap of bra triples, IT-11 N7 with distributed storage), D12 (phase-2 model
   and measurements).
5. m5: rank 0 broadcasts the NAT occupations once C13 lands (`CheckReplicatedSetup` throws on a
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
