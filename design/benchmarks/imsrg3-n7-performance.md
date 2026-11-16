# IMSRG(3)-N7 performance: row tiling, fused driver, triples context (run 3, brief C)

Status: 2026-11-17 03:20 JST, branch `feature/m10-n7-perf-2`. Design: `design/14-imsrg3-plan.md`
section 6; plan (outside the repository): `new-imsrg-reference/plans/m10-n7-perf-plan.md`. Timing runs
follow D52 (timing only, emax 4-6, one operator below ~8 GB, never in CI); counts come from
`n7_bench --count-only` (no operator).

## 1. What changed

| Task | Change | Commit(s) |
|---|---|---|
| C1 | `basis::RowRange/RowSet/RowPartition` (Whole, Balanced zipper, WholeChannels) | develop e70ce17 |
| C2 | per-row offset views (`row_off`, `row_buf`, halo slot) for every 3B read/write | develop e70ce17 |
| C14 | `tools/bench/n7_bench` (timing, `--checksum`, `--count-only`), `N7CostModel`, `N7LineWeights` | develop e70ce17 |
| C3 | Comm223 on work items built from a `RowSet` (`AddComm223Rows`) | e398f01 |
| C6 | batch planner `PlanBatches`, `BuildH`/`BuildW` (W over all pair-third rows, review fix B1) | 4bd13dd |
| loop | `ForEachN7Batch`, term interface `detail/n7_terms.h`, partition/mask test support | 197f9c9, 750854d |
| C7 | Comm232 on the W columns of the batches | 5574c39 (merge fd0eb3b) |
| C8 | Comm331 (H) and Comm231 (W) per batch, one writer per Z1 element | 869421f (merge 749c793) |
| C9 | Comm132 (W, grouped by Z2 column) and Comm330 (per line) | 5252a82 (merge 31b26ed) |
| C10 | Comm133: bra side from line k, ket side via `TransposeOut` | 07ee34d (merge e5fb703) |
| C11 | fused driver `detail::RunN7`: one batch loop, H/W shared by all terms; `Commutator()` uses it | 3ad85ed (merge 50093ee) |
| C5 | `WorkspaceOptions::three_body_context` (kAuto/kSkip), `Workspace::BuildThreeBodyContext()` | a5ac995 |
| C13 | `imsrg3.occ_nat3_cut` for the IMSRG(3) flow (MBPT(2) natural occupations) | 008d5ad |
| C12 | `commutator::TriplesContext`: only channels with an hhh triple, only hhh/ppp triples; no E map | c3a2940 (merge e1aff9e) |
| C4 | `exec::TeamOptions`, `TeamFor`/`TeamReduce`; comm223 launches ported (checksums identical) | adda75c (merge 71b1334) |
| C14c | `n7_bench --partition P` (Balanced) and `--partition-contiguous`: per-part times, sum/whole, max/mean | 4702200 (merge d8803df) |

Every rewritten kernel keeps its element-wise predecessor as `detail::Comm<term>ssReference`.

## 2. Results unchanged (tests)

Comparisons use max|dZ| <= 1e-12 max(1, max|Z|) (review fix M3); measured worst cases:

| Check | Worst deviation |
|---|---|
| 331 / 231 batch vs reference (He4, O16, fractional, dE3max cut, H/A/N, budgets 1 line ... default) | 1.4e-15 / 8e-16 (1e-14 on two cases with \|Z\| ~ 5e-15) |
| 132 / 330 batch vs reference | 3.5e-16 / 2.7e-14 |
| 133, 232 batch vs reference | pass at the M3 form (C7/C10 test logs) |
| partition invariance with NaN-masked operands (P = 2, 3, 7; Balanced 4, 16) | 331 6.6e-16, 231 5.9e-16, 132 2.0e-16, 330 1.0e-15, full N7 7.6e-16 |
| fused `RunN7` vs the seven `terms::` calls | bitwise at the default budget (32 cases); 1.9e-15 at one triple per batch |
| N7 terms + fused commutator vs m-scheme brute force on an OccNat3-cut space | 1.3e-15 |
| triples: `TriplesContext` vs full context (He4/O16 emax 2, O16 emax 3, dE3max, OccNat3) | 3e-16 relative |
| IT-9 emax 2 (He4 / O16) | 250 / 147 steps, E0 -25.233798919033614 / -145.48982708228132, identical before/after |
| IT-9 emax 3 (slow, run once by the C11 worker) | He4 248, O16 173 steps (= goldens); dE0 vs goldens -3.8e-10 / -3.0e-10 |
| IT-8 (driver_triples) | unchanged (pass) |

`tools/ci.sh dev-openmp` and `dev-serial` PASS; `tools/lint.sh` clean (e1aff9e).

## 3. Timings before/after (O16, AH operands, ms)

Measured 2026-11-17 02:55-03:15 on the quiet machine (18 threads, `OMP_PROC_BIND=false`,
`VECLIB_MAXIMUM_THREADS=1`, release-openmp). "Before" = develop e739622 + n7_bench (binary in worktree
`agent-a9d112cd8777f9b77`); "after" = a5b1987. Each entry is the best of 5 repetitions after a
warm-up; emax 3 and 4: the better of two interleaved before/after pairs; emax 5: one pair. Every
run started at a 1-minute load below 6 (the load rises to 9-20 during an 18-thread run itself).

| emax / E3max | variant | 133 | 330 | 223 | 232 | 331 | 231 | 132 | full N7 | start load |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 / 9 | before | 141.2 | 0.36 | 16.0 | 54.1 | 31.1 | 2.70 | 5.93 | 255.1 | 2.2-2.5 |
| 3 / 9 | after | 28.1 | 0.55 | 15.9 | 75.5 | 27.0 | 63.5 | 64.4 | **104.8** | 2.2-2.5 |
| 4 / 12 | before | 4306 | 5.31 | 282.0 | 1228 | 378.8 | 9.74 | 59.4 | 6289 | 5.6-5.9 |
| 4 / 12 | after | 311.9 | 8.27 | 269.2 | 642.4 | 344.5 | 550.8 | 562.9 | **1147** | 4.1-5.9 |
| 5 / 12 | before | 32481 | 33.9 | 1868 | 10662 | 1002 | 38.9 | 412.9 | 46629 | 6.0 |
| 5 / 12 | after | 1705 | 47.6 | 1826 | 3170 | 1961 | 2623 | 2654 | **6218** | 5.7 |

* **Full N7 commutator** (the production path, `Commutator()` with `TermMask::Imsrg3N7()`,
  including the IMSRG(2) terms): 2.4x faster at emax 3, 5.5x at emax 4, 7.5x at emax 5. Counted
  multiply-adds per second (section 4 totals over the full time): 6.3e9 / 1.3e10 / 1.7e10 after,
  2.6e9 / 2.3e9 / 2.3e9 before.
* **Single-term calls** (`terms::Comm*ss`) rebuild the intermediates they need (M10-Q73): 231 and
  132 now contain a full H + W build (about 0.55 s at emax 4, 2.6 s at emax 5) and are slower
  than their element-wise predecessors; 331 contains the H build. In the fused driver H and W are
  built once per batch and shared, which is where the speed-up comes from (133 alone is 14x / 19x
  faster at emax 4 / 5).
* 223 is unchanged (bitwise; C3 only changed its work items). 330 is ~1.5x slower per call (per-line
  tile lookup), negligible in the total.
* emax 6 was not timed (a full N7 commutator holds about 12-14 GB there; the before path would take
  minutes per repetition).

## 4. Cost model (O16, multiply-adds per N7 commutator with 3B parts in X and Y)

From `n7_bench --count-only` (git 50093ee); "H", "W" are the shared intermediates of C6.

| emax / E3max | 223 (+epilogue) | H | W | 232 | 331 | 133 | 330 | 231+132 | total | packed op |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 / 12 | 1.67e9 | 1.19e9 | 3.43e9 | 5.40e9 | 7.99e8 | 1.99e9 | 7.2e7 | 1.53e8 | 1.47e10 | 0.29 GB |
| 6 / 12 | 3.27e10 | 1.66e10 | 4.91e10 | 1.37e11 | 1.44e10 | 3.10e10 | 9.5e8 | 1.52e9 | 2.84e11 | 3.8 GB |
| 10 / 16 | 2.59e12 | 8.09e11 | 2.41e12 | 1.50e13 | 9.04e11 | 1.71e12 | 3.9e10 | 4.07e10 | 2.35e13 | 157 GB |
| 12 / 16 | 3.79e12 | 9.01e11 | 2.69e12 | 2.29e13 | 1.20e12 | 2.10e12 | 4.4e10 | 3.92e10 | 3.37e13 | 177 GB |

At emax 12 / E3max 16 (1,744,097 rows, 711,296 triples, 5.18e6 pair-third states, largest channel
46,188 rows / 137,729 states): one N7 commutator costs 673 node-seconds at 5e10 multiply-adds/s
(135 s at 2.5e11); six packed operators take 1,064 GB (+ 7 GB 2B). One node cannot hold one
operator: the M12b phase-2 row distribution is required (the kernels are now ready for it: every
term works on line batches and passes the masked-operand test).

## 5. Perturbative triples (C12, `n7_bench --triples --count-only`, O16, E3max = 3 emax)

| emax | channels | ppp triples | rows | states | context | tile work |
|---|---|---|---|---|---|---|
| 3 | 28 | 1,800 | 4,190 | 11,348 | 1.2 MB | 4.1e5 |
| 8 | 28 | 297,202 | 819,732 | 2.41e6 | 0.24 GB | 3.0e8 |
| 12 | 28 | 2,388,180 | 7,199,014 | 2.13e7 | 2.16 GB | 3.5e9 |

The plan's 5.4e9 at emax 12 is not reproduced (3.47e9 = 2.62e9 W sums + 0.85e9 epilogue); the
counts of channels and triples match. Building the context at emax 12 takes ~150 s (host, 4.4 GB
peak): a follow-up.

## 6. Simulated partitions

`n7_bench --partition P` runs `detail::RunN7` part by part on replicated operands. Smoke runs
(He4/O16 emax 2-3, P up to 64, one repetition on the busy machine): the summed result equals the
whole within 9e-16 (M3 form), P = 1 is bitwise equal to the whole run. The timing ratios of
those runs (sum/whole 0.7-6.3, max/mean 1.05-1.9) are noise at these sizes; the planned
measurement (emax 5, P = 4, 16, 64, quiet machine; targets sum/whole <= 1.10 and max/mean <= 1.10
at P = 16) is done in section 8.

## 7. Review and hand-off

Opus review of e739622..e1aff9e: no blocker. Fixed since (a5b1987): output ownership checked for
every term (`PartitionChecks::ownership`; 133 bra side per part with the transpose outputs kept
apart), masked check per single-triple batch (`PartitionChecks::per_batch`), the 133 slot table and
one-body channel size cached per call (`Comm133Cache`; per-batch work proportional to the batch,
checksums identical), the budget documented (it bounds H + W of one batch; allocation max(H) +
max(W), plus TransposeOut and slot tables), `DeviceBuffer<TileWorkItem>` instead of an int32
reinterpretation, the triples test in the M3 form with a fractional case, the OccNat3 driver test
asserting fewer kept rows. `imsrg3.occ_nat3_cut` also works with the perturbative triples (C13b;
`results.json` gained `perturbative_triples.kept_kets`/`total_kets`).

A second short Opus review (C4 and the partition mode, 93e2d9f..338b0b9) found no blocker; its
major item (vector length AUTO was built as 1 in the facade; no effect on Serial/OpenMP) and the
minor items (level-0 scratch for the old `TeamFor` overload, one launch-options object per comm223
launch, `InvariantError` in the facade test, NaN-safe partition check, no partitions with a dense
Z) are fixed in 6cc1d57 and c3218b4 (checksums unchanged).

Open items, in priority order:
1. `CommutatorOptions::imsrg3_batch_bytes` after the M12b merge (M10-Q74).
2. C15 (223 locality, M10-Q41/Q42; needs quiet-machine timings at emax 4-6).
3. The simulated-partition timings of section 6 (quiet machine, emax 5).
4. The H build looks like the least efficient piece: estimated from the single-term times at
   emax 4 (331 = H + contraction, 231 = H + W + contraction), H runs at about 4e9 multiply-adds/s
   and W at about 1.7e10. `BuildH` reads B through `ThreeBodyLocal` once per E-row entry; a tiled
   H build (load the tile's rows into scratch first) is the next kernel optimization to try.
5. Factor the group/leg table builders duplicated in `commutator/triples.cc` (M10-Q75 note); the
   `TriplesContext` build at emax 12 takes ~150 s and 4.4 GB on the host.
6. `n7_bench --partition` holds one unused rank-3 output besides the summed and the whole Z
   (memory at emax 5); reuse it.

## 8. Quiet-machine measurements (2026-11-17)

Simulated row partitions of section 6, measured 04:54-05:15 JST on the idle machine (git 7a78f4d,
`release-openmp` with `-DIMSRG_BUILD_BENCH=ON`, Apple M5 Pro, 18 threads, `OMP_PROC_BIND=false`,
`VECLIB_MAXIMUM_THREADS=1`). O16, emax 5, E3max 12 (144,162 rows, 63,698 triples, 1.63 GB per
operand; counts of section 4 style: 1.055e11 multiply-adds), operands `AH`, seed 1, the seven 3B
terms of `detail::RunN7` (the 3B part of the full N7 commutator). Every run is one process with
`--reps 5`: each part, and then the whole space, is timed as the best of 5 repetitions after a
warm-up. A run of 3-4 minutes in total (a single whole commutator takes 6.1 s), so emax 5 was kept.
Every run started at a 1-minute load below 6 (the loads in the table; the 5-minute loads of 8-9
are the aftermath of the previous 18-thread run). The load average was 2.5 at the first run.
Deviation of the summed result from the whole: bitwise for P = 1, 5.6e-16 to 8.7e-16 (M3 form) for
all others.

Commands (script outside the repository, same options for every row):

```
export OMP_PROC_BIND=false VECLIB_MAXIMUM_THREADS=1
build/release/tools/bench/n7_bench --emax 5 --e3max 12 --count-only
build/release/tools/bench/n7_bench --emax 5 --e3max 12 --reps 5 --partition P [--partition-contiguous]
```

Baseline: the whole space, P = 1 run: **6116.1 ms** (the other runs' wholes: 6128.9, 6148.3,
6140.4, 6143.0, 6194.6 and, for the last run, 8407, which is an outlier: the whole of that run is
ignored and sum/whole is taken against 6116.1 everywhere below).

Balanced partition (`RowPartition::Balanced` with `N7LineWeights`, the R5 weights):

| P | load at start | per-part best ms (min / mean / max) | max/mean measured | max/mean predicted (compute / memory) | sum of parts, ms | sum/whole |
|---|---|---|---|---|---|---|
| 1 | 2.54 | 6283.0 | 1.000 | 1.000 | 6283.0 | 1.027 |
| 4 | 5.74 | 1434.9 / 1552.8 / 1804.3 | 1.162 | 1.020 / 1.025 | 6211.1 | 1.016 |
| 16 | 4.89 | 328.0 / 370.8 / 486.4 | 1.312 | 1.038 / 1.064 | 5933.3 | 0.970 |
| 64 | 5.31 | 78.4 / 94.4 / 138.1 | 1.463 | 1.044 / 1.098 | 6039.7 | 0.988 |

Whole-channel (contiguous) partition (`--partition-contiguous`, P contiguous ranges of the triple
list, cut at channel borders only):

| P | load at start | per-part best ms (min / mean / max) | max/mean measured | max/mean predicted (compute) | sum of parts, ms | sum/whole |
|---|---|---|---|---|---|---|
| 4 | 5.27 | 827.3 / 1489.2 / 2115.9 | 1.421 | 1.469 | 5956.7 | 0.974 |
| 16 | 5.63 | 70.1 / 373.0 / 591.5 | 1.586 | 1.737 | 5967.7 | 0.976 |
| 64 | 5.69 | 8.4 / 101.5 / 205.0 | 2.020 | 2.451 | 6494.4 | 1.062 (outlier run) |

Per-part times, balanced P = 4: 1439.2, 1434.9, 1532.6, 1804.3 ms (rows 20,803 / 24,336 / 29,008 /
70,015; stored doubles 50.0e6 / 50.5e6 / 51.3e6 / 52.3e6). Balanced P = 16 (rows in brackets):
337.9 (5,053), 336.8 (5,066), 343.6 (5,321), 341.9 (5,363), 328.0 (5,511), 334.2 (5,951), 361.5
(6,366), 374.0 (6,508), 372.5 (6,786), 356.6 (7,085), 351.6 (6,845), 395.8 (8,292), 359.2 (9,313),
447.5 (13,820), 405.6 (16,714), 486.4 (30,168). Contiguous P = 16: 171.6, 179.4, 404.6, 335.0,
575.7, 454.9, 533.1, 580.9, 477.1, 591.5, 421.3, 421.8, 334.6, 248.0, 168.4, 70.1 ms.

Predicted imbalance: computed from the same `N7LineWeights` weights the partition is built with
(`basis::PartitionImbalance` for the balanced rows, the sums of the weights over the same triple
ranges for the contiguous rows), with a scratch program linked against the release libraries
(not in the repository); `max/mean` here is 1 + the printed imbalance.

Reading:

* **Overhead of the summed parts**: none measurable. Sum/whole is 0.97 to 1.02 for P up to 64
  (balanced) and 0.97 for contiguous P = 4, 16; the sums are not above the whole because every part
  works on a smaller working set, so the 3 % below one is cache effect, and the 1-3 % above one at
  P = 1 and 4 is the noise level (P = 1 compares two identical runs: 6283 vs 6116 ms). The
  target of at most 1.10 holds with a margin. The replicated per-part setup (halo slots, `H`/`W`
  tables) does not show.
* **Imbalance**: the target of max/mean <= 1.10 at P = 16 is **not met** (balanced 1.31). The
  cost model predicts 1.04 for the balanced partition, so the R5 weights are not good enough: the
  parts with the most rows (largest channels, the last parts) are up to 48 % slower than parts with
  the same predicted work and the same stored size, and the time per part grows monotonically with
  the part's row count (P = 16: 328 ms at 5.1k rows, 486 ms at 30.2k rows). The weights therefore
  undercount something that grows with the rows of a part. For the contiguous partition the model
  has the right shape and over-predicts by 3-18 % (1.42 vs 1.47, 1.59 vs 1.74, 2.02 vs 2.45), so
  it is a safe upper bound there and the balanced zipper is clearly better (1.31 vs 1.59 at P = 16,
  1.46 vs 2.02 at P = 64).

Conclusion for run 4: the simulated partitions cost nothing in total work (sum/whole <= 1.02,
checked up to P = 64 with bitwise or 9e-16 results), so row distribution is viable. The R5
compute weights are **not yet good enough to balance the parts**: the measured max/mean is 1.16 /
1.31 / 1.46 at P = 4 / 16 / 64 against a predicted 1.02 / 1.04 / 1.04, and the error is systematic
(time grows with the part's number of rows at equal predicted work), so it will not average out
across ranks. Do not hard-wire the R5 weights as the balance criterion for D-phase2: first
calibrate the per-triple weights from the per-part times (a term proportional to the rows of a
triple, or per-term line weights fitted to `n7_bench --terms` times at emax 5; the per-part times
above are the data for a first fit), and then re-measure max/mean at P = 16 with the target 1.10.
With the present weights plan for 1.3-1.5x the mean part time as the slowest rank's time; the
memory balance of the zipper is fine (stored doubles within 5 % at P = 4). No code was changed.

### Row-cost model with a per-row term (2026-11-17)

Follow-up to the conclusion above (branch `feature/n7-row-cost-model`). `n7_bench --partition`
now prints, per part, the modeled multiply-adds (`model_ma`, sum of `N7TripleCost::Total` over the
part's triples) and the triple count next to rows, stored doubles and `best_ms`; the new option
`--row-overhead MADD` overrides the per-row weight.

Fit over all 84 parts of the balanced partitions P = 4, 16, 64 (baseline weights, `--reps 3`,
same setup as above, 18 threads, release build), `t_part = alpha cost_part + beta rows_part`:

| model | alpha, ms per multiply-add | beta, ms per row | beta / alpha | R^2 | max abs. residual |
|---|---|---|---|---|---|
| cost only | 5.68e-8 | - | - | 0.984 | - |
| cost + rows | 4.79e-8 | 6.62e-3 | 1.38e5 | 0.998 | 60 ms (22 % of that part) |
| cost + rows, fit of t/t_part (relative) | 4.87e-8 | 5.87e-3 | 1.21e5 | - | rms 7.5 %, max 23 % |
| cost + rows + triples | 5.13e-8 | -1.1e-2 (negative) | gamma = 3.5e-2 | 0.9987 | 47 ms |

A triples term does not help (the fit makes beta negative and the largest relative residual
worse), so the model keeps two terms. Refit on the re-measured parts (new weights, `--reps 5`):
alpha 4.97e-8, beta 6.09e-3, beta/alpha 1.23e5, R^2 0.998, so the ratio is stable under the
change of the partition. The row term is about 15 % of the modeled time (1.3e5 x 144,162 rows =
1.9e10 multiply-add equivalents against 1.05e11 counted).

Weights: `N7LineWeights(ms3, row_overhead_madd = kRowOverheadMadd)` adds
`row_overhead_madd` x rows of the triple to its compute weight; the memory weight is unchanged.
`commutator::kRowOverheadMadd = 1.3e5` multiply-add equivalents per row, about the mean of the
three fitted ratios above (1.21e5, 1.23e5, 1.38e5; median 1.23e5) (measured on this machine; refit it on other hardware, any value >= 0 keeps the
partition valid).

Before / after, balanced partition, O16 emax 5 E3max 12, best of 3 (before) and 5 (after), whole
space 6.1-6.3 s. The 1-minute load before each run was 8-17 because of the previous 18-thread run
of the same script (each run started right after the last, as in the 04:54 series, but not on a
cool machine); the first run of each series started at 3.4 (before) and 10.1 (after, P = 16):

| P | max/mean before | max/mean after | load before / after | sum/whole after |
|---|---|---|---|---|
| 4 | 1.127 (1.162 at 05:00) | 1.039 | 15.2 / 16.3 | 0.986 |
| 16 | 1.312 | **1.068** | 3.4 / 10.1 | 0.974 |
| 64 | 1.431 | 1.160 | 15.5 / 16.7 | 0.979 |

The target of max/mean <= 1.10 at P = 16 is met (1.068; per-part times 336-398 ms for 5.2k-26.8k
rows, the last part was 486 ms before). P = 64 improves from 1.43 to 1.16 but stays above 1.10:
its parts are small (80-110 ms), the residual of the two-term fit (up to 23 %) and run-to-run noise
(the load) are of the order of the remaining spread, and the slowest parts are no longer the ones
with the most rows. Not investigated further: a per-term fit of `--terms` times. The whole
(unpartitioned) path is untouched: `n7_bench --emax 3 --e3max 9 --checksum --terms full` prints
the same hashes before and after (`full AH 0xe5f3b639070ccd2f`).

Commands (binaries of the commit before / after; `OMP_PROC_BIND=false`,
`VECLIB_MAXIMUM_THREADS=1`, `sysctl -n vm.loadavg` before each):

```
n7_bench --emax 5 --e3max 12 --reps {3|5} --partition {4|16|64}
python fit: least squares of best_ms on (model_ma, rows) over all "part" lines
```
