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

Pending: `n7_bench --partition P` (plan section 3.13) is not implemented yet.

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

Open items, in priority order:
1. `CommutatorOptions::imsrg3_batch_bytes` after the M12b merge (M10-Q74).
2. C4 (exec `TeamFor`/`TeamReduce`, M10-Q44/Q65/Q77) and C15 (223 locality, M10-Q41/Q42; needs
   quiet-machine timings at emax 4-6).
3. `n7_bench --partition P` and the simulated-partition table (section 6).
4. The H build looks like the least efficient piece: estimated from the single-term times at
   emax 4 (331 = H + contraction, 231 = H + W + contraction), H runs at about 4e9 multiply-adds/s
   and W at about 1.7e10. `BuildH` reads B through `ThreeBodyLocal` once per E-row entry; a tiled
   H build (load the tile's rows into scratch first) is the next kernel optimization to try.
5. Factor the group/leg table builders duplicated in `commutator/triples.cc` (M10-Q75 note); the
   `TriplesContext` build at emax 12 takes ~150 s and 4.4 GB on the host.
