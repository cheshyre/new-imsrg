# IMSRG(3)-N7 performance: row tiling, fused driver, triples context (run 3, brief C)

Status: **draft** (2026-11-17 02:25 JST), branch `feature/m10-n7-perf-2`. Design: `design/14-imsrg3-plan.md`
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

## 3. Timings before/after (O16, AH operands, best of 5, ms)

"Before" = develop e739622 + n7_bench (binary kept in worktree `agent-a9d112cd8777f9b77`),
measured by the first lead at load 8-11, **to be repeated at load <= 6**. "After" = this branch,
**pending** (the machine was loaded by the parallel team's builds, load 12-22, at 02:15).

| emax / E3max | 133 | 330 | 223 | 232 | 331 | 231 | 132 | full N7 | load |
|---|---|---|---|---|---|---|---|---|---|
| 3 / 9 before | 141 | 0.37 | 15.9 | 54.5 | 31.2 | 2.7 | 5.9 | 258 | 8-11 |
| 3 / 9 after | | | | | | | | | |
| 4 / 12 before | 4282 | 5.2 | 280 | 1239 | 374 | 9.5 | 60 | 6269 | 8-11 |
| 4 / 12 after | | | | | | | | | |

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

## 7. Hand-off (open items, in priority order)

1. Timings of section 3 (load <= 6, interleaved before/after, emax 3-5; emax 6 per term).
2. Review findings (Opus review of e739622..e1aff9e, no blocker): (a) partition helper should
   check output ownership (`OnlyRowsNonzero`) for every term, not only 223; (b) masked check with
   one batch per part (a read of another batch's lines in the same part is not caught today);
   (c) 133 rebuilds its slot table over whole channels per batch and copies `ob_offsets` to the
   host per batch (performance at emax 12); (d) batch buffers sized from per-batch maxima of H and
   W separately and TransposeOut not counted in the budget; (e) `CommutatorOptions::
   imsrg3_batch_bytes` not added (the budget is `kN7BatchBytes`; plumbing needs a signature change
   that conflicts with M12b); (f) `test_triples_context` uses 1e-14 relative instead of the M3 form;
   (g) `three_body.cc` reinterprets an int32 buffer as `TileWorkItem*`; (h) group/leg table
   builders duplicated in `triples.cc`; (i) `row_mask.h` doc vs `SyncToHost`; (j) the
   occ_nat3_cut driver test should assert the kept-row count.
3. `occ_nat3_cut` for the perturbative triples (E-PLANNED kept; C12 now provides the context).
4. C4 (exec `TeamFor`/`TeamReduce`), C15 (223 locality), `n7_bench --partition`.
