# 27Al restart check (D75 acceptance, production conditions)

Question: is a flow killed by SIGTERM and restarted from its checkpoint bit-identical to an
uninterrupted one, on a GPU at emax 10? Case: v3 N2LOsat (`tools/al27/n2losat_v3.toml`, emax 10,
H_0 from v2's `prepared/`, 42 Omegas, 17 GB host RSS), one MI250X GCD per run, frontier-hip.

## Setup

* Binary: `imsrg.gpu-20261127-183256` (the copy the reference run used, frontier-hip at 42a46f7)
  for all runs, so that only the interruption differs. The branch also builds with
  `tools/al27/build_hip.sh` (checked: links); that build was not used for the runs.
* Reference: `$IMSRG_WORK/runs/al27/v3/N2LOsat/` (job 6015996).
* Runs in `$IMSRG_WORK/runs/al27/restart-check/`: `v3-N2LOsat-straight/` (one link) and
  `v3-N2LOsat-killed/`, both through `tools/al27/flows_chain.sh` with `MAX_LINKS=1`. Killed: link 1
  `WALLTIME=00:06:00 SIGNAL_LEAD=300` (Slurm sends TERM about 60 s after job start, i.e. during the
  Magnus steps: the flow itself runs from ~3 s to ~110 s after the process start), link 2 submitted
  by hand (`CHAIN_LINK=2 sbatch -t 00:30:00 chain-*.sh`; the chain continues with
  `checkpoint.restart=true`).
* Attempt 1 (jobs 6017687, 6017688, 6017694) ran before `flows_chain.sh` overrode
  `output.flow_file`; see "Incident". Attempt 2 is the clean one; attempt 1 is kept in
  `*-attempt1/` and gave the same results.

## Where TERM landed

| attempt | link 1 job | TERM sent | landed | exit |
|---|---|---|---|---:|
| 1 | 6017688 | 21:08:02 (about 36 s into the flow) | stage 2 step 11 (s = 13.19, 13 Omegas) | 75 |
| 2 | 6017813 | 21:22:04 (about 32 s into the flow) | stage 2 step 3 (s = 13.08, 10 Omegas), checkpoint 6.7 s | 75 |

Both inside stage 2 (shell_model_atan) with 10-13 Omegas pushed, not during the preparation. Link 2
(6017827, 21:24:39, 11:03) read `checkpoint/prepared` and the flow state, finished the flow and
wrote everything.

## Comparison (attempt 2; attempt 1 identical in all rows but the flow file)

| Quantity | killed vs reference | straight vs reference | killed vs straight |
|---|---|---|---|
| E0 (`results.json`, -0x1.acdb86c42a9c5p+7 = -214.428763) | identical | identical | identical |
| 42 Omega files `*_omega_000..041.opb` (293 MB each, `cmp`) | identical | identical | identical |
| `N2LOsat.snt`, `N2LOsat_TCM.snt` (`cmp`) | identical | identical | identical |
| `N2LOsat_omegas.toml`, `_occ.dat`, `_occ_no.dat` | identical | identical | identical |
| TCM imsrg value and Omega count (42) in `results.json` | identical | identical | identical |
| flow table `flow.txt`, all columns except `wall_s` | 143 rows written after the restart equal the straight rows (see issue) | not comparable (original lost, see incident) | identical on those 143 rows |

47 files compared per pair (42 Omegas + 5), 0 differences. Run-to-run determinism of the GPU flow
holds: the straight re-run equals the reference bit for bit (E0, all 42 Omegas, both `.snt`).
Stage-end table (flow_summary.py): killed and straight both end at s = 48.85381, E0 -214.428763,
42 Omegas; the straight table has 45 + 147 rows as the reference summary (-131.608497 /
-214.428763).

## Result

A killed (SIGTERM mid-Magnus) and restarted v3 flow at emax 10 on the GPU is bit-identical in E0,
every native Omega, the H and TCM `.snt` files and the Omega manifest. D75 acceptance holds in
production conditions.

## Issue (not numerical): flow.txt after a restart

The restarted link rewrites `flow.txt` from the restart step on: the killed run's `flow.txt` has 143
data rows (stage 2, step 4 to 146) and none of link 1's rows (stage 1, stage 2 steps 0-3), so
`flow_summary.py` then shows only stage 2. The rows that exist are bit-identical to the straight
run. Recorded as WE-R-1 in `design/QUESTIONS.md`; a fix would open the flow file in append mode on
restart (src/, not changed here).

## Incident: production flow.txt overwritten (fixed)

The v3 configs name `output.flow_file` as an absolute path in the production run directory and
`flows_chain.sh` overrode only `output.directory` and `checkpoint.directory`, so attempt 1 wrote
`runs/al27/v3/N2LOsat/flow.txt` (several concurrent writers: the file was inconsistent). Nothing
else below `runs/al27/{v1,v2,v3}` changed after 20:50 (`find -newer`). Fix on this branch:
`flows_chain.sh` now also sets `output.flow_file=RUN_DIR/flow.txt` (the other path keys are
inputs: `checkpoint.prepared_from`, `interaction.*.file`, `operators.from_file`); attempt 2 used it
and left the production directory untouched. Because the straight re-run of attempt 2 is
bit-identical to the reference, the production `flow.txt` was restored from the straight run's own
flow file (same rows, `wall_s` column from the re-run).

| file | sha256 |
|---|---|
| corrupt (attempt 1; kept as `restart-check/flow.txt.corrupt-attempt1`) | 1a882017416806e15b5ce688f5d14cfb6b9e4cb58480683f8bbe0a9040082981 |
| restored `v3/N2LOsat/flow.txt` (= `restart-check/v3-N2LOsat-straight/flow.txt`) | 1feac99bbcf38c05346dd2a23836bad85395564f5ee1f8ea76342f679f235b42 |

## Cost

About 0.8 node-hours for both attempts (`design/benchmarks/al27-jobs.md`, "Restart check").
