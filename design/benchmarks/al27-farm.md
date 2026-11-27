# 27Al GPU production path and farm timings (Brief A, 27Al weekend, 2026-11-27)

Branch `feature/al27-gpu-production` (lead A). GPU build: preset `frontier-hip` (hipcc / ROCm
7.0.2, Kokkos 5.3.0 HIP gfx90a **with an OpenMP host space**, install
`$IMSRG_DEPS_PREFIX/kokkos-hip-openmp`, D73), rocBLAS device GEMMs, one rank per GCD with 7
OpenMP host threads. CPU build: `frontier-cpu-release` (Kokkos OpenMP, default libsci setup).

## 1. What the GPU path is

No new kernels were needed: operators live in `exec::DeviceBuffer`s in the default memory space,
and the commutators (scalar and tensor), BCH, norms and axpys of the merged M14 code run on the
default execution space, which is the GCD in the HIP build. End to end that gives:

* **Flow** (`driver::Run`, Magnus): H, eta, Omega stay on the device; per step the generator
  reads H's host mirror (`H.SyncToHost()`, ~0.3 GB at emax 10, then host-parallel on the rank's
  OpenMP threads) and writes eta back (`SyncToDevice`). Checkpoints and Omega files sync to the
  host before writing (`io::WriteOperatorBinary`), so their formats are unchanged.
* **Farm** (`transform::TransformItem`): the Omegas sit once per node in the MPI shared window
  (host); `SharedOmegaStack::Omega(k)` materializes one at a time as an `Operator` and syncs it to
  the device, the BCH series runs on the device, the finished operator syncs to the host for the
  writers.
* **One flow / one farm rank per GCD**: `srun ... --gpus-per-task=1 --gpu-bind=closest -c 7`
  (`tools/al27/farm_sweep.sh`), `tools/al27/flows_gpu.sh` (up to 4 flows per node, 14 cores
  each), Kokkos picks the one visible device.
* Fix found on the way: the lazy host-mirror creation of `DeviceBuffer` was not thread-safe; with
  the OpenMP host space host-parallel loops really run threaded in the HIP build (separate
  mirror), so it is now guarded (atomic flag + mutex).

## 2. GPU vs CPU equality (acceptance item 4)

`tools/al27/gpu_check/run_check.sh cpu|gpu DIR` runs the production configuration shrunk to
emax 4 (`tools/al27/gpu_check/al27_emax4_v1.toml`: 27Al, sd shell, NAT, compat
`imsrg_fork_79e9330`, unsplit Magnus, EM500 + cE test inputs, e3max 12), writes the Omegas, then
runs the farm over them for FM0p_150 (L = 0), FSigmap1p_150 (L = 1), FSigmap5n_150 and
FDelta5p_300 (L = 5) and TCM with `transform.dump_native = true` (full-precision `.opb` of every
finished operator; the `.snt` files carry 6-9 digits). `compare_runs.py` compares (2026-11-26,
node frontier03560):

| Quantity | max deviation (relative to max magnitude) | tolerance |
|---|---|---|
| flow file, every column but wall_s, 116 rows (stage 1: 20 steps, stage 2: 94 steps) | 1.7e-14 (worst: norm_eta_1b) | 1e-10 |
| final E0 (-641.31626284446861 vs ...838) | 3.5e-16 | 1e-10 |
| Omega 0 / Omega 1 (native, 68057 values each) | 6.1e-15 / 2.8e-14 | 1e-10 |
| H `.snt`, TCM `.snt` (printed digits) | 0 | 1e-10 |
| farm FM0p_150 (L = 0) | 1.6e-16 | 1e-12 |
| farm FSigmap1p_150 (L = 1) | 3.6e-16 | 1e-12 |
| farm FSigmap5n_150 (L = 5) | 4.9e-16 | 1e-12 |
| farm FDelta5p_300 (L = 5) | 2.0e-16 | 1e-12 |
| farm TCM | 6.1e-16 | 1e-12 |
| farm `.snt` files (printed digits) | 0 | - |

Timings of this check (not benchmarks, node shared with builds and tests): GPU flow 8.3 s for
114 steps; CPU flow 1283 s (the CPU run shared the node with a hipcc build and GPU tests; the
threaded libsci behind the BLAS lock degrades badly under oversubscription, AL-E-Q3).

## 3. Farm items at emax 10 on one GCD (item 5)

Job 6006593 (dedicated node frontier00021, idle before; four farm runs side by side, each with one
GCD and 7 host cores, `--exact`): one rank, `tools/al27/timing_job.sh`. Seconds per item:

| item | L | N2LOsat native | N2LOsat ME2JP | DN2LOGO native | DN2LOGO ME2JP | CPU 7 cores (orchestrator farm-test1) |
|---|---:|---:|---:|---:|---:|---:|
| FM0p_150 | 0 | 3.10 | 3.06 | 3.07 | 3.04 | 27.2 |
| FSigmap1p_150 | 1 | 19.2 | 19.1 | 19.1 | 19.1 | 373 (FSigmap1n_150) |
| FM2p_150 | 2 | 40.8 | 40.8 | 39.7 | 39.5 | |
| FSigmap3p_150 | 3 | 69.0 | 69.2 | 69.1 | 69.1 | |
| FM4p_150 | 4 | 95.6 | 95.7 | 95.8 | 95.7 | |
| FSigmap5p_150 | 5 | 125.5 | 125.6 | 125.6 | 125.6 | |
| FM0p_1050 | 0 | 2.64 | 2.63 | 2.71 | 2.62 | |
| FSigmap5p_1050 | 5 | 50.1 | 50.3 | 52.3 | 52.3 | |
| TCM | 0 | 749 | | | | |

* The Omega format does not matter for the time (both are materialized as float64 operators); the
  setup (frame + Omegas into the shared window) takes 1.5-12 s. Farm process: 10.9 GB host RSS
  per rank (8 per node: 87 GB of 512 GB).
* Items with every q at rank 5 (all 426 J = 5 items of each Hamiltonian, jobs 6006670/6006671,
  3 nodes = 24 GCDs each): 426 items in 1003 s (N2LOsat) and 995 s (DN2LOGO), i.e. at most
  56 GCD-seconds per J = 5 item on average (q = 150: 125 s, q = 1050: 50 s).
* GPU vs CPU (same 7 cores): 8.8x for rank 0, 19x for rank 1.
* TCM is host-bound (built at basis.emax 14: Moshinsky table to e2max 28 and the two-body
  part, on 7 cores) and is the longest item; every flow writes TCM already (WE-A-3).
* Every timed item logged "BCH transform not converged after 12 nested commutators" (the
  warning threshold; the limit is 40).

**J = 5 NaN-norm check (AL-A-Q6).** The one-body norm keeps imsrg++'s integer degeneracy, which
is negative for (2j_p, 2j_q, L) = (21, 11, 5) at emax 10; a negative weighted sum gives a NaN
norm (imsrg++: 40 nested commutators; the rewrite: stops with the warning "non-finite nested
commutator"). In all runs above (every timed item, and the rank-0 items of the two J = 5 jobs:
18 of 426 per Hamiltonian) no non-finite norm and no 40-term series occurred. **Incomplete:** the
farm logs only on rank 0 (`Logger::set_rank`), so the warnings of the other ranks' 408 items per
Hamiltonian were not printed; the per-item log line of Brief B (BCH terms) or a comparison of the
852 J = 5 outputs (`$IMSRG_WORK/runs/al27/v1/{N2LOsat,DN2LOGO}/farm_j5check-native/`, native
Omegas, v1) with the archive settles it.

## 4. Cost model (AL-F3) and farm walltime

`CommutatorWeight(L)` at emax 10 grows 1 : 1.97 : 2.86 : 3.61 : 4.20 : 4.57 (L = 0..5), the
measured times 1 : 6.2 : 13.1 : 22.3 : 30.9 : 40.4. Their ratio is 4.55 for L = 0 and
7.6 + 6.5 L for L = 1..5 (within 1 %), so `ItemCost` now multiplies by
`TensorCostFactor(L)` = 1 (L = 0), 1.67 + 1.43 L (L >= 1) and adds TCM's preparation
(`kTcmPreparationCost` = 500 scalar commutator weights, once per item). The q dependence (rank 5: 0.4x at q = 1050 vs
q = 150) is not modeled.

**One farm** (one flow, one Omega format: 2414 NREFT items + TCM): from the q = 150 times
(upper bound, 142 items per family) 149,600 GCD-s = **42 GCD-hours = 5.2 node-hours**; with the
measured J = 5 average (<= 56 s instead of 125 s, the same q factor for every rank) ~65,000 GCD-s
= **18 GCD-hours = 2.3 node-hours**. Longest item: TCM 750 s (without TCM: rank 5 at low q,
~125 s). Walltime per job (+ ~10 s setup, items packed longest first):

| nodes | GCDs | walltime estimate | node-hours | note |
|---:|---:|---|---:|---|
| 1,200 | 9,600 | 12.5 min (TCM-bound; 2.2 min without TCM) | 250 (45) | most GCDs idle |
| 600 | 4,800 | 12.5 min (2.2 min without TCM) | 125 (22) | most GCDs idle |
| 302 | 2,416 | 12.5 min (2.2 min without TCM) | 63 (11) | one item per GCD |
| 76 | 608 | 12.5 min (2-4 min without TCM) | 16 (3-5) | |
| 38 | 304 | 12.5 min (4-8 min without TCM) | 8 (3-5) | recommended (WE-A-2) |

Request 2x the estimate (30 min). The v1 sweep (2 Hamiltonians x 2 formats) plus v2 and v3
(2 x 1 each) is 8 farms: 8 x 2.3-5.2 = 18-42 node-hours of GCD work. 19,320 ranks = 8 farms x
2415 items, one item per GCD: 2415 nodes for ~13 min (~520 node-hours, TCM-bound) in one job.

## 5. Whole v1 flow (N2LOsat, emax 14/10) on one GCD

Job 6006799 (`tools/al27/flows_gpu.sh`, one GCD, 14 host cores, frontier-hip with the OpenMP host
space; run directory `$IMSRG_WORK/runs/al27/v1-gpu/N2LOsat`) against the v1 CPU flow
(56 cores, design/benchmarks/al27-production.md):

| | GPU, 1 GCD + 14 cores | CPU, 56 cores | speed-up |
|---|---:|---:|---:|
| preparation (start to flow step 0: me2j, NO2B, HF, NAT, MBPT) | ~22 min | 16-17 min | 0.75x |
| stage 1, s per step (37 steps) | 1.24 | 13.8 | 11x |
| stage 2, s per step (272 steps) | 2.10 | 26.4 | 12.6x |
| whole run incl. TCM, writers, Omegas in two formats | 40 min | ~1 h 55 min (+ restart) | 2.9x |
| peak host RSS | 66.1 GB | 65.8 GB | |

Results: every common row of the flow files (194 (stage, step) rows: the CPU file of the
restarted run) agrees to 3.5e-14 in E0, 2.6e-13 in ||Omega_2b||, 5.4e-13 in ||eta_2b|| (relative
to the column maximum); final E0 -213.961459717498 vs -213.961459717501; the two native Omegas
agree to 2.8e-13 and 2.1e-12 (36.7 million values each, identical headers). The `.snt` files are
not comparable digit by digit: the CPU files of v1 were written by the imsrg-devel writer, the GPU
run by the merged fork writer (`TokyoNumberFormat::kFork79e9330`, 6-7 digits, Aref 16), so they
differ by the rounding (max 5.0e-7). Four flows per node (one per GCD) make the preparation the
largest part of a flow; restarts redo it (AL-D1, Brief B).
