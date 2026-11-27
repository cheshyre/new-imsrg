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

## 3. Farm items at emax 10 on one GCD (item 5; preliminary)

v1 N2LOsat Omegas (2 Omegas, native float64, manifest `N2LOsat_omegas_no.toml`), one rank, one
GCD, 7 host threads, `tools/al27/time_farm_gpu.sh`. **Preliminary:** node frontier03560 shared
with the lead's CI (load average 7-15, GPU 0 also used by GPU tests at the start).

| item | L | GPU (s) | CPU 7 cores (s, orchestrator farm-test1) |
|---|---:|---:|---:|
| FM0p_150 | 0 | 3.3 | 27.2 |
| FSigmap1p_150 | 1 | 20.2 | 373 (FSigmap1n_150) |
| FM2p_150 | 2 | 45.5 | |
| FSigmap3p_150 | 3 | 77.8 | |
| FM4p_150 | 4 | 103.7 | |
| FSigmap5p_150 | 5 | 129.4 | |
| TCM | 0 | 786 | |

Every item logged "BCH transform not converged after 12 nested commutators" (the warning
threshold; the limit is 40). TCM is dominated by its host preparation at basis.emax 14 (Moshinsky
table to e2max 28 and the two-body build, ~2 min on 56 cores, here on 7); host RSS peak 22 GB.
