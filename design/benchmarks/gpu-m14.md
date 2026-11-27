# IMSRG(2) commutators on one MI250X GCD (M14 / C11, 2026-11-26)

Branch `feature/m14-gpu-commutators`. One Frontier node of the Slurm job `imsrg-a` (8 GCDs, 56
usable cores), shared with other agents' builds (CPU numbers can be noisy by ~10 %). Release
builds; benchmarks `tools/bench/commutator_bench` (random anti-Hermitian X, Hermitian Y, O16
reference), `tools/bench/tensor_commutator_bench` (scalar X, Hermitian rank-L tensor Y),
`tools/bench/bch_bench` (one BCH transform). Times are means of the reps after one warm-up call.

* **GPU:** preset `frontier-hip` + `-DIMSRG_BUILD_BENCH=ON` (hipcc / ROCm 7.0.2, Kokkos 5.3.0 HIP
  gfx90a, device BLAS rocBLAS, `-ffp-contract=off`), one GCD: `srun --overlap -n1 -c7 --gpus=1`.
* **CPU:** preset `frontier-cpu-release` + `-DIMSRG_BUILD_BENCH=ON`, 56 OpenMP threads
  (`OMP_PROC_BIND=spread OMP_PLACES=cores`). Two BLAS setups:
  * **default** (as merged): `libsci_gnu_mp` (threaded cray-libsci), every BLAS call under the
    `IMSRG_BLAS_SERIALIZE` lock, GroupedGemm items one after another;
  * **serial libsci + concurrent items** (new option `IMSRG_BLAS_CONCURRENT`):
    `-DBLAS_LIBRARIES=-DLAPACK_LIBRARIES=$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_gnu.so
    -DIMSRG_BLAS_SERIALIZE=OFF -DIMSRG_BLAS_CONCURRENT=ON` (one single-threaded BLAS call per host
    thread, as with Accelerate on the Mac). This is the fair CPU baseline below.

## Scalar IMSRG(2) commutator (full `Commutator()`, ms)

| emax | GPU, 1 GCD | CPU 56 cores, serial libsci + concurrent | CPU 56 cores, default BLAS | CPU 16 cores, default | speed-up GCD / 56 cores |
|-----:|-----------:|-----------------------------------------:|---------------------------:|----------------------:|------------------------:|
| 8 | **13.0** | 65 | 430-590 | 170-186 | 5.0x |
| 10 | **58.0** | 339 | - | - | 5.8x |

Per term at emax 10 (ms): GPU / CPU (serial libsci, 56): comm221 12.8 / 137, comm122 12.6 / 56,
pp_hh 9.7 / 150, ph 34.2 / 127, pp_hh_221 14.9 / 156. GPU history at emax 10: 105 ms with the
host-shaped kernels (only the GEMMs moved to rocBLAS) -> 103 (team-per-column variants, no gain)
-> 78 ms (flat element kernels) -> 64 ms (comm122 bra part as dense GEMMs of the lifted one-body
operators) -> 58 ms (comm221ss / comm121ss one team per (i, j)). Remaining hot spot: the ph inverse transform (25 ms, team per ket). A hoisted
variant of the tensor phst inverse (9j box and cc indices per element, private arrays) was
slower (2133 -> 2367 ms at emax 10 rank 5, register spills) and was dropped. At emax 8 the host-shaped kernels took 102 ms (ph inverse alone
73 ms: one thread per ket starves the GPU).

**Finding for the CPU (FP-Q6):** with the default BLAS setup the CPU commutator is 6-7x slower
than with the serial libsci and concurrent GroupedGemm items (emax 8: 430 vs 65 ms on 56 cores;
16 threads: 170 ms vs Mac/Accelerate 30.5 ms earlier). Proposal: make `libsci_gnu` +
`IMSRG_BLAS_SERIALIZE=OFF` + `IMSRG_BLAS_CONCURRENT=ON` the `frontier-cpu*` default (the serial
libsci is reentrant, FP-Q6), after checking the Python thread test with it.

## Tensor commutator (all scalar-tensor terms, ms)

| emax | rank | GPU, 1 GCD | CPU 56 cores, final code | CPU 56 cores, before M14 | GCD / CPU final |
|-----:|-----:|-----------:|-------------------------:|-------------------------:|----------------:|
| 8 | 5 | **261** (first call 3.1 s) | 1134 (first call 17.8 s) | 5298 | 4.3x |
| 10 | 2 | **875** (first call 3.2 s) | 3888 (first call 13.7 s) | 6752 | 4.4x |
| 10 | 5 | **1509** (first call 11.3 s) | 7408 (first call 72.7 s) | 21534 | 4.9x |

"Before M14" = the merged M15 code (host tensor ladder, element-wise phst inverse) with the serial
libsci and concurrent GroupedGemm. Two M14 changes help the CPU as much as the GPU: the tensor
ladder (occupation weights folded into scaled copies of the scalar X, grouped GEMMs: emax 8 rank
5 comm221st 2074 -> 135 ms, comm222_pp_hhst 2097 -> 134 ms) and the quadruple form of the phst
inverse transform (below: emax 10 rank 5 18.9 -> 4.9 s, rank 2 5.3 -> 2.4 s). Lead A measured
63 s at 16 threads (default BLAS) for emax 10 rank 5 before M14.

Per term, emax 10 rank 5 (ms), GPU / CPU final: comm121st 89 / 18, comm122st 114 / 1477,
comm221st 98 / ~1100, pp_hh 102 / ~1100, comm222_phst 1206 / 4861. Device memory at emax 10 rank
5: peak 23.2 GB of the 64 GB HBM (sysfs `mem_info_vram_used`, sampled every 0.2 s; the comm122st
GEMM path adds four Z-sized scratch buffers), host RSS 3.6 GB; it fits.

GPU history at emax 8 rank 5: 1105 ms (GEMMs on rocBLAS, host-shaped kernels) -> 505 ms
(comm221st team reduction over orbits, 319 -> 18.5 ms; phst inverse NestedFor, 617 -> 310 ms)
-> 467 ms (comm122st NestedFor, 136 -> 98 ms) -> 389 ms (comm122st as GEMMs: the lifted scalar
one-body operators L, R per channel and the part-2 lists folded into B, K, then
G1 = L_b Y + B X_kk, G2 = X_bb K - Y R_k^T, four grouped GEMM calls; 98 -> 21 ms; emax 10 rank 5
883 -> 114 ms) -> 261 ms (phst inverse in quadruple form: one team per ket pair gathers the
Zbar^{J3 J4} values once into team scratch for all (J1, J2) blocks instead of once per block,
9j box looked up once per (J1, J2); bitwise equal to the element form; 310 -> 180 ms, emax 10
rank 5 2150 -> 1206 ms). On the device the quadruple form is used for L >= 3 only (emax 10,
element / quad ms: L = 1 275 / 583, 2 675 / 775, 3 1173 / 944, 4 1642 / 1094, 5 2150 / 1206);
the host uses it always. Remaining hot spot: the phst inverse (~80 % at emax 10 rank 5).

## Full IMSRG(2) Magnus flow (O16, IT-2 settings, 2N only, atan generator)

`apps/imsrg/imsrg 2bme=<EM500 me2j> emax=E reference=O16 valence_space=O16 method=magnus` (the
settings of `tools/bench/flow_bench.sh`), flow time = last `wall_s` of the flow file; emax 8 is a
D40 benchmark exception.

| emax | GPU, 1 GCD | CPU 56 cores (serial libsci + concurrent) | speed-up | steps, E0 |
|-----:|-----------:|------------------------------------------:|---------:|-----------|
| 6 | **8.6 s** | 17.3 s | 2.0x | 219, -171.232126375213 (both, to 2e-13) |
| 8 | **28.0 s** | 81.1 s | 2.9x | 209, -172.796123844736 (both, to 3e-13) |

History of the GPU flow: emax 6 24.0 s / emax 8 41.6 s before the norm fix -> 8.6 / 28.0 s. The
two-body norm (`WeightedBlockSumOfSquares`, 4096-element chunks per work item) was a third of
the flow's kernel time (3.4 ms per call at emax 8, ~16 norms per step); with 16-element chunks
on the device it takes 0.5 ms. comm221ss and comm121ss (one thread per (i, j) pair) became one
team per pair (emax 8: 2.6 -> ~0.6 ms and 1.1 -> 0.15 ms). At emax 6 the remaining host time is
launch overhead (HSA runtime, ~250 small rocBLAS calls per commutator). With an OpenMP host space
in the HIP Kokkos (experiment, AL-E-Q5; 7 threads) the emax 8 flow takes 23.9 s instead of 28.0 s.

## Device vs host results

`commutator_bench --dump` / `tensor_commutator_bench --dump` write [X, Y] as raw doubles;
`tools/bench/compare_dumps.py host.bin device.bin` compares them (max|dZ| / max(1, max|Z|) per
part, tolerance 1e-12). GPU (rocBLAS, device shapes) vs CPU (OpenMP, libsci): scalar emax 8
1.0e-15, emax 10 2.0e-15; tensor rank 5 emax 8 3.5e-15, rank 2 emax 6 1.7e-15 (worst part).
All goldens pass on the GCD (after merging develop 2026-11-26 14:00: 1352 of 1357 unit tests, all
85 integration tests with three at a time; the 5 failures are the bitwise atan tests of AL-E-Q2).
The HIP build's host space is Serial (AL-E-Q5), so host-side parts (tables, generator, IMSRG(3)
host loops) run on one core there.

## Device BLAS: rocBLAS vs Kokkos Kernels (owner rule: vendor path only at >= 5x)

* **rocBLAS** (implemented, default for hip): `rocblas_dgemm` on the Kokkos stream; grouped calls
  fork onto 4 side streams (largest problem first, least loaded stream) and join with events.
  GEMM share of the kernel time: scalar emax 10 12.5 ms of ~110 ms (11 %, 248 GEMMs per
  commutator) before the comm122 GEMMs, ~26 ms of ~64 ms (~40 %) after them; tensor emax 10 rank 2 ~14 % (3888 GEMMs per three calls).
* **Kokkos Kernels** (measured 2026-11-26, branch `feature/m14-kokkos-kernels-measure`):
  Kokkos Kernels 5.3.0 built with `tools/bootstrap_deps.sh --backends hip` (native kernels, no TPLs,
  ETI double/LayoutLeft; build ~5 min on the node), frontier-hip configured with
  `-DIMSRG_DEVICE_BLAS=KOKKOSKERNELS` in `build/hip-kk` against `build/hip-rocblas` (ROCBLAS),
  `-DIMSRG_BUILD_BENCH=ON`, one GCD, back to back, same node:

  | benchmark | rocBLAS | Kokkos Kernels | rocBLAS benefit |
  |-----------|--------:|---------------:|----------------:|
  | scalar commutator emax 8 (ms) | 13.0 | 60.2 | 4.6x |
  | scalar commutator emax 10 (ms) | 58.4 | 366.7 | 6.3x |
  | tensor rank 5 emax 10, all terms (ms) | 1537 | 5556 | 3.6x |
  | tensor first call (tables, s) | 11.3 | 15.0 | - |

  Tensor terms (ms, rocBLAS / KK): comm121st 87 / 92, comm122st 108 / 1903, comm221st 97 / 1882,
  comm222_pp_hhst 100 / 1884, comm222_phst 1233 / 1683 (the GEMM-heavy terms carry the gap).
  Results: [X, Y] dumps of the two builds agree to 1.1e-15 / 2.1e-15 (scalar emax 8 / 10) and
  2.3e-15 (tensor rank 5 emax 10), relative, limit 1e-12. Unit tests with Kokkos Kernels on the
  GCD: 1353 of 1358 pass, the same 5 `flow_generator3_kernel` Bitwise failures as with rocBLAS
  (AL-E-Q2); all commutator, linalg and tensor tests pass.
* **Owner rule** (verbatim): "Try both. If there is a significant performance benefit (5x or more)
  prefer the vendor-tuned path. But portability is better long term." Outcome: rocBLAS wins by
  4.6x (scalar emax 8), 6.3x (scalar emax 10) and 3.6x (tensor rank 5 emax 10). Only the large
  scalar case crosses 5x, the tensor commutator and the small scalar one do not. **Selection:**
  the scalar emax-10 case is the one where the GPU matters for flows and clears the threshold, so
  the rule picks rocBLAS there; the tensor case alone (3.6x) would pick Kokkos Kernels. We keep
  **rocBLAS as the hip default** with Kokkos Kernels as the portable option (`-DIMSRG_DEVICE_BLAS=
  KOKKOSKERNELS`); owner may overrule (borderline, see AL-E-Q1 follow-up).
  (An earlier bound estimate of the GEMM share, ~40 % scalar / ~14 % tensor, understated the gap:
  KK GEMMs are far slower than 5x for the grouped small-matrix shapes.)

**Proposed D-row (pending the KK measurement):** "M14 device BLAS: `linalg::DeviceGemm` /
`DeviceGroupedGemm` with configure-time backends HOST (CPU), ROCBLAS, KOKKOSKERNELS. Default
for hip: KOKKOSKERNELS if its GEMM time on the emax-10 scalar and the emax-10 rank-2 tensor
commutator is within 5x of rocBLAS (owner: portability first), otherwise ROCBLAS. Measured: KK is 4.6x / 6.3x / 3.6x slower, so ROCBLAS stays the default."

## Device residency across a BCH series

`rocprofv3 --memory-copy-trace` of `bch_bench --emax 8 --omega-norm 2` (warm-up + one timed
transform, 4 nested commutators each): all host-device copies larger than a few kB happen during
setup (random operators written on the host, then `SyncToDevice`); the two BCH transforms make
10 device-to-host copies of at most 8 us in total (one-body norms read on the host). No operator
data moves per commutator. Time per nested commutator inside BCH 22.7 ms vs 21.7 ms for the bare
commutator (allocation, norms, axpy).

## C12 preparation: 8 ranks x 1 GCD vs 1 rank x 1 GCD (scalar commutator, emax 10)

| ranks x GCDs | full commutator | of which combine (reduce + bcast) |
|-------------:|----------------:|----------------------------------:|
| 1 x 1 | **77.6 ms** | - |
| 2 x 2 | 454 ms | 290 ms |
| 8 x 8 | 656 ms | 497 ms |

The M12b phase-1 combine copies Z (0.29 GB at emax 10) to the host and reduces it with host
MPI on every commutator, and the distributed parts run the host-shaped kernels (the device shapes
are only used for undistributed work), so the per-rank terms do not even shrink. **For flows on a
node: one rank per GCD, each running its own flow (8 independent flows per node)**; distributing
one commutator over GCDs needs GPU-aware MPI and device shapes for the owned-column paths first.

## Reproduce

```
tools/ci.sh frontier-hip-run      # unit + integration tests on one GCD
cmake --preset frontier-hip -B build/hip-bench -DIMSRG_BUILD_BENCH=ON && cmake --build build/hip-bench
srun --jobid=<imsrg-a> --overlap -N1 -n1 -c7 --gpus=1 build/hip-bench/tools/bench/commutator_bench --emax 10 --reps 3
srun ... build/hip-bench/tools/bench/tensor_commutator_bench --emax 10 --rank 5 --reps 2
srun ... -n 8 --gpus-per-task=1 --gpu-bind=closest build/hip-bench/tools/bench/commutator_bench --emax 10 --reps 3
```
