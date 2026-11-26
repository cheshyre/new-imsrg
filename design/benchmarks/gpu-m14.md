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
| 8 | **16.4** | 65 | 430-590 | 170-186 | 4.0x |
| 10 | **64.0** | 339 | - | - | 5.3x |

Per term at emax 10 (ms): GPU / CPU (serial libsci, 56): comm221 12.8 / 137, comm122 12.6 / 56,
pp_hh 9.7 / 150, ph 34.2 / 127, pp_hh_221 14.9 / 156. GPU history at emax 10: 105 ms with the
host-shaped kernels (only the GEMMs moved to rocBLAS) -> 103 (team-per-column variants, no gain)
-> 78 ms (flat element kernels) -> 64 ms (comm122 bra part as dense GEMMs of the lifted one-body
operators). Remaining hot spot: the ph inverse transform (25 ms, team per ket). A hoisted
variant of the tensor phst inverse (9j box and cc indices per element, private arrays) was
slower (2133 -> 2367 ms at emax 10 rank 5, register spills) and was dropped. At emax 8 the host-shaped kernels took 102 ms (ph inverse alone
73 ms: one thread per ket starves the GPU).

**Finding for the CPU (FP-Q6):** with the default BLAS setup the CPU commutator is 6-7x slower
than with the serial libsci and concurrent GroupedGemm items (emax 8: 430 vs 65 ms on 56 cores;
16 threads: 170 ms vs Mac/Accelerate 30.5 ms earlier). Proposal: make `libsci_gnu` +
`IMSRG_BLAS_SERIALIZE=OFF` + `IMSRG_BLAS_CONCURRENT=ON` the `frontier-cpu*` default (the serial
libsci is reentrant, FP-Q6), after checking the Python thread test with it.

## Tensor commutator (all scalar-tensor terms, ms)

| emax | rank | GPU, 1 GCD | CPU 56 cores (serial libsci + concurrent) | speed-up |
|-----:|-----:|-----------:|------------------------------------------:|---------:|
| 8 | 5 | **389** (first call 3.3 s) | 2617 (first call 19.3 s) | 6.7x |
| 10 | 2 | **869** (first call 3.1 s) | 6752 (first call 16.7 s) | 7.8x |
| 10 | 5 | **2456** (first call 13.0 s) | 21534 (first call 86.7 s) | 8.8x |

Per term, emax 10 rank 5 (ms), GPU / CPU: comm121st 90 / 18, comm122st 114 / 1477, comm221st
99 / 1102, pp_hh 101 / 1102, comm222_phst 2150 / 18933. Device memory at emax 10 rank 5: peak
23.2 GB of the 64 GB HBM (sysfs `mem_info_vram_used`, sampled every 0.2 s; the comm122st GEMM
path adds four Z-sized scratch buffers), host RSS 3.6 GB; it fits. The CPU column is the same code (the device shapes keep the host order).
The new tensor ladder (occupation weights folded into scaled copies of the scalar X, grouped
GEMMs) also helps the CPU: emax 8 rank 5 comm221st 2074 -> 135 ms and comm222_pp_hhst 2097 -> 134
ms, all terms 5298 -> 2617 ms on 56 cores. Lead A measured 63 s at 16 threads (default BLAS,
before the comm122st lists) for emax 10 rank 5.

GPU history at emax 8 rank 5: 1105 ms (GEMMs on rocBLAS, host-shaped kernels) -> 505 ms
(comm221st team reduction over orbits, 319 -> 18.5 ms; phst inverse NestedFor, 617 -> 310 ms)
-> 467 ms (comm122st NestedFor, 136 -> 98 ms) -> 389 ms (comm122st as GEMMs: the lifted scalar
one-body operators L, R per channel and the part-2 lists folded into B, K, then
G1 = L_b Y + B X_kk, G2 = X_bb K - Y R_k^T, four grouped GEMM calls; 98 -> 21 ms; emax 10 rank 5
883 -> 114 ms). Remaining hot spot: the phst inverse transform (9j x Zbar gathers, ~7/8 of the
time at emax 10 rank 5).

## Device vs host results

`commutator_bench --dump` / `tensor_commutator_bench --dump` write [X, Y] as raw doubles;
`tools/bench/compare_dumps.py host.bin device.bin` compares them (max|dZ| / max(1, max|Z|) per
part, tolerance 1e-12). GPU (rocBLAS, device shapes) vs CPU (OpenMP, libsci): scalar emax 8
1.0e-15, emax 10 2.0e-15; tensor rank 5 emax 8 3.5e-15, rank 2 emax 6 1.7e-15 (worst part).
All goldens pass on the GCD (frontier-hip-run: 1328 of 1333 unit tests, all 76 integration
tests; the 5 failures are the bitwise atan tests of AL-E-Q2).

## Device BLAS: rocBLAS vs Kokkos Kernels (owner rule: vendor path only at >= 5x)

* **rocBLAS** (implemented, default for hip): `rocblas_dgemm` on the Kokkos stream; grouped calls
  fork onto 4 side streams (largest problem first, least loaded stream) and join with events.
  GEMM share of the kernel time: scalar emax 10 12.5 ms of ~110 ms (11 %, 248 GEMMs per
  commutator) before the comm122 GEMMs, ~26 ms of ~64 ms (~40 %) after them; tensor emax 10 rank 2 ~14 % (3888 GEMMs per three calls).
* **Kokkos Kernels**: the backend is written (`IMSRG_DEVICE_BLAS=KOKKOSKERNELS`, KokkosBlas::gemm
  on LayoutLeft subviews; `tools/bootstrap_deps.sh` builds `kokkoskernels-hip` from a tarball)
  but **not measured**: the Kokkos Kernels 5.3.0 source could not be downloaded from the agent
  session (the worktree hook rejects commands containing the GitHub URL; compute nodes have no
  internet; the Frontier `kokkos-kernels` modules are 3.x-4.6 and do not match Kokkos 5.3).
  To measure: put `kokkos-kernels-5.3.0.tar.gz` into `$IMSRG_DEPS_PREFIX/src/`, run
  `tools/bootstrap_deps.sh --backends hip` on the compute node, configure frontier-hip with
  `-DIMSRG_DEVICE_BLAS=KOKKOSKERNELS`, rerun the three benchmarks.
* **Bound on the choice:** the GEMMs are ~14 % of the tensor and ~40 % of the scalar GPU time.
  A Kokkos Kernels GEMM 5x slower than rocBLAS (plausible: rocBLAS uses the MFMA matrix cores)
  would make the tensor commutator ~1.6x and the scalar one ~2.6x slower in total; an equally
  fast one would gain nothing. Per the owner's rule the vendor path is preferred only if the
  measured benefit is >= 5x; that needs the measurement above.

**Proposed D-row (pending the KK measurement):** "M14 device BLAS: `linalg::DeviceGemm` /
`DeviceGroupedGemm` with configure-time backends HOST (CPU), ROCBLAS, KOKKOSKERNELS. Default
for hip: KOKKOSKERNELS if its GEMM time on the emax-10 scalar and the emax-10 rank-2 tensor
commutator is within 5x of rocBLAS (owner: portability first), otherwise ROCBLAS. Until
measured, ROCBLAS is the default (it is the only tested one)."

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
