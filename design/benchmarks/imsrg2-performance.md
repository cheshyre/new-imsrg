# IMSRG(2) single-node CPU performance vs imsrg++ (M12a)

Owner decision D40: first reach imsrg++ parity on one node, then explore how far one node
goes, judged mainly at larger emax. This page compares the IMSRG(2) commutator and a full
IMSRG(2) Magnus flow (O16, hw 16, 2N only) with imsrg-devel on the same machine at emax 4, 6
and 8, gives the Workspace memory and the thread scaling (1, 6, 18 threads), and lists what
changed and what did not help. The M7 micro-benchmark that started this is
`imsrg2-commutators.md`.

## Summary

- **Commutator** (full IMSRG(2) commutator, random operators): this code is about 4x faster
  than imsrg++ on one thread at emax 4-8, 2.3x / 3.0x / 3.2x on six threads at emax 4 / 6 / 8,
  and 2.4-3.0x on 18 threads at emax 6-8 (table below). Before M12a (`develop`) it was 1.2-1.4x
  slower than imsrg++.
- **Full IMSRG(2) flow** (same step counts and E0 to all printed digits in both codes): 2.6-3.8x
  faster than imsrg++ at emax 6 and 8 on 1, 6 and 18 threads (emax 8 on six threads: 60 s vs
  192 s), 4x at emax 4 on one thread and 2.3x on six (table below).
- **Attribution.** imsrg++ links Homebrew OpenBLAS, this code Accelerate. Built against the
  same OpenBLAS, this code is still 2.0-2.6x faster on the commutator at emax 6-8 (1.6x at emax
  6 on six threads): the particle-hole term is 3.2x faster with the same BLAS. The ladder terms
  are GEMM-bound: with OpenBLAS they take 0.7-0.8x imsrg++'s time, with Accelerate (AMX) they
  are 4.5-6x faster on one thread.
- **Memory.** Workspace scratch at emax 6: 10.4x -> 5.5x a two-body operator (emax 8: 558 ->
  288 MB). Peak RSS of the whole flow: 169 vs 431 MiB at emax 6, 1.10 vs 2.51 GB at emax 8 (this code vs imsrg++, six threads).
- **Threads.** On this machine (6 performance + 12 efficiency cores) 6 -> 18 threads gives
  1.3-1.6x at emax 6-8 for both codes; at emax 4 the 18-thread runs are slower than the
  6-thread ones in both codes (efficiency cores, many small kernels).

Full commutator, ms per call (best of N; N and loads in the per-term tables):

| emax | threads | before (develop) | this code | this code, OpenBLAS | imsrg++ | ours / imsrg++ |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 6.708 | 1.160 | 1.508 | 5.212 | 0.22 |
| 4 | 6 | 2.176 | 0.649 | 0.961 | 1.526 | 0.43 |
| 4 | 18 | - | 1.276 | - | 1.425 | 0.90 |
| 6 | 1 | 100.3 | 18.2 | 27.4 | 70.6 | 0.26 |
| 6 | 6 | 24.9 | 5.112 | 9.745 | 15.2 | 0.34 |
| 6 | 18 | - | 3.887 | - | 9.470 | 0.41 |
| 8 | 1 | 865.6 | 168.6 | 307.8 | 670.9 | 0.25 |
| 8 | 6 | 198.7 | 49.1 | 77.3 | 154.8 | 0.32 |
| 8 | 18 | - | 31.2 | - | 94.7 | 0.33 |

Full IMSRG(2) Magnus flow, seconds (best of N):

| emax | threads | this code flow [s] | imsrg++ flow [s] | ratio | total [s] (ours / ref) | steps | E0 [MeV] (both) | peak RSS [MiB] (ours / ref) | runs | load (1/5/15) |
|---:|---:|---:|---:|---:|---|---:|---|---|---:|---|
| 4 | 1 | 2.06 (max 2.29) | 8.26 (max 9.28) | 0.25 | 2.1 / 8.4 | 256 | -164.648772308 | 36 / 75 | 3 | 10.82/5.75/3.21; 11.25/5.66/3.15; 11.27/5.57/3.10; 3.74/11.15/13.03 |
| 4 | 6 | 1.23 (max 1.91) | 2.85 (max 3.12) | 0.43 | 1.7 / 2.9 | 256 | -164.648772308 | 36 / 80 | 4 | 10.21/5.86/3.29; 10.32/5.81/3.26; 2.97/11.26/13.09; 3.37/11.21/13.06 |
| 4 | 18 | 2.47 (max 2.89) | 2.67 (max 2.75) | 0.93 | 2.6 / 2.7 | 256 | -164.648772308 | 39 / 81 | 3 | 3.84/10.93/12.93; 9.37/5.89/3.34; 9.49/5.85/3.32; 9.79/5.85/3.30 |
| 6 | 1 | 28.73 (max 30.90) | 89.74 (max 97.53) | 0.32 | 29.3 / 91.7 | 219 | -171.232126375 | 168 / 398 | 3 | 11.57/7.13/4.14; 4.44/9.15/12.00; 5.31/9.80/12.33; 7.07/5.66/3.35; 9.37/5.89/3.34 |
| 6 | 6 | 7.87 (max 8.88) | 24.26 (max 24.27) | 0.32 | 8.4 / 25.0 | 219 | -171.232126375 | 169 / 445 | 2 | 4.01/10.63/12.79; 4.02/10.85/12.89; 5.00/10.32/12.61; 5.02/10.14/12.52 |
| 6 | 18 | 6.26 | 16.68 | 0.38 | 6.8 / 17.3 | 219 | -171.232126375 | 170 / 485 | 1 | 3.60/7.61/11.12; 4.38/7.65/11.09 |
| 8 | 6 | 59.79 | 192.31 | 0.31 | 65.3 / 197.5 | 209 | -172.796123845 | 1097 / 2507 | 1 | 3.23/6.17/10.03; 5.33/6.17/9.74 |
| 8 | 18 | 40.49 | 153.18 | 0.26 | 46.1 / 157.1 | 209 | -172.796123845 | 1098 / 2590 | 1 | 13.98/8.72/9.86; 8.58/7.17/9.41 |

## Setup

- Machine: Apple M5 Pro, 6 performance + 12 efficiency cores, 48 GiB, macOS 26.6.2, shared
  with another lead's builds: every number is the best of N runs (N given; "max" is the
  slowest of the N), with the 1/5/15-minute load averages at the time. `OMP_PROC_BIND=false`,
  so 6 threads may land on any core type and 18 threads use the efficiency cores as well.
- Compiler: Apple clang 21.0.0 with Homebrew libomp, for both codes.
- This code: branch `feature/m12a-perf`, preset `release-openmp` (Kokkos 5.3.0 OpenMP, `-O3
  -DNDEBUG -mcpu=native`), BLAS = Accelerate. "This code, OpenBLAS": the same sources configured
  with `-DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/openblas` (Homebrew OpenBLAS,
  OpenMP-threaded via libomp, as in the imsrg++ build; `linalg::GroupedGemm` then runs its
  problems one after another, M12-Q5). "Before (develop)": the commutator of `develop` 7d56da2.
  Commutator: `tools/bench/commutator_bench` (`-DIMSRG_BUILD_BENCH=ON`); flow: `apps/imsrg`
  through `tools/bench/flow_bench.sh`.
- imsrg++: imsrg-devel built by `tools/reference/build_imsrg_devel.sh` (CMake `Release`, `-O3
  -DNDEBUG`, no `-mcpu=native`, OpenMP via libomp, Armadillo on Homebrew OpenBLAS).
  Commutator: `refharness bench-commutator` (pristine build `harness-build-bench`); flow: the
  D35-patched `build-d35/imsrg++` (the e2e goldens' build, same kernels).
- Commutator inputs: random scalar operators (X anti-Hermitian, Y Hermitian), O16 reference;
  two-body sizes 67,156 (emax 4), 927,680 (emax 6) and 7,033,580 elements (emax 8). Times are the
  mean per call over the repetitions (100, 20, 3) after one warm-up call. Our full row is
  `commutator::Commutator` (it resets and reuses Z); imsrg++'s allocates Z per call.
- Flow inputs: the IT-2 settings at the given emax, identical arguments for both codes:
  `2bme=VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz file2e1max=14 file2e2max=28
  file2lmax=14 hw=16 emax=E reference=O16 valence_space=O16 method=magnus`. "flow" is the
  Magnus loop alone (ours: the last `wall_s` of the flow file; imsrg++: last minus first time
  column of its flow file), "total" the whole process (reading the 2N file, HF, MP2, flow,
  output); peak RSS from `/usr/bin/time -l`.

## Commutator per term

The small terms (comm110ss, comm220ss, comm111ss, comm121ss) are below 1% of the total; at 6
and 18 threads comm110ss is dominated by the launch of a parallel region (16-60 microseconds).

### emax 4, 1 thread (ms per call, best of 4; loads 4.32 14.14 14.21, 4.53 14.35 14.28, 7.61 3.51 2.20)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.000 | 0.000 (0.000) | 0.000 | 0.000 | - |
| comm220ss | 0.034 | 0.003 (0.053) | 0.003 | 0.019 | 0.15 |
| comm111ss | 0.001 | 0.001 (0.002) | 0.002 | 0.003 | 0.36 |
| comm121ss | 0.018 | 0.018 (0.032) | 0.018 | 0.043 | 0.42 |
| comm221ss | 0.578 | 0.096 (0.157) | 0.181 | 0.451 | 0.21 |
| comm122ss | 0.767 | 0.222 (0.336) | 0.224 | 0.765 | 0.29 |
| comm222_pp_hhss | 0.563 | 0.088 (0.100) | 0.172 | 0.400 | 0.22 |
| comm222_phss | 5.255 | 0.805 (0.876) | 1.041 | 3.732 | 0.22 |
| comm222_pp_hh_221ss | 0.587 | 0.109 | 0.195 | 0.476 | 0.23 |
| commutator (full) | 6.708 | 1.160 | 1.508 | 5.212 | 0.22 |

### emax 4, 6 threads (ms per call, best of 4; loads 4.19 13.79 14.08, 8.22 4.08 2.46)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.016 | 0.016 (0.054) | 0.016 | 0.000 | - |
| comm220ss | 0.021 | 0.020 (0.058) | 0.022 | 0.053 | 0.39 |
| comm111ss | 0.001 | 0.001 (0.002) | 0.002 | 0.003 | 0.36 |
| comm121ss | 0.022 | 0.022 (0.063) | 0.023 | 0.030 | 0.72 |
| comm221ss | 0.341 | 0.121 (0.263) | 0.216 | 0.203 | 0.60 |
| comm122ss | 0.253 | 0.098 (0.158) | 0.105 | 0.326 | 0.30 |
| comm222_pp_hhss | 0.322 | 0.098 (0.201) | 0.187 | 0.191 | 0.51 |
| comm222_phss | 1.432 | 0.328 (0.598) | 0.559 | 0.829 | 0.40 |
| comm222_pp_hh_221ss | 0.355 | 0.121 (0.269) | 0.211 | 0.232 | 0.52 |
| commutator (full) | 2.176 | 0.649 (1.326) | 0.961 | 1.526 | 0.43 |

### emax 4, 18 threads (ms per call, best of 4; loads 4.01 13.59 14.01, 9.34 4.46 2.61, 9.71 4.61 2.67)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | - | 0.064 (0.111) | - | 0.000 | - |
| comm220ss | - | 0.061 (0.106) | - | 0.144 | 0.42 |
| comm111ss | - | 0.001 (0.002) | - | 0.003 | 0.38 |
| comm121ss | - | 0.056 (0.116) | - | 0.065 | 0.87 |
| comm221ss | - | 0.259 (0.484) | - | 0.262 | 0.99 |
| comm122ss | - | 0.167 (0.263) | - | 0.227 | 0.73 |
| comm222_pp_hhss | - | 0.205 (0.361) | - | 0.224 | 0.92 |
| comm222_phss | - | 0.540 (0.832) | - | 0.632 | 0.85 |
| comm222_pp_hh_221ss | - | 0.264 (0.501) | - | 0.298 | 0.88 |
| commutator (full) | - | 1.276 (2.304) | - | 1.425 | 0.90 |

### emax 6, 1 thread (ms per call, best of 4; loads 4.32 14.14 14.21, 4.38 13.99 14.15, 7.24 3.50 2.20, 7.38 3.60 2.24)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.000 | 0.000 (0.000) | 0.000 | 0.001 | 0.29 |
| comm220ss | 0.434 | 0.010 (0.435) | 0.010 | 0.136 | 0.08 |
| comm111ss | 0.002 | 0.003 (0.003) | 0.013 | 0.014 | 0.18 |
| comm121ss | 0.085 | 0.086 | 0.085 | 0.191 | 0.45 |
| comm221ss | 9.322 | 1.628 (1.865) | 6.130 | 9.133 | 0.18 |
| comm122ss | 14.5 | 3.997 (4.340) | 3.989 | 6.692 | 0.60 |
| comm222_pp_hhss | 9.191 | 1.924 (2.107) | 6.354 | 8.880 | 0.22 |
| comm222_phss | 72.4 | 11.1 (15.6) | 15.7 | 45.6 | 0.24 |
| comm222_pp_hh_221ss | 9.776 | 2.064 (2.432) | 6.586 | 9.318 | 0.22 |
| commutator (full) | 100.3 | 18.2 (25.7) | 27.4 | 70.6 | 0.26 |

### emax 6, 6 threads (ms per call, best of 4; loads 4.19 13.79 14.08, 8.22 4.08 2.46)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.016 | 0.016 (0.045) | 0.017 | 0.001 | 23.29 |
| comm220ss | 0.110 | 0.048 (0.110) | 0.065 | 0.175 | 0.28 |
| comm111ss | 0.003 | 0.002 (0.007) | 0.013 | 0.014 | 0.17 |
| comm121ss | 0.044 | 0.045 (0.092) | 0.044 | 0.067 | 0.66 |
| comm221ss | 3.248 | 0.869 (1.247) | 2.825 | 2.436 | 0.36 |
| comm122ss | 3.381 | 0.815 (1.085) | 0.819 | 2.458 | 0.33 |
| comm222_pp_hhss | 3.311 | 0.860 (1.213) | 2.922 | 2.631 | 0.33 |
| comm222_phss | 17.5 | 2.829 (4.265) | 5.429 | 9.403 | 0.30 |
| comm222_pp_hh_221ss | 3.344 | 0.965 (1.366) | 2.994 | 2.751 | 0.35 |
| commutator (full) | 24.9 | 5.112 (6.982) | 9.745 | 15.2 | 0.34 |

### emax 6, 18 threads (ms per call, best of 4; loads 4.01 13.59 14.01, 9.71 4.61 2.67)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | - | 0.061 (0.106) | - | 0.001 | 101.67 |
| comm220ss | - | 0.096 (0.120) | - | 0.259 | 0.37 |
| comm111ss | - | 0.003 (0.007) | - | 0.014 | 0.17 |
| comm121ss | - | 0.064 (0.122) | - | 0.073 | 0.87 |
| comm221ss | - | 0.807 (2.184) | - | 1.954 | 0.41 |
| comm122ss | - | 0.524 (0.624) | - | 1.192 | 0.44 |
| comm222_pp_hhss | - | 0.782 (1.206) | - | 2.159 | 0.36 |
| comm222_phss | - | 2.034 (2.554) | - | 5.090 | 0.40 |
| comm222_pp_hh_221ss | - | 0.867 (1.298) | - | 2.257 | 0.38 |
| commutator (full) | - | 3.887 (5.417) | - | 9.470 | 0.41 |

### emax 8, 1 thread (ms per call, best of 3; loads 3.01 12.14 13.46, 3.10 12.31 13.52, 7.38 3.60 2.24, 7.43 3.67 2.28, 7.54 3.81 2.34)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.000 | 0.000 (0.000) | 0.000 | 0.001 | 0.50 |
| comm220ss | 3.188 | 0.031 (3.214) | 0.030 | 1.013 | 0.03 |
| comm111ss | 0.011 | 0.011 (0.012) | 0.075 | 0.080 | 0.13 |
| comm121ss | 0.348 | 0.305 (0.377) | 0.307 | 0.594 | 0.51 |
| comm221ss | 84.7 | 19.7 (23.3) | 109.9 | 136.6 | 0.14 |
| comm122ss | 149.6 | 37.9 (40.8) | 37.5 | 55.8 | 0.68 |
| comm222_pp_hhss | 84.4 | 21.0 (25.2) | 111.3 | 135.6 | 0.16 |
| comm222_phss | 627.5 | 104.1 (111.2) | 151.3 | 477.7 | 0.22 |
| comm222_pp_hh_221ss | 85.4 | 22.6 (26.3) | 113.6 | 138.8 | 0.16 |
| commutator (full) | 865.6 | 168.6 (185.0) | 307.8 | 670.9 | 0.25 |

### emax 8, 6 threads (ms per call, best of 3; loads 2.93 11.97 13.39, 3.10 11.85 13.34, 8.84 4.28 2.53, 9.34 4.46 2.61)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.017 | 0.015 (0.063) | 0.014 | 0.001 | 15.40 |
| comm220ss | 0.731 | 0.037 (0.764) | 0.029 | 1.012 | 0.04 |
| comm111ss | 0.012 | 0.011 (0.012) | 0.079 | 0.084 | 0.13 |
| comm121ss | 0.094 | 0.104 (0.235) | 0.097 | 0.162 | 0.65 |
| comm221ss | 31.4 | 10.4 (15.8) | 28.8 | 34.0 | 0.31 |
| comm122ss | 30.5 | 9.215 | 8.077 | 20.1 | 0.46 |
| comm222_pp_hhss | 31.3 | 10.6 (16.5) | 29.4 | 35.2 | 0.30 |
| comm222_phss | 135.3 | 26.9 (29.0) | 36.9 | 91.8 | 0.29 |
| comm222_pp_hh_221ss | 32.5 | 11.4 (16.6) | 28.7 | 34.8 | 0.33 |
| commutator (full) | 198.7 | 49.1 (63.7) | 77.3 | 154.8 | 0.32 |

### emax 8, 18 threads (ms per call, best of 3; loads 10.05 4.77 2.74, 3.10 11.85 13.34, 9.71 4.61 2.67)

| term | before (develop) | this code | this code, OpenBLAS | imsrg++ | ratio |
|---|---:|---:|---:|---:|---:|
| comm110ss | - | 0.068 (0.139) | - | 0.001 | 112.83 |
| comm220ss | - | 0.087 (0.411) | - | 1.086 | 0.08 |
| comm111ss | - | 0.012 (0.034) | - | 0.146 | 0.08 |
| comm121ss | - | 0.086 (0.184) | - | 0.109 | 0.79 |
| comm221ss | - | 10.2 (19.1) | - | 23.8 | 0.43 |
| comm122ss | - | 4.020 (5.082) | - | 18.8 | 0.21 |
| comm222_pp_hhss | - | 10.8 (13.5) | - | 26.0 | 0.42 |
| comm222_phss | - | 15.2 (19.7) | - | 45.4 | 0.33 |
| comm222_pp_hh_221ss | - | 12.1 (14.7) | - | 25.8 | 0.47 |
| commutator (full) | - | 31.2 (41.1) | - | 94.7 | 0.33 |

## Full IMSRG(2) flow

Both codes take the same number of steps and end at the same E0 to the printed digits (the
flow itself is validated to 1e-10 MeV by the IT-2 golden test at emax 4). A step costs about
seven commutators in both codes (BCH product for Omega plus the BCH transform of H), so the
flow ratio follows the commutator ratio; the rest of a step (generator, norms, operator
arithmetic, allocation of the BCH temporaries) is a few percent at emax >= 6. The pre-flow part
(2N file, HF, MP2) is at most 2 s at emax 4-6 in both codes and 4-6 s at emax 8.

## Memory

Workspace scratch (`commutator_bench` prints it; `commutator_robustness.scratch_footprint`
for emax 4 and 6), in multiples of one two-body operator of the same model space:

| emax | two-body operator | ladder before -> after | Pandya before -> after | total before -> after | 6j table |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.5 MB | 3.48x -> 2.04x | 7.97x -> 4.34x | 11.4x -> 6.4x | 0.5 MB |
| 6 | 7.4 MB | 3.75x -> 2.01x | 6.64x -> 3.49x | 10.4x -> 5.5x | 3.8 MB |
| 8 | 56.3 MB | 3.86x -> 2.00x | 6.06x -> 3.12x | 9.9x -> 5.1x (558 -> 288 MB) | 17.0 MB |

Ladder: closed-shell pp channels (contiguous listed kets with weight 1) feed X and Y to the
GEMM in place instead of gathered copies; what remains is the two product buffers Mpp and Mhh
(2x). Pandya: no separate ZB buffer (W is combined in place) and YK stored for one pair order
only; what remains is XK + YK (Ncc nk each) and W (2 Ncc^2) per cc channel. Open-shell
(fractional) references keep the gathered ladder operands (M12-Q4).

## What changed (M12a, `feature/m12a-perf`)

- `comm222_phss` (the largest term): row-wise kernels instead of one work item per element
  (each item searched its block and divided indices); the X and Y Pandya transforms of the fill
  share one J'' loop with the 6j values from a contiguous table row (`SixJView::row`); ZB folded
  into W in place; the inverse transform runs over pairs of kets of a (parity, Tz) class and
  forms Z for every J of the pair from one gather of the J-independent zbar values, with the cc
  kets of orbit pairs from a small descriptor table (norb^2 entries); YK stored for one pair
  order, the right half of W from a second GEMM after an in-place row-pair flip.
  emax 6, one thread: 72 -> 13 ms.
- `comm122ss`: one item per column; the bra parts go column-wise into a scratch (the ladder's
  pp product buffer, zeroed again after use) and a second pass forms Z, instead of strided reads
  of rows of X and Y. emax 8, six threads: 30 -> 9 ms.
- Ladder terms: per-column gather and antisymmetrization with the Z update fused; closed-shell
  pp operands read in place; `linalg::GroupedGemm` runs the per-channel GEMMs concurrently on
  the host threads (Accelerate only, M12-Q5): at emax 8 the sequential loop had left five of six
  threads idle during the GEMMs.
- `comm220ss` restricted to hole-hole rows and the pp list (bitwise identical).
- Operator norms: one chunked device reduction for the two-body norm instead of one blocking
  reduction per channel (1.7 -> 0.17 ms at emax 6 on six threads). `No2bThreeBody::get_pn`: dense
  orbit table and tabulated isospin CGs (183 -> 65 ns per call; M6-Q4, M12-Q1).
- Facade: `exec::ParallelForDynamic`, `exec::HostParallelForDynamic`,
  `IMSRG_FORCEINLINE_FUNCTION`, int32 `FindBlock`; `commutator_bench --checksum` (FNV hash of
  [A,H] and [N,H]) to check that a kernel change leaves the results bitwise identical.

## Bitwise changes

Every kernel change was checked with `commutator_bench --checksum` against the previous binary
at emax 2 and 4 with 1 and 4 threads. All are bitwise identical except the YK halving
(db50d4d): the emax-2 [A,H] hash changes because Accelerate's dgemm takes another code path for
the new n x n shapes than for n x 2n (round-off only: goldens <= 1.5e-15, fast vs
`commutator_ref` <= 6.4e-16, as before); emax 4 is identical. The concurrent GroupedGemm
writes each C with one BLAS call, so its results do not depend on the schedule (stable hashes
over repeated runs). The norm reduction reassociates sums (5e-16 relative). All unit,
integration, MPI, Python and `slow` tests pass; IT step counts are unchanged.

## What did not help, or was not done

- Force-inlining the particle-hole helpers: about 2% (kept, it is free).
- A stored element-wise sparse recoupling map for the Pandya transforms (the design/12 section
  4.2 plan): not built. The inverse map alone would hold about eight (index, coefficient)
  pairs per stored Z element, about six two-body operators of memory; contiguous 6j rows and the
  per-pair descriptor table give most of the gain without it.
- Concurrent GEMMs with OpenBLAS/MKL: not enabled (oversubscription, thread-count-dependent
  rounding in MKL; M12-Q5).
- Not tried (follow-ups): `dsyr2k` for the Hermitian ladder products (half the GEMM flops when
  hZ = +1, but another summation order and it needs exactly symmetric storage); reusing the BCH
  temporaries across flow steps (a few percent at emax 8; the flow module is shared with the
  IMSRG(3) work); team-scratch variants of the per-pair J' buffers for GPUs (M14).

## Reproduce

```
cmake --preset release-openmp -B build/rel -DIMSRG_BUILD_BENCH=ON
cmake --build build/rel --target commutator_bench imsrg
OMP_PROC_BIND=false OMP_NUM_THREADS=T build/rel/tools/bench/commutator_bench --emax E --reps R
(cd <reference> && OMP_PROC_BIND=false OMP_NUM_THREADS=T harness-build-bench/refharness bench-commutator emax=E reps=R)
tools/bench/flow_bench.sh --emax E --threads T --code both --reps N --out <scratch dir>
```
