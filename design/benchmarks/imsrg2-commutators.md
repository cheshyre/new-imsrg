# IMSRG(2) commutator micro-benchmark: this code vs imsrg++ (M7 T5)

Measured per design/30 section 5. Random scalar rank-2 operators, X anti-Hermitian and
Y Hermitian, O16 reference, hw 16, IMSRG(3) off in both codes. All times are wall-clock
milliseconds per call, averaged ("mean") or best ("min") over the repetitions after one
warm-up call. A ratio above 1 means this code is slower than imsrg++.

## Setup

- Machine: Apple M5 Pro, 6 performance + 12 efficiency cores, 48 GiB, macOS 26 (Darwin 25.6).
  Other load during the runs: about two cores (load average 2.0-2.3, no compile jobs running).
- Compiler: Apple clang 21.0.0 (`/usr/bin/clang++`), libomp from Homebrew, for both codes.
- This code: branch `feature/m7-commutator` at `14217b3`, plus the uncommitted working-tree
  edits to `src/imsrg/commutator*/` present at the time (made by another worker); preset
  `release-openmp` (Kokkos 5.3.0 OpenMP backend, `-O3 -DNDEBUG -mcpu=native`).
  Binary: `build/t5-bench/tools/bench/commutator_bench` (target `commutator_bench`,
  `-DIMSRG_BUILD_BENCH=ON`). Timing: `std::chrono::steady_clock` around
  `exec::fence` calls.
- imsrg++: imsrg-devel (pristine, src sha256 `ddcbf1af...`) built through
  `tools/reference/build_imsrg_devel.sh harness` into a separate build directory, CMake
  `Release` (`-O3 -DNDEBUG`, no `-mcpu=native`; the reference build system does not set it),
  OpenMP via libomp. Timing: `omp_get_wtime`. Binary:
  `refharness bench-commutator` (`tools/reference/rh_bench.cpp`).
- Model space: `emax 4`: 30 orbits, 67156 two-body elements; `emax 6`: 56 orbits,
  927680 two-body elements (identical in both codes).
- Threads: `OMP_NUM_THREADS=1` and `6`, `OMP_PROC_BIND=false`, so the six threads may land on
  any core type; no pinning to performance cores was attempted.
- Repetitions: 200 (emax 4) and 20 (emax 6) per term.

Commands:

```
cmake --preset release-openmp -B build/t5-bench -DIMSRG_BUILD_BENCH=ON -DIMSRG_BUILD_TESTS=OFF
cmake --build build/t5-bench --target commutator_bench
HARNESS_BUILD_DIR=<reference>/harness-build-bench tools/reference/build_imsrg_devel.sh harness
OMP_PROC_BIND=false OMP_NUM_THREADS=T build/t5-bench/tools/bench/commutator_bench --emax E --reps R
OMP_PROC_BIND=false OMP_NUM_THREADS=T <reference>/harness-build-bench/refharness bench-commutator emax=E reps=R
```

What is and is not inside the timed region:

- Term functions (`terms::` and `Commutator::comm...`) add into Z, which accumulates across the
  repetitions; the full `commutator()` resets Z first (ours). imsrg++'s
  `Commutator::Commutator(X, Y)` allocates a new Z each call, so its full time includes that
  allocation while ours reuses Z (the flow-loop case).
- Our Workspace construction (6j table, ladder and Pandya data) is timed once and listed
  separately. imsrg++'s one-time 6j precalculation ("Precalculating SixJ's", 0.17M and larger
  tables) happens before the timed region and is not measured here, so the two setup costs
  are not comparable and are not part of the term tables.
- The "full" row of our code is `commutator()` with the default `CommutatorOptions` (IMSRG(2)
  terms); imsrg++ uses `SetUseIMSRG3(false)`. The sum of the term rows is not exactly the
  full row (the full call also does scheduling, hermiticity completion and the like).

## Results


### emax 4, OMP_NUM_THREADS=1 (200 reps)

Workspace construction (ours, once): 0.57 ms.

| term | ours mean ms | ours min ms | imsrg++ mean ms | imsrg++ min ms | ratio (mean) |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.0002 | 0.0001 | 0.0004 | 0.0000 | 0.50 |
| comm220ss | 0.0359 | 0.0356 | 0.0183 | 0.0169 | 1.96 |
| comm111ss | 0.0012 | 0.0010 | 0.0027 | 0.0019 | 0.44 |
| comm121ss | 0.0192 | 0.0174 | 0.0399 | 0.0379 | 0.48 |
| comm221ss | 0.5521 | 0.5064 | 0.4287 | 0.4129 | 1.29 |
| comm122ss | 4.2305 | 4.1087 | 0.7901 | 0.7129 | 5.35 |
| comm222_pp_hhss | 0.5338 | 0.5008 | 0.3770 | 0.3698 | 1.42 |
| comm222_phss | 7.0338 | 6.9170 | 3.5363 | 3.4680 | 1.99 |
| comm222_pp_hh_221ss | 0.5564 | 0.5164 | 0.4435 | 0.4330 | 1.25 |
| commutator (full) | 11.9295 | 11.7580 | 4.8486 | 4.7541 | 2.46 |

### emax 6, OMP_NUM_THREADS=1 (20 reps)

Workspace construction (ours, once): 4.76 ms.

| term | ours mean ms | ours min ms | imsrg++ mean ms | imsrg++ min ms | ratio (mean) |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.0003 | 0.0001 | 0.0007 | 0.0000 | 0.43 |
| comm220ss | 0.4076 | 0.4055 | 0.1279 | 0.1249 | 3.19 |
| comm111ss | 0.0026 | 0.0023 | 0.0134 | 0.0129 | 0.19 |
| comm121ss | 0.0815 | 0.0792 | 0.1804 | 0.1631 | 0.45 |
| comm221ss | 8.9201 | 8.6143 | 8.7316 | 8.5242 | 1.02 |
| comm122ss | 76.0358 | 75.4166 | 6.2822 | 6.1820 | 12.10 |
| comm222_pp_hhss | 8.9079 | 8.5844 | 8.4801 | 8.3790 | 1.05 |
| comm222_phss | 81.5434 | 80.3770 | 43.8661 | 41.7700 | 1.86 |
| comm222_pp_hh_221ss | 9.1812 | 8.9317 | 8.9627 | 8.8320 | 1.02 |
| commutator (full) | 167.3451 | 166.2838 | 59.2263 | 58.4421 | 2.83 |

### emax 4, OMP_NUM_THREADS=6 (200 reps)

Workspace construction (ours, once): 1.26 ms.

| term | ours mean ms | ours min ms | imsrg++ mean ms | imsrg++ min ms | ratio (mean) |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.0169 | 0.0110 | 0.0004 | 0.0000 | 42.25 |
| comm220ss | 0.0198 | 0.0173 | 0.0556 | 0.0489 | 0.36 |
| comm111ss | 0.0011 | 0.0009 | 0.0027 | 0.0019 | 0.41 |
| comm121ss | 0.0231 | 0.0155 | 0.0227 | 0.0150 | 1.02 |
| comm221ss | 0.2396 | 0.2234 | 0.1789 | 0.1659 | 1.34 |
| comm122ss | 0.8138 | 0.8025 | 0.3062 | 0.2959 | 2.66 |
| comm222_pp_hhss | 0.2328 | 0.2163 | 0.1741 | 0.1669 | 1.34 |
| comm222_phss | 1.4267 | 1.3945 | 0.7487 | 0.7310 | 1.91 |
| comm222_pp_hh_221ss | 0.2571 | 0.2407 | 0.2090 | 0.1960 | 1.23 |
| commutator (full) | 2.6288 | 2.5704 | 1.3789 | 1.3468 | 1.91 |

### emax 6, OMP_NUM_THREADS=6 (20 reps)

Workspace construction (ours, once): 6.01 ms.

| term | ours mean ms | ours min ms | imsrg++ mean ms | imsrg++ min ms | ratio (mean) |
|---|---:|---:|---:|---:|---:|
| comm110ss | 0.0178 | 0.0123 | 0.0007 | 0.0000 | 25.43 |
| comm220ss | 0.1088 | 0.1066 | 0.1584 | 0.1512 | 0.69 |
| comm111ss | 0.0028 | 0.0023 | 0.0134 | 0.0129 | 0.21 |
| comm121ss | 0.0389 | 0.0335 | 0.0482 | 0.0410 | 0.81 |
| comm221ss | 2.4986 | 2.4487 | 2.3299 | 2.2888 | 1.07 |
| comm122ss | 14.0941 | 13.7754 | 2.1377 | 2.1000 | 6.59 |
| comm222_pp_hhss | 2.5745 | 2.5345 | 2.4838 | 2.4672 | 1.04 |
| comm222_phss | 16.6512 | 15.7169 | 7.9441 | 7.7620 | 2.10 |
| comm222_pp_hh_221ss | 2.6639 | 2.5671 | 2.6070 | 2.5809 | 1.02 |
| commutator (full) | 33.4269 | 32.6140 | 13.6149 | 13.3989 | 2.46 |

## Observations

These are readings of the numbers above; the kernels were not profiled or tuned in this task, so
the causes named are hypotheses.

- The full IMSRG(2) commutator is 1.9-2.8x slower than imsrg++ at both emax and both thread
  counts (emax 6, 1 thread: 167 ms vs 59 ms; 6 threads: 33 ms vs 14 ms). Nothing here shows our
  code ahead on the full commutator.
- The gap is concentrated in two terms. `comm122ss` is about 12x slower at emax 6 with one
  thread (76 ms vs 6.3 ms) and about 6.6x with six threads; `comm222_phss` is 1.9x slower (82 ms
  vs 44 ms) and 2.1x with six threads. Together they are about 95% of our full time at emax 6.
  Likely causes are the element-wise accumulation and gather/scatter in the `comm122ss` kernel
  against imsrg++'s matrix-multiply formulation, and the Pandya-transform data movement
  (gather into recoupled blocks) in `comm222_phss`. The 6j table fill is not the bottleneck in
  the timed calls, since the table is built in the Workspace (4-6 ms at emax 6).
- `comm221ss`, `comm222_pp_hhss` and `comm222_pp_hh_221ss`, the BLAS-dominated terms, are on par
  with imsrg++ (ratio 1.0-1.1 at emax 6, up to 1.3 at emax 4 with six threads), which suggests the linear-algebra backend itself is not the
  problem. Our six-thread scaling of those terms is about 3.6x (comm221ss, emax 6), as for
  imsrg++ (about 3.7x), consistent with 6 threads landing partly on efficiency cores.
- Our `comm122ss` scales better with threads than imsrg++'s (5.4x against 2.9x at emax 6), but
  from a much slower serial base. The small terms (`comm110ss`, `comm220ss`, `comm121ss`) are a
  few hundredths of a millisecond; at six threads our `comm110ss` costs 0.017 ms against a
  negligible serial time (parallel-region launch overhead), and at emax 6 with one thread
  `comm220ss` (0.41 ms) is 3.2x and `comm121ss` (0.08 ms) is 0.45x of imsrg++. Their share of
  the full commutator is under 1%.
- Workspace construction is 0.6-1.3 ms at emax 4 and 5-6 ms at emax 6, i.e. less than the cost
  of one commutator call at emax 6 and negligible over a flow.
- Caveats: one machine, one run per configuration (no repeat to quantify run-to-run spread;
  mean and min agree within a few percent for the large terms), `OMP_PROC_BIND=false` on a
  heterogeneous CPU, the two codes are compiled with different native-architecture flags, and
  X, Y differ between the codes (different random generators; sizes and sparsity are identical).

## After the first optimization pass (M7 T7, commit b202e81, 2026-11-12)

Two exact work reductions, no change of results (all golden and fast-vs-reference tests
unchanged, deviations below 2e-15): `comm122ss` now uses precomputed "lifted one-body" lists
per (channel, ket) (target ket, orbit pair, normalization ratio times exchange phase), so each
element is a short sum of direct block reads instead of ~32 index-computing lookups, and it
computes the upper triangle and mirrors with hZ; `comm222_phss` fills XK and YK from one
Pandya evaluation per (pair, ket) point using Xbar^T = hX Xbar and F Ybar F = hY Phi Ybar Phi
(about 3x less fill work), and its inverse transform computes the upper triangle only.

Same machine, same binaries and commands as above (ours rebuilt at b202e81 with
`release-openmp -DIMSRG_BUILD_BENCH=ON`; imsrg++ harness unchanged), both codes run back to back
in one script. The machine was shared with other builds (load average 6-8 on 18 cores), so
expect a few percent of noise; one run per configuration.

### emax 4, OMP_NUM_THREADS=1 (100 reps), after T7

| term | ours mean ms | imsrg++ mean ms | ratio |
|---|---:|---:|---:|
| comm110ss | 0.0002 | 0.0004 | 0.50 |
| comm220ss | 0.0345 | 0.0177 | 1.95 |
| comm111ss | 0.0011 | 0.0027 | 0.41 |
| comm121ss | 0.0181 | 0.0429 | 0.42 |
| comm221ss | 0.5712 | 0.4573 | 1.25 |
| comm122ss | 0.7624 | 0.7605 | 1.00 |
| comm222_pp_hhss | 0.5601 | 0.3956 | 1.42 |
| comm222_phss | 4.2988 | 3.8024 | 1.13 |
| comm222_pp_hh_221ss | 0.5874 | 0.4846 | 1.21 |
| commutator (full) | 5.7371 | 5.2293 | 1.10 |

Workspace construction (ours): 0.88 ms.

### emax 4, OMP_NUM_THREADS=6 (100 reps), after T7

| term | ours mean ms | imsrg++ mean ms | ratio |
|---|---:|---:|---:|
| comm110ss | 0.0237 | 0.0004 | 59.25 |
| comm220ss | 0.0205 | 0.0594 | 0.35 |
| comm111ss | 0.0012 | 0.0027 | 0.44 |
| comm121ss | 0.0210 | 0.0233 | 0.90 |
| comm221ss | 0.2617 | 0.1955 | 1.34 |
| comm122ss | 0.2113 | 0.3037 | 0.70 |
| comm222_pp_hhss | 0.2847 | 0.1899 | 1.50 |
| comm222_phss | 1.0386 | 0.7819 | 1.33 |
| comm222_pp_hh_221ss | 0.3520 | 0.2187 | 1.61 |
| commutator (full) | 1.6390 | 1.4469 | 1.13 |

Workspace construction (ours): 1.54 ms.

### emax 6, OMP_NUM_THREADS=1 (10 reps), after T7

| term | ours mean ms | imsrg++ mean ms | ratio |
|---|---:|---:|---:|
| comm110ss | 0.0004 | 0.0006 | 0.67 |
| comm220ss | 0.4347 | 0.1375 | 3.16 |
| comm111ss | 0.0030 | 0.0135 | 0.22 |
| comm121ss | 0.0884 | 0.1832 | 0.48 |
| comm221ss | 9.5102 | 9.5468 | 1.00 |
| comm122ss | 14.5002 | 6.8525 | 2.12 |
| comm222_pp_hhss | 9.2844 | 9.1138 | 1.02 |
| comm222_phss | 54.9399 | 47.6614 | 1.15 |
| comm222_pp_hh_221ss | 9.6493 | 9.7740 | 0.99 |
| commutator (full) | 79.7536 | 66.6579 | 1.20 |

Workspace construction (ours): 6.52 ms.

### emax 6, OMP_NUM_THREADS=6 (10 reps), after T7

| term | ours mean ms | imsrg++ mean ms | ratio |
|---|---:|---:|---:|
| comm110ss | 0.0171 | 0.0007 | 24.43 |
| comm220ss | 0.1090 | 0.1900 | 0.57 |
| comm111ss | 0.0028 | 0.0135 | 0.21 |
| comm121ss | 0.0360 | 0.0604 | 0.60 |
| comm221ss | 2.6869 | 2.5739 | 1.04 |
| comm122ss | 2.8853 | 2.4184 | 1.19 |
| comm222_pp_hhss | 2.8795 | 2.6793 | 1.07 |
| comm222_phss | 11.0953 | 8.6634 | 1.28 |
| comm222_pp_hh_221ss | 2.7947 | 2.7710 | 1.01 |
| commutator (full) | 16.7004 | 14.7446 | 1.13 |

Workspace construction (ours): 7.56 ms.

Summary after T7: the full commutator is 1.10-1.20x the imsrg++ time (emax 4: 5.7 vs 5.2 ms on
one thread, 1.6 vs 1.4 ms on six; emax 6: 80 vs 67 ms on one thread, 16.7 vs 14.7 ms on six),
down from 1.9-2.8x. Remaining gaps: `comm222_phss` (1.15-1.33x; next steps: precomputed sparse
recoupling maps instead of per-entry 6j/index arithmetic, design/12 section 4.2, and fewer
passes over W/ZB), `comm122ss` serially at emax 6 (2.1x; it scales better than imsrg++ with
threads), and the ladder GEMMs at emax 4 with six threads (1.3-1.6x: per-call gathers and the
symmetrization pass on small blocks). These are M12 (performance) items.
