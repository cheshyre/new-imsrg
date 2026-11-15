# IMSRG(3) three-body storage layout: benchmark and decision (M10 step 1)

Benchmark of the three candidate layouts of `14-imsrg3-plan.md` §2 for the three-body (3B) blocks
of a scalar operator, run on 2026-11-15 (run 2), and the layout decision it supports (**D43**).
Prototypes: `tools/bench/imsrg3_layout/` (benchmark target, outside the production library and
the unit label); imsrg++ timings and values through `refharness comm223`
(`tools/reference/rh_imsrg3.cpp`).

## 1. Result

**Layout A (canonical) is chosen**: the imsrg++ kets |(ab)Jab, c; J⟩ with a ≤ b ≤ c and all
allowed Jab per (2J, parity, 2Tz) channel, the kets failing dE3max/OccNat3Cut dropped at
enumeration (D34), one Hermitian (or anti-Hermitian) block per channel stored as a packed
triangle (as imsrg++). The [2,2]→3 term and the perturbative triples go through **pair-third
tiles** (§3): one intermediate W per pair of orbit triples, contracted with a sparse recoupling
map R into the canonical rows.

* **C fails the memory gate**: 8.8× the memory of A (702 GB per operator at emax 12 / E3max 14,
  dense) and 1.7× A's [2,2]→3 time at emax 3.
* **B brings no measurable gain**: 2.1 % less memory than A at emax 12 / E3max 14 (13 % at
  emax 3), the same [2,2]→3 and triples times within the run-to-run spread, and it needs dense
  basis maps for every access by orbit labels.
* **A** compares element by element with imsrg++, needs no basis maps, and is within 2 % of B
  everywhere; the decision rule (below) then prefers A.
* The pair-third tiling makes [2,2]→3 **54× faster than imsrg++ `comm223ss`** at emax 3 on 18
  threads (8.94 ms vs 482 ms; 53× on one thread). The cost model attributes 5-7× of that to the
  algorithm (each orbit-triple pair's inner sum is computed once, not for every (J1, J2)); the
  rest is implementation and may shrink at large emax (§6).

Decision rule (fixed before the runs, from the planning notes): (1) memory gate: six operators
plus scratch within 50 % of an assumed 4 × 256 GB (M10-Q2); (2) lowest extrapolated time of the
[2,2]→3 hot spot; (3) within 15 %, prefer A.

## 2. Setup

* Machine: Apple M5 Pro, 6 performance + 12 efficiency cores, 48 GiB, macOS 26. Other agents
  built on the machine during the session; the runs below were taken at a 1-minute load of 2-5
  (other agents' work), and runs taken under heavy load (up to 30) were discarded and repeated.
* Build: `release-openmp` (Apple clang 21, `-O3 -DNDEBUG -mcpu=native`, Kokkos 5.3.0 OpenMP),
  `-DIMSRG_BUILD_BENCH=ON`, branch `feature/m10-layout`. imsrg++: imsrg-devel (src sha256
  `ddcbf1af...`), CMake `Release`, libomp, refharness build `harness-build-m10`.
* Threads: `OMP_NUM_THREADS` 18, 6 and 1, `OMP_PROC_BIND=false` (18 threads use both core types),
  `VECLIB_MAXIMUM_THREADS=1`.
* Spaces: O16 (hw 16) at emax 2 (E3max 6) and emax 3 (E3max 9), plus emax 3 with E3max 7 and with
  dE3max 3, and He4 at emax 2. Production-target sizes (emax 10-12) are **counted** only
  (`--count-only`: kets, triples and operation counts, no operator; M10-Q1).
* Inputs: random scalar two-body operators (`op::RandomOperator`), X anti-Hermitian (seed 1), Y
  Hermitian (seed 2); triples use the one-body diagonal ε_i = (n_i > ½ ? −30 : 10) + 2 e_i +
  0.1 (i mod 7) MeV, holes below particles. The same operators, written as dumps, feed imsrg++.
* Timing: one warm-up call, then 10 calls (K0: 3; imsrg++: 5) between fences; the tables give
  the **best minimum over all process runs** (10 for our kernels at 18 threads, 5 for imsrg++, 3
  otherwise) and in parentheses the spread of the per-run minima (max/min − 1).

Reproduce (all commands from the repository root):

```
cmake --preset release-openmp -B build/m10-rel -DIMSRG_BUILD_BENCH=ON -DIMSRG_BUILD_TESTS=OFF
cmake --build build/m10-rel --target imsrg3_layout_bench
OMP_NUM_THREADS=18 OMP_PROC_BIND=false build/m10-rel/tools/bench/imsrg3_layout/imsrg3_layout_bench --emax 3 --check
build/m10-rel/tools/bench/imsrg3_layout/imsrg3_layout_bench --count-only --emax 12 --e3max 14
imsrg3_layout_bench --emax 3 --write-dumps D   # then, in the reference harness build:
OMP_NUM_THREADS=18 refharness comm223 emax=3 e3max=9 reference=O16 x=D/X.dump y=D/Y.dump epsilon=D/epsilon.txt reps=5
```

Checks: `imsrg3_layout_check` (label `slow`, 37 tests): the 6j table against `Wigner6j` on all
45,056 arguments (emax 3); recoupling matrices orthogonal, M_ACB a symmetric involution for b = c
and (1 + M_CBA + M_ACB)/3 a projector for a = b = c; counts; layout round trips and norm
identities; K0 self-checks (mirror, [Y,X] = −[X,Y], anti-Hermitian diagonal 0, null kets 0);
every layout's K1 against K0 on four spaces and two hermiticity pairs.

## 3. Layouts and kernels

| | A (canonical) | B (orthonormal) | C (pair-third) |
|---|---|---|---|
| Rows of a channel | kept kets (a ≤ b ≤ c, Jab) | per triple an orthonormal basis of the antisymmetric subspace | all \|(pq)Jpq, r⟩, p ≤ q, any r |
| Values | imsrg++'s (unnormalized, antisymmetrized) | s_T s_T′ V_Tᵀ z_A V_T′, s_T = √(sym_T/6) | E z_A Eᵀ (fully antisymmetrized) |
| Internal 3B sums | weight w_T = sym_T/6 | plain matrix products | over the canonical rows only, or through R_C = E R |
| Conversion to imsrg++ | identity | V_T (dense, only triples with b = c) | gather of the canonical rows |

B differs from A only for triples with b = c (a < b = c and a = b = c), where the canonical kets
are overcomplete: V_T are the eigenvalue-1 eigenvectors of Q = (1 + M_ACB)/2 resp.
(1 + M_CBA + M_ACB)/3 (checked: Q symmetric, eigenvalues 0 or 1 within 1e-10 at every tested
space). For a = b < c the Pauli-allowed kets are already orthonormal.

Kernels (all in `tools/bench/imsrg3_layout/kernels/`):

* **K0**, the reference: imsrg++ `comm223ss` (IMSRG3Commutators.cc:5012-5376) ported loop by
  loop: per canonical element, 3 × 3 permutations, J1′, J2′ and the internal orbit a.
* **K1**, the tiled kernel: a team per bra orbit triple T; for each ket triple T′ ≤ T of the
  channel, the pair-third tile
  W[(P z J1′), (Q w J2′)] = Ĵ1′Ĵ2′ Σ_a {j_w j_a J1′; j_z J J2′} (X_{P,(wa)} Ỹ_{(za),Q} −
  Y_{P,(wa)} X̃_{(za),Q}), then the layout epilogue z_L[T, T′] = L_T W L_T′ᵀ with the layout's
  outer map L_T (A: R_T, the comm223ss recoupling; B: s_T V_Tᵀ R_T; C: E_T R_T), and the mirror
  h_Z z_L[T, T′]ᵀ. W is identical for all layouts; only the epilogue differs. `W only` times the
  tiles without the epilogue.
* **Triples**: E = Σ_ch (2J+1) Σ_{T′<T} f_TT′ Σ (ω_i ω_j z_ij)², f = (n_T n̄_T′ − n_T′ n̄_T)/(ε_T
  − ε_T′), computed tile by tile without 3B storage (`TriplesTiled`), and element by element
  with K0's arithmetic and imsrg++'s skips (`TriplesReference`).
* **pack**: dense ↔ packed-triangle conversion of all channel blocks of A (the cost of storing A
  packed and unpacking a channel for GEMM-type terms).

**Agreement** (all ≤ 1e-10 required, measured): K1 in A, B, C against K0 ≤ 7e-16 relative; the
imsrg++ 3B norm (D12) agrees across layouts to 0 (emax 3); triples ≤ 3e-16. Against imsrg++
`comm223ss` on the same operators: 3B norm 340.71339955428834 vs 340.71339955428823 (emax 2,
3e-16) and 2174.8102180586966 vs 2174.8102180586943 (emax 3, 1e-15); triples energy
−64.610022003823076 vs −64.610022003823076 (emax 2) and −1451.9275912120736 vs
−1451.9275912120786 (emax 3, 3e-15). These are quadratic quantities, blind to a global sign;
element-level checks come with the m-scheme reference (`14` §3).

## 4. Memory and metadata

Per operator, O16, dense (packed in parentheses); measured at emax 2-3, counted at emax 10-12.

| emax / E3max / dE3max | ΣN_A | max N_A | A | B | C | B/A | C/A |
|---|---|---|---|---|---|---|---|
| 2 / 6 / – | 2,099 | 113 | 1.18 MB (0.60) | 0.96 MB (0.49) | 8.72 MB (4.38) | 0.82 | 7.42 |
| 3 / 9 / – | 13,223 | 554 | 36.1 MB (18.1) | 31.5 MB (15.8) | 286 MB (143) | 0.87 | 7.93 |
| 3 / 7 / – | 8,909 | 449 | 18.3 MB (9.2) | 16.7 MB (8.4) | 151 MB (76) | 0.91 | 8.27 |
| 3 / 9 / 3 | 5,832 | 270 | 7.9 MB (4.0) | 7.3 MB (3.7) | 66.8 MB (33.5) | 0.93 | 8.44 |
| 10 / 14 / – | 762,308 | 22,583 | 76.0 GB (38.0) | 74.4 GB (37.2) | 671 GB (336) | 0.979 | 8.83 |
| 12 / 14 / – | 782,388 | 23,083 | 79.5 GB (39.7) | 77.8 GB (38.9) | 702 GB (351) | 0.979 | 8.84 |
| 10 / 16 / – | 1,626,081 | 43,484 | 313 GB (157) | 307 GB (154) | 2.77 TB (1.39) | 0.982 | 8.86 |
| 12 / 16 / – | 1,744,097 | 46,188 | 355 GB (177) | 349 GB (174) | 3.15 TB (1.57) | 0.983 | 8.87 |
| 10-12 / 16 / 8 | 242,118 | 7,930 | 8.75 GB (4.38) | 8.48 GB (4.24) | 76.6 GB (38.3) | 0.969 | 8.76 |
| 12 / 16 / 6 | 73,633 | 2,739 | 0.94 GB (0.47) | 0.90 GB (0.45) | 8.15 GB (4.08) | 0.958 | 8.67 |

(B/A and C/A are ratios of ΣN².) With dE3max ≤ 8 the space no longer grows with emax: the
emax-10 and emax-12 counts are identical. The counts reproduce imsrg++'s pn storage (A at emax 3:
2,261,446 packed doubles, the size imsrg++ allocates). The planning pass predicted C 3 % larger;
it also counted the pair-third states of triples that keep no canonical row in a channel, which
vanish under antisymmetrization and are left out here.

**Gate.** A Magnus step of IMSRG(3)-N7 holds about six 3B operators. At emax 12 / E3max 14, A
needs 238 GB packed (477 GB dense) for six, within 50 % of 4 × 256 GB; C needs 2.1 TB packed
and fails by a factor 4 even packed. At E3max 16 without a dE3max cut, even A needs 1.06 TB
packed for six (about 9 nodes of 256 GB at the 50 % rule); with dE3max 8 it needs 26 GB. Which
cuts the owner uses at emax 10-12 decides the node count, not the layout (M10-Q2).

**Metadata** (emax 3): A 1.13 MB (row and channel tables plus its outer map R), B 1.16 MB (plus
V_T), C 3.12 MB (pair-third tables and E R), against 36 MB for one A operator; shared tables:
canonical space 0.30 MB, pair-third states 0.64 MB, R 0.97 MB, E 1.10 MB, 3B 6j table 0.36 MB
(451 MB at emax 12). Build times (emax 3, 18 threads): canonical space 0.30 ms, pair-third
states with R and E 3.6 ms, layout A 0.29 ms, B 1.25 ms (eigendecompositions), C 3.0 ms, 6j table
0.35 ms.

## 5. Timings

### 5.1 [2,2]→3 (ms per call; best minimum over 5 runs, spread of the run minima in parentheses)

| Kernel | O16 emax 2, 18 thr | O16 emax 3, 18 thr | O16 emax 3, 1 thr |
|---|---|---|---|
| imsrg++ `comm223ss` | 10.93 (1 %) | 482 (2 %) | 6411 (3 %) |
| K0 (port of `comm223ss`) | 6.53 (9 %) | 360 (9 %) | 4497 (1 %) |
| K1, layout A | 0.392 (7 %) | 8.94 (3 %) | 120.4 (0.3 %) |
| K1, layout B | 0.379 (9 %) | 8.80 (4 %) | 119.4 (0.2 %) |
| K1, layout C | 0.494 (9 %) | 14.88 (3 %) | 188.0 (0.5 %) |
| K1, W only (any layout) | 0.34 | 7.58 | 100.6 |
| pack + unpack A | 0.105 | 0.391 | 1.48 |

18-thread columns: 10 process runs for our kernels (5 imsrg++); 1-thread column: 3 runs.

Other spaces and 6 threads (ms, best of 3 runs; taken at a load of 4-5, spread up to 30 %):

| Space, threads | imsrg++ | K0 | K1 A | K1 B | K1 C | W only | triples A / B / C |
|---|---|---|---|---|---|---|---|
| O16 emax 3, 6 thr | 1191 | 851 | 21.1 | 20.9 | 34.4 | 17.9 | 0.448 / 0.440 / 0.503 |
| O16 emax 3, E3max 7, 18 thr | – | 160 | 4.91 | 4.89 | 7.73 | 4.20 | 0.365 / 0.349 / 0.376 |
| O16 emax 3, dE3max 3, 18 thr | – | 76.4 | 2.33 | 2.31 | 3.41 | 2.01 | 0.336 / 0.315 / 0.340 |
| He4 emax 2, 18 thr | – | 6.56 | 0.406 | 0.397 | 0.528 | 0.361 | 0.175 / 0.160 / 0.171 |

### 5.2 Perturbative triples (ms per call)

| Kernel | O16 emax 2, 18 thr | O16 emax 3, 18 thr | O16 emax 3, 1 thr |
|---|---|---|---|
| imsrg++ (`comm223ss`, `perturbative_triples`) | 0.418 (3 %) | 5.40 (7 %) | 69.8 |
| element-wise (K0 arithmetic) | 0.228 | 3.53 | 42.3 |
| tiled, layout A | 0.196 | 0.407 | 2.04 |
| tiled, layout B | 0.186 | 0.401 | 2.03 |
| tiled, layout C | 0.185 | 0.443 | 2.33 |

Only tile pairs with one triple fully occupied and the other fully empty contribute, so the tiled
triples cost 5 % of a full [2,2]→3 and need no 3B storage.

### 5.3 Reading

* A and B differ by 1-4 % in every kernel, inside the run-to-run spread (2-12 %). B's smaller
  epilogue (d_T ≤ n_T rows) is the only systematic gain.
* C pays for its 2.8× larger rows in the epilogue: K1 − W = 7.3 ms for C against 1.4 (A) and
  1.2 ms (B) at emax 3. A C-native GEMM formulation of W (planning item K2) was not built; it
  would not change the memory verdict.
* Thread scaling at emax 3 (1 → 6 → 18 threads, 6 P + 12 E cores): K1 5.7× and 13.5×, K0 5.3× and
  12.5×, imsrg++ 5.4× and 13.3×.
* Storing A packed costs one pack/unpack per operator use: 4 % of a [2,2]→3 at emax 3, and less
  at larger emax (O(N²) against O(N² n_a) and O(N³) kernels).

## 6. Extrapolation to the production target

Operation counts (`--count-only`, summed over all channels and orbit-triple pairs): `w_cost`, the
(row, column, a) terms of the W tiles; `k0_cost`, the terms of imsrg++'s loop structure; and the
epilogue (dense-per-triple approximation).

| emax / E3max / dE3max | w_cost | k0_cost | k0/w | epilogue A (= B) | epilogue C |
|---|---|---|---|---|---|
| 2 / 6 / – | 1.56e6 | 6.54e6 | 4.2 | 1.55e6 | 6.48e6 |
| 3 / 9 / – | 8.16e7 | 4.06e8 | 5.0 | 5.83e7 | 2.55e8 |
| 12 / 14 / – | 1.80e12 | 1.10e13 | 6.1 | 1.73e11 | 8.13e11 |
| 12 / 16 / – | 8.06e12 | 5.75e13 | 7.1 | 8.52e11 | 4.01e12 |
| 12 / 16 / 8 | 1.96e11 | 1.14e12 | 5.8 | 1.86e10 | 8.50e10 |

Scaling the emax-3 times (18 threads, one node) with these counts gives, at emax 12 / E3max 14,
about 170 s per [2,2]→3 evaluation in layouts A and B (W 167 s, epilogue 4 s), 190 s in C, and
about 3.6 h for imsrg++ `comm223ss`; with dE3max 8, about 19 s (A) against 23 min (imsrg++).
These are **order-of-magnitude** estimates (the review found them optimistic for our kernels):
at emax 3 the operands sit in cache and the orbit groups are small; 60-75 % of the group terms
have a triangle-forbidden 6j at large emax, which imsrg++ skips and K1 does not yet; and the
benchmark's gather tables (2 × 7.7 GB at emax 12) must be replaced by per-tile gathers in
production. None of this depends on the layout: W is the same computation in A, B and C.

## 7. Other findings

* **6j trap (verified).** `angmom::SixJTable` covers {ja jb J; jc jd J′} with all four j single
  particle; a three-body total J (up to 6 emax + 3 over 2) in its fifth slot silently reads 0.0.
  The only production user (`commutator::Workspace`, Pandya 6j) never passes one, so no
  production code is affected. The 3B kernels need the family {j1 j2 J12; j3 J J23}
  (`SixJ3Table`: 0.36 MB at emax 3, 451 MB at emax 12), which covers the recoupling coefficients
  and the [2,2]→3 sum.
* **Null kets.** imsrg++ keeps canonical kets that vanish identically under antisymmetrization
  (36 at O16 emax 3, all with b = c; its stretched-state rules remove only some). They stay
  zero in every operator; layout B drops them; layout A keeps them for comparability.
* **Hermiticity shortcut.** K1 reads the ket leg X̃_{(za),Q} as h_X X_{Q,(za)} from the same
  gathered rows, exact for operators of definite hermiticity; the general (non-Hermitian, D36)
  path needs a second gather.
* No imsrg++ bug was found in `comm223ss`; K0 reproduces it to rounding, with and without the
  perturbative-triples mode (6j hash map for 2J > 2 emax + 1 included).

## 8. Deviations from the plan

* [2,3]→3 and [3,3]→2 were not prototyped (time); their layout dependence is the GEMM dimension
  (ΣN³: B ≈ 0.97 A, C ≈ 26 A at emax 12) and the orbit-label access (identity in A, V_T in B,
  native in C). They are next steps on layout A (`14` §4).
* No C-native GEMM kernel (planning K2): C is timed with the shared W tiles and its E R epilogue.
* K1 gathers the two-body rows once per call into global tables instead of per tile (a benchmark
  shortcut; 1.2 MB at emax 3, 15 GB at emax 12; production gathers per tile).
* Counts at emax 10-12 are metadata-only runs (M10-Q1).
