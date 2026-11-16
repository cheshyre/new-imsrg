# 14 — IMSRG(3) plan

IMSRG(3) is the **top-priority extension** *(user)*. Implement it after the core, 3N input, and
natural orbitals are validated. **Test only at emax = 2 and 3** *(user)*. The variants that
matter most *(user)*:

1. **IMSRG(3)-N7**: only the N⁷-scaling 3B terms (imsrg++ `imsrg3_n7`).
2. **IMSRG(3f2)**: factorized double commutators. They avoid storing 3B operators by
   evaluating the [[η,Γ]₃, ·] contributions through 2B intermediates
   (imsrg++ `FactorizedDoubleCommutator.cc`, used from BCH). **Definition (owner, D32):** plain
   IMSRG(2) plus hunter-gatherer Magnus, `omega_norm_max` 0.1, the factorized correction in BCH,
   1b intermediates on and 2b intermediates off. In pyIMSRG terms:
   `imsrgsolver.SetHunterGatherer(True)`, `imsrgsolver.SetOmegaNormMax(0.1)`,
   `BCH.SetUseFactorizedCorrection(True)`,
   `Commutator.FactorizedDoubleCommutator.SetUse_1b_Intermediates(True)`,
   `Commutator.FactorizedDoubleCommutator.SetUse_2b_Intermediates(False)` (imsrg++ defaults: both
   intermediates on, `FactorizedDoubleCommutator.cc:15-16`).
3. **Perturbative triples**: the energy correction from the induced [2,2]→3 part
   (imsrg++ `perturbative_triples`, which reuses `comm223ss` and takes denominators from the 1B part).

Full IMSRG(3) with all terms and cuts is lower priority, but the term registry must allow it.

## 1. Reference material

`reference/02-commutators.md` §1, §4 and §8 catalog every IMSRG(3) term, the truncation flags,
the hotspots (`comm223ss` dominates; then `comm232`, `comm333_pph`, `comm233_ph`), and the
known bugs:

* copy-paste errors in the dE3max/OccNat cuts in `comm231ss` and `comm233_pp_hhss`;
* `comm331ss` applies its cuts asymmetrically (to one external index only);
* `comm233_phss_tz` is flagged in its own header as failing a unit test, yet it is the live path
  for some channel-changing operators;
* the factorized kernels assume η is anti-Hermitian and apply no 3B truncation;
* every scalar commutator with IMSRG(3) enabled allocates and zero-fills Z's full 3B storage,
  even when the 3B terms are skipped.

Fix-and-document policy *(user; confirmed for these IMSRG(3) bugs on 2026-11-15, D34)*: where a
bug changes results, the new code is correct, and the validation for that path uses a brute-force
m-scheme reference (below) instead of imsrg-devel. Tolerances are those of D19 (terms 1e-10
relative, perturbative triples 1e-4 MeV, N7/3f2 E0 1e-3 MeV).

**Production target (owner, D34):** IMSRG(3)-N7 at emax 10–12 on several nodes. Tests stay at
emax 2 and 3; the layout, its memory per operator and MPI phase 2 must be judged against that target.
**Refined (owner, 2026-11-16, D45):** E3max 16 without a dE3max cut as the starting point (about
177 GB per packed layout-A operator at emax 12, about 1.06 TB for the six operators of an N7 step),
nodes with 256 GB or less (sometimes 100 GB), at most 50 % of a node for 3B operators, many nodes:
single channels are split across ranks (row-wise, cost-balanced distribution as the baseline), and
the N7 kernels are tiled over row ranges of distributed blocks. Production also needs
`occ_nat3_cut` (D49). Timing-only kernel runs at emax 4–6 are allowed (D52).

### 1.1 The N7 term set (imsrg++ `imsrg3_n7`) and its imsrg++ bugs (inventory, run 2)

`IMSRG3=true imsrg3_n7=true` (imsrg++.cc:1048-1049) calls `Commutator::SetUseIMSRG3(true)`
(Commutator.cc:99-109: all 13 scalar three-body terms on) and then `SetUseIMSRG3N7(true)`
(111-129: the six N8/N9 terms 332_ppph_hhhp, 332_pphh, 233_pp_hh, 233_ph, 333_ppp_hhh,
333_pph_hhp off, the seven N7 terms on). `CommutatorScalarScalar` (322-487) gives Z particle rank
3, runs the IMSRG(2) terms, then, iff `X.Norm() > threebody_threshold` and `Y.Norm() >
threebody_threshold` (396), the three-body block in this order (403-430):

| Term | Ranks | imsrg++ (`IMSRG3Commutators.cc`) | Ours | Brute force / imsrg-devel |
|---|---|---|---|---|
| comm133ss | [1,3]+[3,1]→3 | 4517-4989 (dense embedding GEMM per 3B channel) | `kernels/comm133.cc` | 1e-15 / 2.1e-16 |
| comm330ss | [3,3]→0 | 72-186 | `kernels/comm330.cc` | 2.7e-15 / 1e-10 (E3max 6, the E3max-3 value is 0) |
| comm223ss | [2,2]→3 | 5012-5376 | tiles of `kernels/comm223.cc` | 2e-15 / 2.2e-15 |
| comm232ss | [2,3]+[3,2]→2 | 814 → `comm232ss_srs_optimized` 867 → `_old` 1305-2059 (GEMM per one-body channel) or `comm232ss_slow` 2065-2253 | `kernels/comm232.cc` (pair-third blocks + element-wise reference) | 7.4e-16 / 6.1e-16 |
| comm331ss | [3,3]→1 | 199-411 | `kernels/comm331.cc` | 6.1e-15 / 3.1e-16 |
| comm231ss | [2,3]+[3,2]→1 | 426-596 | `kernels/comm231.cc` | 2.1e-15 / 1.2e-16 |
| comm132ss | [1,3]+[3,1]→2 | 615-811 | `kernels/comm132.cc` | 1.1e-15 (emax 2) / 4.3e-16 |

The generator's three-body part is `Generator::ConstructGenerator_SingleRef_3body`
(Generator.cc:342-427) with `Get3bDenominator` (231-239). The kernels evaluate the m-scheme
formulas of `commutator/imsrg3.h` for any hermiticity; the deviations are measured against the
m-scheme brute force (`commutator_ref::mscheme`) and against imsrg-devel (`refharness comm3`,
`tests/golden/n7`).

imsrg++ IMSRG(3) bugs and quirks (section 1 list, re-checked, plus new findings):

* `comm231ss` cut copy-paste (`d_ed` from `ec`, 544; `occnat_c/d` from the bra, 545-546) and its
  `> cut` instead of `< cut` (521-522, 551-552), and `comm331ss`'s cut on (a, b, j) but never
  (a, b, i) (289-292): only with dE3max/OccNat3Cut active; impossible here because cut kets do
  not exist (D34). The cut paths are validated by the brute force (comm232ss with E3max and
  dE3max cuts) and by the comm223 kept-space tests.
* NEW: `comm232ss_srs_optimized_old` obtains the annihilator side of the term from the creator
  side through `hermX * hermY` (2047), wrong for a non-Hermitian operand (imsrg++ has no
  non-Hermitian three-body storage, so its flows never hit it). Ours computes both sides
  explicitly; the non-Hermitian brute-force cases cover it (`30` §6 M10-N7-1).
* NEW: `comm232ss` drops occupation factors below 1e-6 (1552); we drop only exact zeros (differs
  only for occupations within 1e-6 of 0 or 1, `30` §6 M10-N7-2).
* NEW (harness): `comm331ss` and `comm231ss` abort with "Updating SixJList inside a parallel
  loop" unless the 6j cache was filled before (they rely on an earlier `comm223ss` or the first
  pass); `refharness comm3` calls `PreCalculateSixJ()` first.
* NEW: the three-body denominator has no floor and no `denominator_delta` (M10-Q60); copied.
  **To change (owner, D49):** the sign-keeping floor of D21 is extended to the 3B denominator
  (no `denominator_delta`); no golden changes, a deviation entry in `30` §6.
* `comm233_pp_hhss` cut copy-paste and the failing `comm233_phss_tz`: N8 terms, not implemented.
* The factorized kernels assume an anti-Hermitian eta and apply no 3B truncation: the 3f2 track
  (D32), unchanged.
* Z's full 3B storage is allocated and zero-filled on every call (Commutator.cc:344-345): ours is
  allocated once when Z is reused in the flow loop (set to zero per call, as imsrg++).

Next steps towards the production target (emax 10-12): (1) comm232ss caps its intermediate
blocks at 512 MiB per batch but never splits a channel (about 33 GB for one block set at the
target) and runs all Z2 elements through every batch: tile by column ranges and clamp J3 per
batch (review findings 2, 3; M10-Q63); (2) the element-wise readers of 133, 132, 231, 331 recouple
every element with binary searches: precompute the per-row recoupling expansion, or move them to
pair-third tiles like 223/232; (3) the pair-third tables of comm232ss are rebuilt per call and
belong in the context (M10-Q64); (4) the kernels call Kokkos range/team policies directly (as
comm223), a candidate for the exec facade (M10-Q65); (5) MPI distribution of the 3B blocks
(D14 phase 2, M12b).
**Status after the 2026-11-16 interview:** items (1) and (3) were done by the comm232 follow-up
47aa8b2 (column tiles split channels, J3 clamped per batch, tables in `ThreeBodyContext`). Run 3
(D53) continues with: tiling all N7 kernels over **row ranges** of channel blocks, so that a
row-distributed block (D45) needs no kernel change; (2); `exec::TeamFor`/`TeamReduce` options
(M10-Q44/Q65); a lazy `ThreeBodyContext` for IMSRG(2)-only use (M10-Q66); per-tile 6j slices and
a ket cache for `Comm223`, judged with timing-only runs at emax 4–6 (M10-Q41/Q42, D52);
perturbative triples at emax ≥ 8 (R per tile, no E map, only ppp/hhh blocks, M10-Q46) and the
split-stack order fix `BchProduct(Omega_n, omega)` (M10-Q40); `occ_nat3_cut` with NAT occupations
passed into the 3B space (M10-Q62); an IT-9 Rp2 operator golden from the patched reference
(M10-Q61). Then (5) in M12b phase 2.

## 2. Step 1: 3B layout benchmark (decide by measurement) *(user)*

Prototype these candidate layouts for the 3B blocks of a (2J3, P, 2Tz3) channel, behind one
small interface (`ThreeBodyLayout` concept: build kets, map (a,b,c,Jab) to row/coefficients,
block extents):

| Layout | Kets per channel | Pros | Cons |
|--------|------------------|------|------|
| **A. Canonical (imsrg++-like)** | a ≤ b ≤ c (orbit order) × allowed Jab; overcomplete when orbits coincide | Smallest storage; one-to-one comparison with imsrg++ | Every access to another ordering recouples (6j sums); gather-heavy kernels |
| **B. Orthonormal antisymmetric basis** | Per channel, eigenvectors (eigenvalue 1) of the antisymmetrizer on the a ≤ b ≤ c, Jab set | Minimal dimension; dense, GEMM-friendly blocks for [3,3]→3 and the 3B-diagonal parts of [2,3]→3 | Needs basis-change maps (dense, per channel) to reach orbit labels; extra up-front cost |
| **C. Partially antisymmetric** | \|(ab)Jab c⟩ with a < b (or a ≤ b), all c | Simple kernels; 2B operators act directly on the (ab) pair | About 3× memory; antisymmetrizer applied explicitly |

Benchmark protocol (**emax = 2 and 3 only**, owner rule; E3max as in the tests):

* kernels: [2,2]→3 (`comm223`-type), [2,3]→3 (`comm233`-type, pp/hh and ph), [3,3]→2, and the
  perturbative-triples energy;
* metrics: memory per operator, build time for the layout metadata, kernel time (OpenMP, 18
  threads), and agreement with each other and with imsrg-devel / brute force to 1e-10.

Record the results in `design/benchmarks/imsrg3-layout.md`. **The session chooses the layout
itself** *(owner: "decide and continue")*: take the best speed at acceptable memory, document the
reasoning and the numbers, add a decision D-row to `00-requirements-and-decisions.md`, and continue
with the IMSRG(3) terms. Keep the non-chosen prototypes out of the
production library (delete them or move them to `tools/bench/`).

**Result (run 2, 2026-11-15, D43): layout A**, stored as packed channel blocks, with the [2,2]→3
term and the perturbative triples computed through pair-third tiles. Numbers and reasoning:
`benchmarks/imsrg3-layout.md`. In short: C needs 8.8× A's memory (702 GB per operator at emax 12 /
E3max 14) and fails the memory gate; B saves 2 % memory at the target and is within the run-to-run
spread of A in every kernel; A compares element by element with imsrg++. The pair-third tiling
reproduces imsrg++ `comm223ss` to 1e-15 and is 54× faster at emax 3 (18 threads). The prototypes
stay in `tools/bench/imsrg3_layout/` (benchmark target only); [2,3]→3 and [3,3]→2 were not
prototyped and move to the N7 work below.

## 3. Step 2: brute-force m-scheme reference (validation backbone)

Independent of imsrg++, implement a **tiny m-scheme engine** for validation only (in
`commutator_ref`, or in Python with numpy for clarity and speed of writing):

* expand J-coupled 1B/2B/3B operators to m-scheme at emax ≤ 2 (He4/O16 references);
* evaluate commutators in m-scheme by direct second-quantized algebra (normal-ordered
  Wick contractions coded generically for ranks ≤ 3);
* recouple the result back to J-scheme and compare term by term.

This validates J-coupled kernels where imsrg-devel is buggy or unavailable, and it validates
normal ordering of 3B operators. imsrg-devel's UnitTest.cc has a version of this approach to
learn from (`reference/05` §3), but note its shadowed `random_me` bug, which zeroes most random
3B matrix elements.

**Done (M10 step 3, run 2):** `commutator_ref::mscheme` (test-only module `commutator_ref`):
m-states ordered by (orbit, 2m), antisymmetric tensors on sorted index sets, J→m expansion
(plain CG expansion of the unnormalized elements) and m→J contraction, and ONE generic Wick
engine for any [A^(nA), B^(nB)]^(nC) with nA, nB, nC <= 3 (all cross contractions with n and
1 - n, the pattern sign computed by permutation parity once per pattern, shuffles for the output
antisymmetrization), so no term-specific formula enters the oracle. Its own validation: the
round trip Contract(Expand(X)) = X on physical rank-3 operators (1.1e-15), the IMSRG(2) terms
of `commutator_ref` and comm223 (validated against imsrg-devel) to 1.2e-15. Random three-body
test operands must be physical (layout A keeps linearly dependent kets for triples with equal
orbits): `tests/support/three_body_test_support.h` builds them as Comm223 of two random two-body
operators. Each N7 part runs in under 1 s at emax 1 (16 m-states); emax 2 takes ~13 s for four
terms with E3max 4.

## 4. Step 3: implementation order

**Order (owner, D39):** IMSRG(3f2) needs no 3B storage, so it is developed **in parallel with the
layout benchmark** (item 3 below, as its own track); then perturbative triples → N7 on the chosen
layout.

0. **Production pieces of the layout decision (D43)**, each a small task with the benchmark
   prototype (`tools/bench/imsrg3_layout/`) as the reference implementation:
   * `basis`: the kept canonical space (kets failing dE3max/OccNat3Cut dropped at enumeration,
     orbit-triple blocks with their symmetry factors; prototype `CanonicalSpace`), as part of
     `ModelSpace::WithThreeBody` with the cuts in `ThreeBodyCuts`;
   * `angmom`: the 3B 6j family {j1 j2 J12; j3 J J23} (prototype `SixJ3Table`); the existing
     `SixJTable` reads 0 for a three-body J and must not be used for it;
   * `operator`: `op::ThreeBody` in layout A, packed channel blocks, `imsrgpp_norm` (D12), dump
     support; the pair-third states with the maps R and E (prototype `PairThirdSpace`), which
     the [2,2]→3 term and the readers of recoupled 3B elements (232, 331, 231, 132) share;
   * `commutator`: `comm223ss` as the tiled kernel K1, with the production changes the review
     asked for: a per-tile gather instead of the benchmark's global gather tables (15 GB at
     emax 12), the internal orbit a restricted to the 6j-allowed j range (60-75 % of the group
     terms vanish), team-level parallelism and staging through device buffers, and writes of
     the packed triangle only; a second gather for non-Hermitian operands (D36).
1. **Perturbative triples**: the [2,2]→3 term streamed channel by channel. It needs no full 3B
   storage, because the energy accumulates per channel. Validate against imsrg-devel
   (`perturbative_triples=true`) on the O16/He4 emax 2/3 tests. The tiled prototype
   (`TriplesTiled`) already agrees with imsrg++ to 3e-15 at O16 emax 3 through
   `refharness comm223 ... epsilon=` and costs 5 % of a full [2,2]→3.
   **Done (M10 step 2, run 2) for items 0 and 1:** `basis::ThreeBodySpace` (kept canonical space,
   built by `ModelSpace::WithThreeBody`, cuts in `ThreeBodyCuts`), `angmom::SixJ3Table` and
   `three_body_recoupling.h` (`SixJTable` now throws on the host and returns NaN on the device for
   arguments beyond its coverage, e.g. a three-body J), `op::ThreeBody` (packed lower triangle for
   H/A, dense for N; `ImsrgppNorm`; the OperatorDump 3B section), `op::PairThirdSpace` (R, E),
   `commutator::Comm223` and `PerturbativeTriplesEnergy` (`commutator/three_body.h`: per-tile
   gathers, 6j-allowed j blocks, team scratch, packed writes, a general ket-leg gather for
   non-Hermitian operands), `flow::PerturbativeTriples` and the driver call after stage 1
   (`results.json` schema 2). Against imsrg-devel: [2,2]→3 element-wise 2.2e-15 of max|Z| (O16
   emax 2, E3max 3, `tests/golden/imsrg3`), 3B norms <= 1.6e-14 and triples energies <= 1.3e-15
   relative (emax 2 and 3); IT-8 He4 emax 2 and O16 emax 3: same step counts (192, 154), dE within
   1e-15 MeV of the reference harness (`harness_triples.json`). The production kernel takes
   15.2 ms against the prototype K1's 9.0 ms at O16 emax 3 (18 threads; the per-tile gathers,
   M10-Q41), the triples 0.32 against 0.43 ms; device tables 4.2 MB and 16.6 KB of scratch per
   team replace the global gather tables (15 GB at emax 12). Open: M10-Q40..Q46.
2. **IMSRG(3)-N7 terms**: implement the 3B part of the operator (chosen layout), all terms in
   the N7 preset, CommutatorOptions presets, and 3B cuts (E3max, dE3max, OccNat3Cut). Kets that
   fail dE3max or OccNat3Cut are **dropped when the 3B kets are enumerated** (owner, D34, M3-Q9):
   the cut defines the 3B space and acts identically in every term. Validate
   term by term against imsrg-devel on random operators (emax 2), against brute force where
   imsrg-devel is buggy, then end to end (E0 of a Magnus IMSRG(3)-N7 flow at emax 2 and 3).
   **Done (M10 step 3, run 2):** the seven terms of the inventory in section 1.1 on layout A
   (`commutator/imsrg3.h`, kernels `kernels/comm{330,331,231,132,232,133}.cc`, comm223ss through
   the tiles of item 0, now accumulating in place), `TermMask::Imsrg3N7()`/`Imsrg3Full()` (the
   N8/N9 terms are registered and throw NotImplementedError), `CommutatorOptions::
   threebody_threshold`, the IMSRG(3) truncation in `Commutator()` (rank-3 Z, imsrg++'s term
   order; three-body operands with an IMSRG(2) term set still throw), the generator's three-body
   part, the rank-3 Magnus flow and the driver's `[imsrg3] in_flow = true, variant = "n7"`. The
   brute-force m-scheme reference of section 3 is `commutator_ref::mscheme`. Validation: every
   term against the brute force (emax 1: He4, C12, fractional He4; emax 2 fractional for the
   terms with one-body legs; Hermitian, anti-Hermitian and non-Hermitian operands) <= 6.1e-15
   relative; against imsrg-devel (`refharness comm3`, `tests/golden/n7`, He4 and O16 emax 2)
   element by element at E3max 3 <= 6.1e-16 and the E3max-6 norms <= 1e-10; the full N7
   commutator against imsrg++'s 5.7e-16; the generator's three-body part exactly. IT-9: He4 and
   O16 emax 2 reproduce the D35-build goldens with identical step counts (250, 147) and dE0 =
   -3e-11 / -3e-10 MeV; emax 3 (label `slow`, pristine-build goldens): He4 248 steps (248), dE0
   -3.8e-10 MeV in 720 s, O16 173 steps (173), dE0 -3.0e-10 MeV in 644 s (dev build; imsrg++
   30 and 21 min with 8 threads), final Omega norms equal to all 9 printed digits. Time per
   term, O16 emax 2, E3max 6 (dev build, 8 threads): 133 7 ms,
   132 1 ms, 231 0.9 ms, 331 1.6 ms, 232 2.6 ms (release, 18 threads: 1.7 ms); IT-9 He4 emax 2
   takes 28-45 s against imsrg++'s 46 s (8 threads), O16 12.5 s against 24 s. Open: M10-Q60..Q66.
   The kernels other than 223 and 232 read recoupled elements one by one (binary searches in the
   channel's triples); they are fast at emax <= 3 but are not the production design for emax
   10-12 (next steps in section 1.1).
3. **IMSRG(3f2)**: the factorized double-commutator terms inside BCH. Validate against
   imsrg-devel (`UnitTestFactorized.py` cases) and against explicit IMSRG(3) evaluation of the
   same double commutator at emax 2. imsrg++ has no executable parameter for it (only
   `BCH::SetUseFactorizedCorrection`, reachable from pyIMSRG), so golden flow data come from the
   harness (`refharness flow --factorized`). In our code it is `BchOptions::factorized`, with the
   1b/2b-intermediate switches. **Prerequisites:** hunter-gatherer Magnus (an M8 follow-up) and
   refharness flags for hunter-gatherer, `omega_norm_max` and the 2b-intermediate switch; the
   IT-10 goldens are regenerated with the owner's settings (D32) before they are used.
   **Done (M10 3f2, run 2):** 1b intermediates only (`13` §6-7); the 2b intermediates are
   E-PLANNED (QUESTIONS M10-Q20). The explicit-IMSRG(3) cross-check is the harness's `direct`
   route (imsrg++ comm223ss + comm231ss + comm232ss), which agrees with imsrg++'s full factorized
   result to 1.3e-15 (`tests/golden/factorized/README.md`); our own explicit 3B check waits for the
   N7 track's 3B storage.
4. Later: the full IMSRG(3) term set; MPI distribution of 3B blocks (D14 phase 2, in M12b: evaluate
   the row-wise, cost-balanced block distribution of Hagen and Nam, Prog. Theor. Phys. Suppl. 196,
   102 (2012), arXiv:1203.3765, against distribution by channel, which is likely unbalanced; D40);
   GPU tuning after M12.

## 5. Data-structure requirements fixed now (so core work does not block IMSRG(3))

* `Operator::three_body` is `std::optional<ThreeBody>`. `ThreeBody` hides its layout behind a
  small interface, so the layout choice does not leak into the commutator signatures. The layout
  is A (D43): canonical kets, packed channel blocks; kernels see it through the pair-third maps.
* `ThreeBodyLayout` provides `imsrgpp_norm()`, which reproduces imsrg++'s n3 for any layout (D12).
* `ThreeBodyChannels` lives in `basis` and is built lazily on request
  (`ModelSpace::WithThreeBody(cuts)`).
* `CommutatorOptions` already carries `Truncation`, `TermMask`, and `ThreeBodyCuts`
  {e3max, de3max, occnat3_cut, threebody_threshold}.
* `Workspace` owns per-thread and per-team scratch sized from the 3B channel extents. It is
  never allocated inside kernels.
