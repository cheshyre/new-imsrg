# 14 — IMSRG(3) plan

IMSRG(3) is the **top-priority extension** *(user)*. Implement it after the core, 3N input, and
natural orbitals are validated. **Test only at emax = 2 and 3** *(user)*. The variants that
matter most *(user)*:

1. **IMSRG(3)-N7**: only the N⁷-scaling 3B terms (imsrg++ `imsrg3_n7`).
2. **IMSRG(3f2)**: factorized double commutators. They avoid storing 3B operators by
   evaluating the [[η,Γ]₃, ·] contributions through 2B intermediates
   (imsrg++ `FactorizedDoubleCommutator.cc`, used from BCH).
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

Fix-and-document policy *(user)*: where a bug changes results, the new code is correct, and the
validation for that path uses a brute-force m-scheme reference (below) instead of imsrg-devel.

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

## 4. Step 3: implementation order

1. **Perturbative triples**: the [2,2]→3 term streamed channel by channel. It needs no full 3B
   storage, because the energy accumulates per channel. Validate against imsrg-devel
   (`perturbative_triples=true`) on the O16/He4 emax 2/3 tests.
2. **IMSRG(3)-N7 terms**: implement the 3B part of the operator (chosen layout), all terms in
   the N7 preset, CommutatorOptions presets, and 3B cuts (E3max, dE3max, OccNat3Cut). Validate
   term by term against imsrg-devel on random operators (emax 2), against brute force where
   imsrg-devel is buggy, then end to end (E0 of a Magnus IMSRG(3)-N7 flow at emax 2 and 3).
3. **IMSRG(3f2)**: the factorized double-commutator terms inside BCH. Validate against
   imsrg-devel (`UnitTestFactorized.py` cases) and against explicit IMSRG(3) evaluation of the
   same double commutator at emax 2. imsrg++ has no executable parameter for it (only
   `BCH::SetUseFactorizedCorrection`, reachable from pyIMSRG), so golden flow data come from the
   harness (`refharness flow --factorized`). In our code it is `BchOptions::factorized`.
4. Later: the full IMSRG(3) term set; MPI distribution of 3B blocks (D14 phase 2); GPU tuning.

## 5. Data-structure requirements fixed now (so core work does not block IMSRG(3))

* `Operator::three_body` is `std::optional<ThreeBody>`. `ThreeBody` hides its layout behind a
  small interface, so the layout choice does not leak into the commutator signatures.
* `ThreeBodyLayout` provides `imsrgpp_norm()`, which reproduces imsrg++'s n3 for any layout (D12).
* `ThreeBodyChannels` lives in `basis` and is built lazily on request
  (`ModelSpace::with_three_body(cuts)`).
* `CommutatorOptions` already carries `Truncation`, `TermMask`, and `ThreeBodyCuts`
  {e3max, de3max, occnat3_cut, threebody_threshold}.
* `Workspace` owns per-thread and per-team scratch sized from the 3B channel extents. It is
  never allocated inside kernels.
