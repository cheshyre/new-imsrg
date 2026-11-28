# 13 — Physics modules: what to implement and what must match

This is a condensed specification. The detailed behaviour of the reference, with file:line
citations, is in `design/reference/0*.md`. **Read the relevant reference file before
implementing a module.**

## 1. basis (reference: `reference/01` §1–2)

* Orbit ordering follows the imsrg++ formula (D11): shell N ascending, l descending, j
  descending, proton (tz2 = −1) then neutron; proton index even, neutron = proton + 1.
  Orbits are sorted by HO energy.
* Truncations: `emax`, `lmax`, and the optional `emax_unocc`/`emax_imsrg` from the old
  parameters. There is **no E2max cut on 2B kets** (E2max is used only by file readers).
  E3max: the imsrg++ *driver* default is **12** (`Parameters.hh:130`, applied at `imsrg++.cc:335`);
  the `min(14, 3*emax)` in the ModelSpace constructor is always overwritten by the driver. Use 12
  as the legacy default, and make it explicit in the config.
* Reference: nucleus names such as `O16`, `Ca48`, `He4` map to Z and N. Filling is by HO energy
  ordering, with equal filling of open shells, as in imsrg++, unless an occupation file is given.
  After HF, orbits are re-filled by HF energy (`freeze_occupations=false` is the default).
* Valence spaces: named spaces (`p-shell`, `sd-shell`, `fp-shell`, `sdfp`, `psd`, ...), plus
  "nucleus as valence space" for single reference, plus custom spaces. The authoritative list
  is the table in imsrg++'s ModelSpace; copy it exactly.
* 2B channels (J, P, Tz) and kets p ≤ q in lexicographic order. Per-channel index subsets
  (hh, ph, pp, cc, vc, qc, vv, qv, qq) are computed from the Partition and Reference.
* 3B channels: (2J3, P, 2Tz3). The 3B ket set (which orderings and which Jab) is
  **layout-dependent**; see `14-imsrg3-plan.md`. Cuts: E3max, dE3max, OccNat3Cut.

## 2. operator (reference: `reference/01` §3–5)

* 2B normalization: matrix elements are stored normalized. imsrg++ getters return
  `sqrt((1+δpq)(1+δrs)) * V` for the "unnormalized" accessors; implement the same pair of
  accessors (`get_normalized`, `get_unnormalized`) and name them explicitly.
* Norms (D12): 1B part weighted by (2j+1), 2B part by (2J+1)². The total is √(n1² + n2² + n3²)
  **without** the zero-body part (`Operator.cc:1644-1661`). n3 is imsrg++'s Frobenius norm over its
  canonical pn storage (`ThreeBodyStorage_pn.cc:341`); every 3B layout must reproduce that value
  (`ThreeBodyLayout::imsrgpp_norm()`, tested against harness dumps). These norms drive step sizes,
  Omega splitting and BCH termination, so they must match imsrg++ exactly.
* Normal ordering: vacuum ↔ reference for 2B (0, 1, 2 body) and 3B → 2B, 1B, 0B. NO2B with
  fractional occupations is included.
* Known reference bugs to fix (fix + document policy): `Subtract` in isospin 3B storage adds;
  `AddToTBME` misses the anti-Hermitian sign; `ChangeNormalization` swaps rows and columns; the
  reduced 2B normal-ordering branch zeroes non-Hermitian entries. Write a unit test for each
  that demonstrates the correct behaviour.

**Tensor operators (M15, implemented 2026-11-26).** `OperatorSpec` accepts rank_J 0..(2 emax+1),
parity ±, delta_Tz 0..2 (particle_rank <= 2, Hermitian or anti-Hermitian); every non-scalar spec is
stored REDUCED as imsrg++ stores it (M15-Q1..Q3; the m-scheme oracle shows this is plain Edmonds
reduction for 1B and pairs alike, AL-A-Q5). `TwoBody` keeps one block per channel pair ch_bra <= ch_ket
allowed by `ConnectsChannels` (D25); getters with (J_bra, J_ket) apply h (-1)^(J_bra-J_ket) for the
other orientation. Norms: 2B weight 1/(2L+1), x2 off-diagonal as imsrg++; the tensor 1B norm is by default
degeneracy-weighted, sum O_pq^2 (2j_p+1)(2j_q+1)/(2L+1) (`op::TensorOneBodyNorm::
kDegeneracyWeighted`), and imsrg++'s integer degeneracy incl. its quirk AL-A-Q6 (can be
negative, NaN norm) only as `kImsrgppInteger` (`BchOptions::tensor_one_body_norm`, compat
`tensor_one_body_norm_imsrgpp`). `TransformBasis` handles tensors (D_bra^T V D_ket). `NormalOrderTensor`
/ `ReNormalOrder` (ensemble -> core) sum both channel orientations; the fork's diagonal omission is
`TensorNormalOrderOptions::fork79e9330_diagonal_bug` (design/30 section 6, M15-1). I/O: raw `.tdump`
reader/writer (`tensor_dump.h`), `io::WriteTensorTokyo` (fork 79e9330 and devel formats).

## 3. io (reference: `reference/04` §4–5)

Phase-1 formats, in this order:

1. **me2j** (gz or plain text). The loop order is in `reference/04` §4 and in imsrg-devel
   `ReadWrite.cc:832ff`. The first line is a comment (for the test file:
   `(*** nn_transforms v0.2.0 -- generated on 03/07/2022, 10:56:07 -- me2j-f2 ***)`) that does
   not contain the truncations. `file2e1max`, `file2e2max` (default 2·file2e1max) and
   `file2lmax` (default file2e1max) must be given correctly or parsing silently desynchronizes.
   Provide a `expected_value_count(e1max, e2max, lmax)` function (same loops, 4 values per J),
   and when the whole file is read, check the count at EOF. For the test file,
   `file2e1max=14 file2e2max=28 file2lmax=14`; **verified**: the file holds exactly 168,066,656
   values after the header, which equals the predicted count for these truncations. Reading stops after the prefix
   needed for the calculation's emax. Parse the file as a **token stream** (the test file has 10 values per text line, unrelated to
   the record structure): 4 values per (abcd, J) record (`T0 pn, nn, T1 pn, pp`). Parse each value
   into a `float` first, then widen to double, exactly as imsrg++ does (`ReadWrite.cc:882`);
   otherwise read values differ at the 1e-7 relative level. Convert to pn exactly as
   `Set_pn_TBME_from_iso` does.
2. **NO2B** (`*.stream.bin`, float, or half with `no2b_precision=half`). The test file is
   `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin` with `file3e1max=6
   file3e2max=12 file3e3max=18`. imsrg++ takes the type from `3bme_type=no2b` (`imsrg++.cc:477`); the file name only selects
   the encoding. Our config key is `[interaction.three_body] type = "no2b"`. Verified: the test
   file holds exactly 32,685,840 floats, the expected count for (6, 12, 18) in single precision.
3. **Tokyo `.snt`** and **NuShellX `.int/.sps`** writers for valence-space interactions and
   operators. They must match imsrg++ output **byte-for-byte modulo floating-point formatting**;
   compare numerically, line by line.
4. **OperatorDump**: our own simple, documented text format (header with emax, orbit list,
   spec; then 0B, 1B, 2B blocks), used for golden data and debugging. imsrg-devel gets a small
   companion dumper (in `tools/reference/`, built against the old library) that writes the same
   format.
5. Later: full me3j (isospin), Navratil/Miyagi operator formats (needed, per the owner),
   binary scratch and checkpoint files (with a magic number, version and endianness tag; imsrg++
   has none). **The owner's ME1J/ME2JP/ME3JP formats** (pn-resolved ME2J/ME3J; readers and
   writers, text, gzip and later binary) carry the evolved H, stored Omegas and checkpoints:
   specification in `19-mexj-formats.md` (D54), milestone after tensor operators (D53).
6. **Dropped** *(owner)*: HDF5 3N, Oslo, OakRidge, "Jason", "Takayuki" and the other older readers.

**As implemented (M5).** `imsrg/io/input_stream.h` (`TextInput`: zlib `gzread` for plain and
gzip text alike, tokens parsed straight to `float`; `GzipWriter`), `imsrg/io/me2j.h`
(`me2j_value_count`, `read_me2j(path, {Me2jTruncation{e1max, e2max, lmax}, read_whole_file}, ms)`
returning a new Hermitian rank-2 operator), `imsrg/io/no2b.h` (`no2b_value_count`,
`read_no2b(path, {No2bFileTruncation, No2bPrecision}, op::No2bThreeBody&)`; binary when the
file name contains `stream.bin`, size checked against the count; the container is M4's
`op::No2bThreeBody`). Only the prefix needed for the model space is decoded: emax 4 reads 27,336
me2j values in 4 ms and 2.7 M NO2B values in 0.05 s; the whole me2j file (168,066,656 values,
count check) takes 9 s (test label `slow`). No caching is needed.
Writers: `imsrg/io/valence_writers.h` (`write_tokyo`, `write_nushellx_int`,
`write_nushellx_sps`; `ValenceWriteOptions` carries the provenance of the header lines and the
mode `Hamiltonian` or `IsospinAveragedOperator` = imsrg++ `"op"`). Golden files from the new
harness subcommand `refharness write-valence` (`tests/golden/io/`, HF HNO at emax 2 in the
sd shell): every number identical to imsrg++, comment lines identical except the version line.

## 4. hf (reference: `reference/04` §1)

* Algorithm to match: Fock matrix from the monopole of V (plus the 3N monopole from NO2B),
  per-channel `eigh`, a sign fix making the diagonal of C positive, density mixing with weight
  0.8 on the new density (check this in the code; record the exact scheme), convergence on
  `‖Δε‖_F / norb < 1e-8`, at most 1000 iterations. imsrg++'s DIIS has off-by-one and dead-code
  issues. **Owner: DIIS from the start.** Implement a correct DIIS (Pulay, on the commutator
  [F, ρ] error vectors) in M6, keep plain mixing selectable (needed to compare against imsrg++'s
  default path), and test that both reach the same HF solution within 1e-8 MeV.
* Output: `HFResult {E_HF, spe, C, occupations, iterations, converged}` plus the
  normal-ordered H in the HF basis (`D^T (V2 + V3NO) D` for the 2B part; 3B → NO2B).
* Known reference bugs: the dE3max cut uses `op->l` for all three orbits (HartreeFock.cc:1368);
  the SCF uses `std::rand()`. Fix both and document.
* **As implemented (M6).** `imsrg/hf/hartree_fock.h`: `hf::solve(H, const op::No2bThreeBody*
  v3, HFOptions) -> HFResult {e_hf, e1, e2, e3, iterations, converged, spe, C, rho, F,
  occupations, ediff_history}`; `Mixing::kLinear` reproduces imsrg++ (weight 0.2 on the old
  density, `FillLowestOrbits` schedule, convergence `‖Δε‖_F / norb < 1e-8`, unfrozen occupations
  by default) and `Mixing::kDiis` is Pulay DIIS on the Fock matrix (error F ρ − ρ F, metric
  weighted by 2j+1, `diis_start` = 2, subspace 8, history reset when occupations change). The
  NO2B monopole uses `No2bThreeBody::get_pn` with the `e3max` cut (imsrg++ `ModelSpace` E3max,
  default 12). Deviations M6-1/M6-2 in `30` §6 (rescue path beyond iteration 100 only).
  **Owner, 2026-11-15 (D36, D37):** the DIIS design is confirmed; the driver gets a
  `hartree_fock.mixing` key in run 2, defaulting to DIIS in TOML configs and to linear mixing in
  the legacy layer.

## 5. mbpt (reference: `reference/04` §2)

* MP2 and MP3 energies (for printing and tests).
* Natural orbitals: MBPT(2) one-body density matrix (pp, hh, ph diagrams, two-level regulator),
  diagonalized per (l, j, tz) channel, with orbit ordering by occupation (default) or energy
  (`NAT_order`). Optional `use_NAT_occupations`: threshold 0.02, back-fill 0.005 (check the
  code). Natural orbitals are built on top of the HF basis (HF first, then NAT), as imsrg++
  does; verify the exact pipeline in `imsrg++.cc`.
* MP4 and Van Vleck are out of the near-term scope.

**As implemented (M9).** `mbpt::Mp2Energy`, `mbpt::ComputeMp3Energy` (pp, hh, ph, total),
`mbpt2_density_matrix` (HFMBPT.cc:421-777 with the regulated denominators),
`natural_orbitals` (per-channel `eigh`, descending occupation, columns signed so C(i,i) >= 0,
trace check against A), `reference_density_ho`, `matmul`, `to_nat_basis`. mbpt does not depend on
hf: the NAT-basis mean-field state comes from `hf::SolutionInBasis(H, v3, C_ho2nat, occupations)` (C = C_ho2hf C_hf2nat, rho = reference density in the NAT basis, F[rho], E_HF), and `hf::NormalOrderedHamiltonian` then gives the NAT-basis Hamiltonian; the driver (`pipeline::PrepareHamiltonian`, `basis.kind = "nat"`) and `tests/unit/mbpt/mbpt_test_support.h` both use this public API (run 2; before, the test drove `hf::detail`). Not implemented: `use_NAT_occupations`,
`NAT_order` energy/mp2 (M9-Q3; owner: not now, added when a job needs them, D36). Verified: MP2/MP3 at emax 2 (2N and 2N+3N) to 1e-15; IT-1 EMP2
to 3e-9; IT-5 NAT E_HF to 2e-10 and NAT E_MP2 to 4e-9. In the driver (run 2): IT-5/smax10 (IT-4 + `basis=NAT`) reproduces the same step count (28), E0(s=10) to 4e-10 MeV, NAT E_HF to 2e-10 MeV and NAT EMP2 to 4e-10 MeV.

## 6. commutator (reference: `reference/02`)

* IMSRG(2) scalar terms: comm110, comm111, comm121, comm122, comm220, comm221, comm222_pp_hh,
  comm222_ph (names as in imsrg++). Each has a reference implementation in `commutator_ref`
  and term-by-term tests on random (anti-)Hermitian operators against imsrg-devel golden data.
* IMSRG(3) terms: see `14-imsrg3-plan.md`.
* Term selection uses an enum bitset `TermMask` in `CommutatorOptions` (no string keys) and
  named presets: `imsrg2`, `imsrg3_n7`, `imsrg3_full`. IMSRG(3f2) is **not** a TermMask preset:
  it acts at the BCH level (it reuses the nested commutator from two steps back, `BCH.cc:182-191`)
  and is switched on with `BchOptions::factorized` (`[imsrg3] factorized = true`). The owner's
  3f2 recipe (D32): hunter-gatherer Magnus, `omega_norm_max` 0.1, 1b intermediates on, 2b
  intermediates off; `factorized = true` sets these as defaults (`16` §2.8). Implemented in M10 3f2
  (kernels, `BchOptions`, config keys V-56/V-57, driver wiring, IT-10).
* **As implemented (M10 3f2):** header `imsrg/commutator/factorized.h`:
  `FactorizedDoubleCommutator(eta, gamma, Z, FactorizedOptions, ws)` (overwrites Z, rank 2, gamma's
  hermiticity; eta anti-Hermitian) evaluates imsrg++'s `comm223_231_chi1b` + `comm223_232_chi1b`
  through two ladder-product pairs `eta P eta`, `gamma P eta` (`detail::LadderProducts`, shared with
  the ladder terms), comm221ss-like contractions into four one-body intermediates, a comm121ss-shaped
  one-body pass and a comm122ss-shaped two-body pass over the lifted lists (no Pandya, no 6j; about
  half a commutator). `FactorizedEta` holds the eta-only intermediates so a BCH series computes them
  once. 2b intermediates throw NotImplementedError (D32 has them off). imsrg++'s occupation skip
  thresholds are not applied (`30` §6 M10-3f2-1). Validation: imsrg++ goldens
  (`tests/golden/factorized/`, `refharness factorized d32`) <= 6.7e-16; the plain-loop transcription
  `commutator_ref::FactorizedDoubleCommutator` (also <= 2.7e-16 against the goldens) agrees with the
  fast form to 5.2e-16 on closed-shell and fractional references. Cost (release-openmp, 8 threads,
  O16 emax 4, 2N): 35.6 ms per 3f2 Magnus step against 32.5 ms per plain hunter-gatherer IMSRG(2)
  step (1.10x; imsrg++ He4 emax 2: 1.6x); 3f2 converges there in 205 steps (E0 -164.158 MeV) against
  298 (-165.001).
* **As implemented (M7, tag `m7-api-ready`):** headers `imsrg/commutator/{options,workspace,
  commutator}.h`. `Commutator(X, Y, Z, opts, ws)` overwrites Z (reusing its storage when the
  spec matches) and an allocating overload returns Z; `terms::Comm*ss` add one term each;
  `commutator_spec` gives Z's spec (rank max of the inputs, hermiticity rule of imsrg++). The
  `Workspace` (one per model space, eagerly built, one call at a time) holds the device tables,
  the dense 6j table, the ladder ket lists and the Pandya data, and all scratch. Every fast term
  writes every element of its output part (no final symmetrization); X and Y may each be
  Hermitian, anti-Hermitian or general (D36, M7-Q2; done in run 2: a NonHermitian input takes
  the general paths below, Z is then general, tested against `commutator_ref`), and Z must
  carry the hermiticity `commutator_spec` gives. Ladders: per channel gathered GEMMs
  `X P Y` with P = diag(nbar nbar) and diag(n n), symmetrized with `Y P X = hX hY (X P Y)^T`,
  shared with the 221 term. Particle-hole: Pandya transform of X and Y into the cross-coupled
  channels (6j from the table), one GEMM `Ncc x nk x 2Ncc` per channel over the ordered pairs
  with n_a != n_b, and an inverse transform that reads only the stored half of Zbar through
  the flip symmetry `F W F = -hX hY Phi W Phi` (so, unlike imsrg++, it does not depend on the
  lexicographic ket order, reference/02 P-12). General paths (either input NonHermitian): the
  ladders take both products `X P Y` and `Y P X` (the second GEMM accumulates with alpha = -1, no extra
  buffer); the particle-hole term fills the Pandya matrices for all `2 Ncc`
  ordered pairs on both sides (two GEMMs `2Ncc x nk x 2Ncc` per channel into one result, `A0 = D Xbar, B0 = Ybar`, then
  `A1 = -D Ybar, B1 = Xbar`, lazily allocated scratch) and inverse-transforms every stored element
  without mirroring; comm122 applies the lifted one-body list to the bra and, with the one-body
  matrix transposed, to the ket; comm110/220 keep their sums for a general Z (they vanish only
  for an anti-Hermitian Z). The (anti-)Hermitian paths are unchanged. `commutator_ref` (tests only) evaluates the same
  formulas with plain loops for any hermiticity. Validation: imsrg-devel goldens at emax 1-2
  (all hermiticity pairs, closed and open shell) and fast vs reference on random operators,
  all deviations below 2e-15 relative; benchmark in `design/benchmarks/imsrg2-commutators.md`.

**Scalar-tensor commutators (M15).** `commutator/tensor.h`: `TensorCommutator(X, Y)` for scalar X
and tensor Y with comm111st, 121st, 122st, 222_pp_hh_221st (1B/2B parts separable) and 222_phst in
imsrg-devel's formulas and triangle/mirror conventions; `Commutator()` dispatches tensor operands
(so `BchTransform` and `MagnusSolver::Transform` work for tensors). Tables per rank: {J1 J2 L; ja jb jc}
and a dense 9j box table whose entries below `ninej_threshold` are zeroed (devel 1e-10, fork 1e-8).
The ph term Pandya-transforms over all ordered cross-coupled pairs and stores Zbar for c3 <= c4.
Validation: imsrg-devel goldens (emax 1-2, all terms, four hermiticity pairs, BCH), m-scheme brute
force for ranks 0-5 and fractional occupations. Cost (O16, 16 threads, rank 5): 0.9 s at emax 6, 9 s
at emax 8 per commutator (`tools/bench/tensor_commutator_bench`); memory and GPU follow-ups AL-A-Q2.

## 7. flow (reference: `reference/03`)

* **As implemented (F2b, `flow/ode.{h,cc}`, `OdeSolver`):** `flow_rk4` and `flow` integrate
  dH/ds = [eta(H), H] with the requested operators flowing alongside (`dO/ds = [eta, O]`; the driver
  builds them before the flow, imsrg++.cc:1074, and stage 2 integrates [s1, s1 + smax] for every method, D47).
  `flow_rk4` is imsrg++ `Solve_flow_RK4` (ds = min(dsmax, smax - s), eta recomputed at every stage,
  `shell-model-atan` cutoff 1.0 until ||eta|| < 1); `flow` is boost odeint 1.92 `integrate_adaptive`
  with `make_controlled<runge_kutta_dopri5>` reimplemented step by step (6 right-hand-side
  evaluations per try, FSAL, `decrease_step` 0.9 err^(-1/3) >= 0.2, `increase_step` for err < 0.5), with
  imsrg++'s error norm: sum over the state's operators of `||(|xerr|)/(eps_abs + eps_rel(|x| +
  |dt||dxdt|))||` (zero-body excluded), built from the element-wise operations of
  `operator/elementwise.h` (abs, divide, add a scalar to every element: NOT `operator+=(Operator&,
  double)`, which touches E0 only). `eta_criterion` and `ode_tolerance` are doubles (D47; imsrg++
  rounds both through float, quirk P7). As in imsrg++,
  after ||eta|| < eta_criterion dxdt = 0 and the integration runs on to smax. The status callback of
  `flow` fires for every right-hand-side evaluation, like the imsrg++ flow file; the flow log's E_MP2
  evaluates the H of that line. Deviations: M8-Q40. Goldens: `tests/golden/e2e/IT-14`. Every flowing
  operator stays in the error norm (owner, M8-Q50, D47).
* Methods, phase 1: `magnus` (the default: an Euler step on Omega,
  `Omega ← BCH_Product(η ds, Omega)`, then `H ← BCH_Transform(H0, Omega)`) and `flow_rk4` /
  `flow` (DP45 on H). Later: `magnus_modified_euler`, `magnus_adaptive`, `restore_4th_order`.
* Step size: `ds = min(domega/‖η‖, domega/‖η‖/(‖Ω‖+1e-9), (omega_norm_max+1e-5)/‖η‖, dsmax)`,
  recomputed every step, then capped with `ds = min(ds, smax - s)`; replicate the `domega` bump
  too (`IMSRGSolver.cc:290`, `:335`). `ds_0` is effectively unused in magnus; keep it for flow methods.
* Omega stack: when `‖Ω‖ > omega_norm_max`, push Ω and restart from the transformed H.
  Operators are transformed afterwards by replaying the stack. Storing the Omega stack (ME1J/ME2JP/
  ME3JP files, a `transform` task) and checkpoint/restart form the milestone after tensor operators
  (D53, D54, `19-mexj-formats.md`).
* **Hunter-gatherer** (`imsrg.omega.hunter_gatherer`, imsrg++ `IMSRGSolver.cc:85-115`): only two
  Omegas are kept; the hunter accumulates η ds by BCH products and, when its norm exceeds
  `omega_norm_max`, is gathered into the gatherer (`gatherer = BCH_Product(hunter, gatherer)`) and
  reset. **As implemented (M8, `SolverOptions::hunter_gatherer`):** a verbatim copy of imsrg++'s
  semantics (decision of the orchestrator, flow review item 3): the stack is `{gatherer, hunter}`;
  `GatherOmega` with a one-entry stack appends a zero hunter, else `gatherer = BchProduct(hunter,
  gatherer)` if `‖hunter‖ > 1e-12`, then zeroes the hunter and rebuilds `H_saved` from `H_0` through
  the gatherer; a gather inside the loop leaves `H(s)` untouched, and `Solve` gathers once more after
  the loop (also `set_generator` gathers if the hunter is non-zero) and then, unless the flow
  diverged, **sets `H(s) = H_saved`** (D48, below). `SolveResult::num_gathers` counts them as the
  `gathering Omega.` lines of imsrg++ (loop gathers + the final one); `results.json` carries
  `num_gathers` per stage and in total. Goldens: `tests/golden/e2e/IT-2/hunter_gatherer`, `IT-10`. A
  **prerequisite of IMSRG(3f2)** (D32). **D48 (owner, M8-Q10; imsrg++ deviation, `30` §6 row
  D48-1):** after the final gather H(s) is the rebuilt `H_saved` (H(s) = e^Ω H0 e^−Ω with the
  gathered Ω), so E0 and every transformed operator come from one Ω and stage 2 starts from it
  (imsrg++ leaves H(s) at the last hunter step, up to tens of keV away). Unit test
  `MagnusSolver.HunterGathererHamiltonianIsRebuiltWithGatheredOmega` checks `H(s) == Transform(H0)`;
  the D35 patch of imsrg-devel and `refharness flow` carry the same line and the hunter-gatherer
  goldens were regenerated from them (ΔE0 vs the old goldens in the commit message).
* IMSRG(3f2) (`BchOptions::factorized`, M10 3f2): `BchTransform` adds F(Omega, nested_{i-2}) to
  every nested term from the second on (before the series update and the epsilon test);
  `BchProduct` adds F(X, Y)/12 to the 1/12 term and F(Y, N_{k-2}) (N_0 = X) to N_k for k >= 2, as
  imsrg++ BCH.cc:153-191, 312-359. Magnus steps, gathers and operator transforms all inherit it.
  Validated against `refharness bch` goldens (<= 2e-17) and IT-10 (|dE0| <= 1.1e-13 MeV, identical
  step and gather counts).
* BCH transform: nested commutators with 1/i!, terminating on a threshold epsilon, at most 40
  terms. BCH product: implement exactly the terms imsrg++ includes; its comment lists a
  −1/24 [Y,[X,[X,Y]]] term that the code does **not** implement. That is a truncation choice,
  not a bug, so reproduce it and document it.
* Generators, phase 1: `white`, `atan`, `imaginary-time`, `wegner`, `shell-model`,
  `shell-model-atan`, `shell-model-imaginary-time`, `hartree-fock`, `qtransfer` (exact block
  definitions in `reference/03` §4). Denominators: Epstein-Nesbet (default) and
  Møller-Plesset, plus `denominator_delta` (and its orbit-restricted variant). The denominator
  floor in imsrg++ loses the sign of the denominator. **Owner: this is a bug, so fix it** (keep the
  sign), add it to the deviation log, and test it (QUESTIONS Q7). The three-body denominator
  (IMSRG(3) generator) gets the same sign-keeping floor, without `denominator_delta` (owner,
  M10-Q60, D49).
* Two-step decoupling: the core generator first, then the valence generator, with a new Omega.
  **D47 (owner):** s continues across the stages and stage 2 integrates [s1, s1 + smax] for every
  method. imsrg++ doubles `smax` for the second Magnus stage only, so its `flow_RK4` stage 2 is
  empty after a stage 1 that reached smax (`30` §6 row D47-1).
* Convergence: `‖η‖ < eta_criterion` or `s ≥ smax`.
* Flow log: CSV with a header (s, E0, ‖η‖, ‖Ω‖, ds, MP2 estimate, wall time) plus the same
  lines to stdout. The integration tests parse the CSV.

### 7.3 Checkpoint and restart (design/50 C7, 2026-11-26)

* Solver side (`flow/magnus.h`): `Solve(on_status, on_step)` calls `on_step(MagnusLoopState)` after
  every completed step; `MagnusLoopState` = stage, step, s, ds, the E0 of the divergence guard, gather
  counts and the step options in force (domega bump included). With H(s), H_saved and the Omega stack
  this fixes the rest of the stage. `MagnusSolver::Restore(state, generator, omegas, h_saved, h)` puts a
  freshly constructed solver into that state; the next `Solve` recomputes eta from H(s) and continues
  from the next step (no domega bump, no istep-0 line). The loop itself is unchanged. With the exact
  operators the continuation is bit for bit (test `MagnusSolver.RestoreContinuesBitForBit`).
  Hunter-gatherer restores are refused (the gatherer changes in place; later).
* Driver side (`driver/pipeline/checkpoint.{h,cc}`, `[checkpoint]` of design/16 §2.11):
  `FlowCheckpointer` writes, when `interval_minutes` of wall time have passed, the closed Omegas once,
  the H_saved of the current piece once, the current Omega and H(s), then `state.txt` atomically, then
  removes the previous per-step files, so the directory always holds one complete checkpoint. Only
  rank 0 writes. A restart (`checkpoint.restart`) re-runs HF/NAT (deterministic), checks the flow
  configuration hash, reads the checkpoint on every rank, restores the finished stage records and
  continues the stage in progress; stage 2 is entered directly when the checkpoint is in stage 2.
  The flow file stays continuous: a restart from a flow checkpoint keeps the rows of the existing
  file with (stage, istep) <= the checkpoint's (the restarted flow logs from istep + 1), drops later
  rows (written between the checkpoint and the kill) and appends; a missing file or one with another
  header starts afresh (warning), and a restart from `prepared/` only or a non-restart run writes a
  fresh file.
* Formats: native float64 (`io::WriteOperatorBinary`, bit-exact restart) or ME1J + float32 binary
  ME2JP (restart within float32 precision, owner answer to O4).
* **Prepared state (D75, 2026-11-27; `driver/pipeline/prepared_state.{h,cc}`).** With a checkpoint
  directory, `Run` writes `prepared/` once after stage 1 (as `prepared.tmp/`, then renamed):
  `hno.opb` (H_0 normal ordered in the IMSRG space, always native f64), `c_hf.bin` / `c_nat.bin`,
  and `prepared.toml` (written last: `preparation_hash`, mass/proton number, e_reference, e_mp2 as
  hexfloats, occupations and reference names of the IMSRG and normal-ordering spaces, HF/NAT
  scalars and occupations, natural occupations). A restart with `prepared/` (and the same
  `PreparationHash`) reads it on every rank instead of running readers, HF and NAT; the model spaces
  are rebuilt from the config and the stored occupations as `PrepareHamiltonian` builds them, so the
  restart stays bit for bit. Not stored: `HFResult::spe/rho/F/ediff_history` (unused downstream).
  A checkpoint without `prepared/` (older builds) is restarted with preparation; one with
  `prepared/` but no `state.txt` (killed before the first checkpoint) starts the flow from it.
  `checkpoint.prepared_from` starts a fresh flow from another run's `prepared/` (27Al v3 from v2).
  `PreparationHash` = `FlowConfigHash` without `[operators]`, `[transform]`, the `[imsrg]` keys other
  than `imsrg.space`, `run.task`, `run.scratch_dir`.
* **SIGTERM (D75).** `driver::InstallTerminationHandler()` (called by the executable's `main` only)
  sets a `sig_atomic_t` flag and nothing else. After every Magnus step all ranks combine it
  (`AllreduceMax`); if set, `FlowCheckpointer::WriteNow` writes a checkpoint (if a directory is
  configured) and `InterruptedError` is thrown; `main` returns exit status 75 (EX_TEMPFAIL) without
  aborting MPI. ODE methods do not check the flag. `tools/al27/run_flow.sh` forwards the batch
  shell's TERM (`#SBATCH --signal=B:TERM@300`) to srun.

## 8. opslib (reference: `reference/05` §1)

Phase 1 (scalar): kinetic energy T (with A-dependence, `A` parameter), intrinsic kinetic energy
`Trel`, `Tcm`, `HCM` (Lawson term `BetaCM`, `hwBetaCM`), `Rp2`, `Rn2`, `Rm2`, `Rso`/Darwin-Foldy
pieces as in imsrg++, `OBD`/`TBD` (later). Later: tensor (E2, M1, GT, Fermi; the first extension
after IMSRG(3), D39), dagger, M0nu, DM NREFT. The name grammar and legacy names are in `16-config-and-cli.md`.

**As implemented (M6).** Namespace `imsrg::opslib`, every physics parameter explicit (no hidden
model-space state): `kinetic_energy(ms)` (Tlab), `t_cm(ms, A)`, `t_rel(ms, A)` (T_int used in
H), `r2_cm(ms, A)`, `h_cm(ms, A, hw_trap)`, `r2_one_body/r2_two_body(ms, Species)`,
`rp2(ms, A, Z)` (imsrg++ "Rp2" includes `RpSpinOrbitCorrection`; ours does not, D35 below), `rn2(ms, A, Z)`, `rm2(ms, A)`. Equal to the imsrg-devel dumps within 1.1e-14 (emax 2
and 4); emax 6 builds take 0.4 s (t_rel) and 0.7 s (rp2).

**Owner decisions 2026-11-15 (D35), implemented in run 2:** `t_cm` keeps all two-body entries
(imsrg++ drops |v| ≤ 1e-7, which also enters T_int and thus H); `rp2` loses `include_spin_orbit`
and never includes the spin-orbit correction, which stays available as the separate `Rso` operator
(`RpSpinOrbitCorrection`, catalog name `Rso`; a charge radius adds `Rso/Z` explicitly). No imsrg++ compatibility switch: the goldens come from a patched
imsrg-devel build (`30` §3).

**DM NREFT (Al27 night, 2026-11-26; design/50 C2).** `opslib/nreft.h` ports the fork's
`DM_NREFT` (79e9330): HO radial integrals `jho/jdmho/jdpho` in y = q^2/(4 hw m_N)
(`detail/nreft_radial.h`), the blocks M, Ms(J,L), Mg(J,L), Phi', Phi'', tilde-Phi' = Phi' + Ms(J,J)/2,
Sigma, Sigma', Sigma'', Delta, Delta' (`Family`), isospin selections p/n/+/-, the fork's channel sets
and J = 0 non-reduced storage, as dense reduced one-body tables (`OneBodyTable`), and the shorthands
`F<family><L><p|n>_<q>` (`ParseShorthand`; sqrt(4 pi) on FM, FPhipp and (owner, design/50 §4)
FPhitp, `PhitpShorthandPrefactor()`; all Hermitian). Scalar (L = 0) shorthands build operators
(`BuildScalarShorthand`, driver catalog P1 rows FM0p/n, FPhipp0p/n); L > 0 waits for the M15 tensor
storage (planned catalog rows). Tested against the fork dumps (`tests/golden/nreft`, test
`opslib_nreft`): 1e-12 relative or 1e-14 of the largest element at the same q (observed 7.8e-15).

**Charge radius (D35, run 2; `driver/pipeline/charge_radius.h`).** The operator contract: `Rp2` is the
intrinsic point-proton radius squared WITHOUT the spin-orbit term, `Rso` the spin-orbit correction (so
imsrg++'s old `Rp2` equals our `Rp2 + Rso/Z` to 3.5e-17). The driver's output stage builds the charge
radius whenever the plain `Rp2` is requested (not `Rp2Z`/`Rp2AZ`; operators of other families are
untouched): `Rch2 = Rp2 + Rso/Z + r_p^2 + (N/Z) r_n^2 + Darwin-Foldy` (fm^2) with `N = A - Z` of the
Rp2 operator and `r_p^2 = 0.707`, `r_n^2 = -0.1161`, Darwin-Foldy = 0.033172 (`imsrg::constants`, from
`PhysicalConstants.hh`; the imsrg++ lines are `imsrg++.cc:845` and `:1367`). `Rch = sqrt(Rch2)` (`null` in
`results.json` if negative; imsrg++'s `|Rp2| < 1e-6 -> 0` guard of the HF printout is not copied).
`Rso` is evaluated internally if it was not requested: for Magnus it is one more operator replayed with the
Omega stack, for the ODE methods one more operator flowing alongside H; it gets no `operators` entry and no
file. Both the HF (`reference`) and the final (`imsrg`) values are written, `results.json` key
`charge_radius` (`16` §2.10). Validated against the pristine imsrg++ (its `Rp2` still held the spin-orbit
term), `flow_RK4`/`flow` runs, which are the only methods for which it prints a charge radius: HF radius to
2e-9 fm, IMSRG radius to the 6 printed decimals (He4 emax 2 and O16 emax 3; tests in `driver_flow_methods`
and `driver_charge_radius`).

## 9. driver

The pipeline replicates `imsrg++.cc` (see `reference/03` §1):

parse config → build ModelSpace → read 2N (and 3N) → build H = T_int + V (+ V3) → HF (or HO, or
NAT) → normal order → IMSRG (SR or VS; 1 or 2 steps) → transform requested operators → write
outputs (valence-space files, operator files, results summary JSON, flow CSV) → optional
perturbative triples or IMSRG(3)-at-end.

Every branch point is a typed config option. The resolved config is written next to the outputs.
