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
   has none).
6. **Dropped** *(owner)*: HDF5 3N, Oslo, OakRidge, "Jason", "Takayuki" and the other older readers.

**As implemented (M5).** `imsrg/io/input_stream.hpp` (`TextInput`: zlib `gzread` for plain and
gzip text alike, tokens parsed straight to `float`; `GzipWriter`), `imsrg/io/me2j.hpp`
(`me2j_value_count`, `read_me2j(path, {Me2jTruncation{e1max, e2max, lmax}, read_whole_file}, ms)`
returning a new Hermitian rank-2 operator), `imsrg/io/no2b.hpp` (`no2b_value_count`,
`read_no2b(path, {No2bFileTruncation, No2bPrecision}, op::No2bThreeBody&)`; binary when the
file name contains `stream.bin`, size checked against the count; the container is M4's
`op::No2bThreeBody`). Only the prefix needed for the model space is decoded: emax 4 reads 27,336
me2j values in 4 ms and 2.7 M NO2B values in 0.05 s; the whole me2j file (168,066,656 values,
count check) takes 9 s (test label `slow`). No caching is needed.

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
* **As implemented (M6).** `imsrg/hf/hartree_fock.hpp`: `hf::solve(H, const op::No2bThreeBody*
  v3, HFOptions) -> HFResult {e_hf, e1, e2, e3, iterations, converged, spe, C, rho, F,
  occupations, ediff_history}`; `Mixing::Linear` reproduces imsrg++ (weight 0.2 on the old
  density, `FillLowestOrbits` schedule, convergence `‖Δε‖_F / norb < 1e-8`, unfrozen occupations
  by default) and `Mixing::Diis` is Pulay DIIS on the Fock matrix (error F ρ − ρ F, metric
  weighted by 2j+1, `diis_start` = 2, subspace 8, history reset when occupations change). The
  NO2B monopole uses `No2bThreeBody::get_pn` with the `e3max` cut (imsrg++ `ModelSpace` E3max,
  default 12). Deviations M6-1/M6-2 in `30` §6 (rescue path beyond iteration 100 only).

## 5. mbpt (reference: `reference/04` §2)

* MP2 and MP3 energies (for printing and tests).
* Natural orbitals: MBPT(2) one-body density matrix (pp, hh, ph diagrams, two-level regulator),
  diagonalized per (l, j, tz) channel, with orbit ordering by occupation (default) or energy
  (`NAT_order`). Optional `use_NAT_occupations`: threshold 0.02, back-fill 0.005 (check the
  code). Natural orbitals are built on top of the HF basis (HF first, then NAT), as imsrg++
  does; verify the exact pipeline in `imsrg++.cc`.
* MP4 and Van Vleck are out of the near-term scope.

## 6. commutator (reference: `reference/02`)

* IMSRG(2) scalar terms: comm110, comm111, comm121, comm122, comm220, comm221, comm222_pp_hh,
  comm222_ph (names as in imsrg++). Each has a reference implementation in `commutator_ref`
  and term-by-term tests on random (anti-)Hermitian operators against imsrg-devel golden data.
* IMSRG(3) terms: see `14-imsrg3-plan.md`.
* Term selection uses an enum bitset `TermMask` in `CommutatorOptions` (no string keys) and
  named presets: `imsrg2`, `imsrg3_n7`, `imsrg3_full`. IMSRG(3f2) is **not** a TermMask preset:
  it acts at the BCH level (it reuses the nested commutator from two steps back, `BCH.cc:182-191`)
  and is switched on with `BchOptions::factorized` (`[imsrg3] factorized = true`).

## 7. flow (reference: `reference/03`)

* Methods, phase 1: `magnus` (the default: an Euler step on Omega,
  `Omega ← BCH_Product(η ds, Omega)`, then `H ← BCH_Transform(H0, Omega)`) and `flow_rk4` /
  `flow` (DP45 on H). Later: `magnus_modified_euler`, `magnus_adaptive`, `restore_4th_order`.
* Step size: `ds = min(domega/‖η‖, domega/‖η‖/(‖Ω‖+1e-9), (omega_norm_max+1e-5)/‖η‖, dsmax)`,
  recomputed every step, then capped with `ds = min(ds, smax - s)`; replicate the `domega` bump
  too (`IMSRGSolver.cc:290`, `:335`). `ds_0` is effectively unused in magnus; keep it for flow methods.
* Omega stack: when `‖Ω‖ > omega_norm_max`, push Ω and restart from the transformed H.
  Operators are transformed afterwards by replaying the stack. Optional scratch-to-disk.
* BCH transform: nested commutators with 1/i!, terminating on a threshold epsilon, at most 40
  terms. BCH product: implement exactly the terms imsrg++ includes; its comment lists a
  −1/24 [Y,[X,[X,Y]]] term that the code does **not** implement. That is a truncation choice,
  not a bug, so reproduce it and document it.
* Generators, phase 1: `white`, `atan`, `imaginary-time`, `wegner`, `shell-model`,
  `shell-model-atan`, `shell-model-imaginary-time`, `hartree-fock`, `qtransfer` (exact block
  definitions in `reference/03` §4). Denominators: Epstein-Nesbet (default) and
  Møller-Plesset, plus `denominator_delta` (and its orbit-restricted variant). The denominator
  floor in imsrg++ loses the sign of the denominator. **Owner: this is a bug, so fix it** (keep the
  sign), add it to the deviation log, and test it (QUESTIONS Q7).
* Two-step decoupling: the core generator first, then the valence generator, with a new Omega.
  imsrg++ doubles `smax` for the second Magnus stage; keep that.
* Convergence: `‖η‖ < eta_criterion` or `s ≥ smax`.
* Flow log: CSV with a header (s, E0, ‖η‖, ‖Ω‖, ds, MP2 estimate, wall time) plus the same
  lines to stdout. The integration tests parse the CSV.

## 8. opslib (reference: `reference/05` §1)

Phase 1 (scalar): kinetic energy T (with A-dependence, `A` parameter), intrinsic kinetic energy
`Trel`, `Tcm`, `HCM` (Lawson term `BetaCM`, `hwBetaCM`), `Rp2`, `Rn2`, `Rm2`, `Rso`/Darwin-Foldy
pieces as in imsrg++, `OBD`/`TBD` (later). Later: tensor (E2, M1, GT, Fermi), dagger, M0nu,
DM NREFT. The name grammar and legacy names are in `16-config-and-cli.md`.

**As implemented (M6).** Namespace `imsrg::opslib`, every physics parameter explicit (no hidden
model-space state): `kinetic_energy(ms)` (Tlab), `t_cm(ms, A)`, `t_rel(ms, A)` (T_int used in
H), `r2_cm(ms, A)`, `h_cm(ms, A, hw_trap)`, `r2_one_body/r2_two_body(ms, Species)`,
`rp2(ms, A, Z, include_spin_orbit = true)` (imsrg++ "Rp2" includes `RpSpinOrbitCorrection`,
M6-Q1), `rn2(ms, A, Z)`, `rm2(ms, A)`. Equal to the imsrg-devel dumps within 1.1e-14 (emax 2
and 4); emax 6 builds take 0.4 s (t_rel) and 0.7 s (rp2).

## 9. driver

The pipeline replicates `imsrg++.cc` (see `reference/03` §1):

parse config → build ModelSpace → read 2N (and 3N) → build H = T_int + V (+ V3) → HF (or HO, or
NAT) → normal order → IMSRG (SR or VS; 1 or 2 steps) → transform requested operators → write
outputs (valence-space files, operator files, results summary JSON, flow CSV) → optional
perturbative triples or IMSRG(3)-at-end.

Every branch point is a typed config option. The resolved config is written next to the outputs.
