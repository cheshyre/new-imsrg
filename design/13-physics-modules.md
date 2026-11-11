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
  E3max defaults to `min(14, 3*emax)` in imsrg++; we keep that default but make it explicit in
  the config.
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
* Norms (D12): 1B part weighted by (2j+1), 2B part by (2J+1)². These norms drive step sizes and
  BCH thresholds, so they must match imsrg++ exactly.
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
   needed for the calculation's emax. Values arrive in isospin form, 4 floats per J line
   (`T0 pn, nn, T1 pn, pp`); convert to pn exactly as `Set_pn_TBME_from_iso` does.
2. **NO2B** (`*.stream.bin`, float, or half with `no2b_precision=half`). The test file is
   `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin` with `file3e1max=6
   file3e2max=12 file3e3max=18`. imsrg++ detects the type from the file name; we take it from
   config (`three_body.format = "no2b"`) and only warn on a name mismatch.
3. **Tokyo `.snt`** and **NuShellX `.int/.sps`** writers for valence-space interactions and
   operators. They must match imsrg++ output **byte-for-byte modulo floating-point formatting**;
   compare numerically, line by line.
4. **OperatorDump**: our own simple, documented text format (header with emax, orbit list,
   spec; then 0B, 1B, 2B blocks), used for golden data and debugging. imsrg-devel gets a small
   companion dumper (in `tools/reference/`, built against the old library) that writes the same
   format.
5. Later: full me3j (isospin), Navratil/Miyagi operator formats, binary scratch and checkpoint
   files (with a magic number, version and endianness tag; imsrg++ has none).

## 4. hf (reference: `reference/04` §1)

* Algorithm to match: Fock matrix from the monopole of V (plus the 3N monopole from NO2B),
  per-channel `eigh`, a sign fix making the diagonal of C positive, density mixing with weight
  0.8 on the new density (check this in the code; record the exact scheme), convergence on
  `‖Δε‖_F / norb < 1e-8`, at most 1000 iterations. DIIS exists in imsrg++ but has off-by-one
  and dead-code issues; implement it later, optionally.
* Output: `HFResult {E_HF, spe, C, occupations, iterations, converged}` plus the
  normal-ordered H in the HF basis (`D^T (V2 + V3NO) D` for the 2B part; 3B → NO2B).
* Known reference bugs: the dE3max cut uses `op->l` for all three orbits (HartreeFock.cc:1368);
  the SCF uses `std::rand()`. Fix both and document.

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
  named presets: `imsrg2`, `imsrg3_n7`, `imsrg3_full`, `imsrg3f2`.

## 7. flow (reference: `reference/03`)

* Methods, phase 1: `magnus` (the default: an Euler step on Omega,
  `Omega ← BCH_Product(η ds, Omega)`, then `H ← BCH_Transform(H0, Omega)`) and `flow_rk4` /
  `flow` (DP45 on H). Later: `magnus_modified_euler`, `magnus_adaptive`, `restore_4th_order`.
* Step size: `ds = min(domega/‖η‖, domega/‖η‖/(‖Ω‖+1e-9), (omega_norm_max+1e-5)/‖η‖, dsmax)`,
  recomputed every step. `ds_0` is effectively unused in magnus; keep it for flow methods.
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

## 9. driver

The pipeline replicates `imsrg++.cc` (see `reference/03` §1):

parse config → build ModelSpace → read 2N (and 3N) → build H = T_int + V (+ V3) → HF (or HO, or
NAT) → normal order → IMSRG (SR or VS; 1 or 2 steps) → transform requested operators → write
outputs (valence-space files, operator files, results summary JSON, flow CSV) → optional
perturbative triples or IMSRG(3)-at-end.

Every branch point is a typed config option. The resolved config is written next to the outputs.
