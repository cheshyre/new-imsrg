# Golden data from imsrg-devel

Reference outputs of the old code (imsrg++ / imsrg-devel), used by the tests of the rewrite.
Produced by `tools/reference/` (milestone M1); see `design/30-testing-and-validation.md` §3–4
for how the tests use them and §2 for the tolerances.

## Reference version and toolchain

* **Reference source:** `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel` (read-only; the tree has
  no `.git`, so the version is a checksum). Version string used in every golden file:
  `imsrg-devel src-sha256:ddcbf1aff166414e746113d5f3de2ddc1e72783fb7c201829cd58e317717f40f`
  = sha256 over `shasum -a 256` of every file under `src/` (relative paths, sorted, `.DS_Store`
  ignored); recompute with `tools/reference/build_imsrg_devel.sh checksum`.
  The old code reports its own build version as `N/A_N/A` (no git).
* **Build:** `tools/reference/build_imsrg_devel.sh` (2026-11-11): pristine copy of `src/` into
  `/Users/cheshyre/Downloads/new-imsrg-reference/imsrg-devel-src/`, Release (`-O3 -DNDEBUG`,
  plus the old code's `-frounding-math`, C++14), Apple clang 21.0.0, OpenMP through Homebrew
  libomp (`OpenMP_ROOT`), OpenBLAS 0.3.x, GSL 2.8, Boost 1.92 headers, system zlib, macOS SDK
  26.5 (the default SDK 27.0 does not link, see `design/20-build-and-tooling.md` §5), no Python
  bindings. Executables: `new-imsrg-reference/build/imsrg++` and
  `new-imsrg-reference/harness-build/refharness` (`tools/reference/refharness.cpp` + `rh_*.cpp`).
* **Inputs** (never copied into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  2N `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` (`file2e1max=14 file2e2max=28
  file2lmax=14`), 3N `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin`
  (`3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18`, single precision). `hw=16`.
* **Threads:** runs use `OMP_NUM_THREADS=8` unless noted. Results can differ from run to run with a
  different thread count at the level of floating-point summation order (≲1e-13 relative).
* Wall times below are from the dev machine (Apple M-series, 18 cores) and are indicative only.

## Formats

* **OperatorDump v1** (`*.dump`): `design/15-operator-dump-format.md`. The harness writes every
  nonzero one-body element and every nonzero normalized two-body element (both `(pq,rs)` and
  `(rs,pq)`), `three_body none`. `refharness roundtrip IN OUT` rewrites a dump byte-identically
  (`tools/reference/check_roundtrip.sh`).
* **Norms JSON** (`norms.json`, `*_norms`): `zero_body`, `one_body_norm`, `two_body_norm`,
  `three_body_norm`, `norm` = imsrg++ `Operator::{ZeroBody, OneBodyNorm, TwoBodyNorm,
  ThreeBodyNorm, Norm}` (1B weighted by 2j+1, 2B by (2J+1)², zero-body excluded from `norm`).
* **imsrg-basis-dump v1** (`basis/*.txt`), one record per line, single spaces, `%.17g`:

  | Record | Fields |
  |--------|--------|
  | `# imsrg-basis-dump v1` | mandatory first line |
  | `source` | free text (reference version) |
  | `basis` | `emax <e> e2max <e2> e3max <e3> lmax <l> hw <hw>` (`lmax` = min(Lmax, emax)) |
  | `reference` | `<reference arg> valence_space <valence arg or none>` (imsrg++ `reference`, `valence_space`) |
  | `counts` | `Aref Zref Acore Zcore target_mass target_Z` (ModelSpace getters; Acore counts full shells) |
  | `norbits N`, then N × `orbit` | `<index> <n> <l> <2j> <2tz> <occupation> <cvq>` (cvq 0 core, 1 valence, 2 q-space; 2tz = -1 proton) |
  | `holes`, `particles`, `core`, `valence`, `qspace` | `<count> <orbit indices...>` |
  | `nkets N`, then N × `ket` | `<index> <p> <q>`, all kets p ≤ q, index = imsrg++ `Index2(p,q)` |
  | `ntwobody_channels N`, then per channel `channel` | `<ch> <J> <parity> <Tz> <nkets>`, ch = 6J + 2(Tz+1) + parity, Tz = (tz2_p+tz2_q)/2 |
  | `channel_kets` | `<ch> <global ket indices in local order>` |
  | `channel_subset` | `<ch> <pp|hh|ph|cc|vc|qc|vv|qv|qq> <count> <local indices>` (occupation classes pp/hh/ph use OCC_CUT; c/v/q classes use cvq) |
  | `nph_channels N`, then `ph_channel` | `<ch> <J> <parity> <Tz> <nkets>` for the cross-coupled (Pandya) channels; Tz = \|tz2_p − tz2_q\|/2; a CC matrix has dimension 2·nkets (local index + nkets = swapped ket \|qp⟩) |
  | `ph_channel_kets` | `<ch> <global ket indices (p ≤ q)>` |
  | `nthreebody_channels N`, then `threebody_channel` | `<ch> <2J> <parity> <2Tz> <nkets>`: kets \|pqr; Jpq⟩, p ≤ q ≤ r, 2(n_p+n_q+n_r)+l_p+l_q+l_r ≤ e3max, with imsrg++'s Pauli cuts (`ThreeBodyChannel::CheckChannel_ket`) |
  | `nkets3` | `<number of 3B kets> sum_channel_kets3 <sum over channels>` |
  | `end` | |

* **hf.json**: `EHF`, `e1hf`, `e2hf`, `e3hf` (MeV), `iterations`, `spe` (HF single-particle energies,
  orbit order), `occupation`, `orbit_n/l/2j/2tz`, `C` (row = HO orbit, column = HF orbit; imsrg++
  `HartreeFock::C` after its phase convention), `EMP2`, `EMP3_pp/hh/ph`, `EMP3` (imsrg++
  `Operator::GetMP2_Energy` / `GetMP3_Energy` on the HF normal-ordered H), `HNO_norms`.

## Files

### `basis/` (M3) — `tools/reference/gen_golden_basis.sh`

`refharness dump-basis emax=<e> reference=<ref> valence_space=<val> hw=16` (e3max = 12 = imsrg++
default, lmax = 99999 = no cut), < 1 s each:

| File | reference | valence_space | Notes |
|------|-----------|---------------|-------|
| `emax{2,4}_O16.txt` | O16 | O16 | closed shell |
| `emax{2,4}_He4.txt` | He4 | He4 | closed shell |
| `emax{2,4}_Ne20.txt` | Ne20 | Ne20 | open shell: 0d5/2 occupation 1/3 (ensemble filling) |
| `emax{2,4}_sd-shell.txt` | default | sd-shell | core O16, valence sd |
| `emax{2,4}_p-shell.txt` | default | p-shell | core He4, valence p |
| `emax{2,4}_Ne20_sd-shell.txt` | Ne20 | sd-shell | reference ≠ core (IT-7 setup) |
| `emax4_O16_e3max7.txt` | O16 | O16 | `e3max=7`: an active E3max cut on the 3B kets (added in M3) |
| `emax4_Ne20_sd-shell_lmax2_e3max8.txt` | Ne20 | sd-shell | `lmax=2 e3max=8`: truncated orbit indexing (index ≠ hash), open shell, E3max cut (added in M3) |

The e3max 12 files never cut 3B kets (3·emax ≤ 12), hence the two extra cases (same command with
the extra `e3max=`/`lmax=` arguments; the ph channel `Tz` field is the channel-index slot, so the
empty Tz = -1 slots print -1).

### `hamiltonian/emax2_O16/` (M4, M5) — `tools/reference/gen_golden_hamiltonian.sh`

`refharness dump-hamiltonian emax=2 reference=O16 valence_space=O16 hw=16 2bme=<2N file>
file2e1max=14 file2e2max=28 file2lmax=14 3bme=<3N file> 3bme_type=no2b file3e1max=6
file3e2max=12 file3e3max=18 ops=Rp2,random_H random_seed=1` (~1 s). HO basis, reference O16:

| File | Content |
|------|---------|
| `T.dump` | `imsrg_util::Trel_Op` (intrinsic kinetic energy, 1B + 2B, A = 16) |
| `V2.dump` | 2N interaction from the me2j file (`ReadWrite::ReadBareTBME_Darmstadt`) |
| `H.dump` | bare H = T + V2, as `imsrg++` builds it without 3N |
| `V3_NO2B.dump` | 3N alone (no2b storage) normal ordered w.r.t. O16 in the HO basis: its 0B, 1B, 2B parts (`Operator::DoNormalOrdering`) |
| `Rp2.dump` | `imsrg_util::OperatorFromString(ms, "Rp2")` (point-proton radius², HO basis, not normal ordered) |
| `random_H.dump` | `UnitTest::RandomOp(ms, 0, 0, 0, 2, +1)` after `SetRandomSeed(1)` (Hermitian random 1B + 2B) |
| `norms.json` | imsrg++ norms of each of the above |

### `hf/` (M6, MBPT) — `tools/reference/gen_golden_hamiltonian.sh`

`refharness hf emax=<e> reference=O16 valence_space=O16 hw=16 <2N args> [<3N args>]
dump_hno=<e==2>` (HF as imsrg++ with `basis=HF`, occupations unfrozen, plain density mixing),
< 1 s each:

| Directory | Setup | E_HF (MeV) | iterations | EMP2 | EMP3 |
|-----------|-------|-----------:|-----------:|-----:|-----:|
| `emax2_O16_2N` | 2N only | -134.03044657048093 | 20 | -8.0345791646758009 | -0.99474361308108639 |
| `emax2_O16_2N3N` | 2N + NO2B 3N | -206.46003654384998 | 22 | -12.843884653514557 | -3.8779725559665277 |
| `emax4_O16_2N` | 2N only (IT-1) | -147.89357642845789 | 30 | -15.579113783434599 | -0.22908993771133579 |
| `emax4_O16_2N3N` | 2N + NO2B 3N (IT-4 HF part) | -313.16848141064463 | 25 | -16.305529474946081 | -2.5073906145000517 |

Each holds `hf.json`; the emax-2 directories also hold `HNO.dump` (the HF-basis normal-ordered H,
`HartreeFock::GetNormalOrderedH(2)`). Cross-check: `imsrg++ ... emax=4 method=MP3` with the 3N file
prints `EHF = -313.1684814`, `EMP2 = -16.30552947`, `EMP3 = -2.50739061` (identical).
The emax-4 dumps (`hamiltonian`, `HNO`) are regenerated on demand into `generated/` (gitignored)
with `gen_golden_hamiltonian.sh --generated`.

### `commutator/emax2_O16/` (M7) — `tools/reference/gen_golden_commutator.sh`

Single IMSRG(2) commutator terms evaluated by the old code (`refharness commutator TERM X Y Z`)
on (eta_white, H) and (random_A, random_H) at emax 2; inputs, outputs and norms. Terms:
`comm110ss comm220ss comm111ss comm121ss comm221ss comm122ss comm222_pp_hhss comm222_phss
comm222_pp_hh_221ss` and `all` (`Commutator::Commutator`, IMSRG(3) off). Commands, checks and the
one omitted output (comm111ss of eta_white with the HF H, ~1e-8, fails the hermiticity check): see
`commutator/README.md`.

### `e2e/` (integration tests) — `tools/reference/gen_golden_e2e.sh`

Output of the `imsrg++` executable for IT-1 … IT-9 (`result.json`, flow file, `.snt` files,
stdout). Commands, key numbers, wall times and the investigation of the diverging flows
(IT-4/IT-5 with the contact-only 3N file, IT-8 at emax 2) with the replacement goldens
`IT-4/smax10`, `IT-5/smax10`, `IT-8/He4_emax2`: see `e2e/README.md`.

### `e2e/IT-2/trajectory.csv`, `e2e/IT-10/` — `tools/reference/gen_golden_flow.sh`

`refharness flow` (a copy of `IMSRGSolver::Solve_magnus_euler` using the old library, `%.17g`
CSV `step,s,E0,norm_eta,norm_omega`). IT-2 reproduces the executable (256 steps, final s
124.45796230102064, E0 -164.6487723082966, |dE0| = 3e-10 vs `IT-2/result.json`);
`IT-2/trajectory.csv` has steps 0-9 and the last step. IT-10 (IMSRG(3f2), `--factorized`):
`O16_emax2`, `He4_emax2`, `O16_emax3` (full trajectories, `result.json`). See `e2e/README.md`.

### `operator/emax2/` (M4) — `tools/reference/gen_golden_operator.sh`

Normal ordering (`Operator::DoNormalOrdering` / `UndoNormalOrdering`) and the HF-basis
transformation (`HartreeFock::TransformToHFBasis`) of the old code. Commands, contents and
numbers: see `operator/README.md`.

### `io/` (M5, valence writers) — `tools/reference/gen_golden_io.sh`

tokyo `.snt` and NuShellX `.int` / `.sp` files of the old code for the O16 emax-2 HF Hamiltonian
on the sd-shell valence space, plain and isospin-averaged (`op`) mode. See `io/README.md`.

## Missing (not produced in M1)

* IMSRG(3) commutator-term goldens: OperatorDump has no 3B section yet (reserved for M10).
* Tensor operators are not dumped (reduced-ME convention still open, design/15).

**Commutator, emax 1 (M7-T2).** `commutator/emax1_He4/` (closed shell He4) and
`commutator/emax1_He6/` (open shell He6, neutron 0p3/2 occupation 0.5) add the term-by-term
commutators for the (A,H), (H,A), (H,H) and (A,A) pairs at emax 1, plus per-directory
`norms.json` and a sum-versus-`all` check. Pairs, terms, hermiticity, the one dropped He6 dump
and the regeneration command (`REFHARNESS` must point at `harness-build-comm`) are in
`commutator/README.md`, section "emax 1, other hermiticity pairs and an open-shell reference".
Script: `tools/reference/gen_golden_commutator_extra.sh`.
