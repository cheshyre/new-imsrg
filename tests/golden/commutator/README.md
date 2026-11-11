# Golden data: commutators (M7)

Term-by-term IMSRG(2) commutators of the OLD imsrg++ (scalar-scalar, IMSRG(3) off) for O16 at
emax 2, hw 16, N3LO EM500 lambda 1.8 NN interaction. All files are OperatorDump v1 (design/15).
Regenerate with `REFHARNESS=<harness-build>/refharness tools/reference/gen_golden_commutator.sh`
(default `refharness` is the lead's `harness-build`). Total size about 0.7 MB.

## Inputs (`emax2_O16/`)

| file | content | hermiticity |
|---|---|---|
| `H.dump` | copy of `tests/golden/hf/emax2_O16_2N/HNO.dump` (normal-ordered HF Hamiltonian) | H |
| `eta_white.dump` | `refharness generator white H.dump` | A |
| `eta_atan.dump` | `refharness generator atan H.dump` (input only, no commutators taken) | A |
| `random_A.dump` | `refharness dump-hamiltonian ... ops=random_A random_seed=7` | A |
| `random_H.dump` | `refharness dump-hamiltonian ... ops=random_H random_seed=11` | H |

Generators use the IMSRGSolver defaults: Epstein_Nesbet denominators, cutoff 1e-6.
`dump-hamiltonian` common arguments: `emax=2 reference=O16 valence_space=O16 hw=16
2bme=VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz file2e1max=14 file2e2max=28 file2lmax=14`.

## Outputs

`Z_<TERM>__<X>__<Y>.dump` is `[X, Y]` restricted to TERM, for (X,Y) = (eta_white, H) and
(random_A, random_H). Command: `refharness commutator TERM X.dump Y.dump Z.dump`
(`refharness commutator --list` prints the terms). Terms, in the order of
`Commutator::CommutatorScalarScalar`:

`comm110ss comm220ss comm111ss comm121ss comm221ss comm122ss comm222_pp_hhss comm222_phss
comm222_pp_hh_221ss all`

`all` is the full `Commutator::Commutator(X, Y)` with `SetUseIMSRG3(false)`, which calls the
combined `comm222_pp_hh_221ss` (not the separate pp_hh and 221) plus the other terms.
A single-term Z is built as in `CommutatorScalarScalar`: particle rank 2, not reduced,
hermiticity from X and Y, then only that term function accumulates into it.

Hermiticity of every Z: A,A or H,H gives A; H,A or A,H gives H. Here both pairs are (A,H), so
**every Z file has hermiticity H** (header `hermiticity H`), including partial terms.

`norms.json`: imsrg++ norms (`refharness norms`) of every dump, keyed by file stem. One dump,
`Z_comm111ss__eta_white__H`, is deliberately absent: its entries are rounding noise (~1e-16, the
term vanishes for this pair) and it fails the reader's relative hermiticity check (1e-12 * max
entry), so it cannot be read back. The generator script drops any dump that fails that check.

## Consistency checks (computed on the dumps, all matrix elements)

| check | eta_white,H | random_A,random_H |
|---|---|---|
| max abs dev of (110+220+111+121+122+222_pp_hh_221+222_ph) from `all` | 4.4e-16 (max abs Z 16.0) | 5.7e-14 (max abs Z 318.8) |
| ratio to max abs Z | 2.8e-17 | 1.8e-16 |
| max abs dev of `222_pp_hh_221` from `222_pp_hh` + `221` | 0 (max 2.9) | 0 (max 75.4) |

The sum over terms reproduces `all` to within 1e-10 * max|Z| (actual ratio below 2e-16).
