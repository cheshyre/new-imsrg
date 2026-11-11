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
`Z_comm111ss__eta_white__H`, is deliberately absent: its entries are small (~1e-8: the one-body
parts of eta_white and the HF H nearly commute; M7 measured this, design/QUESTIONS.md M7-Q1) and
it fails the reader's relative hermiticity check (1e-12 * max entry), so it cannot be read back.
The tests check this term against `Z_all` minus the other terms. The generator script drops any dump that fails that check.

## Consistency checks (computed on the dumps, all matrix elements)

| check | eta_white,H | random_A,random_H |
|---|---|---|
| max abs dev of (110+220+111+121+122+222_pp_hh_221+222_ph) from `all` | 4.4e-16 (max abs Z 16.0) | 5.7e-14 (max abs Z 318.8) |
| ratio to max abs Z | 2.8e-17 | 1.8e-16 |
| max abs dev of `222_pp_hh_221` from `222_pp_hh` + `221` | 0 (max 2.9) | 0 (max 75.4) |

The sum over terms reproduces `all` to within 1e-10 * max|Z| (actual ratio below 2e-16).

## emax 1, other hermiticity pairs and an open-shell reference (M7-T2)

`tools/reference/gen_golden_commutator_extra.sh` adds term-by-term goldens at emax 1, hw 16,
same N3LO EM500 lambda 1.8 two-body file, `file2e1max=14 file2e2max=28 file2lmax=14`. They cover
the pairs the emax-2 O16 set lacks: (H,A), (H,H) and (A,A), and an open-shell reference. Dumps
are small at emax 1, so they are committed.

Regenerate (the default `refharness` in `harness-build` lacks the `commutator` subcommand; use
the build that has it):

    REFHARNESS=/Users/cheshyre/Downloads/new-imsrg-reference/harness-build-comm/refharness \
        tools/reference/gen_golden_commutator_extra.sh

### `emax1_He4/` (closed shell, reference He4, valence He4)

Inputs from `dump-hamiltonian emax=1 reference=He4 valence_space=He4 hw=16 ... ops=random_A
random_seed=21` (A1), `random_seed=22` (A2), `ops=random_H random_seed=23` (H1) and `24` (H2).

Pairs (X,Y): (A1,H1), (H1,A1), (H1,H2), (A1,A2). Each pair has 8 Z files,
`Z_<TERM>__<X>__<Y>.dump`, for the terms `comm110ss comm220ss comm111ss comm121ss comm221ss
comm122ss comm222_pp_hhss comm222_phss`. That is 32 Z files. The split 222 terms replace the
combined `comm222_pp_hh_221ss` of `all`, so the consistency check is a sum over the split terms.

Hermiticity: A,A or H,H gives A; A,H or H,A gives H. So (A1,A2) and (H1,H2) give A, and
(A1,H1) and (H1,A1) give H. No dump was dropped. Size 152 kB: 4 inputs, 32 Z files and
`norms.json` (36 entries).

### `emax1_He6/` (open shell, reference He6, valence He6)

Inputs from `dump-hamiltonian emax=1 reference=He6 valence_space=He6 hw=16 ...`: `random_A1`
(seed 31), `random_H1` (33), `random_H2` (34). In every dump, neutron orbit 0p3/2 (n 0, l 1, 2j 3,
2tz +1) has occupation 0.5, so this is a fractional-occupation (ensemble-filled) open shell.

Pairs: (A1,H1), (H1,H2). Terms as above. Dropped: `Z_comm121ss__random_H1__random_H2.dump`,
because the reader's hermiticity check failed ("hermiticity check failed: max deviation
1.6003776271653047e-15", line 23): its entries are rounding noise and the term is not kept.
Remaining: 15 Z files, 18 norm entries in `norms.json`. Size 76 kB.

Hermiticity: (A1,H1) gives H; (H1,H2) gives A.

### Totals and checks

Both directories together: 228 kB, far below the 600 kB limit. The largest dump is about 4 kB.

Consistency check (scratch script, not committed): for each pair, the sum of the eight term dumps
compared with the `all` dump, all matrix elements, pass criterion 1e-10 * max|Z_all|:

| directory | pair (X,Y) | max abs dev | max abs Z_all | ratio |
|---|---|---|---|---|
| emax1_He4 | A1,H1 | 3.553e-15 | 35.01 | 1.0e-16 |
| emax1_He4 | H1,A1 | 3.553e-15 | 35.01 | 1.0e-16 |
| emax1_He4 | H1,H2 | 3.553e-15 | 29.95 | 1.2e-16 |
| emax1_He4 | A1,A2 | 0 | 25.63 | 0 |
| emax1_He6 | A1,H1 | 8.882e-16 | 22.85 | 3.9e-17 |
| emax1_He6 | H1,H2 | 1.776e-15 | 26.06 | 6.8e-17 |

All six pass. The He6 (H1,H2) sum uses the regenerated `comm121ss` term, not the dropped file.
