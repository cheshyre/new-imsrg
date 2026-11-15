# IMSRG(3)-N7 goldens of imsrg++ (milestone M10 step 3)

Written only by `tools/reference/gen_golden_n7.sh` (golden rule 7, D42) from the D35 patched
reference harness (`refharness comm3`, `refharness generator3`; version string in
`values.json: reference_version`). They validate the N7 terms of the new code
(design/14-imsrg3-plan.md section 4 item 2) against imsrg++.

## Inputs

Cases `He4_emax2` and `O16_emax2` (hw 16, valence space = reference, `E3max` 3 and 6, no dE3max /
OccNat cut). The operators are `testing::RandomThreeBodyOperator(ThreeBodyContext(
MakeThreeBodySpace(emax, reference, {e3max})), h, seed)` of `tests/support/three_body_test_support.h`
(one- and two-body scale 1, PHYSICAL three-body part), written by `tools/bench/n7_inputs.cc`
(`imsrg_n7_inputs --emax E --reference R --e3max M --seed S --herm A|H|N --out F`):

| operator | hermiticity | seed |
|---|---|---|
| `X` | anti-Hermitian | 11 |
| `Y` | Hermitian | 12 |
| `H` | Hermitian | 13 (generator only) |

The dumps of X, Y and H are NOT committed: the tests regenerate them in process from the seeds.
`values.json` holds their imsrg++ norms (`inputs`) as a drift check (a changed
`RandomThreeBodyOperator` shows up there first).

## Files of `<case>/`

| file | content |
|---|---|
| `comm330ss.dump`, `comm331ss.dump`, `comm231ss.dump`, `comm132ss.dump`, `comm232ss.dump`, `comm133ss.dump`, `comm223ss.dump` | `Z = [X, Y]` of that single imsrg++ term (`Commutator::<term>(X, Y, Z)` on a fresh Z), OperatorDump v1 with a `three_body` section; `E3max 3` |
| `n7.dump` | full `Commutator::Commutator(X, Y)` after `SetUseIMSRG3(true)`, `SetUseIMSRG3N7(true)`: the IMSRG(2) terms plus the seven N7 terms (comm133, 330, 223, 232, 331, 231, 132 in imsrg++'s call order); `E3max 3` |
| `eta_white.dump` | imsrg++ `Generator` (`white`, single reference, Epstein-Nesbet, defaults) applied to the rank-3 `H`, `E3max 3` (`ConstructGenerator_SingleRef_3body` runs) |
| `eta_white_e3max6.dump` | the same at `E3max 6` (kept because it is small: only hhh-ppp elements) |
| `values.json` | per `E3max` (3, 6): the norms of `X`, `Y`, `H` and, for each term, `n7` and the generator: `zero_body`, `one_body_norm`, `two_body_norm`, `three_body_norm` (imsrg++ `OneBodyNorm`, `TwoBodyNorm`, `ThreeBodyNorm`) as numbers and `%.17g` strings (`*_str`), and `dump` naming the file (null: norms only) |

`Z` is created exactly as in `Commutator::CommutatorScalarScalar` with IMSRG(3) on: particle rank 3,
Hermitian (anti-Hermitian X times Hermitian Y), not reduced, `ThreeBody.SetMode("pn")`. The 3B
section lists every nonzero element of every channel block, both triangles (design/15). Every
file is below 1 MB (largest 461 KB).

## Things to know

* At `E3max 3` the hhh-ppp couplings do not exist (an even-parity ppp ket needs `e1 + e2 + e3 >= 4` for He4 and `>= 6` for O16),
  so `comm330ss` (0B), `comm331ss`, `comm231ss` (1B) of He4 and O16, and the 3B part of `eta_white`
  are zero in the `E3max 3` dumps. Use the `E3max 6` rows of `values.json` for them. `O16_emax2` at
  `E3max 3` has 1B parts only from `n7` (through the IMSRG(2) terms).
* `n7`'s 3B norm is not the quadrature sum of `comm133ss` and `comm223ss`: the 3B parts overlap.
* The terms are parallel sums. Results change in the last digits from run to run and with the
  thread count (e.g. `n7`, He4, `E3max 3`, 2B element (1, 1): `3.4421068778002937` and
  `3.4421068778002946`). Compare to 1e-12 relative.
* `comm331ss` and `comm231ss` die in imsrg++ ("Updating SixJList inside a parallel loop") unless
  the 6j table is filled first; the harness calls `ModelSpace::PreCalculateSixJ()` before every
  term (`comm223ss` fills it itself).
* `refharness roundtrip` reproduces every 3B dump byte for byte (the harness reader stores the
  records through `ThreeBodyME::SetME_pn_ch`; records of the unstored triangle are checked against
  the mirrored value).

## Regenerate

```
cmake --preset release-openmp -B build/golden-n7 -DIMSRG_BUILD_BENCH=ON
HARNESS_BUILD_DIR_D35=<dir> tools/reference/build_imsrg_devel.sh patched-harness   # once
HARNESS_BUILD_DIR_D35=<dir> tools/reference/gen_golden_n7.sh
```

(`BENCH_BUILD_DIR` selects another build directory of the bench targets; the script configures and
builds `imsrg_n7_inputs` on demand.)
