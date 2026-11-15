# Golden data: factorized double commutator and BCH (M10 track 3f2, D32)

Produced by the OLD imsrg++ built from the D35-patched reference sources (harness build
`new-imsrg-reference/harness-build-3f2`, source sha `...+d35-patch:cb9d11ba30f579d4` in each dump's
`source` line). Regenerate with `tools/reference/gen_golden_factorized.sh` (two runs are
byte-identical). All files are OperatorDump v1 (design/15), `three_body none`. Total size 732 KB.

## Definitions

`Z = d32(Eta, Gamma)`: `refharness factorized d32 ETA.dump GAMMA.dump Z.dump`, i.e.
`FactorizedDoubleCommutator::SetUse_1b_Intermediates(true)`, `SetUse_2b_Intermediates(false)`, then
`comm223_231(Eta, Gamma, Z)` and `comm223_232(Eta, Gamma, Z)`: the factorized part of the double
commutator `[Eta,[Eta,Gamma]_3b]_{1b,2b}` that D32 adds to BCH. Z is a new rank-2 scalar operator,
not reduced, with Gamma's hermiticity (hZ = hGamma). Other parts (`chi1b_231 chi1b_232 chi2b_231
chi2b_232 all direct`) exist in the harness but are only used for the self-checks below.

## `emax2_He6/` (open shell He6, emax 2, hw 16, valence_space He6)

Inputs from `refharness dump-hamiltonian emax=2 reference=He6 valence_space=He6 hw=16
2bme=VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz file2e1max=14 file2e2max=28 file2lmax=14
ops=OPS random_seed=SEED`: `random_A1` (random_A, 41), `random_A2` (random_A, 42), `random_H1`
(random_H, 43). Outputs `Z_d32__random_A1__random_H1.dump`, `Z_d32__random_A1__random_A2.dump`.

## `emax2_O16/`

Inputs are read in place from `tests/golden/commutator/emax2_O16/` (not copied). Outputs
`Z_d32__eta_white__H.dump`, `Z_d32__random_A__random_H.dump`, `Z_d32__eta_white__random_A.dump`.

## `bch_emax2_O16/`

`refharness bch transform|product A.dump B.dump OUT.dump scale_a=SA scale_b=SB [opts]`, A and B from
`tests/golden/commutator/emax2_O16/`, scaled by `Operator * double`. Norms (imsrg++ `Norm()`):
||H|| = 330.92558, ||eta_white|| = 1.0764164, ||random_A|| = 241.13063. "d32" = `--factorized
--factorized-1b-intermediates=1 --factorized-2b-intermediates=0`; "none" = no `--factorized`.
`bch_scaled_norms.txt` holds the exact command and %.17g norms of scaled inputs and result.

| file | op | A (scale) | B (scale) | ||A|| | ||B|| | ||OUT|| |
|---|---|---|---|---|---|---|
| `transform_H__eta_white_x0.0929__d32` | transform | H (1) | eta_white (0.0929) | 330.926 | 0.0999991 | 330.100 |
| `transform_H__eta_white_x1.86__d32` | transform | H (1) | eta_white (1.86) | 330.926 | 2.00213 | 302.657 |
| `transform_H__eta_white_x1.86__none` | transform | H (1) | eta_white (1.86) | 330.926 | 2.00213 | 330.228 |
| `product_eta_white_x0.0464__random_A_x0.000415__d32` | product | eta_white (0.0464) | random_A (0.000415) | 0.0499457 | 0.100069 | 0.108172 |
| `product_eta_white_x0.0929__random_A_x0.00829__d32` | product | eta_white (0.0929) | random_A (0.00829) | 0.0999991 | 1.99897 | 1.99307 |
| `product_eta_white_x0.0929__random_A_x0.00829__none` | product | eta_white (0.0929) | random_A (0.00829) | 0.0999991 | 1.99897 | 1.99325 |

Observation: for the transform with ||Omega|| ~ 2 imsrg++ prints "BCH_Transform not converged after
12 nested commutators" (with the factorized correction on), and the corrected result differs from
the uncorrected by about 8% of the norm; this case is a stress test, not a physical regime.

`norms.json` in each directory: imsrg++ norms of every dump there (read-back check by
`refharness norms` passed for all files).

## Self-checks (computed on temporary dumps, not kept)

Max abs difference over all 1b and 2b elements, relative to max |Z| (imsrg++'s UnitTest criterion
is < 1e-6):

| pair | `all` vs `direct` (max abs / max abs Z / rel) | `d32` - (`chi1b_231` + `chi1b_232`) |
|---|---|---|
| He6 (random_A1, random_H1) | 2.05e-12 / 1590.3 / 1.3e-15 | 0 (exact) |
| He6 (random_A1, random_A2) | 1.36e-12 / 1579.9 / 8.6e-16 | 0 (exact) |
| O16 (eta_white, H) | 7.8e-16 / 0.867 / 9.0e-16 | 0 (exact) |

`direct` is the route of `UnitTest::TestFactorizedDoubleCommutators`: `comm223ss` into a rank-3
pn-mode intermediate (antihermitian iff -hEta*hH < 0, as in UnitTest), then `comm231ss` and
`comm232ss` into Z. `d32` includes only the 1b-intermediate topologies, so it differs from `all`
(which adds the 2b-intermediate topologies).
