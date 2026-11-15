# IMSRG(3) goldens of imsrg++ (milestone M10)

Written only by `tools/reference/gen_golden_imsrg3.sh` (golden rule 7) from the D35 patched
reference harness (`refharness comm223`, `refharness flow --perturbative-triples`).

## Random-operator commutator: `<case>/`

Cases `He4_emax2`, `O16_emax2`, `O16_emax3`. Operators exactly as the layout benchmark makes them:
`X = op::RandomOperator(ms, OperatorSpec::Scalar(2, kAntiHermitian), {seed 1, one_body_scale 1,
two_body_scale 1})`, `Y` the same with `kHermitian` and seed 2 (`tools/bench/imsrg3_layout`,
`--write-dumps`, which writes `X.dump`, `Y.dump` and `epsilon.txt`). Model space: hw 16, valence =
reference, `E3max` as listed, no dE3max / OccNat cut.

| file | content |
|---|---|
| `X.dump`, `Y.dump` | the operators (OperatorDump v1, 1B + 2B); emax 2 cases only. The tests regenerate X and Y in process with the same seeds |
| `epsilon.txt` | single-particle energies of the perturbative-triples mode, one per orbit (`%.17g`) |
| `Z3.dump` | three-body part of the full-mode `comm223ss(X, Y)` of imsrg++ as an OperatorDump v1 with a `three_body` section (`particle_rank 3`, Hermitian; one- and two-body parts are zero). Emax 2 cases, `E3max 3` (the largest E3max below 1 MB; `E3max 4` is 1.6 MB) |
| `values.json` | `three_body_norm` (imsrg++ `ThreeBodyNorm`) and `triples_energy` (`comm223ss` with `perturbative_triples`) for each listed `E3max` (`%.17g` strings), `dump` names `Z3.dump` for the entry it belongs to |

* `Z3.dump` of He4 and O16 differ only in the header (reference name, occupations): `comm223ss`
  does not depend on the reference.
* `triples_energy` is 0 at `E3max 3` (no hhh-ppp kets coupled); use the `E3max 6` / `9` entries.
  It is a parallel sum: it changes in the last digit with the thread count (e.g. O16 emax 2:
  `-64.610022003823076` with 18 threads, `-64.610022003823104` with 8). Compare to 1e-12 relative.
  The 3B norm is bit-reproducible.
* `Z3.dump` is read by `tests/unit/operator/test_golden_imsrg3_dump.cc` (norm vs. `values.json`).

## Perturbative triples of the IT-8 flows: `../e2e/IT-8/<case>/harness_triples.json`

`refharness flow --perturbative-triples` with the settings of the IT-8 jobs of
`gen_golden_e2e.sh` (`method=magnus`, defaults otherwise, same 2N file): `dE`, `E0`, `E0_plus_dE`
(`%.17g`), `steps` and the command. They agree with the imsrg++ log values in `result.json` of the
same directory (8 decimals) and with its step count.
