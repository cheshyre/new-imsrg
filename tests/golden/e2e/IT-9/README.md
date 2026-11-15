# IT-9: IMSRG(3)-N7 Magnus flow of imsrg++

`imsrg++ ... IMSRG3=true imsrg3_n7=true method=magnus reference=<He4|O16> valence_space=<same>`,
2N only (command and numbers in each `result.json`; flow-file columns in
`tests/golden/e2e/README.md`). Test: `tests/unit/driver/test_imsrg3_n7.cc` (`driver_imsrg3_n7`;
the emax-3 cases are the `slow`-label target `driver_imsrg3_n7_slow`).

| job | E0 (flow end) | steps | final s | provenance |
|---|---|---|---|---|
| `He4_emax2` | -25.233798919 | 250 | 124.40366 | D35 patched build (`tools/reference/gen_golden_e2e.sh IT-9/He4_emax2 IT-9/O16_emax2`, 49 s) |
| `O16_emax2` | -145.489827082 | 147 | 70.12647 | same (25 s) |
| `He4_emax3` | -26.984606380 | 248 | 123.08995 | pristine build (30.8 min run); not regenerated |
| `O16_emax3` | -158.254072732 | 173 | 83.86824 | pristine build (21.1 min run); not regenerated |

The emax-2 jobs were regenerated from the D35 patched build on 2026-11-16; the numbers equal those
of the pristine build to all printed digits (E0, steps, final s, `|Omega|` norms), only the version
string and the timing columns differ. The emax-3 jobs were left as they are (30 and 21 minute runs):
the D35 patch changes T_int/Rp2 only, and the emax-2 jobs show no effect on this flow at the printed
precision.
