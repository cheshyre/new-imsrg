# End-to-end golden data (old code)

Output of the OLD `imsrg++` executable (imsrg-devel, `REFERENCE_SRC_SHA256` recorded in every
`result.json`) for the integration tests IT-1..IT-9 of `design/30-testing-and-validation.md` section 4.
Regenerate with `bash tools/reference/gen_golden_e2e.sh [IT-n | IT-n/sub ...]` (all jobs if none given;
run directories in `$IMSRG_REFERENCE_ROOT/runs/e2e/<job>/`, input files from `$IMSRG_TEST_DATA_DIR`,
`OMP_NUM_THREADS=8`, wall-time cap `IT_TIMEOUT_S`, default 3600).  Input files are never copied into the repo.

Per job directory: `result.json` (parsed by `tools/reference/parse_imsrg_log.py`; numbers are the values as
printed, `*_str` keys keep the text), `flow.dat` (the flow file), `*.snt` (decoupled operator files),
`stdout.log` (if < 200 kB).  `wall_time_s` is the only timing kept; it is machine/load dependent.
Flow-file columns: `istep s E0 |H| err |Om1| |Om2| |Om3| |Eta1| |Eta2| |Eta3 | Ncomm E_MP2 N_Ops`, then
time and memory; the columns touch for large values, so split by number format, not whitespace.

Common arguments C: `2bme=<VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz> file2e1max=14 file2e2max=28
file2lmax=14 hw=16`; N: `3bme=<NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin> 3bme_type=no2b
file3e1max=6 file3e2max=12 file3e3max=18`; every command also has `flowfile=<rundir>/flow.dat intfile=<rundir>/int`
(exact line in `result.json["command"]`).

| Job | Job-specific arguments | E_HF | E_MP2 | steps / final s | final E0 (flow) | wall s |
|---|---|---|---|---|---|---|
| IT-1 | `emax=4 reference=O16 valence_space=O16 method=MP3` | -147.8935764 | -15.57911378 | no flow | EMP3 -0.22908994 (E3 term) | 0.08 |
| IT-2 | same, `method=magnus` | -147.8935764 | -15.57911378 | 256 / 124.45796 | -164.648772308 | 2.7 |
| IT-3 | IT-2 with `valence_space=sd-shell` | -147.8935764 | -15.57911378 | 421 (2 stages) / 178.79052 | -164.648772308 | 4.6 |
| IT-4 | IT-2 + N | -313.1684814 | -16.30552947 | 436 / 37.93571 (diverged) | -1.0079e9 | 5.6 |
| IT-5 | IT-4 + `basis=NAT` | -313.1684814 | -16.98517513 | 437 / 38.85789 (diverged) | -1.0453e9 | 5.7 |
| IT-6 | IT-2 + `Operators=Rp2` | -147.8935764 | -15.57911378 | 256 / 124.45796 | -164.648772308 | 2.9 |
| IT-7 | `reference=Ne20 valence_space=sd-shell method=magnus` | -170.4417504 | -13.82250927 | 431 (2 stages) / 188.21746 | -190.532734189 | 5.5 |
| IT-8/emax2 | `emax=2 reference=O16 valence_space=O16 method=magnus perturbative_triples=true` | -134.0304466 | -8.03457916 | 536 / 72.59151 (diverged) | -1.028e9 | 1.6 |
| IT-8/emax3 | same, `emax=3` | -147.4333326 | -9.11096312 | 154 / 74.3663 | -157.847764763 | 0.6 |
| IT-9/He4_emax2 | `emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4` | -22.9897017 | -1.76404391 | 250 / 124.40366 | -25.233798919 | 46.5 |
| IT-9/O16_emax2 | same with `reference=O16 valence_space=O16` | -134.0304466 | -8.03457916 | 147 / 70.12647 | -145.489827082 | 23.8 |
| IT-9/He4_emax3 | `emax=3`, He4 | -22.9897017 | -3.48332643 | 248 / 123.08995 | -26.984606380 | 1850.2 (30.8 min) |
| IT-9/O16_emax3 | `emax=3`, O16 | -147.4333326 | -9.11096312 | 173 / 83.86824 | -158.254072732 | 1268.9 (21.1 min) |

All emax=3 runs finished in under 60 min and are kept; nothing was skipped.

## Notes per job

* **IT-1.** Acceptance check: E_HF matches `tests/golden/hf/hf.json` (-147.89357642845789) to the printed
  8 decimals (stdout prints 7 decimals for EHF; the difference is printing precision only, < 1e-7).
  MBPT block (printed twice by the old code) in `result.json["mbpt"]`.  No flow file for MP3.
* **IT-2.** 6 Omegas pushed; `Core Energy = -164.648772` equals the flow's final E0.  Final
  `|Eta2|` etc. in `flow.last`.
* **IT-3.** Two flow stages (first the standard IMSRG(2) to the O16 core, then the valence decoupling);
  `int.snt` has Zero body term -164.648772 (the same core energy as IT-2).  30 Omegas pushed.
* **IT-4/IT-5.** The 3N file with `3bme_type=no2b` gives e1hf 469.8, e2hf -581.1, e3hf -201.9 and
  an unphysical E_HF; the flow diverges (E0 ~ -1e9, |H| ~ 1e9 at s ~ 38).  Recorded as-is, see
  design/QUESTIONS.md Q1.  Use only HF numbers and the early flow steps for comparisons.  IT-5
  (NAT): `Hbare_0b` differs from the HF value (-313.00439075 vs -313.16848141), as expected after the
  basis change.
* **IT-6.** `intRp2.snt` (zero body 4.45016791 transformed; HF value of Rp2 4.294907).  The
  single-reference run writes only the operator `.snt`, not the Hamiltonian.
* **IT-7.** `int.snt` zero body term -161.03272 (Ne20 reference, sd-shell valence space).
* **IT-8.** emax=2 diverges like IT-4/IT-5; its `Perturbative triples: 0.00000000 -> -inf` line is
  recorded (`E0_plus_dE: null` because -inf is not a JSON number; `E0_plus_dE_str` keeps it).  emax=3
  converges: dE_triples -0.53867396, E0+dE -158.38643873.
* **IT-9.** `Norm of 3-body` values in `norm_of_3body_str`.  `imsrg3_n7` flow stages and Omega
  pushes in `result.json`.

## Old-code oddities

* The old code spells "perterbative" in a log line, prints the MBPT block twice, and prints EHF with 7
  decimals and Core Energy with 6.
* Flow-file columns run together for large values (a plain `split()` fails); parser uses number formats.
* The single-reference run only writes the operator file `intRp2.snt`, no Hamiltonian file.
* `perturbative_triples=true` after a diverged flow prints `-inf` rather than failing.
