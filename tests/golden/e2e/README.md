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
| IT-4/smax10 | IT-4 + `smax=10` (use this, not IT-4, for E0) | -313.1684814 | -16.30552947 | 29 / 10.0 | -343.761780688 | 0.6 |
| IT-5/smax10 | IT-5 + `smax=10` (use this, not IT-5, for E0) | -313.0043908 (NAT `Hbare 0b`) | | 28 / 10.0 | -343.644138806 | 0.7 |
| IT-6 | IT-2 + `Operators=Rp2` | -147.8935764 | -15.57911378 | 256 / 124.45796 | -164.648772308 | 2.9 |
| IT-7 | `reference=Ne20 valence_space=sd-shell method=magnus` | -170.4417504 | -13.82250927 | 431 (2 stages) / 188.21746 | -190.532734189 | 5.5 |
| IT-8/emax2 | `emax=2 reference=O16 valence_space=O16 method=magnus perturbative_triples=true` | -134.0304466 | -8.03457916 | 536 / 72.59151 (diverged) | -1.028e9 | 1.6 |
| IT-8/He4_emax2 | `emax=2 reference=He4 valence_space=He4 method=magnus perturbative_triples=true` (use this, not IT-8/emax2) | -22.9897017 | | 192 / 95.39922 | -25.169841888; dE_triples -0.05209710, E0+dE -25.22193898 | 0.3 |
| IT-8/emax3 | same, `emax=3` | -147.4333326 | -9.11096312 | 154 / 74.3663 | -157.847764763 | 0.6 |
| IT-9/He4_emax2 | `emax=2 IMSRG3=true imsrg3_n7=true method=magnus reference=He4 valence_space=He4` | -22.9897017 | -1.76404391 | 250 / 124.40366 | -25.233798919 | 46.5 |
| IT-9/O16_emax2 | same with `reference=O16 valence_space=O16` | -134.0304466 | -8.03457916 | 147 / 70.12647 | -145.489827082 | 23.8 |
| IT-9/He4_emax3 | `emax=3`, He4 | -22.9897017 | -3.48332643 | 248 / 123.08995 | -26.984606380 | 1850.2 (30.8 min) |
| IT-9/O16_emax3 | `emax=3`, O16 | -147.4333326 | -9.11096312 | 173 / 83.86824 | -158.254072732 | 1268.9 (21.1 min) |

All emax=3 runs finished in under 60 min and are kept; nothing was skipped. The rows IT-4/smax10,
IT-5/smax10 and IT-8/He4_emax2 were added by the M1 lead after the divergence investigation below.

## Diverging flows: investigation and what to compare (M1-Q1)

**IT-4/IT-5 (2N + NO2B 3N).** The 3N input is a *contact-only (cE) component file*, not a full
3N force, so its large attractive contribution (e3hf = -201.9 MeV at emax 4, -72 MeV at emax 2)
and the diverging flow are properties of the Hamiltonian, not of the reader:
* `tools/reference/no2b_isospin_scan.py` replays the imsrg++ NO2B reader loops over the file:
  the 91 800 elements with e1 <= 2 (exactly the count imsrg++ allocates at emax 2) have 2T = 1
  elements up to 7.0 MeV (rms 0.43) but **all 2T = 3 (T = 3/2) elements vanish** (max 2e-7,
  float noise), and no Pauli-forbidden element is nonzero. A regulated 3N contact term has a
  totally symmetric spatial kernel and therefore no T = 3/2 matrix elements; the pi-exchange
  (c1, c3, c4) and cD terms do. The name `N2LO_cE_NonLocal4_394` (cE term, nonlocal regulator
  n = 4, Lambda = 394 MeV = 2.0 fm^-1) agrees. The LEC value used for the file is unknown (a unit
  LEC is plausible: the 1.8/2.0(EM) value cE = -0.120 would turn -202 MeV into about +24 MeV).
* The reader layout matches the file (element count, zero Pauli violations); imsrg++ applies no
  LEC or scale factor to `3bme_type=no2b` input (`ReadWrite::SetLECs_preset` is not used by the
  no2b storage); `no2b_precision` is `single` by default. refharness and imsrg++ agree to all
  printed digits (E_HF = -313.1684814, EMP2 = -16.30552947).
* The flow approaches E0 ~ -354 MeV (s ~ 21) and then runs away (s ~ 25 on).
* **Compare:** HF numbers (`tests/golden/hf/emax4_O16_2N3N/hf.json`, %.17g) and the flow stopped
  at s = 10 (`IT-4/smax10`, `IT-5/smax10`: E0(s=10) and the whole 29/28-step trajectory). The full
  IT-4/IT-5 runs are kept only as a record of the divergence.

**IT-8/emax2 (O16, emax 2, 2N only).** A genuine IMSRG(2) instability of O16 in this tiny space,
not an option of the triples run: `perturbative_triples` only adds a post-processing step, and a
plain IMSRG(2) Magnus run with exactly the IT-2 settings at emax 2 diverges identically (537 flow
lines). It also diverges with `core_generator=white` and with `denominator_partitioning=MP`, while
He4 at emax 2 (192 steps, E0 = -25.169841888), O16 at emax 3 (IT-8/emax3) and the O16 emax-2
IMSRG(3)-N7 flow (IT-9/O16_emax2) converge. The emax-2 flow reaches E0 ~ -150.6 MeV with
|Eta_2| down to 0.074 near s ~ 46 and then runs away; the mechanism (e.g. a closing
Epstein-Nesbet denominator in the sd-only particle space) was not pinned down. **Compare:**
IT-8/He4_emax2 (E0 and dE_triples at emax 2) and IT-8/emax3 (O16).

## Notes per job

* **IT-1.** Acceptance check: E_HF matches `tests/golden/hf/hf.json` (-147.89357642845789) to the printed
  8 decimals (stdout prints 7 decimals for EHF; the difference is printing precision only, < 1e-7).
  MBPT block (printed twice by the old code) in `result.json["mbpt"]`.  No flow file for MP3.
* **IT-2.** 6 Omegas pushed; `Core Energy = -164.648772` equals the flow's final E0.  Final
  `|Eta2|` etc. in `flow.last`.
* **IT-3.** Two flow stages (first the standard IMSRG(2) to the O16 core, then the valence decoupling);
  `int.snt` has Zero body term -164.648772 (the same core energy as IT-2).  30 Omegas pushed.
* **IT-4/IT-5.** The cE-only 3N file gives e1hf 469.8, e2hf -581.1, e3hf -201.9; the full flow
  diverges (E0 ~ -1e9 at s ~ 38). See "Diverging flows" above; compare with IT-4/smax10, IT-5/smax10.  IT-5
  (NAT): `Hbare_0b` differs from the HF value (-313.00439075 vs -313.16848141), as expected after the
  basis change.
* **IT-6.** `intRp2.snt` (zero body 4.45016791 transformed; HF value of Rp2 4.294907).  The
  single-reference run writes only the operator `.snt`, not the Hamiltonian.
* **IT-7.** `int.snt` zero body term -161.03272 (Ne20 reference, sd-shell valence space).
* **IT-8.** emax=2 diverges (see "Diverging flows"; use IT-8/He4_emax2 instead); its `Perturbative triples: 0.00000000 -> -inf` line is
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

## Trajectories and IT-10 (refharness flow)

`bash tools/reference/gen_golden_flow.sh [refharness]`, `OMP_NUM_THREADS=8`. `refharness flow` runs the
imsrg++ pipeline (me2j 14/28/14, hw=16, HF, normal ordering, Magnus with the executable's default
step control, atan generator) with a verbatim copy of `Solve_magnus_euler` and prints every step as
`step,s,E0,norm_eta,norm_omega` (`%.17g`). Options: `emax reference valence_space generator smax
--imsrg3 --n7 --factorized`.

* **IT-2 check:** `refharness flow emax=4 reference=O16 valence_space=O16 hw=16`: 256 steps, s =
  124.45796230102064, E0 = -164.6487723082966 vs golden -164.648772308: |dE0| = 3e-10 (<= 1e-8).
  Wall 6.3 s (8 threads, includes file read). `IT-2/trajectory.csv` = steps 0-9 + step 256;
  `IT-2/refharness_result.json` is the harness result.
* **IT-10** (IMSRG(3f2) = IMSRG(2) + `BCH::SetUseFactorizedCorrection(true)`; no executable switch):

| case | E_HF | E0 | steps / s_final | wall (s) |
|---|---|---|---|---|
| IT-10/O16_emax2 | -134.03044657 | -147.389459022 | 407 / 200 (smax reached) | 3.8 |
| IT-10/He4_emax2 | -22.98970175 | -25.153996210 | 193 / 95.90493 (|eta| < 1e-6) | 2.1 |
| IT-10/O16_emax3 | -147.43333256 | -157.777153257 | 152 / 73.38877 (|eta| < 1e-6) | 5.0 |

O16 emax 2 does NOT diverge with 3f2 (unlike 2N-only IMSRG(2)); it needs the full smax = 200 and
ends with |eta| = 2.5e-5, i.e. slow tail convergence. He4 emax 2 is kept as an extra case.
