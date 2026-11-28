# 27Al v1 response farm: comparison with the 2024 archive

Lead C, branch `feature/al27-v1-farm`. Variant v1 (`run.compat = "imsrg_fork_79e9330"`, unsplit
Magnus) from the stored v1 Omegas (`$IMSRG_WORK/runs/al27/v1/<H>/`, manifests `<H>_omegas_no.toml`,
`<H>_me2jp_omegas_no.toml`); outputs in `$IMSRG_WORK/runs/al27/v1/<H>/farm-<format>/`. Tables are
written by `tools/al27/compare_farm.py` (archive element vs our element, absolute; rows present in
one file only compared against 0). Acceptance: every archived element to 1e-4 absolute.

## Summary

* **Step 1 (2026-11-26, CPU, job 6006462, 32 nodes, 31 min):** J = 0 and J = 1 operators (FM0,
  FPhipp0, FSigmap1, FSigmapp1, FDelta1; p and n; 71 q) and TCM, both Hamiltonians, native Omegas:
  1,422 files, **all columns PASS; max |diff| 1.0e-9 (N2LOsat), 1.0e-10 (DN2LOGO)**, i.e. the last
  printed digit (7 significant digits). No row missing on either side. TCM: all 158 archived 2b
  elements and the 1b part identical (AL-D3 resolved by the fork tokyo format under compat).
* **v1 native complete (2026-11-27 00:52, CPU, jobs 6006462 + 6006851):** all 34 operators x 71 q
  + TCM for both Hamiltonians (4,830 files, 0 failed). **Every archived element of every column
  matches to <= 1.0e-9 (N2LOsat: 2,111 archived files; DN2LOGO: 2,105), all 31 archived columns
  PASS at 1e-4**; no row missing on either side (tables below).
* **v1 ME2JP complete (2026-11-27 01:38, GPU, job 6007467, 60 nodes, 19.6 min, frontier-hip with
  lead A's device path, one rank per GCD):** 2 x 2,415 files, 0 failed; **every archived element
  PASS at 1e-4, max |diff| 1.0e-7 (N2LOsat), 1.0e-6 (DN2LOGO)**. Native vs ME2JP (ours vs ours,
  all 35 columns incl. FPhitp and the q gaps): max 1e-7 (N2LOsat) / 1e-6 (DN2LOGO, the zero-body
  term of FPhipp0n at q = 45; 1b/2b <= 1e-7 everywhere), per column 1e-9 ... 1e-7 (tables below):
  the float32 Omegas cost at most the 7th printed digit.
* **GPU vs CPU (production check):** the same job reran the native set on GPUs; GPU vs CPU native
  (job 6006851/6006462) over all 4,830 files: max |diff| 6.4e-18 (N2LOsat), 3.5e-18 (DN2LOGO),
  i.e. identical printed digits.
* Archive gaps (counted as "placeholder"): q = 0 of FM2/4, FPhipp4, FSigmap3/5, FSigmapp3/5,
  FDelta3/5 and q = 15 of FDelta5: the fork wrote no file; ours are identically zero there
  (max |me| <= 1e-7, the q^L limit), so nothing is lost. The six **DN2LOGO FM2p q = 975-1050 MeV**
  (archive job J00011500 died) are produced: max |me| 1.40e-4, 1.12e-4, 8.77e-5, 6.72e-5, 5.03e-5,
  3.67e-5, continuing smoothly from the archived q = 945, 960 (2.08e-4, 1.72e-4).
* **FPhitp2/4** (no archive): 284 files per Hamiltonian, max |me| 1.27 (N2LOsat q = 0, FPhitp2).
* CPU time per item (1 rank x 7 threads, emax 10, 8 ranks per node): L0 27 s, L1 ~430 s, L2 ~740 s,
  L3 ~1160 s, L4 ~1370 s, L5 ~1690 s, TCM 989 s (23 GB RSS). The native v1 set costs ~178 node-hours
  on CPU (6006462 16.5 + 6006851 161).

## C14: isospin averaging of scalar operators (one example, N2LOsat, native Omegas)

The 79e9330 writer (and ours, `output.isospin_average_operators = true`, the default in every
variant) replaces each proton-neutron two-body element <ab|O|cd> of a scalar operator by the mean of
itself and its isospin mirror (p and n swapped in all four orbits); NuShellX needs Vpnpn = Vnpnp.
The same items were written once more with `--set output.isospin_average_operators=false`
(`tools/al27/farm_test.sh` with EXTRA_SET; output `runs/al27/v1/N2LOsat/farm-c14/`):

| item | 1b, pp/nn 2b | pn 2b: max abs(averaged - unaveraged) | worst element (averaged / unaveraged) | average(unaveraged) vs averaged |
|---|---|---|---|---|
| FM0p_450 | identical | 2.98e-3 | (0d3/2 p, 1s1/2 n; 0d5/2 p, 1s1/2 n; J=2): 1.79281e-3 / -1.18913e-3 | 7.0e-10 |
| FPhipp0n_450 | identical | 3.33e-4 | (0d3/2 p, 0d3/2 n; 0d3/2 p, 0d5/2 n; J=3): -4.15581e-4 / -8.26609e-5 | 2.5e-10 |

* Only pn two-body elements change; the changes (up to 3e-3, O(100%) of the element for the
  proton-only FM0p) are far above the 1e-4 acceptance, so the archive is reproduced only with
  averaging (it is: step 1). Averaging our unaveraged file (`compare_snt.py
  --isospin-average-test`) reproduces the written file to the printed digits.
* Physics note for the owner: for an operator acting on protons only (FM0p) the IMSRG-evolved pn
  elements are genuinely not mirror symmetric; the average mixes in the neutron-mirror part. This
  is the fork's (imsrg++ "op" mode) convention, kept in all variants by decision.

<!-- compare_farm:Step 1: J = 0/1 and TCM, N2LOsat, native Omegas, CPU (job 6006462) begin -->
## Step 1: J = 0/1 and TCM, N2LOsat, native Omegas, CPU (job 6006462)

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (N2LOsat) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/N2LOsat/farm-native`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-10 | 75 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-10 | 75 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-11 | 105 | 1.00e-12 | 1.00e-11 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-10 | 120 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-09 | 0 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-09 | 15 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-10 | 0 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-10 | 0 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-09 | 0 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-09 | 105 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 0.00e+00 | - | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |

**Summary:** 11/11 columns pass (tol 1e-04); overall max 1.00e-09; files missing 0

Reference TCM: 420 copies; max |diff| of any copy vs the first 1.00e-10 (33 copies not bit-identical in value); compared against the first copy.
<!-- compare_farm:Step 1: J = 0/1 and TCM, N2LOsat, native Omegas, CPU (job 6006462) end -->

<!-- compare_farm:Step 1: J = 0/1 and TCM, DN2LOGO, native Omegas, CPU (job 6006462) begin -->
## Step 1: J = 0/1 and TCM, DN2LOGO, native Omegas, CPU (job 6006462)

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (DN2LOGO) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/DN2LOGO/farm-native`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 6.72e-16 | 0 | 0.00e+00 | 6.72e-16 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-12 | 930 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-14 | 870 | 0.00e+00 | 1.00e-14 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-13 | 885 | 0.00e+00 | 1.00e-13 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-11 | 225 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-11 | 465 | 1.00e-12 | 1.00e-11 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-10 | 135 | 1.00e-10 | 1.00e-11 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-10 | 450 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-10 | 60 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-12 | 165 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 0.00e+00 | - | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |

**Summary:** 11/11 columns pass (tol 1e-04); overall max 1.00e-10; files missing 0

Reference TCM: 419 copies; max |diff| of any copy vs the first 0.00e+00 (0 copies not bit-identical in value); compared against the first copy.
<!-- compare_farm:Step 1: J = 0/1 and TCM, DN2LOGO, native Omegas, CPU (job 6006462) end -->

<!-- compare_farm:v1 native, all operators, N2LOsat (CPU jobs 6006462 + 6006851) begin -->
## v1 native, all operators, N2LOsat (CPU jobs 6006462 + 6006851)

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (N2LOsat) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/N2LOsat/farm-native`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-10 | 75 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-10 | 75 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM2n | 70 | 0 | 1 | 1.00e-09 | 210 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FM2p | 70 | 0 | 1 | 1.00e-09 | 45 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FM4n | 70 | 0 | 1 | 1.00e-10 | 240 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM4p | 70 | 0 | 1 | 1.00e-09 | 255 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-11 | 105 | 1.00e-12 | 1.00e-11 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-10 | 120 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FPhipp2n | 71 | 0 | 0 | 1.00e-10 | 105 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FPhipp2p | 71 | 0 | 0 | 1.00e-09 | 30 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FPhipp4n | 70 | 0 | 1 | 1.00e-09 | 465 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FPhipp4p | 70 | 0 | 1 | 1.00e-09 | 405 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-09 | 0 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-09 | 15 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap3n | 70 | 0 | 1 | 1.00e-10 | 105 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmap3p | 70 | 0 | 1 | 1.00e-09 | 285 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap5n | 70 | 0 | 1 | 1.00e-10 | 450 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmap5p | 70 | 0 | 1 | 1.00e-10 | 135 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-10 | 0 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-10 | 0 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp3n | 70 | 0 | 1 | 1.00e-11 | 210 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FSigmapp3p | 70 | 0 | 1 | 1.00e-09 | 195 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp5n | 70 | 0 | 1 | 1.00e-10 | 285 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp5p | 70 | 0 | 1 | 1.00e-11 | 300 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-09 | 0 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-09 | 105 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FDelta3n | 70 | 0 | 1 | 1.00e-10 | 285 | 1.00e-13 | 1.00e-10 | 0 | 0 | PASS |
| FDelta3p | 70 | 0 | 1 | 1.00e-10 | 30 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FDelta5n | 69 | 0 | 2 | 1.00e-11 | 360 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FDelta5p | 69 | 0 | 2 | 1.00e-10 | 225 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 0.00e+00 | - | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |

**Summary:** 31/31 columns pass (tol 1e-04); overall max 1.00e-09; files missing 0

Reference TCM: 420 copies; max |diff| of any copy vs the first 1.00e-10 (33 copies not bit-identical in value); compared against the first copy.

Placeholders (absent/empty/zero in reference): FM2n: q=0; FM2p: q=0; FM4n: q=0; FM4p: q=0; FPhipp4n: q=0; FPhipp4p: q=0; FSigmap3n: q=0; FSigmap3p: q=0; FSigmap5n: q=0; FSigmap5p: q=0; FSigmapp3n: q=0; FSigmapp3p: q=0; FSigmapp5n: q=0; FSigmapp5p: q=0; FDelta3n: q=0; FDelta3p: q=0; FDelta5n: q=0,15; FDelta5p: q=0,15
<!-- compare_farm:v1 native, all operators, N2LOsat (CPU jobs 6006462 + 6006851) end -->

<!-- compare_farm:v1 native, all operators, DN2LOGO (CPU jobs 6006462 + 6006851) begin -->
## v1 native, all operators, DN2LOGO (CPU jobs 6006462 + 6006851)

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (DN2LOGO) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/DN2LOGO/farm-native`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 6.72e-16 | 0 | 0.00e+00 | 6.72e-16 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-12 | 930 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| FM2n | 70 | 0 | 1 | 1.00e-12 | 810 | 1.00e-12 | 1.00e-12 | 0 | 0 | PASS |
| FM2p | 64 | 0 | 7 | 1.00e-10 | 180 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM4n | 70 | 0 | 1 | 1.00e-10 | 75 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FM4p | 70 | 0 | 1 | 1.00e-09 | 435 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-14 | 870 | 0.00e+00 | 1.00e-14 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-13 | 885 | 0.00e+00 | 1.00e-13 | 0 | 0 | PASS |
| FPhipp2n | 71 | 0 | 0 | 1.00e-10 | 615 | 1.00e-11 | 1.00e-10 | 0 | 0 | PASS |
| FPhipp2p | 71 | 0 | 0 | 1.00e-09 | 15 | 0.00e+00 | 1.00e-09 | 0 | 0 | PASS |
| FPhipp4n | 70 | 0 | 1 | 1.00e-10 | 90 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FPhipp4p | 70 | 0 | 1 | 1.00e-10 | 60 | 1.00e-12 | 1.00e-10 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-11 | 225 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-11 | 465 | 1.00e-12 | 1.00e-11 | 0 | 0 | PASS |
| FSigmap3n | 70 | 0 | 1 | 1.00e-13 | 915 | 0.00e+00 | 1.00e-13 | 0 | 0 | PASS |
| FSigmap3p | 70 | 0 | 1 | 1.00e-10 | 600 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmap5n | 70 | 0 | 1 | 1.00e-12 | 435 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| FSigmap5p | 70 | 0 | 1 | 1.00e-13 | 900 | 0.00e+00 | 1.00e-13 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-10 | 135 | 1.00e-10 | 1.00e-11 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-10 | 450 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FSigmapp3n | 70 | 0 | 1 | 1.00e-11 | 690 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FSigmapp3p | 70 | 0 | 1 | 1.00e-12 | 840 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| FSigmapp5n | 70 | 0 | 1 | 1.00e-13 | 1020 | 0.00e+00 | 1.00e-13 | 0 | 0 | PASS |
| FSigmapp5p | 70 | 0 | 1 | 0.00e+00 | 15 | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-10 | 60 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-12 | 165 | 0.00e+00 | 1.00e-12 | 0 | 0 | PASS |
| FDelta3n | 70 | 0 | 1 | 0.00e+00 | 15 | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |
| FDelta3p | 70 | 0 | 1 | 1.00e-11 | 165 | 0.00e+00 | 1.00e-11 | 0 | 0 | PASS |
| FDelta5n | 69 | 0 | 2 | 1.00e-11 | 330 | 1.00e-13 | 1.00e-11 | 0 | 0 | PASS |
| FDelta5p | 69 | 0 | 2 | 1.00e-10 | 300 | 0.00e+00 | 1.00e-10 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 0.00e+00 | - | 0.00e+00 | 0.00e+00 | 0 | 0 | PASS |

**Summary:** 31/31 columns pass (tol 1e-04); overall max 1.00e-09; files missing 0

Reference TCM: 419 copies; max |diff| of any copy vs the first 0.00e+00 (0 copies not bit-identical in value); compared against the first copy.

Placeholders (absent/empty/zero in reference): FM2n: q=0; FM2p: q=0,975,990,1005,1020,1035,1050; FM4n: q=0; FM4p: q=0; FPhipp4n: q=0; FPhipp4p: q=0; FSigmap3n: q=0; FSigmap3p: q=0; FSigmap5n: q=0; FSigmap5p: q=0; FSigmapp3n: q=0; FSigmapp3p: q=0; FSigmapp5n: q=0; FSigmapp5p: q=0; FDelta3n: q=0; FDelta3p: q=0; FDelta5n: q=0,15; FDelta5p: q=0,15
<!-- compare_farm:v1 native, all operators, DN2LOGO (CPU jobs 6006462 + 6006851) end -->

<!-- compare_farm:v1 ME2JP Omegas (float32), all operators, N2LOsat (GPU job 6007467) vs archive begin -->
## v1 ME2JP Omegas (float32), all operators, N2LOsat (GPU job 6007467) vs archive

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (N2LOsat) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/N2LOsat/farm-me2jp`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-08 | 270 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-08 | 105 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FM2n | 70 | 0 | 1 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FM2p | 70 | 0 | 1 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FM4n | 70 | 0 | 1 | 1.00e-07 | 375 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FM4p | 70 | 0 | 1 | 1.00e-08 | 240 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-08 | 240 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp2n | 71 | 0 | 0 | 1.00e-07 | 30 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp2p | 71 | 0 | 0 | 1.00e-07 | 105 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp4n | 70 | 0 | 1 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp4p | 70 | 0 | 1 | 1.00e-08 | 195 | 1.00e-09 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-08 | 75 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-07 | 0 | 1.00e-07 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap3n | 70 | 0 | 1 | 1.00e-08 | 390 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap3p | 70 | 0 | 1 | 1.00e-08 | 225 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap5n | 70 | 0 | 1 | 1.00e-09 | 225 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap5p | 70 | 0 | 1 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-09 | 60 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp3n | 70 | 0 | 1 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp3p | 70 | 0 | 1 | 1.00e-08 | 540 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp5n | 70 | 0 | 1 | 1.00e-09 | 255 | 1.00e-10 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp5p | 70 | 0 | 1 | 1.00e-09 | 240 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-07 | 75 | 1.00e-07 | 1.00e-09 | 0 | 0 | PASS |
| FDelta3n | 70 | 0 | 1 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FDelta3p | 70 | 0 | 1 | 1.00e-09 | 420 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta5n | 69 | 0 | 2 | 1.00e-09 | 240 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta5p | 69 | 0 | 2 | 1.00e-09 | 285 | 1.00e-10 | 1.00e-09 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 1.00e-08 | - | 0.00e+00 | 1.00e-08 | 0 | 0 | PASS |

**Summary:** 31/31 columns pass (tol 1e-04); overall max 1.00e-07; files missing 0

Reference TCM: 420 copies; max |diff| of any copy vs the first 1.00e-10 (33 copies not bit-identical in value); compared against the first copy.

Placeholders (absent/empty/zero in reference): FM2n: q=0; FM2p: q=0; FM4n: q=0; FM4p: q=0; FPhipp4n: q=0; FPhipp4p: q=0; FSigmap3n: q=0; FSigmap3p: q=0; FSigmap5n: q=0; FSigmap5p: q=0; FSigmapp3n: q=0; FSigmapp3p: q=0; FSigmapp5n: q=0; FSigmapp5p: q=0; FDelta3n: q=0; FDelta3p: q=0; FDelta5n: q=0,15; FDelta5p: q=0,15
<!-- compare_farm:v1 ME2JP Omegas (float32), all operators, N2LOsat (GPU job 6007467) vs archive end -->

<!-- compare_farm:Native vs ME2JP Omegas (ours vs ours), N2LOsat begin -->
## Native vs ME2JP Omegas (ours vs ours), N2LOsat

`/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/N2LOsat/farm-native` vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/N2LOsat/farm-me2jp`.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-08 | 270 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FM0p | 71 | 0 | 0 | 1.00e-08 | 105 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FM2n | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FM2p | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FM4n | 71 | 0 | 0 | 1.00e-07 | 375 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FM4p | 71 | 0 | 0 | 1.00e-08 | 240 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp0n | 71 | 0 | 0 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp0p | 71 | 0 | 0 | 1.00e-08 | 240 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp2n | 71 | 0 | 0 | 1.00e-07 | 30 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhipp2p | 71 | 0 | 0 | 1.00e-07 | 105 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhipp4n | 71 | 0 | 0 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp4p | 71 | 0 | 0 | 1.00e-08 | 195 | 1.00e-09 | 1.00e-08 | 0 | 0 | - |
| FPhitp2n | 71 | 0 | 0 | 1.00e-08 | 0 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhitp2p | 71 | 0 | 0 | 1.00e-07 | 15 | 1.00e-07 | 1.00e-07 | 0 | 0 | - |
| FPhitp4n | 71 | 0 | 0 | 1.00e-08 | 300 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhitp4p | 71 | 0 | 0 | 1.00e-08 | 150 | 1.00e-09 | 1.00e-08 | 0 | 0 | - |
| FSigmap1n | 71 | 0 | 0 | 1.00e-08 | 75 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmap1p | 71 | 0 | 0 | 1.00e-07 | 0 | 1.00e-07 | 1.00e-09 | 0 | 0 | - |
| FSigmap3n | 71 | 0 | 0 | 1.00e-08 | 390 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmap3p | 71 | 0 | 0 | 1.00e-08 | 225 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmap5n | 71 | 0 | 0 | 1.00e-09 | 225 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FSigmap5p | 71 | 0 | 0 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-09 | 60 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp3n | 71 | 0 | 0 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp3p | 71 | 0 | 0 | 1.00e-08 | 540 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp5n | 71 | 0 | 0 | 1.00e-09 | 255 | 1.00e-10 | 1.00e-09 | 0 | 0 | - |
| FSigmapp5p | 71 | 0 | 0 | 1.00e-09 | 240 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta1n | 71 | 0 | 0 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FDelta1p | 71 | 0 | 0 | 1.00e-07 | 75 | 1.00e-07 | 1.00e-09 | 0 | 0 | - |
| FDelta3n | 71 | 0 | 0 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FDelta3p | 71 | 0 | 0 | 1.00e-09 | 420 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta5n | 71 | 0 | 0 | 1.00e-09 | 240 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta5p | 71 | 0 | 0 | 1.00e-09 | 285 | 1.00e-10 | 1.00e-09 | 0 | 0 | - |
| TCM | 1 | 0 | 0 | 1.00e-08 | - | 0.00e+00 | 1.00e-08 | 0 | 0 | - |

**Summary:** 35/35 columns compared; overall max 1.00e-07; files missing 0
<!-- compare_farm:Native vs ME2JP Omegas (ours vs ours), N2LOsat end -->

<!-- compare_farm:v1 ME2JP Omegas (float32), all operators, DN2LOGO (GPU job 6007467) vs archive begin -->
## v1 ME2JP Omegas (float32), all operators, DN2LOGO (GPU job 6007467) vs archive

Archive `/lustre/orion/nph123/scratch/mheinz/imsrg/inputs/al27/al27_provenance/original/al27_data/output_2024_03_01_Al27CorrFull/raw` (DN2LOGO) vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/DN2LOGO/farm-me2jp`. Placeholder = archive file absent/empty/zero.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-07 | 120 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FM0p | 71 | 0 | 0 | 1.00e-07 | 120 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FM2n | 70 | 0 | 1 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FM2p | 64 | 0 | 7 | 1.00e-07 | 315 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FM4n | 70 | 0 | 1 | 1.00e-07 | 375 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FM4p | 70 | 0 | 1 | 1.00e-08 | 285 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp0n | 71 | 0 | 0 | 1.00e-06 | 45 | 1.00e-09 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp0p | 71 | 0 | 0 | 1.00e-07 | 375 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp2n | 71 | 0 | 0 | 1.00e-07 | 15 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp2p | 71 | 0 | 0 | 1.00e-07 | 90 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp4n | 70 | 0 | 1 | 1.00e-07 | 225 | 1.00e-07 | 1.00e-08 | 0 | 0 | PASS |
| FPhipp4p | 70 | 0 | 1 | 1.00e-08 | 195 | 1.00e-09 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap1n | 71 | 0 | 0 | 1.00e-08 | 15 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap1p | 71 | 0 | 0 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap3n | 70 | 0 | 1 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FSigmap3p | 70 | 0 | 1 | 1.00e-08 | 360 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap5n | 70 | 0 | 1 | 1.00e-09 | 210 | 1.00e-11 | 1.00e-09 | 0 | 0 | PASS |
| FSigmap5p | 70 | 0 | 1 | 1.00e-09 | 165 | 1.00e-11 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-08 | 15 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-08 | 405 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FSigmapp3n | 70 | 0 | 1 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp3p | 70 | 0 | 1 | 1.00e-08 | 495 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp5n | 70 | 0 | 1 | 1.00e-07 | 360 | 1.00e-07 | 1.00e-09 | 0 | 0 | PASS |
| FSigmapp5p | 70 | 0 | 1 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta1n | 71 | 0 | 0 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | PASS |
| FDelta1p | 71 | 0 | 0 | 1.00e-09 | 180 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta3n | 70 | 0 | 1 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | PASS |
| FDelta3p | 70 | 0 | 1 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | PASS |
| FDelta5n | 69 | 0 | 2 | 1.00e-09 | 240 | 1.00e-10 | 1.00e-09 | 0 | 0 | PASS |
| FDelta5p | 69 | 0 | 2 | 1.00e-09 | 330 | 1.00e-10 | 1.00e-09 | 0 | 0 | PASS |
| TCM | 1 | 0 | 0 | 1.00e-08 | - | 0.00e+00 | 1.00e-08 | 0 | 0 | PASS |

**Summary:** 31/31 columns pass (tol 1e-04); overall max 1.00e-06; files missing 0

Reference TCM: 419 copies; max |diff| of any copy vs the first 0.00e+00 (0 copies not bit-identical in value); compared against the first copy.

Placeholders (absent/empty/zero in reference): FM2n: q=0; FM2p: q=0,975,990,1005,1020,1035,1050; FM4n: q=0; FM4p: q=0; FPhipp4n: q=0; FPhipp4p: q=0; FSigmap3n: q=0; FSigmap3p: q=0; FSigmap5n: q=0; FSigmap5p: q=0; FSigmapp3n: q=0; FSigmapp3p: q=0; FSigmapp5n: q=0; FSigmapp5p: q=0; FDelta3n: q=0; FDelta3p: q=0; FDelta5n: q=0,15; FDelta5p: q=0,15
<!-- compare_farm:v1 ME2JP Omegas (float32), all operators, DN2LOGO (GPU job 6007467) vs archive end -->

<!-- compare_farm:Native vs ME2JP Omegas (ours vs ours), DN2LOGO begin -->
## Native vs ME2JP Omegas (ours vs ours), DN2LOGO

`/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/DN2LOGO/farm-native` vs `/lustre/orion/nph123/scratch/mheinz/imsrg/runs/al27/v1/DN2LOGO/farm-me2jp`.

| column | q cmp | missing | placeholder | max abs diff | at q | max 1b | max 2b | only ref | only test | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FM0n | 71 | 0 | 0 | 1.00e-07 | 120 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FM0p | 71 | 0 | 0 | 1.00e-07 | 120 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FM2n | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FM2p | 71 | 0 | 0 | 1.00e-07 | 315 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FM4n | 71 | 0 | 0 | 1.00e-07 | 375 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FM4p | 71 | 0 | 0 | 1.00e-08 | 285 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp0n | 71 | 0 | 0 | 1.00e-06 | 45 | 1.00e-09 | 1.00e-08 | 0 | 0 | - |
| FPhipp0p | 71 | 0 | 0 | 1.00e-07 | 375 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FPhipp2n | 71 | 0 | 0 | 1.00e-07 | 15 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhipp2p | 71 | 0 | 0 | 1.00e-07 | 90 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhipp4n | 71 | 0 | 0 | 1.00e-07 | 225 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhipp4p | 71 | 0 | 0 | 1.00e-08 | 195 | 1.00e-09 | 1.00e-08 | 0 | 0 | - |
| FPhitp2n | 71 | 0 | 0 | 1.00e-07 | 75 | 1.00e-07 | 1.00e-08 | 0 | 0 | - |
| FPhitp2p | 71 | 0 | 0 | 1.00e-07 | 45 | 1.00e-08 | 1.00e-07 | 0 | 0 | - |
| FPhitp4n | 71 | 0 | 0 | 1.00e-08 | 105 | 1.00e-09 | 1.00e-08 | 0 | 0 | - |
| FPhitp4p | 71 | 0 | 0 | 1.00e-08 | 120 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmap1n | 71 | 0 | 0 | 1.00e-08 | 15 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmap1p | 71 | 0 | 0 | 1.00e-08 | 180 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmap3n | 71 | 0 | 0 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmap3p | 71 | 0 | 0 | 1.00e-08 | 360 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmap5n | 71 | 0 | 0 | 1.00e-09 | 210 | 1.00e-11 | 1.00e-09 | 0 | 0 | - |
| FSigmap5p | 71 | 0 | 0 | 1.00e-09 | 165 | 1.00e-11 | 1.00e-09 | 0 | 0 | - |
| FSigmapp1n | 71 | 0 | 0 | 1.00e-08 | 15 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp1p | 71 | 0 | 0 | 1.00e-08 | 405 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FSigmapp3n | 71 | 0 | 0 | 1.00e-08 | 135 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp3p | 71 | 0 | 0 | 1.00e-08 | 495 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FSigmapp5n | 71 | 0 | 0 | 1.00e-07 | 360 | 1.00e-07 | 1.00e-09 | 0 | 0 | - |
| FSigmapp5p | 71 | 0 | 0 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta1n | 71 | 0 | 0 | 1.00e-08 | 165 | 1.00e-08 | 1.00e-08 | 0 | 0 | - |
| FDelta1p | 71 | 0 | 0 | 1.00e-09 | 180 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta3n | 71 | 0 | 0 | 1.00e-08 | 210 | 1.00e-08 | 1.00e-09 | 0 | 0 | - |
| FDelta3p | 71 | 0 | 0 | 1.00e-09 | 165 | 1.00e-09 | 1.00e-09 | 0 | 0 | - |
| FDelta5n | 71 | 0 | 0 | 1.00e-09 | 240 | 1.00e-10 | 1.00e-09 | 0 | 0 | - |
| FDelta5p | 71 | 0 | 0 | 1.00e-09 | 330 | 1.00e-10 | 1.00e-09 | 0 | 0 | - |
| TCM | 1 | 0 | 0 | 1.00e-08 | - | 0.00e+00 | 1.00e-08 | 0 | 0 | - |

**Summary:** 35/35 columns compared; overall max 1.00e-06; files missing 0
<!-- compare_farm:Native vs ME2JP Omegas (ours vs ours), DN2LOGO end -->
