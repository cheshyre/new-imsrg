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
