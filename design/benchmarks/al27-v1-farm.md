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
* CPU time per item (1 rank x 7 threads, emax 10): L = 0 27 s, L = 1 373 s, TCM 989 s (23 GB RSS).

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
