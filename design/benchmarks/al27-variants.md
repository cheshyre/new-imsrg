# 27Al variants v2 and v3: flows, farms and differences

Lead D, weekend run 2026-11-27/30 (design/50 section 7, D76). Variants: **v1** = fork
reproduction (`run.compat = "imsrg_fork_79e9330"`, unsplit Magnus, domega 0.2); **v2** = default
behaviour (no fork emulation: devel tensor normal ordering and odd-L phases, D72 tensor norm,
domega 0.1, devel writer), unsplit Magnus; **v3** = v2 with split Magnus, `omega_norm_max = 0.25`.
Flows: one MI250X GCD each (`tools/al27/flows_chain.sh`), emax-14 preparation, emax-10 flow; farms:
native Omegas, one rank per GCD (`tools/al27/farm_sweep.sh`, `BACKEND=hip`). Run directories (not
in git): `$IMSRG_WORK/runs/al27/{v1,v2,v3}/<H>/` (flows), `.../farm-native/` (2415 `.snt` per
farm), `$IMSRG_WORK/runs/al27/scan/` (section 3). Differences: `tools/al27/diff_variants.py`:
absolute |Δ| for every element; relative |Δ|/|ref| only for |ref| >= 1e-6 (owner's rule); the
per-column tables repeat the relative maximum and the count above 1e-3 for the sizeable elements
|ref| >= 1e-3, because the relative maxima over |ref| >= 1e-6 come from elements just above the cut.
Jobs and node-hours: `design/benchmarks/al27-jobs.md`, section Lead D.

## 1. Summary

* **Flows** (job 6015996, one node, 47 min): all four converged in the first chain link.
  E0 = -213.956872 (v2 N2LOsat), -217.070296 (v2 DN2LOGO), -214.428763 (v3 N2LOsat, 42 Omegas),
  -216.987333 (v3 DN2LOGO, 40 Omegas) MeV. v2 - v1 = +4.6 / -3.9 keV; v3 - v2 = **-472 / +83 keV**.
* **v3 vs v2 is the Magnus(2)-vs-IMSRG(2) truncation difference of the unsplit variant, not a
  defect or a discretization error of v3** (section 3). An ODE IMSRG(2) flow from the same H_0
  gives -214.444235 / -216.989496 MeV: v3 is 15.5 / 2.2 keV from it (7.0 / 0.6 keV at domega
  0.025), v2 487 / 81 keV. The split threshold 0.25 is converged to ~1 keV (0.1 vs 0.25), v3's
  step error is <= 8 keV, v2's <= 4 keV.
* **v2 vs v1** (section 4): E0 and the scalar columns change little (FM0 <= 9e-4 absolute); the
  tensor columns change by up to 0.10 absolute (FPhitp2/FPhipp2 one-body part at q = 0). A
  decomposition farm (v2 Omegas transformed with the fork's operator behaviour) assigns this to the
  fork emulation of v1, i.e. to the fork bugs fixed in the default path (design/30 section 6:
  M15-1, the missing a == b, J_bra != J_ket normal-ordering term, for the even ranks; AL-F1, the
  missing (-1)^L particle-hole phase, for Sigma'/Sigma''/Delta), while the flow part (domega 0.2 ->
  0.1, same operator code) is <= 6e-3 absolute.
* **v3 vs v2 operators** (section 5): (pending)
* **Farms**: v2 2 x 2415 items, 0 failed (job 6016703, 30 nodes, 19.6 min, 9.8 node-hours; 27
  GCD-hours of item work per Hamiltonian, TCM 792 s the longest item); v3: (pending). A v3 item
  costs 3.5-8x a v2 item (job 6016727: ~170 BCH terms over 42 Omegas vs ~28 over 2), TCM ~1x (host
  preparation).
* Writer note: the devel tokyo writer (v2/v3) drops scalar two-body elements below
  1e-7 x ||O_2b|| (imsrg-devel rule); for TCM (||TCM_2b|| ~ 1e4) that is ~1e-3, so v2/v3 TCM files
  have ~140 instead of the fork writer's 158 two-body elements (WE-D-1). The NREFT operators are
  not affected (thresholds <= 1e-7).

## 2. Flows

| flow | stage 1 steps / s / E0 (MeV) | stage 2 steps / s / E0 (MeV) | Omegas | wall (1 GCD) | host RSS |
|---|---|---|---:|---|---:|
| v1 N2LOsat (CPU, reference) | 37 / 13.322 / -131.499115 | 272 / 57.655 / -213.961460 | 2 | | 66 GB |
| v2 N2LOsat | 54 / 13.315 / -131.498815 | 489 / 57.914 / -213.956872 | 2 | 46:52 (22 min preparation) | 66.1 GB |
| v3 N2LOsat | 44 / 13.049 / -131.608497 | 146 / 48.854 / -214.428763 | 42 | 11:35 (H_0 from v2) | 17.4 GB |
| v1 DN2LOGO (CPU, reference) | 30 / 11.306 / -157.357310 | 253 / 47.146 / -217.066365 | 2 | | |
| v2 DN2LOGO | 43 / 11.170 / -157.356574 | 465 / 47.305 / -217.070296 | 2 | 46:14 | 66.1 GB |
| v3 DN2LOGO | 36 / 10.431 / -157.252591 | 135 / 41.727 / -216.987333 | 40 | 11:17 | 17.3 GB |

Steps = last step index of the stage. v3 starts from v2's prepared H_0 (`checkpoint.prepared_from`,
same preparation hash), so v3 - v2 isolates the Magnus splitting. GCD memory (rocm-smi every 5 min):
<= 16.3 GB per GCD; the 40-42 resident v3 Omegas (0.29 GB each) are no issue. v2 needs ~1.8x the
steps of v1 (domega 0.1 vs 0.2). The step is ds = min(domega/|eta|, domega/(|eta| |Omega|),
omega_norm_max/|eta|, dsmax): the split v3 (|Omega| < 0.33) takes much larger steps than v2
(|Omega| up to 8) and needs 3x fewer.

## 3. v3 vs v2: the energy difference (investigation)

Question: v3 - v2 = -472 keV (N2LOsat), +83 keV (DN2LOGO), already ~ +-105 keV at the end of stage
1. Candidates: (a) the time step (v3's steps are up to |Omega| ~ 8 times larger); (b) the split
threshold; (c) the different IMSRG(2) truncation of one large Omega (Magnus(2): the nested
commutators of e^Omega H e^-Omega truncated at the two-body level) and of a product of small
Omegas, which approaches the ODE IMSRG(2) flow (Morris, Parzuchowski, Bogner, PRC 92, 034331
(2015)). Scan (jobs 6016711, 6016712, 6016814; every point from v2's prepared H_0; configs from
`tools/al27/make_scan_configs.py`; the ODE flow is `imsrg.method = "flow"` (dopri5, tolerance
1e-6) on the GCD, 21-24 min per flow):

**N2LOsat**

| point | stage-1 steps | stage-1 E0 | stage-2 steps | final s | final E0 | Omegas | done |
|---|---:|---:|---:|---:|---:|---:|---|
| v2 (unsplit, domega 0.1) | 54 | -131.498815 | 489 | 57.91 | -213.956872 | 2 | yes |
| unsplit, domega 0.05 | 91 | -131.498584 | 931 | 58.04 | -213.957206 | 2 | yes |
| split 1.0, domega 0.1 | 44 | -131.593050 | 146 | 48.91 | -214.411122 | 12 | yes |
| v3 (split 0.25, domega 0.1) | 44 | -131.608497 | 146 | 48.85 | -214.428763 | 42 | yes |
| split 0.25, domega 0.05 | 68 | -131.605162 | 235 | 48.71 | -214.437105 | 41 | yes |
| split 0.25, domega 0.025 | 118 | -131.603752 | 421 | 49.23 | -214.437229 | 45 | yes |
| split 0.1, domega 0.1 | 44 | -131.609869 | 146 | 48.85 | -214.429844 | 122 | yes |
| ODE (dopri5, tol 1e-6) | 102 | -131.603685 | 485 | 200.00 | -214.444235 | 1 | yes |

**DN2LOGO**

| point | stage-1 steps | stage-1 E0 | stage-2 steps | final s | final E0 | Omegas | done |
|---|---:|---:|---:|---:|---:|---:|---|
| v2 (unsplit, domega 0.1) | 43 | -157.356574 | 465 | 47.31 | -217.070296 | 2 | yes |
| unsplit, domega 0.05 | 72 | -157.356156 | 896 | 47.44 | -217.066630 | 2 | yes |
| split 1.0, domega 0.1 | 36 | -157.263653 | 135 | 41.73 | -216.997477 | 12 | yes |
| v3 (split 0.25, domega 0.1) | 36 | -157.252591 | 135 | 41.73 | -216.987333 | 40 | yes |
| split 0.25, domega 0.05 | 57 | -157.252979 | 223 | 42.04 | -216.987045 | 40 | yes |
| split 0.25, domega 0.025 | 100 | -157.253099 | 404 | 41.76 | -216.988880 | 43 | yes |
| split 0.1, domega 0.1 | 36 | -157.252097 | 135 | 41.73 | -216.987269 | 116 | yes |
| ODE (dopri5, tol 1e-6) | 83 | -157.252943 | 443 | 200.00 | -216.989496 | 1 | yes |

(ODE rows: accepted steps from results.json (the flow file also lists trial steps); the ODE stage 1
runs on to s = 100 at constant E0 while |eta| sits at the 1e-3 criterion, WE-D-2, harmless; its
stage 2 starts with steps of ~1e-5 in s and speeds up later.)

**Conclusion.** (1) Time step: v3 at domega 0.1 is within 8 keV (N2LOsat) and 2 keV (DN2LOGO) of
its domega -> 0 plateau; v2's step error is <= 4 keV (domega 0.05: -0.3 / +3.7 keV). (2)
Threshold: 1.0 -> 0.25 -> 0.1 moves E0 by 18 / 1 keV (N2LOsat) and 10 / 0.1 keV (DN2LOGO): 0.25
is converged to ~1 keV, as the owner expected. (3) The split series approaches the ODE IMSRG(2)
result: stage 1 to 0.07 / 0.16 keV, final to 7.0 / 0.6 keV at domega 0.025 (the 7 keV of N2LOsat
are the residual threshold and ODE-tolerance effects). The unsplit Magnus(2) is 105 keV off at
stage 1 and 487 / 81 keV at the end. **The v3 - v2 difference is therefore the known Magnus(2) vs
IMSRG(2) truncation difference carried by the unsplit variant**; its sign and size depend on the
Hamiltonian. v3 is the better approximation to the IMSRG(2) flow. The operator differences v3 vs
v2 (section 5) are read the same way: same H_0 and operator code, different truncation of the
unitary transformation.

## 4. v2 vs v1

### 4.1 Full farms (2415 files per Hamiltonian) and the flow outputs

#### v2 vs v1, N2LOsat

Absolute |test - ref| for all elements; relative only for |ref| >= 1e-06; flag threshold rel > 0.001.


##### Flow (results.json, flow.txt)

| quantity | ref | test | diff |
|---|---|---|---|
| final E0 (results.json) | -213.9614597175 | -213.9568717976 | 4.59e-03 |
| num_omegas (results.json) | 2 | 2 | 0 |
| num_omegas (flow.txt, last row) | 2 | 2 | 0 |
| stage 1 final E0 | n/a | -131.49881501042395 | |
| stage 1 final s | n/a | 13.314645897481741 | |
| stage 1 final step | n/a | 54 | |
| stage 2 final E0 | -213.9614597175 | -213.9568717976 | 4.59e-03 |
| stage 2 final s | 57.655176 | 57.913877 | 2.59e-01 |
| stage 2 final step | 272 | 489 | 217 |

##### Flow directory files (valence H, TCM)

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 2.53e-02 (-) | 2.18e-01 (-) | 96 | 2.18e-01 | 96 | 0 | 0/0 | 1.18e+02 |
| TCM(flow) | 1 | 1.90e-03 (-) | 1.00e+00 (-) | 108 | 1.00e+00 | 108 | 0 | 1/0 | 6.96e+00 |

H by part:

| part | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|
| 0b | 3.49e-03 | 2.95e-05 | 0 | 0 | 0/0 |
| 1b | 7.35e-03 | 2.56e-02 | 2 | 0 | 0/0 |
| 2b | 2.53e-02 | 2.18e-01 | 94 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 5.19e-04 (135) | 1.86e+00 (705) | 6509 | 1.24e-01 | 1502 | 713 | 166/1 | 8.00e+00 |
| FM0p | 71 | 9.01e-04 (135) | 3.96e+00 (555) | 6899 | 1.66e-01 | 1728 | 735 | 160/0 | 8.00e+00 |
| FM2n | 71 | 4.02e-02 (180) | 7.88e+01 (720) | 29824 | 9.16e+00 | 11438 | 870 | 0/78 | 6.63e-01 |
| FM2p | 71 | 3.54e-02 (180) | 3.95e+01 (915) | 30736 | 2.80e+00 | 11704 | 869 | 0/74 | 6.49e-01 |
| FM4n | 71 | 1.20e-02 (495) | 5.42e+01 (135) | 22335 | 1.52e+00 | 7730 | 1025 | 0/175 | 4.00e-01 |
| FM4p | 71 | 1.01e-02 (510) | 4.98e+01 (315) | 23261 | 1.19e+00 | 8209 | 1073 | 0/163 | 3.98e-01 |
| FPhipp0n | 71 | 3.28e-03 (0) | 8.64e+00 (90) | 6039 | 5.61e-01 | 1743 | 1770 | 14/7 | 8.44e-01 |
| FPhipp0p | 71 | 6.31e-03 (0) | 3.74e+00 (120) | 6460 | 3.36e-01 | 1860 | 1781 | 4/0 | 8.61e-01 |
| FPhipp2n | 71 | 8.19e-02 (0) | 3.15e+02 (0) | 26837 | 5.17e+00 | 10797 | 3823 | 1/593 | 1.04e+00 |
| FPhipp2p | 71 | 8.19e-02 (0) | 3.96e+01 (270) | 27179 | 3.01e+00 | 10877 | 3932 | 1/645 | 1.04e+00 |
| FPhipp4n | 71 | 5.10e-03 (390) | 9.96e+00 (330) | 19928 | 6.48e-01 | 5146 | 2886 | 0/436 | 1.91e-01 |
| FPhipp4p | 71 | 4.33e-03 (345) | 2.61e+01 (90) | 20372 | 6.90e-01 | 5273 | 3042 | 0/492 | 1.90e-01 |
| FPhitp2n | 71 | 1.00e-01 (0) | 3.15e+02 (0) | 27689 | 2.03e+00 | 9831 | 3744 | 1/533 | 1.27e+00 |
| FPhitp2p | 71 | 1.00e-01 (0) | 4.61e+01 (630) | 28090 | 1.63e+00 | 10013 | 3949 | 0/553 | 1.27e+00 |
| FPhitp4n | 71 | 9.46e-03 (360) | 3.48e+01 (120) | 20297 | 6.29e-01 | 4813 | 2667 | 0/353 | 2.06e-01 |
| FPhitp4p | 71 | 7.64e-03 (360) | 2.53e+01 (795) | 20564 | 6.71e-01 | 4926 | 2942 | 2/394 | 2.05e-01 |
| FSigmap1n | 71 | 9.39e-03 (150) | 1.15e+03 (0) | 29298 | 3.58e+00 | 8066 | 2499 | 0/301 | 6.22e-01 |
| FSigmap1p | 71 | 1.10e-02 (150) | 2.69e+03 (30) | 29408 | 4.42e+00 | 8165 | 2347 | 2/321 | 6.21e-01 |
| FSigmap3n | 71 | 1.02e-02 (420) | 2.48e+02 (135) | 33796 | 1.69e+00 | 7015 | 2664 | 3/377 | 1.58e-01 |
| FSigmap3p | 71 | 1.09e-02 (420) | 2.59e+02 (180) | 33493 | 9.69e-01 | 7066 | 2869 | 3/410 | 1.57e-01 |
| FSigmap5n | 71 | 3.60e-03 (345) | 1.60e+02 (345) | 16041 | 1.98e+00 | 3085 | 1693 | 0/373 | 1.46e-01 |
| FSigmap5p | 71 | 2.92e-03 (330) | 1.93e+02 (255) | 16020 | 1.79e+00 | 3072 | 1681 | 0/344 | 1.45e-01 |
| FSigmapp1n | 71 | 7.69e-03 (300) | 4.04e+02 (90) | 29489 | 2.09e+00 | 7337 | 2282 | 0/239 | 4.39e-01 |
| FSigmapp1p | 71 | 8.01e-03 (285) | 3.57e+02 (15) | 29375 | 3.17e+00 | 7550 | 2341 | 1/282 | 4.39e-01 |
| FSigmapp3n | 71 | 1.38e-02 (180) | 4.53e+02 (315) | 33826 | 1.29e+00 | 6490 | 2653 | 1/365 | 1.49e-01 |
| FSigmapp3p | 71 | 1.44e-02 (180) | 1.74e+02 (300) | 33652 | 1.39e+00 | 6442 | 2730 | 1/369 | 1.49e-01 |
| FSigmapp5n | 71 | 2.35e-03 (420) | 8.50e+01 (165) | 16140 | 5.65e-01 | 2843 | 1641 | 0/371 | 1.32e-01 |
| FSigmapp5p | 71 | 2.27e-03 (195) | 1.57e+02 (435) | 15967 | 3.46e-01 | 2806 | 1756 | 0/431 | 1.32e-01 |
| FDelta1n | 71 | 1.59e-02 (195) | 2.04e+02 (150) | 24901 | 3.05e+00 | 4399 | 6903 | 6/2500 | 6.31e-01 |
| FDelta1p | 71 | 1.54e-02 (195) | 1.46e+03 (15) | 24711 | 2.14e+00 | 4577 | 7119 | 8/2764 | 6.29e-01 |
| FDelta3n | 71 | 3.09e-03 (315) | 2.05e+02 (225) | 28473 | 6.42e-01 | 2239 | 7904 | 3/2696 | 5.97e-02 |
| FDelta3p | 71 | 3.41e-03 (135) | 8.63e+01 (240) | 28602 | 5.97e-01 | 2317 | 7805 | 2/2755 | 5.93e-02 |
| FDelta5n | 71 | 1.16e-03 (375) | 1.08e+02 (345) | 13426 | 5.35e-01 | 331 | 4190 | 1/1541 | 2.41e-03 |
| FDelta5p | 71 | 9.89e-04 (375) | 3.79e+01 (405) | 13405 | 5.12e-01 | 353 | 4194 | 2/1592 | 2.47e-03 |
| TCM | 1 | 1.90e-03 (-) | 1.00e+00 (-) | 123 | 1.00e+00 | 109 | 0 | 16/0 | 6.96e+00 |

###### Per q

| q | files | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|
| 0 | 34 | 1.00e-01 | FPhitp2n | 1.15e+03 | FSigmap1n |
| 15 | 34 | 9.98e-02 | FPhitp2n | 1.46e+03 | FDelta1p |
| 30 | 34 | 9.81e-02 | FPhitp2p | 2.69e+03 | FSigmap1p |
| 45 | 34 | 9.54e-02 | FPhitp2p | 2.45e+02 | FDelta1p |
| 60 | 34 | 9.15e-02 | FPhitp2p | 3.46e+02 | FDelta1p |
| 75 | 34 | 8.66e-02 | FPhitp2p | 1.67e+02 | FSigmapp1n |
| 90 | 34 | 8.06e-02 | FPhitp2p | 4.62e+02 | FSigmap1n |
| 105 | 34 | 7.38e-02 | FPhitp2p | 3.20e+02 | FSigmapp1n |
| 120 | 34 | 6.63e-02 | FPhitp2p | 1.36e+02 | FSigmapp1p |
| 135 | 34 | 5.83e-02 | FPhitp2n | 5.16e+02 | FSigmap1n |
| 150 | 34 | 5.01e-02 | FPhitp2n | 2.04e+02 | FSigmapp3n |
| 165 | 34 | 4.19e-02 | FPhitp2n | 2.56e+02 | FSigmapp1p |
| 180 | 34 | 4.02e-02 | FM2n | 2.59e+02 | FSigmap3p |
| 195 | 34 | 3.95e-02 | FM2n | 1.77e+02 | FDelta1p |
| 210 | 34 | 3.74e-02 | FM2n | 1.83e+02 | FSigmap1p |
| 225 | 34 | 3.42e-02 | FM2n | 2.05e+02 | FDelta3n |
| 240 | 34 | 3.05e-02 | FM2n | 1.41e+02 | FSigmap5p |
| 255 | 34 | 3.08e-02 | FM2n | 2.71e+02 | FSigmapp3n |
| 270 | 34 | 3.00e-02 | FM2n | 2.02e+02 | FSigmap1n |
| 285 | 34 | 2.83e-02 | FM2n | 9.71e+01 | FSigmapp3p |
| 300 | 34 | 2.59e-02 | FM2n | 1.74e+02 | FSigmapp3p |
| 315 | 34 | 2.31e-02 | FM2n | 4.53e+02 | FSigmapp3n |
| 330 | 34 | 2.24e-02 | FM2n | 1.69e+02 | FSigmap3n |
| 345 | 34 | 2.28e-02 | FM2n | 1.60e+02 | FSigmap5n |
| 360 | 34 | 2.25e-02 | FM2n | 1.46e+02 | FSigmapp3p |
| 375 | 34 | 2.15e-02 | FM2n | 9.53e+01 | FSigmapp3p |
| 390 | 34 | 2.00e-02 | FM2n | 5.95e+01 | FSigmap1n |
| 405 | 34 | 1.81e-02 | FM2n | 1.19e+02 | FSigmap5n |
| 420 | 34 | 1.57e-02 | FM2n | 6.10e+01 | FDelta1p |
| 435 | 34 | 1.32e-02 | FM2n | 1.57e+02 | FSigmapp5p |
| 450 | 34 | 1.18e-02 | FSigmapp3p | 5.94e+01 | FSigmapp3n |
| 465 | 34 | 1.15e-02 | FM4n | 1.06e+02 | FSigmap1p |
| 480 | 34 | 1.19e-02 | FM4n | 6.37e+01 | FSigmap1p |
| 495 | 34 | 1.20e-02 | FM4n | 7.37e+01 | FSigmap3n |
| 510 | 34 | 1.18e-02 | FM4n | 9.52e+01 | FSigmapp1p |
| 525 | 34 | 1.15e-02 | FM4n | 1.10e+02 | FSigmap5p |
| 540 | 34 | 1.09e-02 | FM4n | 9.66e+01 | FSigmap5p |
| 555 | 34 | 1.01e-02 | FM4n | 1.33e+02 | FSigmapp3p |
| 570 | 34 | 9.21e-03 | FM4n | 4.21e+01 | FSigmap1n |
| 585 | 34 | 8.22e-03 | FM4n | 4.46e+01 | FSigmapp3p |
| 600 | 34 | 7.19e-03 | FM4n | 5.80e+01 | FSigmapp5p |
| 615 | 34 | 6.15e-03 | FM4n | 7.98e+01 | FSigmap5p |
| 630 | 34 | 5.15e-03 | FM4n | 4.91e+01 | FSigmap5p |
| 645 | 34 | 4.21e-03 | FM4n | 5.94e+01 | FSigmapp3n |
| 660 | 34 | 3.35e-03 | FM4n | 6.03e+01 | FSigmapp3p |
| 675 | 34 | 2.64e-03 | FM2n | 6.04e+01 | FSigmap3n |
| 690 | 34 | 2.16e-03 | FM2n | 4.20e+01 | FSigmap5n |
| 705 | 34 | 1.70e-03 | FM2n | 6.79e+01 | FSigmap5p |
| 720 | 34 | 1.28e-03 | FM2n | 1.26e+02 | FPhipp2n |
| 735 | 34 | 9.23e-04 | FSigmapp3p | 4.76e+01 | FPhipp2n |
| 750 | 34 | 8.17e-04 | FSigmapp3p | 8.15e+01 | FPhipp2n |
| 765 | 34 | 7.16e-04 | FSigmapp3p | 4.82e+01 | FPhipp2n |
| 780 | 34 | 6.32e-04 | FM2p | 3.33e+01 | FSigmap5n |
| 795 | 34 | 6.00e-04 | FM2p | 2.53e+01 | FPhitp4p |
| 810 | 34 | 5.47e-04 | FM2p | 2.02e+01 | FDelta3p |
| 825 | 34 | 4.82e-04 | FM2p | 1.42e+01 | FSigmapp5n |
| 840 | 34 | 4.12e-04 | FM2p | 4.04e+01 | FSigmap3n |
| 855 | 34 | 3.41e-04 | FM2p | 2.30e+01 | FSigmapp5n |
| 870 | 34 | 2.75e-04 | FM2p | 2.12e+01 | FSigmap3n |
| 885 | 34 | 2.15e-04 | FM2p | 1.67e+01 | FSigmap1n |
| 900 | 34 | 1.65e-04 | FM2p | 1.29e+01 | FSigmap1n |
| 915 | 34 | 1.24e-04 | FM2p | 3.95e+01 | FM2p |
| 930 | 34 | 9.09e-05 | FM2n | 2.56e+01 | FSigmapp3p |
| 945 | 34 | 6.98e-05 | FM2n | 1.33e+01 | FPhipp2p |
| 960 | 34 | 5.14e-05 | FM2n | 1.24e+01 | FSigmap3p |
| 975 | 34 | 3.60e-05 | FM2n | 4.85e+00 | FSigmapp1p |
| 990 | 34 | 2.53e-05 | FSigmap3p | 1.05e+01 | FSigmap3p |
| 1005 | 34 | 1.91e-05 | FSigmap3p | 5.38e+00 | FSigmap3p |
| 1020 | 34 | 1.41e-05 | FSigmap3p | 3.44e+00 | FSigmap3p |
| 1035 | 34 | 1.05e-05 | FM2n | 2.57e+00 | FSigmap1p |
| 1050 | 34 | 8.69e-06 | FM2n | 6.83e+00 | FM2n |

###### By part

| part | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|
| 0b | 1.52e-04 | 2.38e-02 | 12 | 2130 | 0/0 |
| 1b | 1.00e-01 | 4.04e+02 | 15420 | 816 | 10/131 |
| 2b | 8.00e-03 | 2.69e+03 | 757733 | 98146 | 388/22397 |

##### 10 worst elements by relative diff

| file | key | ref | test | value |
|---|---|---|---|---|
| N2LOsat_FSigmap1p_30_2b.snt | 2b:(0, 2, 3, 1),(0, 2, 3, 1),(0, 2, 5, 1),(0, 2, 5, 1),2,2 | 1.016064e-06 | -2.728828e-03 | 2.69e+03 |
| N2LOsat_FSigmap1p_30_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 5, 1),(0, 2, 3, 1),(0, 2, 3, 1),2,2 | 1.016064e-06 | -2.728828e-03 | 2.69e+03 |
| N2LOsat_FDelta1p_15_2b.snt | 2b:(1, 0, 1, -1),(1, 0, 1, 1),(0, 2, 5, 1),(0, 2, 3, -1),1,2 | -1.354319e-06 | 1.970870e-03 | 1.46e+03 |
| N2LOsat_FSigmap1n_0_2b.snt | 2b:(0, 2, 3, -1),(1, 0, 1, -1),(0, 2, 5, -1),(0, 2, 5, -1),1,2 | -1.011374e-06 | 1.160368e-03 | 1.15e+03 |
| N2LOsat_FSigmap1n_135_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 5, -1),(0, 2, 3, -1),2,2 | -1.209562e-06 | 6.229403e-04 | 5.16e+02 |
| N2LOsat_FSigmap1n_135_2b.snt | 2b:(0, 2, 5, -1),(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 3, -1),2,2 | -1.209562e-06 | 6.229403e-04 | 5.16e+02 |
| N2LOsat_FSigmap1n_90_2b.snt | 2b:(0, 2, 5, -1),(0, 2, 3, 1),(0, 2, 5, 1),(0, 2, 3, -1),2,2 | 1.387616e-06 | 6.422110e-04 | 4.62e+02 |
| N2LOsat_FSigmap1n_90_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 3, -1),(0, 2, 5, -1),(0, 2, 3, 1),2,2 | 1.387616e-06 | 6.422110e-04 | 4.62e+02 |
| N2LOsat_FSigmapp3n_315_2b.snt | 2b:(1, 0, 1, -1),(1, 0, 1, 1),(0, 2, 5, -1),(0, 2, 3, 1),1,3 | -1.083311e-06 | -4.920209e-04 | 4.53e+02 |
| N2LOsat_FSigmap1n_90_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 5, -1),(1, 0, 1, 1),2,2 | 4.693617e-06 | -2.117783e-03 | 4.52e+02 |

##### 10 worst elements by absolute diff

| file | key | ref | test | value |
|---|---|---|---|---|
| N2LOsat_FPhitp2n_0_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.577013e-01 | 4.580602e-01 | 1.00e-01 |
| N2LOsat_FPhitp2p_0_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.534997e-01 | 4.538291e-01 | 1.00e-01 |
| N2LOsat_FPhitp2n_15_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.571125e-01 | 4.569058e-01 | 9.98e-02 |
| N2LOsat_FPhitp2p_15_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.528905e-01 | 4.526686e-01 | 9.98e-02 |
| N2LOsat_FPhitp2p_30_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.510252e-01 | 4.491476e-01 | 9.81e-02 |
| N2LOsat_FPhitp2n_30_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.553090e-01 | 4.534080e-01 | 9.81e-02 |
| N2LOsat_FPhitp2p_45_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.477987e-01 | 4.431592e-01 | 9.54e-02 |
| N2LOsat_FPhitp2n_45_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.521868e-01 | 4.474724e-01 | 9.53e-02 |
| N2LOsat_FPhitp2p_60_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.430604e-01 | 4.345604e-01 | 9.15e-02 |
| N2LOsat_FPhitp2n_60_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.475956e-01 | 4.389721e-01 | 9.14e-02 |


#### v2 vs v1, DN2LOGO

Absolute |test - ref| for all elements; relative only for |ref| >= 1e-06; flag threshold rel > 0.001.


##### Flow (results.json, flow.txt)

| quantity | ref | test | diff |
|---|---|---|---|
| final E0 (results.json) | -217.0663647615 | -217.0702962989 | 3.93e-03 |
| num_omegas (results.json) | 2 | 2 | 0 |
| num_omegas (flow.txt, last row) | 2 | 2 | 0 |
| stage 1 final E0 | n/a | -157.35657444544228 | |
| stage 1 final s | n/a | 11.169789227091066 | |
| stage 1 final step | n/a | 43 | |
| stage 2 final E0 | -217.0663647615 | -217.0702962989 | 3.93e-03 |
| stage 2 final s | 47.146391 | 47.305050 | 1.59e-01 |
| stage 2 final step | 253 | 465 | 212 |

##### Flow directory files (valence H, TCM)

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 7.35e-03 (-) | 1.56e-01 (-) | 59 | 1.56e-01 | 59 | 0 | 0/0 | 1.16e+02 |
| TCM(flow) | 1 | 1.03e-03 (-) | 1.87e-01 (-) | 104 | 1.87e-01 | 103 | 0 | 0/0 | 7.08e+00 |

H by part:

| part | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|
| 0b | 2.48e-04 | 2.13e-06 | 0 | 0 | 0/0 |
| 1b | 1.86e-03 | 2.70e-03 | 1 | 0 | 0/0 |
| 2b | 7.35e-03 | 1.56e-01 | 58 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 2.30e-04 (135) | 1.02e+00 (345) | 4709 | 1.46e-01 | 1102 | 1097 | 163/0 | 8.00e+00 |
| FM0p | 71 | 4.98e-04 (120) | 5.28e+00 (135) | 5964 | 2.39e-01 | 1565 | 1075 | 162/0 | 8.00e+00 |
| FM2n | 71 | 4.05e-02 (180) | 4.25e+01 (180) | 23503 | 3.88e+00 | 9345 | 1666 | 0/211 | 6.63e-01 |
| FM2p | 71 | 3.67e-02 (180) | 5.77e+01 (240) | 27210 | 2.94e+00 | 10553 | 1487 | 3/124 | 6.51e-01 |
| FM4n | 71 | 1.16e-02 (315) | 1.77e+01 (135) | 15135 | 1.17e+00 | 4970 | 1919 | 0/250 | 4.20e-01 |
| FM4p | 71 | 1.02e-02 (315) | 1.41e+01 (660) | 19373 | 7.59e-01 | 6613 | 1697 | 0/230 | 4.08e-01 |
| FPhipp0n | 71 | 5.44e-04 (0) | 1.11e+01 (285) | 4356 | 3.65e-01 | 1250 | 2189 | 9/5 | 8.05e-01 |
| FPhipp0p | 71 | 7.13e-04 (0) | 1.53e+02 (30) | 4782 | 4.67e-01 | 1401 | 2218 | 3/1 | 8.31e-01 |
| FPhipp2n | 71 | 9.67e-02 (0) | 1.70e+01 (765) | 19513 | 5.69e+00 | 8590 | 5736 | 0/1272 | 1.00e+00 |
| FPhipp2p | 71 | 9.28e-02 (0) | 2.99e+01 (675) | 22635 | 6.50e+00 | 9776 | 5788 | 0/1220 | 1.00e+00 |
| FPhipp4n | 71 | 3.72e-03 (345) | 1.83e+01 (780) | 11537 | 5.75e-01 | 2785 | 4428 | 0/1049 | 1.88e-01 |
| FPhipp4p | 71 | 3.45e-03 (345) | 2.30e+01 (270) | 14655 | 6.61e-01 | 4057 | 4482 | 1/956 | 1.88e-01 |
| FPhitp2n | 71 | 1.18e-01 (0) | 3.65e+01 (780) | 19599 | 1.47e+00 | 7579 | 5622 | 0/1347 | 1.23e+00 |
| FPhitp2p | 71 | 1.14e-01 (0) | 8.20e+00 (690) | 23426 | 1.58e+00 | 8916 | 4972 | 1/1065 | 1.23e+00 |
| FPhitp4n | 71 | 7.82e-03 (345) | 1.45e+01 (270) | 12569 | 5.66e-01 | 2443 | 4117 | 2/915 | 2.04e-01 |
| FPhitp4p | 71 | 6.19e-03 (345) | 4.82e+01 (660) | 16039 | 6.90e-01 | 3502 | 3558 | 2/758 | 2.03e-01 |
| FSigmap1n | 71 | 1.13e-02 (0) | 3.92e+02 (45) | 28383 | 3.71e+00 | 8392 | 3420 | 0/553 | 6.29e-01 |
| FSigmap1p | 71 | 9.12e-03 (165) | 6.98e+02 (15) | 28527 | 5.09e+00 | 8553 | 3330 | 1/456 | 6.27e-01 |
| FSigmap3n | 71 | 8.11e-03 (420) | 4.13e+02 (225) | 32576 | 1.17e+00 | 7297 | 3850 | 2/657 | 1.58e-01 |
| FSigmap3p | 71 | 9.52e-03 (420) | 2.19e+02 (210) | 32543 | 1.04e+00 | 7421 | 3744 | 5/699 | 1.58e-01 |
| FSigmap5n | 71 | 3.68e-03 (345) | 1.16e+02 (330) | 15338 | 7.32e-01 | 3101 | 2313 | 1/530 | 1.46e-01 |
| FSigmap5p | 71 | 3.15e-03 (210) | 9.58e+01 (270) | 15365 | 6.50e-01 | 3152 | 2302 | 0/517 | 1.46e-01 |
| FSigmapp1n | 71 | 9.09e-03 (105) | 1.14e+03 (75) | 28546 | 2.69e+00 | 7805 | 3266 | 3/550 | 4.45e-01 |
| FSigmapp1p | 71 | 8.97e-03 (105) | 6.63e+02 (105) | 28623 | 3.47e+00 | 7830 | 3179 | 1/465 | 4.44e-01 |
| FSigmapp3n | 71 | 1.23e-02 (180) | 1.99e+02 (90) | 32578 | 1.18e+00 | 6811 | 3845 | 2/718 | 1.50e-01 |
| FSigmapp3p | 71 | 1.38e-02 (180) | 2.06e+02 (165) | 32821 | 1.39e+00 | 6781 | 3561 | 1/645 | 1.50e-01 |
| FSigmapp5n | 71 | 1.97e-03 (195) | 1.30e+02 (300) | 15258 | 4.63e-01 | 2798 | 2331 | 3/573 | 1.33e-01 |
| FSigmapp5p | 71 | 2.69e-03 (195) | 1.76e+02 (285) | 15431 | 4.64e-01 | 2788 | 2266 | 1/562 | 1.32e-01 |
| FDelta1n | 71 | 1.38e-02 (195) | 6.29e+02 (105) | 24139 | 3.03e+00 | 4540 | 7592 | 2/3862 | 6.36e-01 |
| FDelta1p | 71 | 1.46e-02 (195) | 5.30e+02 (30) | 24100 | 3.05e+00 | 4681 | 7687 | 2/3535 | 6.35e-01 |
| FDelta3n | 71 | 3.19e-03 (300) | 1.22e+02 (525) | 27397 | 6.23e-01 | 2242 | 8829 | 0/4339 | 6.05e-02 |
| FDelta3p | 71 | 2.77e-03 (345) | 1.62e+02 (255) | 27201 | 6.32e-01 | 2233 | 8952 | 0/3979 | 6.02e-02 |
| FDelta5n | 71 | 1.08e-03 (360) | 1.38e+02 (360) | 12747 | 5.48e-01 | 367 | 4811 | 0/2294 | 2.32e-03 |
| FDelta5p | 71 | 9.16e-04 (375) | 2.43e+01 (210) | 12747 | 5.28e-01 | 361 | 4850 | 1/2367 | 2.42e-03 |
| TCM | 1 | 1.03e-03 (-) | 1.00e+00 (-) | 112 | 1.87e-01 | 103 | 0 | 8/0 | 7.08e+00 |

###### Per q

| q | files | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|
| 0 | 34 | 1.18e-01 | FPhitp2n | 2.66e+02 | FDelta1n |
| 15 | 34 | 1.18e-01 | FPhitp2n | 6.98e+02 | FSigmap1p |
| 30 | 34 | 1.16e-01 | FPhitp2n | 5.30e+02 | FDelta1p |
| 45 | 34 | 1.13e-01 | FPhitp2n | 5.24e+02 | FSigmapp1n |
| 60 | 34 | 1.09e-01 | FPhitp2n | 3.23e+02 | FSigmap1n |
| 75 | 34 | 1.03e-01 | FPhitp2n | 1.14e+03 | FSigmapp1n |
| 90 | 34 | 9.69e-02 | FPhitp2n | 3.71e+02 | FDelta1n |
| 105 | 34 | 8.97e-02 | FPhitp2n | 6.63e+02 | FSigmapp1p |
| 120 | 34 | 8.19e-02 | FPhitp2n | 4.72e+02 | FDelta1p |
| 135 | 34 | 7.36e-02 | FPhitp2n | 1.33e+02 | FSigmap3p |
| 150 | 34 | 6.50e-02 | FPhitp2n | 2.47e+02 | FSigmap1n |
| 165 | 34 | 5.63e-02 | FPhitp2n | 2.13e+02 | FDelta1n |
| 180 | 34 | 4.77e-02 | FPhitp2n | 2.26e+02 | FDelta1n |
| 195 | 34 | 3.96e-02 | FM2n | 1.72e+02 | FSigmapp1n |
| 210 | 34 | 3.75e-02 | FM2n | 2.19e+02 | FSigmap3p |
| 225 | 34 | 3.42e-02 | FM2n | 4.13e+02 | FSigmap3n |
| 240 | 34 | 3.11e-02 | FM2n | 1.00e+02 | FSigmap3p |
| 255 | 34 | 3.14e-02 | FM2n | 1.71e+02 | FSigmap3n |
| 270 | 34 | 3.05e-02 | FM2n | 3.19e+02 | FSigmap1p |
| 285 | 34 | 2.88e-02 | FM2n | 1.98e+02 | FSigmap3n |
| 300 | 34 | 2.63e-02 | FM2n | 1.61e+02 | FSigmap3n |
| 315 | 34 | 2.34e-02 | FM2n | 9.74e+01 | FDelta1p |
| 330 | 34 | 2.03e-02 | FM2n | 1.16e+02 | FSigmap5n |
| 345 | 34 | 1.72e-02 | FM2n | 9.50e+01 | FSigmap1n |
| 360 | 34 | 1.67e-02 | FM2n | 1.38e+02 | FDelta5n |
| 375 | 34 | 1.64e-02 | FM2n | 1.65e+02 | FSigmapp3n |
| 390 | 34 | 1.56e-02 | FM2n | 9.90e+01 | FSigmap5n |
| 405 | 34 | 1.45e-02 | FM2n | 1.24e+02 | FSigmapp3p |
| 420 | 34 | 1.30e-02 | FM2n | 1.68e+02 | FSigmap1n |
| 435 | 34 | 1.13e-02 | FM2n | 7.75e+01 | FSigmapp1n |
| 450 | 34 | 1.03e-02 | FSigmapp3p | 9.52e+01 | FSigmapp3p |
| 465 | 34 | 9.49e-03 | FSigmapp3p | 7.53e+01 | FSigmap3n |
| 480 | 34 | 8.60e-03 | FSigmapp3p | 5.54e+01 | FSigmapp1n |
| 495 | 34 | 7.68e-03 | FSigmapp3p | 1.25e+02 | FSigmapp1p |
| 510 | 34 | 6.84e-03 | FSigmap3p | 7.18e+01 | FSigmapp5n |
| 525 | 34 | 6.14e-03 | FSigmap3p | 1.33e+02 | FSigmapp3p |
| 540 | 34 | 5.99e-03 | FM4n | 9.47e+01 | FSigmap5p |
| 555 | 34 | 5.93e-03 | FM4n | 7.97e+01 | FSigmapp1p |
| 570 | 34 | 5.74e-03 | FM4n | 4.83e+01 | FSigmapp3p |
| 585 | 34 | 5.45e-03 | FM4n | 4.92e+01 | FSigmapp3n |
| 600 | 34 | 5.07e-03 | FM4n | 4.53e+01 | FSigmap3p |
| 615 | 34 | 4.63e-03 | FM4n | 5.31e+01 | FSigmap3p |
| 630 | 34 | 4.16e-03 | FM4n | 4.15e+01 | FSigmap3n |
| 645 | 34 | 3.68e-03 | FM4n | 5.28e+01 | FSigmap3p |
| 660 | 34 | 3.20e-03 | FM4n | 7.78e+01 | FSigmapp3n |
| 675 | 34 | 2.74e-03 | FM4n | 4.81e+01 | FSigmapp3n |
| 690 | 34 | 2.30e-03 | FM4n | 4.02e+01 | FSigmapp5n |
| 705 | 34 | 1.91e-03 | FM4n | 3.23e+01 | FSigmapp5p |
| 720 | 34 | 1.56e-03 | FM4n | 2.48e+01 | FSigmapp3p |
| 735 | 34 | 1.25e-03 | FM4n | 5.72e+01 | FSigmap5p |
| 750 | 34 | 9.91e-04 | FM4n | 1.95e+01 | FSigmapp3n |
| 765 | 34 | 7.70e-04 | FM4n | 2.14e+01 | FSigmapp3n |
| 780 | 34 | 5.88e-04 | FM4n | 3.65e+01 | FPhitp2n |
| 795 | 34 | 4.41e-04 | FM4n | 6.31e+01 | FSigmapp5p |
| 810 | 34 | 3.24e-04 | FM4n | 1.89e+01 | FSigmap5n |
| 825 | 34 | 2.33e-04 | FM4n | 3.41e+01 | FSigmapp3p |
| 840 | 34 | 1.77e-04 | FM2p | 1.27e+01 | FPhipp2n |
| 855 | 34 | 1.38e-04 | FM2p | 9.57e+00 | FSigmap5p |
| 870 | 34 | 1.06e-04 | FM2n | 1.20e+01 | FPhipp2n |
| 885 | 34 | 8.59e-05 | FM2n | 6.85e+00 | FPhipp4p |
| 900 | 34 | 6.89e-05 | FM2n | 1.76e+01 | FM2p |
| 915 | 34 | 5.47e-05 | FM2n | 7.07e+00 | FSigmap3n |
| 930 | 34 | 4.31e-05 | FM2n | 7.48e+00 | FSigmapp3n |
| 945 | 34 | 3.37e-05 | FM2n | 1.14e+01 | FSigmapp3n |
| 960 | 34 | 2.61e-05 | FM2n | 4.17e+00 | FPhitp2n |
| 975 | 34 | 2.00e-05 | FM2n | 4.15e+00 | FSigmap1n |
| 990 | 34 | 1.52e-05 | FM2n | 3.36e+00 | FSigmap1n |
| 1005 | 34 | 1.24e-05 | FM4p | 2.96e+00 | FSigmap1n |
| 1020 | 34 | 1.06e-05 | FM4p | 1.48e+00 | FM2n |
| 1035 | 34 | 8.97e-06 | FM4p | 1.66e+00 | FM2n |
| 1050 | 34 | 7.45e-06 | FM4p | 1.61e+00 | FM4p |

###### By part

| part | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|
| 0b | 3.79e-04 | 4.38e-01 | 43 | 2135 | 0/0 |
| 1b | 1.18e-01 | 1.99e+02 | 14909 | 1306 | 12/249 |
| 2b | 7.13e-03 | 1.14e+03 | 664485 | 128738 | 367/36455 |

##### 10 worst elements by relative diff

| file | key | ref | test | value |
|---|---|---|---|---|
| DN2LOGO_FSigmapp1n_75_2b.snt | 2b:(0, 2, 5, 1),(1, 0, 1, -1),(0, 2, 3, -1),(0, 2, 3, 1),2,2 | 1.290761e-06 | 1.472162e-03 | 1.14e+03 |
| DN2LOGO_FSigmapp1n_75_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 5, 1),(1, 0, 1, -1),2,2 | 1.290761e-06 | 1.472162e-03 | 1.14e+03 |
| DN2LOGO_FSigmap1p_15_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 3, -1),(0, 2, 3, -1),(1, 0, 1, 1),2,2 | -1.365722e-06 | -9.541912e-04 | 6.98e+02 |
| DN2LOGO_FSigmap1p_15_2b.snt | 2b:(0, 2, 3, -1),(1, 0, 1, 1),(0, 2, 5, 1),(0, 2, 3, -1),2,2 | -1.365722e-06 | -9.541912e-04 | 6.98e+02 |
| DN2LOGO_FSigmapp1p_105_2b.snt | 2b:(0, 2, 5, 1),(1, 0, 1, 1),(0, 2, 3, 1),(0, 2, 3, 1),2,2 | 2.369614e-06 | 1.572732e-03 | 6.63e+02 |
| DN2LOGO_FSigmapp1p_105_2b.snt | 2b:(0, 2, 3, 1),(0, 2, 3, 1),(0, 2, 5, 1),(1, 0, 1, 1),2,2 | 2.369614e-06 | 1.572732e-03 | 6.63e+02 |
| DN2LOGO_FDelta1n_105_2b.snt | 2b:(0, 2, 5, -1),(0, 2, 3, 1),(0, 2, 3, 1),(1, 0, 1, -1),1,1 | -1.646142e-06 | 1.033138e-03 | 6.29e+02 |
| DN2LOGO_FDelta1n_105_2b.snt | 2b:(0, 2, 3, 1),(1, 0, 1, -1),(0, 2, 5, -1),(0, 2, 3, 1),1,1 | -1.646142e-06 | 1.033138e-03 | 6.29e+02 |
| DN2LOGO_FDelta1p_30_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 5, -1),(1, 0, 1, -1),2,2 | -3.809626e-06 | -2.024329e-03 | 5.30e+02 |
| DN2LOGO_FDelta1p_30_2b.snt | 2b:(0, 2, 5, -1),(1, 0, 1, -1),(0, 2, 3, -1),(0, 2, 3, -1),2,2 | -3.809626e-06 | -2.024329e-03 | 5.30e+02 |

##### 10 worst elements by absolute diff

| file | key | ref | test | value |
|---|---|---|---|---|
| DN2LOGO_FPhitp2n_0_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.051852e-01 | 4.236623e-01 | 1.18e-01 |
| DN2LOGO_FPhitp2n_15_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.048980e-01 | 4.227426e-01 | 1.18e-01 |
| DN2LOGO_FPhitp2n_30_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.039936e-01 | 4.199513e-01 | 1.16e-01 |
| DN2LOGO_FPhitp2p_0_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.132985e-01 | 4.269917e-01 | 1.14e-01 |
| DN2LOGO_FPhitp2p_15_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.128427e-01 | 4.259707e-01 | 1.13e-01 |
| DN2LOGO_FPhitp2n_45_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 3.023492e-01 | 4.151992e-01 | 1.13e-01 |
| DN2LOGO_FPhitp2p_30_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.114329e-01 | 4.228625e-01 | 1.11e-01 |
| DN2LOGO_FPhitp2p_45_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.089501e-01 | 4.175433e-01 | 1.09e-01 |
| DN2LOGO_FPhitp2n_60_2b.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 2.997803e-01 | 4.083587e-01 | 1.09e-01 |
| DN2LOGO_FPhitp2p_60_2b.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 3.052209e-01 | 4.098423e-01 | 1.05e-01 |


### 4.2 Decomposition: flow vs fork operator behaviour

A third farm (job 6016893, 136 items per Hamiltonian: all 34 columns at q = 0, 150, 450, 900)
transforms the **v2 Omegas with `run.compat = "imsrg_fork_79e9330"`** (fork operator behaviour and
writer). Against v1 it shows the flow effect alone (v1 vs v2 Omegas, same operator code); against
v2 the operator-side fork behaviour alone (same Omegas). Per-column tables below (N2LOsat;
DN2LOGO is alike). Result: the scalar columns (FM0, FPhipp0) are untouched by the operator
behaviour (<= 4e-7, the writer's rounding); the even-rank tensors change in a few elements, mostly
one-body (0d3/2 and 0d5/2 diagonal: FPhitp2n 0.358 -> 0.458 at q = 0), the signature of the
missing a == b normal-ordering term (M15-1); the odd ranks change in many two-body elements by
<= 1.4e-2 (AL-F1). The flow part is <= 6.3e-3 (FPhipp0p at q = 0), mostly 1e-4 to 1e-3.

#### flow effect (v2 vs v1 Omegas, both with fork operator behaviour), N2LOsat

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| FM0n | 4 | 5.12e-04 (150) | 3.25e-01 (900) | 282 | 8.34e-02 | 91 | 174 | 0/0 | 8.00e+00 |
| FM0p | 4 | 8.77e-04 (150) | 1.46e+00 (450) | 304 | 1.17e-01 | 106 | 174 | 0/0 | 8.00e+00 |
| FM2n | 4 | 7.20e-04 (150) | 6.51e+00 (150) | 1319 | 3.66e-01 | 634 | 22 | 2/2 | 5.91e-01 |
| FM2p | 4 | 8.50e-04 (150) | 5.53e+00 (450) | 1357 | 5.42e-01 | 646 | 18 | 1/0 | 5.88e-01 |
| FM4n | 4 | 2.89e-04 (450) | 2.86e+00 (150) | 1037 | 1.16e-01 | 411 | 10 | 1/0 | 1.86e-01 |
| FM4p | 4 | 4.61e-04 (450) | 5.07e+00 (450) | 1072 | 1.16e-01 | 446 | 15 | 0/0 | 1.74e-01 |
| FPhipp0n | 4 | 3.28e-03 (0) | 6.08e-01 (450) | 387 | 5.61e-01 | 181 | 80 | 0/0 | 8.44e-01 |
| FPhipp0p | 4 | 6.31e-03 (0) | 5.30e-01 (0) | 403 | 3.36e-01 | 187 | 83 | 0/0 | 8.61e-01 |
| FPhipp2n | 4 | 1.35e-03 (0) | 3.15e+02 (0) | 1672 | 1.01e+00 | 940 | 96 | 0/0 | 1.04e+00 |
| FPhipp2p | 4 | 2.50e-03 (0) | 1.80e+00 (0) | 1691 | 7.37e-01 | 962 | 103 | 0/0 | 1.04e+00 |
| FPhipp4n | 4 | 1.60e-04 (150) | 7.94e-01 (450) | 919 | 1.09e-01 | 286 | 74 | 1/2 | 1.48e-01 |
| FPhipp4p | 4 | 2.50e-04 (150) | 7.01e+00 (150) | 920 | 2.24e-01 | 285 | 83 | 1/1 | 1.50e-01 |
| FPhitp2n | 4 | 1.65e-03 (0) | 3.15e+02 (0) | 1692 | 1.01e+00 | 909 | 112 | 1/2 | 1.27e+00 |
| FPhitp2p | 4 | 3.06e-03 (0) | 2.12e+01 (150) | 1693 | 7.37e-01 | 927 | 123 | 0/0 | 1.27e+00 |
| FPhitp4n | 4 | 1.36e-04 (150) | 1.98e+00 (150) | 930 | 7.58e-02 | 279 | 77 | 2/0 | 1.64e-01 |
| FPhitp4p | 4 | 2.67e-04 (150) | 1.50e+00 (150) | 953 | 1.31e-01 | 290 | 79 | 2/3 | 1.65e-01 |
| FSigmap1n | 4 | 7.18e-04 (0) | 1.50e+01 (0) | 1334 | 3.65e-01 | 557 | 90 | 3/3 | 6.22e-01 |
| FSigmap1p | 4 | 1.22e-03 (0) | 2.36e+01 (150) | 1376 | 6.06e-01 | 589 | 67 | 0/1 | 6.21e-01 |
| FSigmap3n | 4 | 1.68e-04 (150) | 1.22e+00 (450) | 1237 | 8.25e-02 | 308 | 58 | 0/0 | 1.42e-01 |
| FSigmap3p | 4 | 2.92e-04 (150) | 3.56e+00 (150) | 1226 | 1.10e-01 | 317 | 86 | 3/0 | 1.43e-01 |
| FSigmap5n | 4 | 1.11e-04 (450) | 3.68e+00 (450) | 539 | 5.37e-02 | 75 | 29 | 1/1 | 9.43e-02 |
| FSigmap5p | 4 | 1.74e-04 (450) | 1.23e+00 (150) | 568 | 4.71e-02 | 83 | 31 | 2/0 | 9.12e-02 |
| FSigmapp1n | 4 | 5.08e-04 (0) | 5.05e+00 (150) | 1404 | 3.20e-01 | 489 | 60 | 0/0 | 4.39e-01 |
| FSigmapp1p | 4 | 8.64e-04 (0) | 5.03e+00 (0) | 1363 | 5.87e-01 | 492 | 56 | 0/0 | 4.39e-01 |
| FSigmapp3n | 4 | 1.78e-04 (150) | 4.67e+00 (450) | 1246 | 7.64e-02 | 284 | 64 | 1/0 | 1.29e-01 |
| FSigmapp3p | 4 | 3.07e-04 (150) | 1.17e+01 (450) | 1230 | 1.17e-01 | 284 | 79 | 3/2 | 1.31e-01 |
| FSigmapp5n | 4 | 1.01e-04 (450) | 1.02e+00 (450) | 524 | 7.93e-02 | 65 | 33 | 0/0 | 8.47e-02 |
| FSigmapp5p | 4 | 1.58e-04 (450) | 2.15e+01 (450) | 538 | 1.57e-01 | 75 | 42 | 3/4 | 8.19e-02 |
| FDelta1n | 4 | 4.01e-04 (0) | 7.39e+00 (0) | 1174 | 2.90e-01 | 341 | 294 | 0/2 | 6.31e-01 |
| FDelta1p | 4 | 3.50e-04 (0) | 7.96e+01 (0) | 1184 | 2.30e-01 | 376 | 294 | 0/4 | 6.29e-01 |
| FDelta3n | 4 | 4.36e-05 (150) | 4.81e+00 (150) | 950 | 2.84e-02 | 92 | 333 | 3/2 | 4.59e-02 |
| FDelta3p | 4 | 7.42e-05 (150) | 8.31e+00 (150) | 957 | 3.37e-02 | 81 | 320 | 2/1 | 4.65e-02 |
| FDelta5n | 4 | 1.14e-06 (450) | 1.54e-01 (450) | 254 | 4.27e-04 | 0 | 129 | 0/1 | 1.81e-03 |
| FDelta5p | 4 | 1.30e-06 (450) | 5.79e-02 (150) | 272 | 6.02e-04 | 0 | 141 | 0/0 | 1.69e-03 |

#### operator-side compat effect (v2 Omegas, default vs fork operator behaviour), N2LOsat

| column | files | max abs (q) | max rel (q) | n rel>0.001 | max rel, |ref|>=0.001 | n rel>0.001, |ref|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|
| FM0n | 4 | 4.10e-07 (150) | 6.84e-07 (900) | 0 | 5.15e-07 | 0 | 174 | 158/0 | 8.00e+00 |
| FM0p | 4 | 3.30e-07 (150) | 4.89e-07 (150) | 0 | 4.89e-07 | 0 | 173 | 158/0 | 8.00e+00 |
| FM2n | 4 | 3.75e-02 (150) | 1.40e+00 (900) | 12 | 7.56e-01 | 8 | 22 | 0/2 | 5.91e-01 |
| FM2p | 4 | 3.29e-02 (150) | 2.58e+00 (900) | 12 | 7.56e-01 | 8 | 28 | 0/6 | 5.89e-01 |
| FM4n | 4 | 1.09e-02 (450) | 1.10e+00 (900) | 6 | 1.02e-01 | 4 | 11 | 0/1 | 1.86e-01 |
| FM4p | 4 | 8.54e-03 (450) | 7.26e-01 (900) | 6 | 7.99e-02 | 4 | 16 | 0/0 | 1.74e-01 |
| FPhipp0n | 4 | 2.96e-07 (0) | 1.32e-06 (0) | 0 | 1.32e-06 | 0 | 79 | 1/0 | 8.44e-01 |
| FPhipp0p | 4 | 3.97e-07 (150) | 3.42e-06 (900) | 0 | 3.03e-06 | 0 | 81 | 3/0 | 8.61e-01 |
| FPhipp2n | 4 | 8.20e-02 (0) | 3.31e+00 (900) | 16 | 9.13e-01 | 11 | 108 | 0/12 | 1.04e+00 |
| FPhipp2p | 4 | 8.19e-02 (0) | 2.98e+00 (900) | 16 | 8.72e-01 | 11 | 113 | 0/14 | 1.04e+00 |
| FPhipp4n | 4 | 4.24e-03 (450) | 1.12e+00 (900) | 5 | 4.86e-01 | 3 | 74 | 0/3 | 1.48e-01 |
| FPhipp4p | 4 | 3.55e-03 (450) | 9.23e-01 (900) | 6 | 4.70e-01 | 4 | 86 | 0/5 | 1.50e-01 |
| FPhitp2n | 4 | 1.00e-01 (0) | 1.34e+00 (450) | 16 | 9.13e-01 | 11 | 117 | 0/9 | 1.27e+00 |
| FPhitp2p | 4 | 1.00e-01 (0) | 1.73e+00 (450) | 15 | 8.72e-01 | 11 | 133 | 0/14 | 1.27e+00 |
| FPhitp4n | 4 | 6.73e-03 (450) | 5.38e-01 (900) | 6 | 1.88e-01 | 4 | 87 | 0/12 | 1.64e-01 |
| FPhitp4p | 4 | 5.73e-03 (450) | 3.29e-01 (900) | 6 | 2.16e-01 | 4 | 92 | 0/15 | 1.65e-01 |
| FSigmap1n | 4 | 9.42e-03 (150) | 1.86e+02 (0) | 1699 | 1.95e+00 | 728 | 98 | 0/14 | 6.22e-01 |
| FSigmap1p | 4 | 1.10e-02 (150) | 4.85e+01 (0) | 1701 | 2.38e+00 | 729 | 75 | 0/6 | 6.21e-01 |
| FSigmap3n | 4 | 9.83e-03 (450) | 1.84e+01 (150) | 1488 | 4.73e-01 | 407 | 69 | 0/9 | 1.42e-01 |
| FSigmap3p | 4 | 1.04e-02 (450) | 2.59e+01 (450) | 1462 | 4.28e-01 | 414 | 89 | 0/6 | 1.43e-01 |
| FSigmap5n | 4 | 2.43e-03 (450) | 2.66e+01 (150) | 730 | 5.16e-01 | 139 | 31 | 0/2 | 9.43e-02 |
| FSigmap5p | 4 | 2.02e-03 (150) | 3.00e+01 (450) | 723 | 4.19e-01 | 135 | 29 | 0/2 | 9.12e-02 |
| FSigmapp1n | 4 | 6.01e-03 (0) | 1.86e+02 (0) | 1733 | 1.95e+00 | 612 | 67 | 0/7 | 4.39e-01 |
| FSigmapp1p | 4 | 5.65e-03 (0) | 4.85e+01 (0) | 1716 | 2.38e+00 | 618 | 63 | 0/4 | 4.39e-01 |
| FSigmapp3n | 4 | 1.28e-02 (150) | 1.79e+02 (150) | 1479 | 5.80e-01 | 361 | 72 | 0/11 | 1.29e-01 |
| FSigmapp3p | 4 | 1.34e-02 (150) | 2.51e+02 (150) | 1462 | 4.11e-01 | 364 | 87 | 0/11 | 1.31e-01 |
| FSigmapp5n | 4 | 2.19e-03 (450) | 1.34e+01 (150) | 728 | 3.52e-01 | 121 | 35 | 0/3 | 8.47e-02 |
| FSigmapp5p | 4 | 1.96e-03 (450) | 2.15e+01 (450) | 720 | 2.37e-01 | 126 | 46 | 0/7 | 8.19e-02 |
| FDelta1n | 4 | 1.38e-02 (150) | 4.10e+02 (0) | 1433 | 2.93e+00 | 445 | 372 | 0/79 | 6.31e-01 |
| FDelta1p | 4 | 1.30e-02 (150) | 1.74e+02 (0) | 1424 | 1.62e+00 | 468 | 376 | 2/82 | 6.29e-01 |
| FDelta3n | 4 | 2.78e-03 (150) | 6.28e+01 (150) | 1178 | 5.15e-01 | 147 | 392 | 0/65 | 4.59e-02 |
| FDelta3p | 4 | 3.28e-03 (150) | 3.34e+01 (450) | 1167 | 4.65e-01 | 138 | 386 | 0/70 | 4.65e-02 |
| FDelta5n | 4 | 9.68e-04 (450) | 4.44e+01 (450) | 600 | 5.33e-01 | 7 | 161 | 0/32 | 1.82e-03 |
| FDelta5p | 4 | 8.27e-04 (450) | 8.19e+00 (450) | 591 | 5.08e-01 | 4 | 162 | 0/21 | 1.69e-03 |

## 5. v3 vs v2

(pending: v3-v2-N2LOsat)

(pending: v3-v2-DN2LOGO)
