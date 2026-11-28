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
with |ref| >= 1e-3 (with the number of such elements), because the relative maxima over elements
with |ref| >= 1e-6 come from elements just above the cut. They also list the number of compared elements
(zero-, one- and two-body, union of the keys of both files) and of rel-eligible ones (|ref| >= 1e-6);
the per-q tables list the same counts per q.
Jobs and node-hours: `design/benchmarks/al27-jobs.md`, section Lead D.

## 1. Summary

* **Flows** (job 6015996, one node, 47 min): all four converged in the first chain link.
  E0 = -213.956872 (v2 N2LOsat), -217.070296 (v2 DN2LOGO), -214.428763 (v3 N2LOsat, 42 Omegas),
  -216.987333 (v3 DN2LOGO, 40 Omegas) MeV. v2 - v1 = +4.6 / -3.9 keV; v3 - v2 = **-472 / +83 keV**.
* **v3 vs v2 is the Magnus(2)-vs-IMSRG(2) truncation difference of the unsplit variant, not a
  defect or a discretization error of v3** (section 3). An ODE IMSRG(2) flow from the same H_0
  gives -214.444235 / -216.989496 MeV: v3 is 15.5 / 2.2 keV from it (7.0 / 0.6 keV at domega
  0.025), v2 487 / 81 keV. The split threshold 0.25 is converged to ~1 keV (0.1 vs 0.25: 1.1 / 0.06 keV). Time step:
  v3 moves by 8.5 / 1.5 keV from domega 0.1 to 0.025 (three points only: 0.1, 0.05, 0.025); v2 moves by
  0.3 / 3.7 keV from domega 0.1 to 0.05, a single halving (no further point was run, so no
  convergence order is established).
* **v2 vs v1** (section 4): E0 and the scalar columns change little (FM0 <= 9e-4 absolute); the
  tensor columns change by up to 0.10 absolute (FPhitp2/FPhipp2 one-body part at q = 0). A
  decomposition farm (v2 Omegas transformed with the fork's operator behaviour, i.e. with
  `run.compat = "imsrg_fork_79e9330"`) assigns this to the operator side of the compat switch,
  while the flow part (domega 0.2 -> 0.1, same operator code) is <= 6e-3 absolute (N2LOsat;
  1.1e-3 DN2LOGO). The compat switch changes a bundle at once (D72 tensor norm, 9j threshold,
  devel writer, fork operator behaviour); the pattern of the differences (one-body 0d diagonal
  elements for the even ranks, many two-body elements for the odd ranks) is what the fork bugs fixed
  in the default path would produce (design/30 section 6: M15-1, the missing a == b,
  J_bra != J_ket normal-ordering term, for the even ranks; AL-F1, the missing (-1)^L particle-hole
  phase, for Sigma'/Sigma''/Delta), but that attribution is inferred from the pattern: the
  bundle was not split into its parts and no individual fix was switched on or off.
* **v3 vs v2 valence H** (`<stem>.snt` of the flow directories; section 5.3): N2LOsat: single-particle
  energies differ by up to 0.843 MeV, the core energy (zero-body term) by 0.898 MeV, two-body
  elements by up to 0.267 MeV; DN2LOGO: 0.624 / 0.186 / 0.201 MeV. (v2 vs v1, N2LOsat: 7.4e-3 /
  3.5e-3 / 2.5e-2 MeV.) The core energy of v2 is 0.89 MeV above that of the ODE IMSRG(2) flow, v3's
  7 keV below (section 5.2).
* **v3 vs v2 operators** (section 5): most elements differ by more than 1e-3 relative (N2LOsat:
  897341 of 904719 rel-eligible elements, 99.2 %; DN2LOGO 98.8 %); the dominant elements by a median
  of 3.8-8.1 % per column (N2LOsat; DN2LOGO 2.7-5.5 %), one-body parts 0.5-6.6 % (DN2LOGO
  0.5-5.7 %). The median is not monotonic in q (FPhipp0p: 11 / 8.6 / 4.8 / 8.9 % at q = 0 / 300 / 600 / 900;
  lower at 600 than at 300 for 31 of 34 columns, higher at 900 than at 300 for 32); the largest |Δ| relative to the largest |ref| of a file occurs at q >= 300 for 32 of 34
  columns (median q 975). FM0 at q = 0 (particle number) agrees to 2e-16. Cause: the same
  Magnus(2)-vs-IMSRG(2) truncation difference. The ODE IMSRG(2) flow carrying FM0p_150 and
  FPhipp0p_0 agrees with v3 to 0.1-0.3 % (median) and with v2 to 7-9 % (N2LOsat); for DN2LOGO
  (Lead E) 0.08-0.4 % and 4-8 %. Only scalars can be checked this way: the ODE flow cannot carry
  a tensor operator (WE-E-1). v3's operators are
  converged in the split threshold (0.25 vs 0.1: 1-7e-4, median) and in the BCH series; v3's
  time-step error at domega 0.1 is 0.09-1.2 % (median, vs domega 0.025; two time steps only).
  The threshold and time-step checks of 5.2 were run for N2LOsat only.
* **Farms**: v2 2 x 2415 items, 0 failed (job 6016703, 30 nodes, 19.6 min, 9.8 node-hours; 27
  GCD-hours of item work per Hamiltonian, TCM 792 s the longest item); v3: 2 x 2415 items, 0 failed (job 6016892, 80 nodes, 40 min,
  53.3 node-hours; 153 / 145 GCD-hours of item work). A v3 item
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

Steps = last step index of the stage (= rows of flow.txt - 1 = `steps` of results.json). v3 starts from v2's prepared H_0 (`checkpoint.prepared_from`,
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

**Conclusion.** (1) Time step: v3 at domega 0.1 differs by 8.5 keV (N2LOsat) and 1.5 keV (DN2LOGO) from
the domega 0.025 run (0.1 -> 0.05 -> 0.025: N2LOsat -214.428763, -214.437105, -214.437229;
DN2LOGO -216.987333, -216.987045, -216.988880; three points, no extrapolation); v2 differs by
only 0.3 keV (N2LOsat) and 3.7 keV (DN2LOGO) between domega 0.1 and 0.05, a single halving (no
smaller domega was run for v2). (2)
Threshold: 1.0 -> 0.25 -> 0.1 moves E0 by 18 / 1 keV (N2LOsat) and 10 / 0.1 keV (DN2LOGO): 0.25
is converged to ~1 keV, as the owner expected. (3) The split series approaches the ODE IMSRG(2)
result: stage 1 to 0.07 / 0.16 keV, final to 7.0 / 0.6 keV at domega 0.025. For N2LOsat only ~1 keV of the 7.0 keV can be
attributed (the threshold change 0.25 -> 0.1 moves E0 by 1.1 keV); no scan of the ODE tolerance
(1e-6) was run, so the rest is not attributed. The unsplit Magnus(2) is 105 keV off at
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

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 165 | 165 | 2.53e-02 (-) | 2.18e-01 (-) | 96 | 2.18e-01 | 96 of 165 | 0 | 0/0 | 1.18e+02 |
| TCM(flow) | 1 | 150 | 150 | 1.90e-03 (-) | 1.00e+00 (-) | 108 | 1.00e+00 | 108 of 150 | 0 | 1/0 | 6.96e+00 |

H by part:

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 1 | 3.49e-03 | 2.95e-05 | 0 | 0 | 0/0 |
| 1b | 6 | 7.35e-03 | 2.56e-02 | 2 | 0 | 0/0 |
| 2b | 158 | 2.53e-02 | 2.18e-01 | 94 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 11712 | 10999 | 5.19e-04 (135) | 1.86e+00 (705) | 6509 | 1.24e-01 | 1502 of 3290 | 713 | 166/1 | 8.00e+00 |
| FM0p | 71 | 11712 | 10977 | 9.01e-04 (135) | 3.96e+00 (555) | 6899 | 1.66e-01 | 1728 of 3336 | 735 | 160/0 | 8.00e+00 |
| FM2n | 71 | 40391 | 39521 | 4.02e-02 (180) | 7.88e+01 (720) | 29824 | 9.16e+00 | 11438 of 15706 | 870 | 0/78 | 6.63e-01 |
| FM2p | 71 | 40389 | 39520 | 3.54e-02 (180) | 3.95e+01 (915) | 30736 | 2.80e+00 | 11704 of 15724 | 869 | 0/74 | 6.49e-01 |
| FM4n | 71 | 31151 | 30126 | 1.20e-02 (495) | 5.42e+01 (135) | 22335 | 1.52e+00 | 7730 of 11204 | 1025 | 0/175 | 4.00e-01 |
| FM4p | 71 | 31151 | 30078 | 1.01e-02 (510) | 4.98e+01 (315) | 23261 | 1.19e+00 | 8209 of 11124 | 1073 | 0/163 | 3.98e-01 |
| FPhipp0n | 71 | 11714 | 9944 | 3.28e-03 (0) | 8.64e+00 (90) | 6039 | 5.61e-01 | 1743 of 2843 | 1770 | 14/7 | 8.44e-01 |
| FPhipp0p | 71 | 11715 | 9934 | 6.31e-03 (0) | 3.74e+00 (120) | 6460 | 3.36e-01 | 1860 of 2846 | 1781 | 4/0 | 8.61e-01 |
| FPhipp2n | 71 | 40960 | 37137 | 8.19e-02 (0) | 3.15e+02 (0) | 26837 | 5.17e+00 | 10797 of 14465 | 3823 | 1/593 | 1.04e+00 |
| FPhipp2p | 71 | 40963 | 37031 | 8.19e-02 (0) | 3.96e+01 (270) | 27179 | 3.01e+00 | 10877 of 14451 | 3932 | 1/645 | 1.04e+00 |
| FPhipp4n | 71 | 31143 | 28257 | 5.10e-03 (390) | 9.96e+00 (330) | 19928 | 6.48e-01 | 5146 of 8166 | 2886 | 0/436 | 1.91e-01 |
| FPhipp4p | 71 | 31144 | 28102 | 4.33e-03 (345) | 2.61e+01 (90) | 20372 | 6.90e-01 | 5273 of 8188 | 3042 | 0/492 | 1.90e-01 |
| FPhitp2n | 71 | 40961 | 37217 | 1.00e-01 (0) | 3.15e+02 (0) | 27689 | 2.03e+00 | 9831 of 13574 | 3744 | 1/533 | 1.27e+00 |
| FPhitp2p | 71 | 40961 | 37012 | 1.00e-01 (0) | 4.61e+01 (630) | 28090 | 1.63e+00 | 10013 of 13598 | 3949 | 0/553 | 1.27e+00 |
| FPhitp4n | 71 | 31145 | 28478 | 9.46e-03 (360) | 3.48e+01 (120) | 20297 | 6.29e-01 | 4813 of 7477 | 2667 | 0/353 | 2.06e-01 |
| FPhitp4p | 71 | 31144 | 28202 | 7.64e-03 (360) | 2.53e+01 (795) | 20564 | 6.71e-01 | 4926 of 7508 | 2942 | 2/394 | 2.05e-01 |
| FSigmap1n | 71 | 32233 | 29734 | 9.39e-03 (150) | 1.15e+03 (0) | 29298 | 3.58e+00 | 8066 of 8211 | 2499 | 0/301 | 6.22e-01 |
| FSigmap1p | 71 | 32233 | 29886 | 1.10e-02 (150) | 2.69e+03 (30) | 29408 | 4.42e+00 | 8165 of 8358 | 2347 | 2/321 | 6.21e-01 |
| FSigmap3n | 71 | 37031 | 34367 | 1.02e-02 (420) | 2.48e+02 (135) | 33796 | 1.69e+00 | 7015 of 7233 | 2664 | 3/377 | 1.58e-01 |
| FSigmap3p | 71 | 37029 | 34160 | 1.09e-02 (420) | 2.59e+02 (180) | 33493 | 9.69e-01 | 7066 of 7333 | 2869 | 3/410 | 1.57e-01 |
| FSigmap5n | 71 | 18125 | 16432 | 3.60e-03 (345) | 1.60e+02 (345) | 16041 | 1.98e+00 | 3085 of 3212 | 1693 | 0/373 | 1.46e-01 |
| FSigmap5p | 71 | 18128 | 16447 | 2.92e-03 (330) | 1.93e+02 (255) | 16020 | 1.79e+00 | 3072 of 3214 | 1681 | 0/344 | 1.45e-01 |
| FSigmapp1n | 71 | 32229 | 29947 | 7.69e-03 (300) | 4.04e+02 (90) | 29489 | 2.09e+00 | 7337 of 7516 | 2282 | 0/239 | 4.39e-01 |
| FSigmapp1p | 71 | 32233 | 29892 | 8.01e-03 (285) | 3.57e+02 (15) | 29375 | 3.17e+00 | 7550 of 7781 | 2341 | 1/282 | 4.39e-01 |
| FSigmapp3n | 71 | 37030 | 34377 | 1.38e-02 (180) | 4.53e+02 (315) | 33826 | 1.29e+00 | 6490 of 6689 | 2653 | 1/365 | 1.49e-01 |
| FSigmapp3p | 71 | 37031 | 34301 | 1.44e-02 (180) | 1.74e+02 (300) | 33652 | 1.39e+00 | 6442 of 6676 | 2730 | 1/369 | 1.49e-01 |
| FSigmapp5n | 71 | 18125 | 16484 | 2.35e-03 (420) | 8.50e+01 (165) | 16140 | 5.65e-01 | 2843 of 2971 | 1641 | 0/371 | 1.32e-01 |
| FSigmapp5p | 71 | 18129 | 16373 | 2.27e-03 (195) | 1.57e+02 (435) | 15967 | 3.46e-01 | 2806 of 2959 | 1756 | 0/431 | 1.32e-01 |
| FDelta1n | 71 | 32194 | 25291 | 1.59e-02 (195) | 2.04e+02 (150) | 24901 | 3.05e+00 | 4399 of 4527 | 6903 | 6/2500 | 6.31e-01 |
| FDelta1p | 71 | 32194 | 25075 | 1.54e-02 (195) | 1.46e+03 (15) | 24711 | 2.14e+00 | 4577 of 4655 | 7119 | 8/2764 | 6.29e-01 |
| FDelta3n | 71 | 37007 | 29103 | 3.09e-03 (315) | 2.05e+02 (225) | 28473 | 6.42e-01 | 2239 of 2392 | 7904 | 3/2696 | 5.97e-02 |
| FDelta3p | 71 | 37004 | 29199 | 3.41e-03 (135) | 8.63e+01 (240) | 28602 | 5.97e-01 | 2317 of 2390 | 7805 | 2/2755 | 5.93e-02 |
| FDelta5n | 71 | 18094 | 13904 | 1.16e-03 (375) | 1.08e+02 (345) | 13426 | 5.35e-01 | 331 of 366 | 4190 | 1/1541 | 2.41e-03 |
| FDelta5p | 71 | 18099 | 13905 | 9.89e-04 (375) | 3.79e+01 (405) | 13405 | 5.12e-01 | 353 of 358 | 4194 | 2/1592 | 2.47e-03 |
| TCM | 1 | 165 | 165 | 1.90e-03 (-) | 1.00e+00 (-) | 123 | 1.00e+00 | 109 of 151 | 0 | 16/0 | 6.96e+00 |

###### Per q

| q | files | elements | rel-eligible | n rel>0.001 | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|---|---|---|
| 0 | 34 | 5706 | 5358 | 4861 | 1.00e-01 | FPhitp2n | 1.15e+03 | FSigmap1n |
| 15 | 34 | 14199 | 11620 | 10315 | 9.98e-02 | FPhitp2n | 1.46e+03 | FDelta1p |
| 30 | 34 | 14243 | 12959 | 11514 | 9.81e-02 | FPhitp2p | 2.69e+03 | FSigmap1p |
| 45 | 34 | 14244 | 13688 | 12230 | 9.54e-02 | FPhitp2p | 2.45e+02 | FDelta1p |
| 60 | 34 | 14244 | 13960 | 12503 | 9.15e-02 | FPhitp2p | 3.46e+02 | FDelta1p |
| 75 | 34 | 14244 | 14088 | 12597 | 8.66e-02 | FPhitp2p | 1.67e+02 | FSigmapp1n |
| 90 | 34 | 14244 | 14132 | 12615 | 8.06e-02 | FPhitp2p | 4.62e+02 | FSigmap1n |
| 105 | 34 | 14244 | 14157 | 12630 | 7.38e-02 | FPhitp2p | 3.20e+02 | FSigmapp1n |
| 120 | 34 | 14244 | 14169 | 12624 | 6.63e-02 | FPhitp2p | 1.36e+02 | FSigmapp1p |
| 135 | 34 | 14244 | 14176 | 12593 | 5.83e-02 | FPhitp2n | 5.16e+02 | FSigmap1n |
| 150 | 34 | 14244 | 14181 | 12537 | 5.01e-02 | FPhitp2n | 2.04e+02 | FSigmapp3n |
| 165 | 34 | 14244 | 14193 | 12627 | 4.19e-02 | FPhitp2n | 2.56e+02 | FSigmapp1p |
| 180 | 34 | 14244 | 14194 | 12625 | 4.02e-02 | FM2n | 2.59e+02 | FSigmap3p |
| 195 | 34 | 14244 | 14194 | 12597 | 3.95e-02 | FM2n | 1.77e+02 | FDelta1p |
| 210 | 34 | 14244 | 14196 | 12602 | 3.74e-02 | FM2n | 1.83e+02 | FSigmap1p |
| 225 | 34 | 14244 | 14199 | 12577 | 3.42e-02 | FM2n | 2.05e+02 | FDelta3n |
| 240 | 34 | 14244 | 14206 | 12590 | 3.05e-02 | FM2n | 1.41e+02 | FSigmap5p |
| 255 | 34 | 14244 | 14206 | 12589 | 3.08e-02 | FM2n | 2.71e+02 | FSigmapp3n |
| 270 | 34 | 14244 | 14195 | 12546 | 3.00e-02 | FM2n | 2.02e+02 | FSigmap1n |
| 285 | 34 | 14244 | 14204 | 12570 | 2.83e-02 | FM2n | 9.71e+01 | FSigmapp3p |
| 300 | 34 | 14244 | 14196 | 12558 | 2.59e-02 | FM2n | 1.74e+02 | FSigmapp3p |
| 315 | 34 | 14244 | 14198 | 12572 | 2.31e-02 | FM2n | 4.53e+02 | FSigmapp3n |
| 330 | 34 | 14244 | 14206 | 12566 | 2.24e-02 | FM2n | 1.69e+02 | FSigmap3n |
| 345 | 34 | 14244 | 14203 | 12521 | 2.28e-02 | FM2n | 1.60e+02 | FSigmap5n |
| 360 | 34 | 14244 | 14203 | 12498 | 2.25e-02 | FM2n | 1.46e+02 | FSigmapp3p |
| 375 | 34 | 14244 | 14194 | 12490 | 2.15e-02 | FM2n | 9.53e+01 | FSigmapp3p |
| 390 | 34 | 14244 | 14195 | 12452 | 2.00e-02 | FM2n | 5.95e+01 | FSigmap1n |
| 405 | 34 | 14244 | 14184 | 12396 | 1.81e-02 | FM2n | 1.19e+02 | FSigmap5n |
| 420 | 34 | 14244 | 14186 | 12363 | 1.57e-02 | FM2n | 6.10e+01 | FDelta1p |
| 435 | 34 | 14244 | 14172 | 12312 | 1.32e-02 | FM2n | 1.57e+02 | FSigmapp5p |
| 450 | 34 | 14244 | 14163 | 12286 | 1.18e-02 | FSigmapp3p | 5.94e+01 | FSigmapp3n |
| 465 | 34 | 14244 | 14157 | 12224 | 1.15e-02 | FM4n | 1.06e+02 | FSigmap1p |
| 480 | 34 | 14244 | 14166 | 12221 | 1.19e-02 | FM4n | 6.37e+01 | FSigmap1p |
| 495 | 34 | 14244 | 14161 | 12164 | 1.20e-02 | FM4n | 7.37e+01 | FSigmap3n |
| 510 | 34 | 14244 | 14152 | 12122 | 1.18e-02 | FM4n | 9.52e+01 | FSigmapp1p |
| 525 | 34 | 14244 | 14150 | 12057 | 1.15e-02 | FM4n | 1.10e+02 | FSigmap5p |
| 540 | 34 | 14244 | 14149 | 11995 | 1.09e-02 | FM4n | 9.66e+01 | FSigmap5p |
| 555 | 34 | 14244 | 14133 | 11919 | 1.01e-02 | FM4n | 1.33e+02 | FSigmapp3p |
| 570 | 34 | 14244 | 14121 | 11815 | 9.21e-03 | FM4n | 4.21e+01 | FSigmap1n |
| 585 | 34 | 14244 | 14094 | 11689 | 8.22e-03 | FM4n | 4.46e+01 | FSigmapp3p |
| 600 | 34 | 14244 | 14077 | 11551 | 7.19e-03 | FM4n | 5.80e+01 | FSigmapp5p |
| 615 | 34 | 14244 | 14044 | 11450 | 6.15e-03 | FM4n | 7.98e+01 | FSigmap5p |
| 630 | 34 | 14244 | 14032 | 11395 | 5.15e-03 | FM4n | 4.91e+01 | FSigmap5p |
| 645 | 34 | 14244 | 14023 | 11367 | 4.21e-03 | FM4n | 5.94e+01 | FSigmapp3n |
| 660 | 34 | 14242 | 13950 | 11284 | 3.35e-03 | FM4n | 6.03e+01 | FSigmapp3p |
| 675 | 34 | 14244 | 13895 | 11286 | 2.64e-03 | FM2n | 6.04e+01 | FSigmap3n |
| 690 | 34 | 14244 | 13883 | 11307 | 2.16e-03 | FM2n | 4.20e+01 | FSigmap5n |
| 705 | 34 | 14244 | 13808 | 11260 | 1.70e-03 | FM2n | 6.79e+01 | FSigmap5p |
| 720 | 34 | 14243 | 13740 | 11159 | 1.28e-03 | FM2n | 1.26e+02 | FPhipp2n |
| 735 | 34 | 14244 | 13634 | 11076 | 9.23e-04 | FSigmapp3p | 4.76e+01 | FPhipp2n |
| 750 | 34 | 14244 | 13565 | 11081 | 8.17e-04 | FSigmapp3p | 8.15e+01 | FPhipp2n |
| 765 | 34 | 14243 | 13425 | 11014 | 7.16e-04 | FSigmapp3p | 4.82e+01 | FPhipp2n |
| 780 | 34 | 14243 | 13322 | 10984 | 6.32e-04 | FM2p | 3.33e+01 | FSigmap5n |
| 795 | 34 | 14241 | 13138 | 10926 | 6.00e-04 | FM2p | 2.53e+01 | FPhitp4p |
| 810 | 34 | 14243 | 12942 | 10785 | 5.47e-04 | FM2p | 2.02e+01 | FDelta3p |
| 825 | 34 | 14243 | 12728 | 10629 | 4.82e-04 | FM2p | 1.42e+01 | FSigmapp5n |
| 840 | 34 | 14244 | 12438 | 10461 | 4.12e-04 | FM2p | 4.04e+01 | FSigmap3n |
| 855 | 34 | 14243 | 12083 | 10190 | 3.41e-04 | FM2p | 2.30e+01 | FSigmapp5n |
| 870 | 34 | 14240 | 11735 | 9935 | 2.75e-04 | FM2p | 2.12e+01 | FSigmap3n |
| 885 | 34 | 14244 | 11301 | 9603 | 2.15e-04 | FM2p | 1.67e+01 | FSigmap1n |
| 900 | 34 | 14241 | 10712 | 9077 | 1.65e-04 | FM2p | 1.29e+01 | FSigmap1n |
| 915 | 34 | 14233 | 10220 | 8656 | 1.24e-04 | FM2p | 3.95e+01 | FM2p |
| 930 | 34 | 14239 | 9582 | 8155 | 9.09e-05 | FM2n | 2.56e+01 | FSigmapp3p |
| 945 | 34 | 14240 | 8965 | 7595 | 6.98e-05 | FM2n | 1.33e+01 | FPhipp2p |
| 960 | 34 | 14233 | 8290 | 6997 | 5.14e-05 | FM2n | 1.24e+01 | FSigmap3p |
| 975 | 34 | 14230 | 7588 | 6380 | 3.60e-05 | FM2n | 4.85e+00 | FSigmapp1p |
| 990 | 34 | 14226 | 6923 | 5788 | 2.53e-05 | FSigmap3p | 1.05e+01 | FSigmap3p |
| 1005 | 34 | 14206 | 6185 | 5100 | 1.91e-05 | FSigmap3p | 5.38e+00 | FSigmap3p |
| 1020 | 34 | 14216 | 5311 | 4364 | 1.41e-05 | FSigmap3p | 3.44e+00 | FSigmap3p |
| 1035 | 34 | 14200 | 4505 | 3656 | 1.05e-05 | FM2n | 2.57e+00 | FSigmap1p |
| 1050 | 34 | 14199 | 3710 | 2969 | 8.69e-06 | FM2n | 6.83e+00 | FM2n |

###### By part

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 2415 | 1.52e-04 | 2.38e-02 | 12 | 2130 | 0/0 |
| 1b | 22844 | 1.00e-01 | 4.04e+02 | 15420 | 816 | 10/131 |
| 2b | 977410 | 8.00e-03 | 2.69e+03 | 757733 | 98146 | 388/22397 |

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

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 165 | 165 | 7.35e-03 (-) | 1.56e-01 (-) | 59 | 1.56e-01 | 59 of 165 | 0 | 0/0 | 1.16e+02 |
| TCM(flow) | 1 | 157 | 157 | 1.03e-03 (-) | 1.87e-01 (-) | 104 | 1.87e-01 | 103 of 156 | 0 | 0/0 | 7.08e+00 |

H by part:

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 1 | 2.48e-04 | 2.13e-06 | 0 | 0 | 0/0 |
| 1b | 6 | 1.86e-03 | 2.70e-03 | 1 | 0 | 0/0 |
| 2b | 158 | 7.35e-03 | 1.56e-01 | 58 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 11712 | 10615 | 2.30e-04 (135) | 1.02e+00 (345) | 4709 | 1.46e-01 | 1102 of 3415 | 1097 | 163/0 | 8.00e+00 |
| FM0p | 71 | 11712 | 10637 | 4.98e-04 (120) | 5.28e+00 (135) | 5964 | 2.39e-01 | 1565 of 3466 | 1075 | 162/0 | 8.00e+00 |
| FM2n | 71 | 40389 | 38723 | 4.05e-02 (180) | 4.25e+01 (180) | 23503 | 3.88e+00 | 9345 of 15996 | 1666 | 0/211 | 6.63e-01 |
| FM2p | 71 | 40389 | 38902 | 3.67e-02 (180) | 5.77e+01 (240) | 27210 | 2.94e+00 | 10553 of 16054 | 1487 | 3/124 | 6.51e-01 |
| FM4n | 71 | 31147 | 29228 | 1.16e-02 (315) | 1.77e+01 (135) | 15135 | 1.17e+00 | 4970 of 11280 | 1919 | 0/250 | 4.20e-01 |
| FM4p | 71 | 31150 | 29453 | 1.02e-02 (315) | 1.41e+01 (660) | 19373 | 7.59e-01 | 6613 of 11352 | 1697 | 0/230 | 4.08e-01 |
| FPhipp0n | 71 | 11715 | 9526 | 5.44e-04 (0) | 1.11e+01 (285) | 4356 | 3.65e-01 | 1250 of 2937 | 2189 | 9/5 | 8.05e-01 |
| FPhipp0p | 71 | 11715 | 9497 | 7.13e-04 (0) | 1.53e+02 (30) | 4782 | 4.67e-01 | 1401 of 2912 | 2218 | 3/1 | 8.31e-01 |
| FPhipp2n | 71 | 40950 | 35214 | 9.67e-02 (0) | 1.70e+01 (765) | 19513 | 5.69e+00 | 8590 of 14819 | 5736 | 0/1272 | 1.00e+00 |
| FPhipp2p | 71 | 40957 | 35169 | 9.28e-02 (0) | 2.99e+01 (675) | 22635 | 6.50e+00 | 9776 of 14680 | 5788 | 0/1220 | 1.00e+00 |
| FPhipp4n | 71 | 31140 | 26712 | 3.72e-03 (345) | 1.83e+01 (780) | 11537 | 5.75e-01 | 2785 of 8255 | 4428 | 0/1049 | 1.88e-01 |
| FPhipp4p | 71 | 31147 | 26665 | 3.45e-03 (345) | 2.30e+01 (270) | 14655 | 6.61e-01 | 4057 of 8217 | 4482 | 1/956 | 1.88e-01 |
| FPhitp2n | 71 | 40954 | 35332 | 1.18e-01 (0) | 3.65e+01 (780) | 19599 | 1.47e+00 | 7579 of 14010 | 5622 | 0/1347 | 1.23e+00 |
| FPhitp2p | 71 | 40952 | 35980 | 1.14e-01 (0) | 8.20e+00 (690) | 23426 | 1.58e+00 | 8916 of 14067 | 4972 | 1/1065 | 1.23e+00 |
| FPhitp4n | 71 | 31142 | 27025 | 7.82e-03 (345) | 1.45e+01 (270) | 12569 | 5.66e-01 | 2443 of 7548 | 4117 | 2/915 | 2.04e-01 |
| FPhitp4p | 71 | 31143 | 27585 | 6.19e-03 (345) | 4.82e+01 (660) | 16039 | 6.90e-01 | 3502 of 7554 | 3558 | 2/758 | 2.03e-01 |
| FSigmap1n | 71 | 32227 | 28807 | 1.13e-02 (0) | 3.92e+02 (45) | 28383 | 3.71e+00 | 8392 of 8545 | 3420 | 0/553 | 6.29e-01 |
| FSigmap1p | 71 | 32232 | 28902 | 9.12e-03 (165) | 6.98e+02 (15) | 28527 | 5.09e+00 | 8553 of 8717 | 3330 | 1/456 | 6.27e-01 |
| FSigmap3n | 71 | 37028 | 33178 | 8.11e-03 (420) | 4.13e+02 (225) | 32576 | 1.17e+00 | 7297 of 7545 | 3850 | 2/657 | 1.58e-01 |
| FSigmap3p | 71 | 37026 | 33282 | 9.52e-03 (420) | 2.19e+02 (210) | 32543 | 1.04e+00 | 7421 of 7689 | 3744 | 5/699 | 1.58e-01 |
| FSigmap5n | 71 | 18121 | 15808 | 3.68e-03 (345) | 1.16e+02 (330) | 15338 | 7.32e-01 | 3101 of 3265 | 2313 | 1/530 | 1.46e-01 |
| FSigmap5p | 71 | 18127 | 15825 | 3.15e-03 (210) | 9.58e+01 (270) | 15365 | 6.50e-01 | 3152 of 3284 | 2302 | 0/517 | 1.46e-01 |
| FSigmapp1n | 71 | 32234 | 28968 | 9.09e-03 (105) | 1.14e+03 (75) | 28546 | 2.69e+00 | 7805 of 7933 | 3266 | 3/550 | 4.45e-01 |
| FSigmapp1p | 71 | 32227 | 29048 | 8.97e-03 (105) | 6.63e+02 (105) | 28623 | 3.47e+00 | 7830 of 8000 | 3179 | 1/465 | 4.44e-01 |
| FSigmapp3n | 71 | 37027 | 33182 | 1.23e-02 (180) | 1.99e+02 (90) | 32578 | 1.18e+00 | 6811 of 6996 | 3845 | 2/718 | 1.50e-01 |
| FSigmapp3p | 71 | 37027 | 33466 | 1.38e-02 (180) | 2.06e+02 (165) | 32821 | 1.39e+00 | 6781 of 7023 | 3561 | 1/645 | 1.50e-01 |
| FSigmapp5n | 71 | 18122 | 15791 | 1.97e-03 (195) | 1.30e+02 (300) | 15258 | 4.63e-01 | 2798 of 2955 | 2331 | 3/573 | 1.33e-01 |
| FSigmapp5p | 71 | 18126 | 15860 | 2.69e-03 (195) | 1.76e+02 (285) | 15431 | 4.64e-01 | 2788 of 2906 | 2266 | 1/562 | 1.32e-01 |
| FDelta1n | 71 | 32160 | 24568 | 1.38e-02 (195) | 6.29e+02 (105) | 24139 | 3.03e+00 | 4540 of 4656 | 7592 | 2/3862 | 6.36e-01 |
| FDelta1p | 71 | 32142 | 24455 | 1.46e-02 (195) | 5.30e+02 (30) | 24100 | 3.05e+00 | 4681 of 4811 | 7687 | 2/3535 | 6.35e-01 |
| FDelta3n | 71 | 36960 | 28131 | 3.19e-03 (300) | 1.22e+02 (525) | 27397 | 6.23e-01 | 2242 of 2427 | 8829 | 0/4339 | 6.05e-02 |
| FDelta3p | 71 | 36959 | 28007 | 2.77e-03 (345) | 1.62e+02 (255) | 27201 | 6.32e-01 | 2233 of 2477 | 8952 | 0/3979 | 6.02e-02 |
| FDelta5n | 71 | 18085 | 13274 | 1.08e-03 (360) | 1.38e+02 (360) | 12747 | 5.48e-01 | 367 of 383 | 4811 | 0/2294 | 2.32e-03 |
| FDelta5p | 71 | 18081 | 13231 | 9.16e-04 (375) | 2.43e+01 (210) | 12747 | 5.28e-01 | 361 of 375 | 4850 | 1/2367 | 2.42e-03 |
| TCM | 1 | 165 | 165 | 1.03e-03 (-) | 1.00e+00 (-) | 112 | 1.87e-01 | 103 of 156 | 0 | 8/0 | 7.08e+00 |

###### Per q

| q | files | elements | rel-eligible | n rel>0.001 | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|---|---|---|
| 0 | 34 | 5706 | 5359 | 4462 | 1.18e-01 | FPhitp2n | 2.66e+02 | FDelta1n |
| 15 | 34 | 14189 | 11584 | 9446 | 1.18e-01 | FPhitp2n | 6.98e+02 | FSigmap1p |
| 30 | 34 | 14243 | 12973 | 10630 | 1.16e-01 | FPhitp2n | 5.30e+02 | FDelta1p |
| 45 | 34 | 14244 | 13685 | 11310 | 1.13e-01 | FPhitp2n | 5.24e+02 | FSigmapp1n |
| 60 | 34 | 14244 | 13947 | 11549 | 1.09e-01 | FPhitp2n | 3.23e+02 | FSigmap1n |
| 75 | 34 | 14244 | 14077 | 11657 | 1.03e-01 | FPhitp2n | 1.14e+03 | FSigmapp1n |
| 90 | 34 | 14244 | 14126 | 11680 | 9.69e-02 | FPhitp2n | 3.71e+02 | FDelta1n |
| 105 | 34 | 14244 | 14155 | 11671 | 8.97e-02 | FPhitp2n | 6.63e+02 | FSigmapp1p |
| 120 | 34 | 14244 | 14171 | 11654 | 8.19e-02 | FPhitp2n | 4.72e+02 | FDelta1p |
| 135 | 34 | 14244 | 14186 | 11694 | 7.36e-02 | FPhitp2n | 1.33e+02 | FSigmap3p |
| 150 | 34 | 14244 | 14195 | 11673 | 6.50e-02 | FPhitp2n | 2.47e+02 | FSigmap1n |
| 165 | 34 | 14244 | 14199 | 11622 | 5.63e-02 | FPhitp2n | 2.13e+02 | FDelta1n |
| 180 | 34 | 14244 | 14206 | 11628 | 4.77e-02 | FPhitp2n | 2.26e+02 | FDelta1n |
| 195 | 34 | 14244 | 14200 | 11668 | 3.96e-02 | FM2n | 1.72e+02 | FSigmapp1n |
| 210 | 34 | 14244 | 14200 | 11673 | 3.75e-02 | FM2n | 2.19e+02 | FSigmap3p |
| 225 | 34 | 14244 | 14206 | 11660 | 3.42e-02 | FM2n | 4.13e+02 | FSigmap3n |
| 240 | 34 | 14244 | 14200 | 11624 | 3.11e-02 | FM2n | 1.00e+02 | FSigmap3p |
| 255 | 34 | 14244 | 14201 | 11601 | 3.14e-02 | FM2n | 1.71e+02 | FSigmap3n |
| 270 | 34 | 14244 | 14204 | 11671 | 3.05e-02 | FM2n | 3.19e+02 | FSigmap1p |
| 285 | 34 | 14244 | 14199 | 11613 | 2.88e-02 | FM2n | 1.98e+02 | FSigmap3n |
| 300 | 34 | 14244 | 14196 | 11583 | 2.63e-02 | FM2n | 1.61e+02 | FSigmap3n |
| 315 | 34 | 14244 | 14193 | 11596 | 2.34e-02 | FM2n | 9.74e+01 | FDelta1p |
| 330 | 34 | 14244 | 14189 | 11592 | 2.03e-02 | FM2n | 1.16e+02 | FSigmap5n |
| 345 | 34 | 14244 | 14197 | 11561 | 1.72e-02 | FM2n | 9.50e+01 | FSigmap1n |
| 360 | 34 | 14244 | 14195 | 11492 | 1.67e-02 | FM2n | 1.38e+02 | FDelta5n |
| 375 | 34 | 14244 | 14187 | 11493 | 1.64e-02 | FM2n | 1.65e+02 | FSigmapp3n |
| 390 | 34 | 14244 | 14187 | 11516 | 1.56e-02 | FM2n | 9.90e+01 | FSigmap5n |
| 405 | 34 | 14244 | 14191 | 11528 | 1.45e-02 | FM2n | 1.24e+02 | FSigmapp3p |
| 420 | 34 | 14244 | 14190 | 11549 | 1.30e-02 | FM2n | 1.68e+02 | FSigmap1n |
| 435 | 34 | 14244 | 14161 | 11527 | 1.13e-02 | FM2n | 7.75e+01 | FSigmapp1n |
| 450 | 34 | 14244 | 14169 | 11527 | 1.03e-02 | FSigmapp3p | 9.52e+01 | FSigmapp3p |
| 465 | 34 | 14244 | 14170 | 11511 | 9.49e-03 | FSigmapp3p | 7.53e+01 | FSigmap3n |
| 480 | 34 | 14244 | 14158 | 11429 | 8.60e-03 | FSigmapp3p | 5.54e+01 | FSigmapp1n |
| 495 | 34 | 14244 | 14140 | 11358 | 7.68e-03 | FSigmapp3p | 1.25e+02 | FSigmapp1p |
| 510 | 34 | 14244 | 14146 | 11314 | 6.84e-03 | FSigmap3p | 7.18e+01 | FSigmapp5n |
| 525 | 34 | 14244 | 14131 | 11196 | 6.14e-03 | FSigmap3p | 1.33e+02 | FSigmapp3p |
| 540 | 34 | 14244 | 14140 | 11119 | 5.99e-03 | FM4n | 9.47e+01 | FSigmap5p |
| 555 | 34 | 14244 | 14119 | 11009 | 5.93e-03 | FM4n | 7.97e+01 | FSigmapp1p |
| 570 | 34 | 14244 | 14117 | 10940 | 5.74e-03 | FM4n | 4.83e+01 | FSigmapp3p |
| 585 | 34 | 14244 | 14104 | 10839 | 5.45e-03 | FM4n | 4.92e+01 | FSigmapp3n |
| 600 | 34 | 14244 | 14088 | 10704 | 5.07e-03 | FM4n | 4.53e+01 | FSigmap3p |
| 615 | 34 | 14244 | 14072 | 10598 | 4.63e-03 | FM4n | 5.31e+01 | FSigmap3p |
| 630 | 34 | 14244 | 14038 | 10457 | 4.16e-03 | FM4n | 4.15e+01 | FSigmap3n |
| 645 | 34 | 14244 | 14027 | 10324 | 3.68e-03 | FM4n | 5.28e+01 | FSigmap3p |
| 660 | 34 | 14244 | 13968 | 10226 | 3.20e-03 | FM4n | 7.78e+01 | FSigmapp3n |
| 675 | 34 | 14244 | 13927 | 10172 | 2.74e-03 | FM4n | 4.81e+01 | FSigmapp3n |
| 690 | 34 | 14244 | 13894 | 10059 | 2.30e-03 | FM4n | 4.02e+01 | FSigmapp5n |
| 705 | 34 | 14243 | 13810 | 9924 | 1.91e-03 | FM4n | 3.23e+01 | FSigmapp5p |
| 720 | 34 | 14242 | 13703 | 9820 | 1.56e-03 | FM4n | 2.48e+01 | FSigmapp3p |
| 735 | 34 | 14244 | 13602 | 9659 | 1.25e-03 | FM4n | 5.72e+01 | FSigmap5p |
| 750 | 34 | 14244 | 13459 | 9539 | 9.91e-04 | FM4n | 1.95e+01 | FSigmapp3n |
| 765 | 34 | 14244 | 13276 | 9452 | 7.70e-04 | FM4n | 2.14e+01 | FSigmapp3n |
| 780 | 34 | 14244 | 13083 | 9257 | 5.88e-04 | FM4n | 3.65e+01 | FPhitp2n |
| 795 | 34 | 14243 | 12808 | 9095 | 4.41e-04 | FM4n | 6.31e+01 | FSigmapp5p |
| 810 | 34 | 14242 | 12462 | 8929 | 3.24e-04 | FM4n | 1.89e+01 | FSigmap5n |
| 825 | 34 | 14243 | 12075 | 8693 | 2.33e-04 | FM4n | 3.41e+01 | FSigmapp3p |
| 840 | 34 | 14243 | 11592 | 8316 | 1.77e-04 | FM2p | 1.27e+01 | FPhipp2n |
| 855 | 34 | 14237 | 10974 | 7814 | 1.38e-04 | FM2p | 9.57e+00 | FSigmap5p |
| 870 | 34 | 14238 | 10381 | 7382 | 1.06e-04 | FM2n | 1.20e+01 | FPhipp2n |
| 885 | 34 | 14236 | 9734 | 6923 | 8.59e-05 | FM2n | 6.85e+00 | FPhipp4p |
| 900 | 34 | 14241 | 9107 | 6516 | 6.89e-05 | FM2n | 1.76e+01 | FM2p |
| 915 | 34 | 14234 | 8464 | 6164 | 5.47e-05 | FM2n | 7.07e+00 | FSigmap3n |
| 930 | 34 | 14231 | 7691 | 5626 | 4.31e-05 | FM2n | 7.48e+00 | FSigmapp3n |
| 945 | 34 | 14237 | 6840 | 5024 | 3.37e-05 | FM2n | 1.14e+01 | FSigmapp3n |
| 960 | 34 | 14224 | 6016 | 4479 | 2.61e-05 | FM2n | 4.17e+00 | FPhitp2n |
| 975 | 34 | 14220 | 5184 | 3876 | 2.00e-05 | FM2n | 4.15e+00 | FSigmap1n |
| 990 | 34 | 14197 | 4289 | 3232 | 1.52e-05 | FM2n | 3.36e+00 | FSigmap1n |
| 1005 | 34 | 14189 | 3507 | 2630 | 1.24e-05 | FM4p | 2.96e+00 | FSigmap1n |
| 1020 | 34 | 14183 | 2794 | 2059 | 1.06e-05 | FM4p | 1.48e+00 | FM2n |
| 1035 | 34 | 14148 | 2091 | 1471 | 8.97e-06 | FM4p | 1.66e+00 | FM2n |
| 1050 | 34 | 14104 | 1537 | 1040 | 7.45e-06 | FM4p | 1.61e+00 | FM4p |

###### By part

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 2415 | 3.79e-04 | 4.38e-01 | 43 | 2135 | 0/0 |
| 1b | 22837 | 1.18e-01 | 1.99e+02 | 14909 | 1306 | 12/249 |
| 2b | 977138 | 7.13e-03 | 1.14e+03 | 664485 | 128738 | 367/36455 |

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
behaviour (<= 4e-7, the writer's rounding; DN2LOGO <= 2.6e-6); the even-rank tensors change in a few
elements, mostly one-body (0d3/2 and 0d5/2 diagonal: FPhitp2n 0.358 -> 0.458 at q = 0), which is
the signature of the missing a == b normal-ordering term (M15-1); the odd ranks change in many
two-body elements by <= 1.4e-2 (AL-F1's missing phase would act like this). Both attributions are
inferred from the pattern: the compat switch changes the D72 tensor norm, the 9j threshold, the
writer and the fork operator behaviour together, and the decomposition farm did not separate them.
The flow part is <= 6.3e-3 (FPhipp0p at q = 0), mostly 1e-4 to 1e-3. The tables below are N2LOsat;
for DN2LOGO the decomposition farm (136 items) gives the same pattern (operator side: largest absolute
difference 0.12 for FPhitp2p/n at q = 0, FPhipp0 <= 5e-7, FM0 <= 2.6e-6; flow part: <= 1.1e-3) but its
tables are not reproduced here.

#### flow effect (v2 vs v1 Omegas, both with fork operator behaviour), N2LOsat

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 4 | 657 | 483 | 5.12e-04 (150) | 3.25e-01 (900) | 282 | 8.34e-02 | 91 of 195 | 174 | 0/0 | 8.00e+00 |
| FM0p | 4 | 657 | 483 | 8.77e-04 (150) | 1.46e+00 (450) | 304 | 1.17e-01 | 106 of 197 | 174 | 0/0 | 8.00e+00 |
| FM2n | 4 | 1732 | 1710 | 7.20e-04 (150) | 6.51e+00 (150) | 1319 | 3.66e-01 | 634 of 860 | 22 | 2/2 | 5.91e-01 |
| FM2p | 4 | 1727 | 1709 | 8.50e-04 (150) | 5.53e+00 (450) | 1357 | 5.42e-01 | 646 of 863 | 18 | 1/0 | 5.88e-01 |
| FM4n | 4 | 1336 | 1326 | 2.89e-04 (450) | 2.86e+00 (150) | 1037 | 1.16e-01 | 411 of 569 | 10 | 1/0 | 1.86e-01 |
| FM4p | 4 | 1336 | 1321 | 4.61e-04 (450) | 5.07e+00 (450) | 1072 | 1.16e-01 | 446 of 582 | 15 | 0/0 | 1.74e-01 |
| FPhipp0n | 4 | 660 | 580 | 3.28e-03 (0) | 6.08e-01 (450) | 387 | 5.61e-01 | 181 of 268 | 80 | 0/0 | 8.44e-01 |
| FPhipp0p | 4 | 660 | 577 | 6.31e-03 (0) | 5.30e-01 (0) | 403 | 3.36e-01 | 187 of 265 | 83 | 0/0 | 8.61e-01 |
| FPhipp2n | 4 | 2296 | 2200 | 1.35e-03 (0) | 3.15e+02 (0) | 1672 | 1.01e+00 | 940 of 1260 | 96 | 0/0 | 1.04e+00 |
| FPhipp2p | 4 | 2294 | 2191 | 2.50e-03 (0) | 1.80e+00 (0) | 1691 | 7.37e-01 | 962 of 1264 | 103 | 0/0 | 1.04e+00 |
| FPhipp4n | 4 | 1333 | 1259 | 1.60e-04 (150) | 7.94e-01 (450) | 919 | 1.09e-01 | 286 of 475 | 74 | 1/2 | 1.48e-01 |
| FPhipp4p | 4 | 1332 | 1249 | 2.50e-04 (150) | 7.01e+00 (150) | 920 | 2.24e-01 | 285 of 475 | 83 | 1/1 | 1.50e-01 |
| FPhitp2n | 4 | 2300 | 2188 | 1.65e-03 (0) | 3.15e+02 (0) | 1692 | 1.01e+00 | 909 of 1215 | 112 | 1/2 | 1.27e+00 |
| FPhitp2p | 4 | 2294 | 2171 | 3.06e-03 (0) | 2.12e+01 (150) | 1693 | 7.37e-01 | 927 of 1220 | 123 | 0/0 | 1.27e+00 |
| FPhitp4n | 4 | 1326 | 1249 | 1.36e-04 (150) | 1.98e+00 (150) | 930 | 7.58e-02 | 279 of 437 | 77 | 2/0 | 1.64e-01 |
| FPhitp4p | 4 | 1323 | 1244 | 2.67e-04 (150) | 1.50e+00 (150) | 953 | 1.31e-01 | 290 of 443 | 79 | 2/3 | 1.65e-01 |
| FSigmap1n | 4 | 1805 | 1715 | 7.18e-04 (0) | 1.50e+01 (0) | 1334 | 3.65e-01 | 557 of 730 | 90 | 3/3 | 6.22e-01 |
| FSigmap1p | 4 | 1810 | 1743 | 1.22e-03 (0) | 2.36e+01 (150) | 1376 | 6.06e-01 | 589 of 753 | 67 | 0/1 | 6.21e-01 |
| FSigmap3n | 4 | 1579 | 1521 | 1.68e-04 (150) | 1.22e+00 (450) | 1237 | 8.25e-02 | 308 of 419 | 58 | 0/0 | 1.42e-01 |
| FSigmap3p | 4 | 1585 | 1499 | 2.92e-04 (150) | 3.56e+00 (150) | 1226 | 1.10e-01 | 317 of 432 | 86 | 3/0 | 1.43e-01 |
| FSigmap5n | 4 | 777 | 748 | 1.11e-04 (450) | 3.68e+00 (450) | 539 | 5.37e-02 | 75 of 145 | 29 | 1/1 | 9.43e-02 |
| FSigmap5p | 4 | 778 | 747 | 1.74e-04 (450) | 1.23e+00 (150) | 568 | 4.71e-02 | 83 of 146 | 31 | 2/0 | 9.12e-02 |
| FSigmapp1n | 4 | 1809 | 1749 | 5.08e-04 (0) | 5.05e+00 (150) | 1404 | 3.20e-01 | 489 of 620 | 60 | 0/0 | 4.39e-01 |
| FSigmapp1p | 4 | 1812 | 1756 | 8.64e-04 (0) | 5.03e+00 (0) | 1363 | 5.87e-01 | 492 of 647 | 56 | 0/0 | 4.39e-01 |
| FSigmapp3n | 4 | 1578 | 1514 | 1.78e-04 (150) | 4.67e+00 (450) | 1246 | 7.64e-02 | 284 of 380 | 64 | 1/0 | 1.29e-01 |
| FSigmapp3p | 4 | 1580 | 1501 | 3.07e-04 (150) | 1.17e+01 (450) | 1230 | 1.17e-01 | 284 of 384 | 79 | 3/2 | 1.31e-01 |
| FSigmapp5n | 4 | 775 | 742 | 1.01e-04 (450) | 1.02e+00 (450) | 524 | 7.93e-02 | 65 of 128 | 33 | 0/0 | 8.47e-02 |
| FSigmapp5p | 4 | 774 | 732 | 1.58e-04 (450) | 2.15e+01 (450) | 538 | 1.57e-01 | 75 of 133 | 42 | 3/4 | 8.19e-02 |
| FDelta1n | 4 | 1737 | 1443 | 4.01e-04 (0) | 7.39e+00 (0) | 1174 | 2.90e-01 | 341 of 447 | 294 | 0/2 | 6.31e-01 |
| FDelta1p | 4 | 1734 | 1440 | 3.50e-04 (0) | 7.96e+01 (0) | 1184 | 2.30e-01 | 376 of 473 | 294 | 0/4 | 6.29e-01 |
| FDelta3n | 4 | 1526 | 1193 | 4.36e-05 (150) | 4.81e+00 (150) | 950 | 2.84e-02 | 92 of 151 | 333 | 3/2 | 4.59e-02 |
| FDelta3p | 4 | 1519 | 1199 | 7.42e-05 (150) | 8.31e+00 (150) | 957 | 3.37e-02 | 81 of 150 | 320 | 2/1 | 4.65e-02 |
| FDelta5n | 4 | 746 | 617 | 1.14e-06 (450) | 1.54e-01 (450) | 254 | 4.27e-04 | 0 of 7 | 129 | 0/1 | 1.81e-03 |
| FDelta5p | 4 | 756 | 615 | 1.30e-06 (450) | 5.79e-02 (150) | 272 | 6.02e-04 | 0 of 4 | 141 | 0/0 | 1.69e-03 |

#### operator-side compat effect (v2 Omegas, default vs fork operator behaviour), N2LOsat

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 4 | 657 | 483 | 4.10e-07 (150) | 6.84e-07 (900) | 0 | 5.15e-07 | 0 of 195 | 174 | 158/0 | 8.00e+00 |
| FM0p | 4 | 657 | 484 | 3.30e-07 (150) | 4.89e-07 (150) | 0 | 4.89e-07 | 0 of 197 | 173 | 158/0 | 8.00e+00 |
| FM2n | 4 | 1732 | 1710 | 3.75e-02 (150) | 1.40e+00 (900) | 12 | 7.56e-01 | 8 of 863 | 22 | 0/2 | 5.91e-01 |
| FM2p | 4 | 1732 | 1704 | 3.29e-02 (150) | 2.58e+00 (900) | 12 | 7.56e-01 | 8 of 866 | 28 | 0/6 | 5.89e-01 |
| FM4n | 4 | 1336 | 1325 | 1.09e-02 (450) | 1.10e+00 (900) | 6 | 1.02e-01 | 4 of 572 | 11 | 0/1 | 1.86e-01 |
| FM4p | 4 | 1336 | 1320 | 8.54e-03 (450) | 7.26e-01 (900) | 6 | 7.99e-02 | 4 of 584 | 16 | 0/0 | 1.74e-01 |
| FPhipp0n | 4 | 660 | 581 | 2.96e-07 (0) | 1.32e-06 (0) | 0 | 1.32e-06 | 0 of 269 | 79 | 1/0 | 8.44e-01 |
| FPhipp0p | 4 | 660 | 579 | 3.97e-07 (150) | 3.42e-06 (900) | 0 | 3.03e-06 | 0 of 265 | 81 | 3/0 | 8.61e-01 |
| FPhipp2n | 4 | 2308 | 2200 | 8.20e-02 (0) | 3.31e+00 (900) | 16 | 9.13e-01 | 11 of 1260 | 108 | 0/12 | 1.04e+00 |
| FPhipp2p | 4 | 2308 | 2195 | 8.19e-02 (0) | 2.98e+00 (900) | 16 | 8.72e-01 | 11 of 1263 | 113 | 0/14 | 1.04e+00 |
| FPhipp4n | 4 | 1335 | 1261 | 4.24e-03 (450) | 1.12e+00 (900) | 5 | 4.86e-01 | 3 of 476 | 74 | 0/3 | 1.48e-01 |
| FPhipp4p | 4 | 1336 | 1250 | 3.55e-03 (450) | 9.23e-01 (900) | 6 | 4.70e-01 | 4 of 476 | 86 | 0/5 | 1.50e-01 |
| FPhitp2n | 4 | 2308 | 2191 | 1.00e-01 (0) | 1.34e+00 (450) | 16 | 9.13e-01 | 11 of 1216 | 117 | 0/9 | 1.27e+00 |
| FPhitp2p | 4 | 2308 | 2175 | 1.00e-01 (0) | 1.73e+00 (450) | 15 | 8.72e-01 | 11 of 1220 | 133 | 0/14 | 1.27e+00 |
| FPhitp4n | 4 | 1336 | 1249 | 6.73e-03 (450) | 5.38e-01 (900) | 6 | 1.88e-01 | 4 of 436 | 87 | 0/12 | 1.64e-01 |
| FPhitp4p | 4 | 1336 | 1244 | 5.73e-03 (450) | 3.29e-01 (900) | 6 | 2.16e-01 | 4 of 443 | 92 | 0/15 | 1.65e-01 |
| FSigmap1n | 4 | 1816 | 1718 | 9.42e-03 (150) | 1.86e+02 (0) | 1699 | 1.95e+00 | 728 of 737 | 98 | 0/14 | 6.22e-01 |
| FSigmap1p | 4 | 1816 | 1741 | 1.10e-02 (150) | 4.85e+01 (0) | 1701 | 2.38e+00 | 729 of 750 | 75 | 0/6 | 6.21e-01 |
| FSigmap3n | 4 | 1588 | 1519 | 9.83e-03 (450) | 1.84e+01 (150) | 1488 | 4.73e-01 | 407 of 423 | 69 | 0/9 | 1.42e-01 |
| FSigmap3p | 4 | 1588 | 1499 | 1.04e-02 (450) | 2.59e+01 (450) | 1462 | 4.28e-01 | 414 of 431 | 89 | 0/6 | 1.43e-01 |
| FSigmap5n | 4 | 778 | 747 | 2.43e-03 (450) | 2.66e+01 (150) | 730 | 5.16e-01 | 139 of 146 | 31 | 0/2 | 9.43e-02 |
| FSigmap5p | 4 | 778 | 749 | 2.02e-03 (150) | 3.00e+01 (450) | 723 | 4.19e-01 | 135 of 144 | 29 | 0/2 | 9.12e-02 |
| FSigmapp1n | 4 | 1816 | 1749 | 6.01e-03 (0) | 1.86e+02 (0) | 1733 | 1.95e+00 | 612 of 623 | 67 | 0/7 | 4.39e-01 |
| FSigmapp1p | 4 | 1816 | 1753 | 5.65e-03 (0) | 4.85e+01 (0) | 1716 | 2.38e+00 | 618 of 639 | 63 | 0/4 | 4.39e-01 |
| FSigmapp3n | 4 | 1588 | 1516 | 1.28e-02 (150) | 1.79e+02 (150) | 1479 | 5.80e-01 | 361 of 380 | 72 | 0/11 | 1.29e-01 |
| FSigmapp3p | 4 | 1588 | 1501 | 1.34e-02 (150) | 2.51e+02 (150) | 1462 | 4.11e-01 | 364 of 382 | 87 | 0/11 | 1.31e-01 |
| FSigmapp5n | 4 | 778 | 743 | 2.19e-03 (450) | 1.34e+01 (150) | 728 | 3.52e-01 | 121 of 129 | 35 | 0/3 | 8.47e-02 |
| FSigmapp5p | 4 | 778 | 732 | 1.96e-03 (450) | 2.15e+01 (450) | 720 | 2.37e-01 | 126 of 133 | 46 | 0/7 | 8.19e-02 |
| FDelta1n | 4 | 1816 | 1444 | 1.38e-02 (150) | 4.10e+02 (0) | 1433 | 2.93e+00 | 445 of 448 | 372 | 0/79 | 6.31e-01 |
| FDelta1p | 4 | 1816 | 1440 | 1.30e-02 (150) | 1.74e+02 (0) | 1424 | 1.62e+00 | 468 of 472 | 376 | 2/82 | 6.29e-01 |
| FDelta3n | 4 | 1588 | 1196 | 2.78e-03 (150) | 6.28e+01 (150) | 1178 | 5.15e-01 | 147 of 150 | 392 | 0/65 | 4.59e-02 |
| FDelta3p | 4 | 1587 | 1201 | 3.28e-03 (150) | 3.34e+01 (450) | 1167 | 4.65e-01 | 138 of 149 | 386 | 0/70 | 4.65e-02 |
| FDelta5n | 4 | 778 | 617 | 9.68e-04 (450) | 4.44e+01 (450) | 600 | 5.33e-01 | 7 of 7 | 161 | 0/32 | 1.82e-03 |
| FDelta5p | 4 | 777 | 615 | 8.27e-04 (450) | 8.19e+00 (450) | 591 | 5.08e-01 | 4 of 4 | 162 | 0/21 | 1.69e-03 |

## 5. v3 vs v2

Farms: job 6016892 (80 nodes, 40 min, 53.3 node-hours), 2 x 2415 items, 0 failed; ~170 BCH terms
per item over 40-42 Omegas (every series converged), 145-153 GCD-hours of item work per
Hamiltonian (5.5x v2). The farm's TCM (42 sequential BCH transforms read from the Omega files)
equals the flow's TCM digit for digit.

### 5.1 Size of the differences

Most elements differ by more than 1e-3 relative (tables in 5.3, columns "n rel>0.001"; farm
elements incl. TCM, N2LOsat: 1002580 compared, 904719 rel-eligible, 897341 (99.2 %) with
rel > 1e-3; of 241698 elements with |ref| >= 1e-3, 239240 (99.0 %); DN2LOGO: 98.8 % and 98.4 %;
for each q the fraction is 98.7-99.6 % (N2LOsat), 98.3-99.4 % (DN2LOGO)). The typical size is a few
percent. The table below gives, per column (N2LOsat, all 71 q; `tools/al27/dominant_rel.py`), the
relative difference of the **dominant elements** (|ref| at least 1 % of the largest |ref| of the
file and at least 1e-6, so the owner's rule holds): the median, the 90th percentile, the maximum, and the
largest |Δ| relative to the largest |ref| of a file (all parts: the zero-body part is not
included). The medians per column are 3.8-8.1 % (N2LOsat; DN2LOGO 2.7-5.5 %); the one-body parts
alone (second table; the ratio is then taken within the one-body part) 0.5-6.6 % (DN2LOGO
0.5-5.7 %). FM0 at q = 0 (the particle-number operator, conserved by every unitary
transformation) agrees to 2e-16, as it must. An operator that vanishes in the reference (no
element with |ref| >= 1e-6, e.g. every L > 0 multipole at q = 0) has no relative difference and is
shown as "vanishes", not as 0. At q = 0 / 300 / 600 / 900 the medians are e.g. 11 / 8.6 / 4.8 /
8.9 % (FPhipp0p), 4.3 / 4.4 / 3.9 / 4.6 % (FDelta1n); the median is non-monotonic: for 31 of the 34 columns the q = 600 value is below the q = 300 one and
for 32 the q = 900 value is above it (the four q values are the only ones tabulated). The largest
absolute difference relative to the largest |ref| of a file occurs at q >= 300 for 32 of the 34 columns (median
q 975). For comparison, v2 vs v1 (same truncation, fork bugs and domega 0.2 vs 0.1) has
medians of 4e-4 to 2.4e-3 (scalar and even-rank columns) and 2.8-6.0e-2 (odd ranks; both
Hamiltonians, same dominant-element definition).

**v3 vs v2, N2LOsat, all parts, dominant elements:**

| column | files | dominant elements | median rel | 90% rel | max rel | max abs / max ref (q) | median rel at q = 0 / 300 / 600 / 900 |
|---|---:|---:|---:|---:|---:|---|---|
| FM0n | 71 | 6055 | 6.1e-02 | 2.3e-01 | 1.5e+00 | 8.1e-02 (795) | 2.2e-16 / 5.3e-02 / 5.4e-02 / 6.4e-02 |
| FM0p | 71 | 6225 | 6.9e-02 | 2.5e-01 | 2.1e+00 | 1.0e-01 (765) | 2.2e-16 / 5.9e-02 / 5.7e-02 / 8.2e-02 |
| FM2n | 71 | 25822 | 5.9e-02 | 2.4e-01 | 3.9e+00 | 1.1e-01 (1050) | vanishes / 5.5e-02 / 5.4e-02 / 9.4e-02 |
| FM2p | 71 | 26310 | 6.8e-02 | 3.0e-01 | 3.2e+00 | 1.1e-01 (1050) | vanishes / 6.6e-02 / 5.9e-02 / 1.2e-01 |
| FM4n | 71 | 21293 | 5.5e-02 | 2.2e-01 | 6.8e+00 | 8.8e-02 (915) | vanishes / 3.9e-02 / 3.8e-02 / 7.2e-02 |
| FM4p | 71 | 21666 | 6.8e-02 | 2.9e-01 | 6.7e+00 | 1.3e-01 (960) | vanishes / 4.5e-02 / 3.7e-02 / 1.0e-01 |
| FPhipp0n | 71 | 6538 | 5.9e-02 | 2.1e-01 | 3.1e+00 | 1.5e-01 (1050) | 6.7e-02 / 6.6e-02 / 4.3e-02 / 9.2e-02 |
| FPhipp0p | 71 | 6641 | 7.0e-02 | 2.6e-01 | 2.4e+00 | 9.6e-02 (975) | 1.1e-01 / 8.6e-02 / 4.8e-02 / 8.9e-02 |
| FPhipp2n | 71 | 27862 | 5.9e-02 | 2.1e-01 | 3.4e+00 | 1.1e-01 (1050) | 6.7e-02 / 5.8e-02 / 3.9e-02 / 9.6e-02 |
| FPhipp2p | 71 | 27913 | 7.3e-02 | 2.7e-01 | 3.2e+00 | 1.3e-01 (1050) | 8.3e-02 / 7.7e-02 / 5.0e-02 / 1.1e-01 |
| FPhipp4n | 71 | 21099 | 4.3e-02 | 1.8e-01 | 2.3e+00 | 1.2e-01 (930) | vanishes / 5.2e-02 / 2.5e-02 / 6.7e-02 |
| FPhipp4p | 71 | 21112 | 5.4e-02 | 2.4e-01 | 5.1e+00 | 1.6e-01 (960) | vanishes / 5.5e-02 / 2.8e-02 / 9.7e-02 |
| FPhitp2n | 71 | 27436 | 6.3e-02 | 2.6e-01 | 5.3e+00 | 1.5e-01 (915) | 6.7e-02 / 5.3e-02 / 4.2e-02 / 1.5e-01 |
| FPhitp2p | 71 | 27591 | 8.1e-02 | 3.2e-01 | 7.7e+00 | 2.1e-01 (1020) | 8.3e-02 / 6.8e-02 / 5.9e-02 / 1.4e-01 |
| FPhitp4n | 71 | 20086 | 5.5e-02 | 2.1e-01 | 3.1e+00 | 2.3e-01 (915) | vanishes / 4.3e-02 / 3.9e-02 / 8.4e-02 |
| FPhitp4p | 71 | 20023 | 6.8e-02 | 2.9e-01 | 4.0e+00 | 5.3e-01 (975) | vanishes / 5.1e-02 / 4.1e-02 / 1.1e-01 |
| FSigmap1n | 71 | 20120 | 5.9e-02 | 2.2e-01 | 3.7e+00 | 1.4e-01 (810) | 6.8e-02 / 6.0e-02 / 4.4e-02 / 7.7e-02 |
| FSigmap1p | 71 | 20433 | 7.5e-02 | 2.8e-01 | 3.7e+00 | 1.7e-01 (810) | 5.3e-02 / 6.9e-02 / 5.9e-02 / 1.3e-01 |
| FSigmap3n | 71 | 23541 | 5.4e-02 | 2.2e-01 | 4.6e+00 | 1.4e-01 (1050) | vanishes / 4.7e-02 / 3.4e-02 / 9.1e-02 |
| FSigmap3p | 71 | 22834 | 6.7e-02 | 2.7e-01 | 5.4e+00 | 1.1e-01 (930) | vanishes / 6.4e-02 / 3.9e-02 / 1.3e-01 |
| FSigmap5n | 71 | 10939 | 4.7e-02 | 1.7e-01 | 2.5e+00 | 2.6e-01 (990) | vanishes / 4.3e-02 / 3.2e-02 / 6.7e-02 |
| FSigmap5p | 71 | 11079 | 6.1e-02 | 2.5e-01 | 3.9e+00 | 5.1e-01 (990) | vanishes / 5.2e-02 / 3.7e-02 / 9.7e-02 |
| FSigmapp1n | 71 | 19910 | 6.0e-02 | 2.4e-01 | 2.7e+00 | 1.2e-01 (1050) | 6.8e-02 / 5.4e-02 / 5.3e-02 / 9.5e-02 |
| FSigmapp1p | 71 | 20227 | 7.2e-02 | 2.8e-01 | 3.4e+00 | 1.5e-01 (1050) | 5.3e-02 / 5.4e-02 / 6.7e-02 / 1.0e-01 |
| FSigmapp3n | 71 | 22773 | 5.7e-02 | 2.4e-01 | 8.0e+00 | 1.8e-01 (1050) | vanishes / 4.7e-02 / 5.1e-02 / 9.7e-02 |
| FSigmapp3p | 71 | 22195 | 6.9e-02 | 2.8e-01 | 5.2e+00 | 1.1e-01 (915) | vanishes / 5.7e-02 / 5.4e-02 / 1.0e-01 |
| FSigmapp5n | 71 | 11107 | 4.8e-02 | 1.8e-01 | 2.4e+00 | 1.3e-01 (990) | vanishes / 4.0e-02 / 2.9e-02 / 5.4e-02 |
| FSigmapp5p | 71 | 11213 | 6.2e-02 | 2.6e-01 | 6.8e+00 | 2.4e-01 (990) | vanishes / 4.0e-02 / 3.9e-02 / 8.2e-02 |
| FDelta1n | 71 | 13258 | 4.5e-02 | 1.6e-01 | 3.0e+00 | 2.1e-01 (1035) | 4.3e-02 / 4.4e-02 / 3.9e-02 / 4.6e-02 |
| FDelta1p | 71 | 13230 | 5.4e-02 | 2.0e-01 | 2.7e+00 | 2.5e-01 (945) | 5.3e-02 / 5.3e-02 / 4.7e-02 / 5.8e-02 |
| FDelta3n | 71 | 18500 | 4.3e-02 | 1.5e-01 | 4.0e+00 | 2.2e-01 (1020) | vanishes / 4.7e-02 / 3.0e-02 / 5.2e-02 |
| FDelta3p | 71 | 18816 | 5.4e-02 | 2.0e-01 | 2.8e+00 | 2.6e-01 (930) | vanishes / 6.2e-02 / 3.3e-02 / 7.0e-02 |
| FDelta5n | 71 | 13514 | 3.8e-02 | 1.4e-01 | 1.8e+00 | 9.9e-02 (30) | vanishes / 4.5e-02 / 2.6e-02 / 3.7e-02 |
| FDelta5p | 71 | 13529 | 3.9e-02 | 1.4e-01 | 2.9e+00 | 1.0e-01 (30) | vanishes / 4.7e-02 / 2.9e-02 / 3.5e-02 |

**v3 vs v2, N2LOsat, one-body parts:**

| column | files | dominant elements | median rel | 90% rel | max rel | max abs / max ref (q) | median rel at q = 0 / 300 / 600 / 900 |
|---|---:|---:|---:|---:|---:|---|---|
| FM0n | 71 | 405 | 4.5e-02 | 1.3e-01 | 1.5e+00 | 8.1e-02 (795) | 2.2e-16 / 2.2e-02 / 2.7e-02 / 4.5e-02 |
| FM0p | 71 | 405 | 4.1e-02 | 1.3e-01 | 2.1e+00 | 1.0e-01 (765) | 2.2e-16 / 1.7e-02 / 3.4e-02 / 4.9e-02 |
| FM2n | 71 | 1105 | 2.7e-02 | 1.4e-01 | 2.4e+00 | 1.3e-01 (720) | vanishes / 1.7e-02 / 1.5e-02 / 6.6e-02 |
| FM2p | 71 | 1098 | 2.5e-02 | 1.1e-01 | 1.9e+00 | 1.2e-01 (735) | vanishes / 7.9e-03 / 1.6e-02 / 6.2e-02 |
| FM4n | 71 | 417 | 2.2e-02 | 1.1e-01 | 1.8e+00 | 2.1e-01 (855) | vanishes / 1.3e-02 / 2.1e-02 / 6.2e-02 |
| FM4p | 71 | 418 | 1.6e-02 | 6.7e-02 | 7.6e+00 | 1.4e-01 (885) | vanishes / 1.2e-02 / 1.7e-02 / 9.3e-02 |
| FPhipp0n | 71 | 376 | 6.3e-02 | 2.9e-01 | 2.5e+00 | 1.5e-01 (1050) | 3.1e-02 / 5.4e-02 / 1.4e-01 / 6.1e-02 |
| FPhipp0p | 71 | 380 | 6.6e-02 | 2.6e-01 | 1.9e+00 | 9.2e-02 (1050) | 4.0e-02 / 3.0e-02 / 1.1e-01 / 7.5e-02 |
| FPhipp2n | 71 | 1050 | 4.9e-02 | 2.1e-01 | 3.4e+00 | 1.1e-01 (1050) | 5.0e-02 / 2.6e-02 / 9.8e-02 / 9.5e-02 |
| FPhipp2p | 71 | 1064 | 4.4e-02 | 2.4e-01 | 1.5e+00 | 1.3e-01 (1050) | 5.6e-02 / 2.1e-02 / 1.4e-01 / 7.7e-02 |
| FPhipp4n | 71 | 383 | 2.1e-02 | 1.8e-01 | 1.4e+00 | 3.2e-01 (915) | vanishes / 1.2e-02 / 1.4e-02 / 1.5e-01 |
| FPhipp4p | 71 | 380 | 2.3e-02 | 4.4e-01 | 3.7e+00 | 7.9e-01 (885) | vanishes / 1.6e-02 / 2.7e-02 / 1.7e+00 |
| FPhitp2n | 71 | 1066 | 5.6e-02 | 2.9e-01 | 5.3e+00 | 1.8e-01 (810) | 5.0e-02 / 4.4e-02 / 7.3e-02 / 1.4e-01 |
| FPhitp2p | 71 | 1044 | 5.7e-02 | 2.5e-01 | 2.3e+00 | 2.1e-01 (1020) | 5.6e-02 / 4.6e-02 / 7.6e-02 / 6.3e-02 |
| FPhitp4n | 71 | 344 | 3.8e-02 | 2.5e-01 | 2.3e+00 | 3.8e-01 (855) | vanishes / 8.7e-03 / 5.4e-02 / 2.6e-01 |
| FPhitp4p | 71 | 344 | 5.0e-02 | 3.1e-01 | 2.1e+00 | 5.8e-01 (960) | vanishes / 1.2e-02 / 1.2e-01 / 2.4e-01 |
| FSigmap1n | 71 | 935 | 4.2e-02 | 1.8e-01 | 1.5e+00 | 1.4e-01 (810) | 2.1e-02 / 3.0e-02 / 1.6e-02 / 8.9e-02 |
| FSigmap1p | 71 | 940 | 4.0e-02 | 2.0e-01 | 3.0e+00 | 1.7e-01 (810) | 1.2e-02 / 2.9e-02 / 2.7e-02 / 7.2e-02 |
| FSigmap3n | 71 | 768 | 2.2e-02 | 9.8e-02 | 3.9e+00 | 1.4e-01 (1050) | vanishes / 1.5e-02 / 2.7e-02 / 4.9e-02 |
| FSigmap3p | 71 | 770 | 2.2e-02 | 8.0e-02 | 3.3e+00 | 8.0e-02 (735) | vanishes / 1.9e-02 / 2.2e-02 / 4.2e-02 |
| FSigmap5n | 71 | 129 | 1.3e-02 | 1.1e-01 | 4.8e+00 | 4.2e-01 (1005) | vanishes / 1.7e-03 / 9.2e-03 / 4.0e-02 |
| FSigmap5p | 71 | 127 | 1.6e-02 | 1.7e-01 | 3.3e+00 | 6.0e-01 (990) | vanishes / 3.1e-03 / 1.2e-02 / 8.6e-02 |
| FSigmapp1n | 71 | 928 | 4.2e-02 | 1.3e-01 | 1.2e+00 | 1.2e-01 (1050) | 2.1e-02 / 1.9e-02 / 3.5e-02 / 1.1e-01 |
| FSigmapp1p | 71 | 936 | 4.4e-02 | 1.3e-01 | 1.3e+00 | 1.5e-01 (1050) | 1.2e-02 / 4.2e-02 / 3.5e-02 / 8.9e-02 |
| FSigmapp3n | 71 | 790 | 2.6e-02 | 9.2e-02 | 8.0e+00 | 1.8e-01 (1050) | vanishes / 1.1e-02 / 2.1e-02 / 4.6e-02 |
| FSigmapp3p | 71 | 785 | 2.5e-02 | 7.5e-02 | 1.7e+00 | 9.8e-02 (1050) | vanishes / 1.1e-02 / 1.8e-02 / 3.7e-02 |
| FSigmapp5n | 71 | 139 | 1.5e-02 | 9.5e-02 | 2.4e+00 | 1.9e-01 (1020) | vanishes / 2.6e-03 / 9.2e-03 / 5.3e-02 |
| FSigmapp5p | 71 | 139 | 1.8e-02 | 1.4e-01 | 6.8e+00 | 2.9e-01 (975) | vanishes / 4.3e-03 / 1.0e-02 / 1.2e-01 |
| FDelta1n | 71 | 717 | 5.0e-02 | 1.9e-01 | 3.0e+00 | 2.9e-01 (960) | 1.0e-02 / 2.5e-02 / 4.9e-02 / 1.5e-01 |
| FDelta1p | 71 | 691 | 4.5e-02 | 1.7e-01 | 1.9e+00 | 2.5e-01 (945) | 8.8e-03 / 2.0e-02 / 4.6e-02 / 1.1e-01 |
| FDelta3n | 71 | 704 | 3.7e-02 | 1.7e-01 | 4.0e+00 | 2.2e-01 (1020) | vanishes / 2.0e-02 / 3.2e-02 / 8.1e-02 |
| FDelta3p | 71 | 713 | 4.7e-02 | 1.3e-01 | 1.5e+00 | 4.0e-01 (900) | vanishes / 2.5e-02 / 3.6e-02 / 6.4e-02 |
| FDelta5n | 71 | 123 | 4.9e-03 | 3.0e-02 | 3.8e-01 | 2.1e-02 (45) | vanishes / 3.9e-03 / 1.3e-02 / 8.1e-03 |
| FDelta5p | 71 | 123 | 1.2e-02 | 2.4e-02 | 1.3e-01 | 2.7e-02 (1005) | vanishes / 1.0e-02 / 1.3e-02 / 2.0e-02 |

DN2LOGO (all parts, same definition): medians 2.7-5.5 % per column (one-body parts 0.5-5.7 %); the
table was regenerated with `dominant_rel.py` and is not reproduced here.

### 5.2 Cause of the differences > 1e-3 (investigation)

Candidates as in section 3: BCH truncation per Omega in the transform, the number of Omegas
(threshold), the time step, and the Magnus(2)-vs-IMSRG(2) truncation of v2. Checks (N2LOsat;
jobs 6017305 and 6017358; 12 items from the worst columns, q = 0 to 990; for DN2LOGO only the ODE
reference, job 6018030, Lead E):

* **BCH truncation per Omega:** every v3 item reports a converged series (`bch_converged=1`; the
  transform stops when a nested commutator falls below 1e-9-scaled epsilon, at most 40 terms);
  each split Omega has |Omega| <= 0.33, so ~4 terms per Omega suffice. Not a source.
* **Threshold, time step:** the same items transformed with the Omega stacks of the split runs
  at threshold 1.0 (12 Omegas), 0.1 (122 Omegas) and at threshold 0.25 with domega 0.025
  (45 Omegas), each compared with v3. Cell = median relative difference of the dominant elements /
  largest |Δ| over largest |ref|:

| item | split 0.25, domega 0.025 | split 0.1, domega 0.1 | split 1.0, domega 0.1 | v2 (unsplit, domega 0.1) |
|---|---|---|---|---|
| FM0p_150 | 2.6e-03 / 5.2e-03 | 1.7e-04 / 1.3e-04 | 1.3e-03 / 1.9e-03 | 8.9e-02 / 3.6e-02 |
| FPhipp0p_0 | 9.0e-04 / 5.6e-03 | 9.2e-05 / 1.1e-04 | 6.1e-04 / 1.6e-03 | 7.0e-02 / 1.2e-02 |
| FM2p_300 | 4.7e-03 / 2.5e-03 | 2.2e-04 / 8.8e-05 | 2.1e-03 / 1.0e-03 | 6.3e-02 / 2.2e-02 |
| FM4p_960 | 1.1e-02 / 3.6e-02 | 4.8e-04 / 1.6e-03 | 5.6e-03 / 1.8e-02 | 1.2e-01 / 1.3e-01 |
| FPhipp2p_0 | 5.0e-03 / 2.9e-03 | 2.7e-04 / 1.6e-04 | 1.5e-03 / 1.5e-03 | 8.7e-02 / 2.3e-02 |
| FPhitp2p_0 | 5.0e-03 / 2.9e-03 | 2.7e-04 / 1.6e-04 | 1.5e-03 / 1.5e-03 | 8.7e-02 / 2.3e-02 |
| FPhitp2p_600 | 3.5e-03 / 3.1e-03 | 2.1e-04 / 2.1e-04 | 1.7e-03 / 2.0e-03 | 6.0e-02 / 5.9e-02 |
| FSigmap1p_90 | 6.7e-03 / 1.9e-03 | 4.6e-04 / 8.0e-05 | 2.3e-03 / 5.1e-04 | 9.3e-02 / 1.6e-02 |
| FSigmap1p_810 | 2.4e-03 / 2.9e-03 | 1.8e-04 / 2.7e-04 | 1.5e-03 / 2.0e-03 | 1.0e-01 / 2.0e-01 |
| FSigmap5p_990 | 9.8e-03 / 7.6e-02 | 7.3e-04 / 2.9e-03 | 6.6e-03 / 4.5e-02 | 1.7e-01 / 3.9e-01 |
| FDelta1p_135 | 2.8e-03 / 1.8e-03 | 1.3e-04 / 4.4e-05 | 1.6e-03 / 4.2e-04 | 6.9e-02 / 1.6e-02 |
| FDelta1p_945 | 1.2e-02 / 3.0e-02 | 6.8e-04 / 1.1e-03 | 6.2e-03 / 1.9e-02 | 1.4e-01 / 2.0e-01 |

  Threshold 0.25 -> 0.1 changes the operators by 0.9-7.3e-4 (median), 1.0 -> 0.25 by 0.6-6.6e-3, so v3
  is converged in the threshold to a few 1e-4. The time step matters more: domega 0.1 -> 0.025
  changes them by 0.09-1.2 % (median; up to 3-8 % of the file maximum for the highest-q L = 4, 5
  items, whose elements are small). v2 differs from v3 by 6-17 % (median), about 10-20 times more.
* **ODE reference:** the ODE IMSRG(2) flow (job 6017305, 36 min on a GCD) evolved FM0p_150 and
  FPhipp0p_0 alongside H. Scalars only: `flow::OdeSolver` cannot carry a tensor operator (it
  rejects non-scalar operators, src/imsrg/flow/ode.cc:214-220; WE-E-1), so no rank L > 0 item
  can be checked this way (an earlier version of this section gave the cost as the reason).
  Against it (same cells as above), N2LOsat:

| quantity (vs ODE) | v3 (split 0.25, domega 0.1) | split 0.25, domega 0.025 | split 0.1, domega 0.1 | v2 (unsplit, domega 0.1) |
|---|---|---|---|---|
| FM0p_150: median rel (dominant) / max abs over max ref | 2.6e-03 / 5.2e-03 | 7.7e-04 / 8.5e-04 | 2.8e-03 / 5.1e-03 | 8.5e-02 / 3.6e-02 |
| FPhipp0p_0: median rel (dominant) / max abs over max ref | 1.3e-03 / 5.4e-03 | 3.9e-04 / 6.5e-04 | 1.4e-03 / 5.4e-03 | 6.9e-02 / 1.0e-02 |
| valence H: median rel (dominant) / max abs over max ref | 8.4e-04 / 1.1e-03 | 1.9e-04 / 3.6e-04 | 8.4e-04 / 1.2e-03 | 4.7e-02 / 1.3e-01 |
| valence H: core energy - ODE (keV; ODE -119.180577 MeV) | -6.9 | +3.4 | -12.6 | +891.1 |

  The split runs agree with the ODE operators and valence H to 0.04-0.3 % (median of the dominant
  elements; domega 0.025 closest), v2 is 5-9 % off, and v2's core energy is 0.89 MeV above the
  ODE value (v3: -7 keV).

  DN2LOGO (job 6018030, Lead E: the same two operators through the ODE flow from v2's prepared
  H_0, config `runs/al27/investigate/configs/DN2LOGO-ode-ops.toml` = the scan's DN2LOGO-ode.toml
  plus the operators, the binary of 6017305, 34 min on a GCD; E0 -216.989496 MeV, 1.3e-7 MeV from
  the ODE flow without operators; 85 + 444 accepted steps vs 83 + 443 without them). The
  split-check Omega stacks exist for N2LOsat only, so the comparison is with v3 and v2:

| quantity (vs ODE), DN2LOGO | v3 (split 0.25, domega 0.1) | v2 (unsplit, domega 0.1) | ODE flow without operators (6016814) |
|---|---|---|---|
| FM0p_150: median rel (dominant) / max abs over max ref | 3.9e-03 / 4.1e-03 | 8.0e-02 / 3.2e-02 | |
| FPhipp0p_0: median rel (dominant) / max abs over max ref | 7.8e-04 / 1.5e-03 | 4.3e-02 / 1.1e-02 | |
| valence H: median rel (dominant) / max abs over max ref | 4.2e-04 / 1.1e-03 | 3.0e-02 / 9.3e-02 | 3.0e-08 / 1.1e-07 |
| valence H: core energy - ODE (keV; ODE -116.479240 MeV) | +2.2 | +187.8 | +0.0 |

  The same picture as for N2LOsat: v3 agrees with the ODE operators to 0.08-0.4 % (median of the
  dominant elements) and with the valence H to 0.04 %, v2 is 4-8 % off (valence H 3 %), and v2's
  core energy is 0.19 MeV above the ODE value (v3: +2.2 keV). (Cells: `tools/al27/ode_check.py`,
  built on `dominant_rel.py`, dominant = |ref| >= max(0.01 max|ref|, 1e-6); it reproduces the
  N2LOsat table above exactly.)

**Conclusion.** The v3-vs-v2 operator differences above 1e-3 relative are the same effect as in
E0: v3 (split Magnus) is an IMSRG(2)-equivalent evolution, v2 (unsplit) is Magnus(2) with one
large Omega, and the two truncations induce different many-body corrections, with operator
differences of a few percent (more at high q). They are not caused by the BCH truncation per
Omega or the number of Omegas (v3 is converged in the threshold to a few 1e-4). v3 differs by
0.09-1.2 % (median) at the default domega 0.1 from the domega 0.025 run (two time steps only, so
this is a difference, not an extrapolated error) (the owner's
v3 definition is unchanged; a smaller domega costs ~1.6x / 2.9x the v3 flow time for 0.05 / 0.025
and nothing extra in the farm, the Omega count stays ~40-45).

### 5.3 Full comparison tables

#### v3 vs v2, N2LOsat

Absolute |test - ref| for all elements; relative only for |ref| >= 1e-06; flag threshold rel > 0.001.


##### Flow (results.json, flow.txt)

| quantity | ref | test | diff |
|---|---|---|---|
| final E0 (results.json) | -213.9568717976 | -214.4287625601 | 4.72e-01 |
| num_omegas (results.json) | 2 | 42 | 40 |
| num_omegas (flow.txt, last row) | 2 | 42 | 40 |
| stage 1 final E0 | -131.4988150104 | -131.6084965803 | 1.10e-01 |
| stage 1 final s | 13.314646 | 13.049405 | 2.65e-01 |
| stage 1 final step | 54 | 44 | -10 |
| stage 2 final E0 | -213.9568717976 | -214.4287625601 | 4.72e-01 |
| stage 2 final s | 57.913877 | 48.853815 | 9.06e+00 |
| stage 2 final step | 489 | 146 | -343 |

##### Flow directory files (valence H, TCM)

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 165 | 165 | 8.98e-01 (-) | 1.49e+01 (-) | 162 | 1.49e+01 | 162 of 165 | 0 | 0/0 | 1.18e+02 |
| TCM(flow) | 1 | 156 | 149 | 1.77e-02 (-) | 5.56e+00 (-) | 148 | 5.56e+00 | 148 of 149 | 7 | 9/7 | 6.96e+00 |

H by part:

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 1 | 8.98e-01 | 7.59e-03 | 1 | 0 | 0/0 |
| 1b | 6 | 8.43e-01 | 2.13e+00 | 6 | 0 | 0/0 |
| 2b | 158 | 2.67e-01 | 1.49e+01 | 155 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 11554 | 11002 | 8.47e-03 (135) | 1.19e+02 (60) | 10895 | 1.50e+00 | 3220 of 3287 | 552 | 5/8 | 8.00e+00 |
| FM0p | 71 | 11554 | 10982 | 1.17e-02 (120) | 1.07e+02 (390) | 10895 | 2.08e+00 | 3290 of 3336 | 572 | 2/2 | 8.00e+00 |
| FM2n | 71 | 40391 | 39528 | 7.90e-03 (240) | 2.23e+02 (60) | 39157 | 1.73e+00 | 15527 of 15730 | 863 | 0/0 | 6.68e-01 |
| FM2p | 71 | 40391 | 39525 | 1.09e-02 (240) | 5.09e+02 (75) | 39249 | 6.35e+00 | 15585 of 15727 | 866 | 0/2 | 6.57e-01 |
| FM4n | 71 | 31151 | 30125 | 2.82e-03 (315) | 2.99e+02 (165) | 29878 | 1.61e+00 | 11095 of 11206 | 1026 | 2/0 | 4.00e-01 |
| FM4p | 71 | 31151 | 30079 | 3.48e-03 (330) | 6.74e+02 (390) | 29855 | 1.88e+00 | 11036 of 11138 | 1072 | 1/0 | 3.98e-01 |
| FPhipp0n | 71 | 11715 | 9944 | 1.12e-02 (120) | 7.53e+01 (300) | 9873 | 1.24e+00 | 2821 of 2845 | 1771 | 4/15 | 8.44e-01 |
| FPhipp0p | 71 | 11715 | 9941 | 1.20e-02 (135) | 1.48e+02 (30) | 9883 | 1.83e+00 | 2828 of 2845 | 1774 | 3/4 | 8.61e-01 |
| FPhipp2n | 71 | 40967 | 37123 | 1.46e-02 (0) | 2.03e+02 (0) | 36791 | 3.19e+00 | 14320 of 14460 | 3844 | 8/8 | 1.05e+00 |
| FPhipp2p | 71 | 40967 | 37042 | 2.38e-02 (0) | 2.87e+02 (345) | 36801 | 5.58e+00 | 14359 of 14451 | 3925 | 7/5 | 1.05e+00 |
| FPhipp4n | 71 | 31151 | 28264 | 3.40e-03 (165) | 2.63e+02 (285) | 27957 | 9.54e-01 | 8059 of 8164 | 2887 | 3/8 | 1.92e-01 |
| FPhipp4p | 71 | 31151 | 28108 | 4.80e-03 (165) | 1.08e+02 (135) | 27815 | 1.55e+00 | 8085 of 8181 | 3043 | 6/7 | 1.91e-01 |
| FPhitp2n | 71 | 40967 | 37208 | 1.78e-02 (0) | 2.63e+02 (30) | 36900 | 4.56e+00 | 13437 of 13578 | 3759 | 8/7 | 1.28e+00 |
| FPhitp2p | 71 | 40967 | 37000 | 2.92e-02 (0) | 8.02e+02 (120) | 36739 | 6.33e+00 | 13493 of 13597 | 3967 | 10/6 | 1.29e+00 |
| FPhitp4n | 71 | 31151 | 28477 | 2.62e-03 (135) | 2.01e+02 (135) | 28230 | 1.00e+00 | 7389 of 7472 | 2674 | 3/6 | 2.11e-01 |
| FPhitp4p | 71 | 31150 | 28204 | 3.72e-03 (135) | 3.57e+02 (135) | 28024 | 1.13e+00 | 7448 of 7514 | 2946 | 7/8 | 2.09e-01 |
| FSigmap1n | 71 | 32234 | 30011 | 5.13e-03 (90) | 6.68e+02 (90) | 29776 | 1.41e+00 | 8249 of 8338 | 2223 | 4/1 | 6.23e-01 |
| FSigmap1p | 71 | 32234 | 29967 | 6.23e-03 (90) | 5.94e+02 (135) | 29788 | 1.42e+00 | 8351 of 8434 | 2267 | 2/3 | 6.22e-01 |
| FSigmap3n | 71 | 37031 | 34632 | 1.81e-03 (345) | 1.86e+02 (405) | 34372 | 6.18e-01 | 7214 of 7291 | 2399 | 6/3 | 1.61e-01 |
| FSigmap3p | 71 | 37031 | 34403 | 2.24e-03 (360) | 1.40e+02 (375) | 34150 | 1.36e+00 | 7240 of 7316 | 2628 | 2/5 | 1.60e-01 |
| FSigmap5n | 71 | 18130 | 16569 | 1.16e-03 (420) | 1.37e+02 (375) | 16406 | 5.52e-01 | 3153 of 3197 | 1561 | 4/5 | 1.46e-01 |
| FSigmap5p | 71 | 18131 | 16562 | 1.51e-03 (405) | 1.36e+02 (360) | 16446 | 6.19e-01 | 3164 of 3190 | 1569 | 5/3 | 1.45e-01 |
| FSigmapp1n | 71 | 32234 | 30002 | 2.96e-03 (0) | 1.16e+02 (90) | 29777 | 1.14e+00 | 7673 of 7752 | 2232 | 3/5 | 4.40e-01 |
| FSigmapp1p | 71 | 32234 | 30028 | 3.27e-03 (180) | 3.10e+02 (255) | 29850 | 9.91e-01 | 7796 of 7855 | 2206 | 5/2 | 4.40e-01 |
| FSigmapp3n | 71 | 37031 | 34507 | 2.35e-03 (330) | 1.90e+02 (390) | 34204 | 4.42e-01 | 6612 of 6705 | 2524 | 1/2 | 1.53e-01 |
| FSigmapp3p | 71 | 37031 | 34441 | 2.91e-03 (330) | 3.45e+02 (435) | 34230 | 6.05e-01 | 6669 of 6737 | 2590 | 1/1 | 1.53e-01 |
| FSigmapp5n | 71 | 18130 | 16565 | 1.09e-03 (420) | 7.14e+01 (285) | 16417 | 4.97e-01 | 2928 of 2970 | 1565 | 5/5 | 1.31e-01 |
| FSigmapp5p | 71 | 18130 | 16530 | 1.40e-03 (405) | 1.67e+02 (195) | 16410 | 5.77e-01 | 2970 of 2998 | 1600 | 3/1 | 1.31e-01 |
| FDelta1n | 71 | 32228 | 25451 | 5.08e-03 (120) | 2.93e+02 (135) | 25193 | 7.32e-01 | 4816 of 4857 | 6777 | 49/40 | 6.38e-01 |
| FDelta1p | 71 | 32233 | 25283 | 5.88e-03 (135) | 1.61e+02 (105) | 25108 | 1.02e+00 | 4847 of 4889 | 6950 | 41/47 | 6.39e-01 |
| FDelta3n | 71 | 37027 | 29472 | 5.30e-04 (270) | 5.56e+01 (315) | 29187 | 2.38e-01 | 2350 of 2383 | 7555 | 29/23 | 6.12e-02 |
| FDelta3p | 71 | 37029 | 29514 | 7.33e-04 (270) | 1.06e+02 (390) | 29285 | 2.73e-01 | 2349 of 2375 | 7515 | 18/27 | 6.09e-02 |
| FDelta5n | 71 | 18112 | 14062 | 1.29e-04 (285) | 2.53e+01 (315) | 13840 | 1.01e-01 | 363 of 370 | 4050 | 22/19 | 2.42e-03 |
| FDelta5p | 71 | 18121 | 14019 | 1.35e-04 (285) | 9.34e+00 (345) | 13812 | 1.01e-01 | 356 of 361 | 4102 | 20/24 | 2.47e-03 |
| TCM | 1 | 156 | 149 | 1.77e-02 (-) | 5.56e+00 (-) | 148 | 5.56e+00 | 148 of 149 | 7 | 9/7 | 6.96e+00 |

###### Per q

| q | files | elements | rel-eligible | n rel>0.001 | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|---|---|---|
| 0 | 34 | 5390 | 5360 | 5319 | 2.92e-02 | FPhitp2p | 2.33e+02 | FSigmap1p |
| 15 | 34 | 14217 | 11661 | 11604 | 2.87e-02 | FPhitp2p | 1.32e+02 | FSigmapp1p |
| 30 | 34 | 14244 | 12971 | 12892 | 2.72e-02 | FPhitp2p | 2.63e+02 | FPhitp2n |
| 45 | 34 | 14244 | 13703 | 13630 | 2.50e-02 | FPhitp2p | 2.11e+02 | FM2p |
| 60 | 34 | 14244 | 14007 | 13929 | 2.21e-02 | FPhitp2p | 2.87e+02 | FDelta1n |
| 75 | 34 | 14244 | 14117 | 14044 | 1.89e-02 | FPhitp2p | 5.09e+02 | FM2p |
| 90 | 34 | 14244 | 14154 | 14086 | 1.55e-02 | FPhitp2p | 6.68e+02 | FSigmap1n |
| 105 | 34 | 14244 | 14172 | 14098 | 1.27e-02 | FPhipp2p | 1.61e+02 | FDelta1p |
| 120 | 34 | 14244 | 14184 | 14113 | 1.18e-02 | FPhipp0p | 8.02e+02 | FPhitp2p |
| 135 | 34 | 14244 | 14196 | 14128 | 1.20e-02 | FPhipp0p | 5.94e+02 | FSigmap1p |
| 150 | 34 | 14244 | 14189 | 14108 | 1.18e-02 | FPhipp0p | 1.76e+02 | FSigmapp1p |
| 165 | 34 | 14244 | 14201 | 14127 | 1.11e-02 | FPhipp0p | 2.99e+02 | FM4n |
| 180 | 34 | 14244 | 14200 | 14098 | 1.01e-02 | FPhipp0p | 1.53e+02 | FM2p |
| 195 | 34 | 14244 | 14204 | 14098 | 8.86e-03 | FPhipp0p | 1.67e+02 | FSigmapp5p |
| 210 | 34 | 14244 | 14208 | 14100 | 9.94e-03 | FM2p | 1.25e+02 | FM4n |
| 225 | 34 | 14244 | 14205 | 14109 | 1.07e-02 | FM2p | 5.82e+01 | FDelta1p |
| 240 | 34 | 14244 | 14206 | 14081 | 1.09e-02 | FM2p | 1.30e+02 | FDelta1n |
| 255 | 34 | 14244 | 14210 | 14097 | 1.06e-02 | FM2p | 3.10e+02 | FSigmapp1p |
| 270 | 34 | 14244 | 14200 | 14090 | 9.97e-03 | FM2p | 1.82e+02 | FPhitp2p |
| 285 | 34 | 14244 | 14205 | 14087 | 9.16e-03 | FM2p | 2.63e+02 | FPhipp4n |
| 300 | 34 | 14244 | 14203 | 14096 | 8.25e-03 | FM2p | 4.39e+02 | FM2p |
| 315 | 34 | 14244 | 14202 | 14110 | 7.31e-03 | FM2p | 1.20e+02 | FSigmap3p |
| 330 | 34 | 14244 | 14200 | 14089 | 6.37e-03 | FM2p | 9.47e+01 | FM2n |
| 345 | 34 | 14244 | 14210 | 14103 | 5.45e-03 | FM2p | 2.87e+02 | FPhipp2p |
| 360 | 34 | 14244 | 14196 | 14076 | 4.55e-03 | FM2p | 1.70e+02 | FM2n |
| 375 | 34 | 14244 | 14191 | 14074 | 3.68e-03 | FM2p | 1.40e+02 | FSigmap3p |
| 390 | 34 | 14244 | 14195 | 14065 | 2.95e-03 | FM2n | 6.74e+02 | FM4p |
| 405 | 34 | 14244 | 14194 | 14062 | 2.55e-03 | FM4p | 1.86e+02 | FSigmap3n |
| 420 | 34 | 14244 | 14191 | 14093 | 2.45e-03 | FM4p | 1.09e+02 | FSigmap1n |
| 435 | 34 | 14244 | 14185 | 14063 | 2.37e-03 | FM4p | 3.45e+02 | FSigmapp3p |
| 450 | 34 | 14244 | 14189 | 14050 | 2.23e-03 | FM4p | 1.57e+02 | FPhipp4n |
| 465 | 34 | 14244 | 14163 | 14039 | 2.07e-03 | FM4p | 2.53e+02 | FM4p |
| 480 | 34 | 14244 | 14173 | 14047 | 1.87e-03 | FM4p | 1.01e+02 | FSigmapp3p |
| 495 | 34 | 14244 | 14155 | 14029 | 1.67e-03 | FM4p | 8.89e+01 | FSigmap5p |
| 510 | 34 | 14244 | 14162 | 14058 | 1.46e-03 | FM4p | 1.03e+02 | FM2n |
| 525 | 34 | 14244 | 14166 | 14026 | 1.26e-03 | FM4p | 1.31e+02 | FM4p |
| 540 | 34 | 14244 | 14160 | 14027 | 1.07e-03 | FM4p | 4.10e+01 | FSigmapp3p |
| 555 | 34 | 14244 | 14146 | 13960 | 8.96e-04 | FM4p | 1.18e+02 | FSigmapp5p |
| 570 | 34 | 14244 | 14141 | 13982 | 7.44e-04 | FM4p | 3.51e+01 | FSigmap1p |
| 585 | 34 | 14244 | 14114 | 13964 | 6.35e-04 | FM4p | 3.58e+01 | FPhitp2n |
| 600 | 34 | 14244 | 14100 | 13931 | 5.39e-04 | FM4p | 2.50e+01 | FSigmapp1p |
| 615 | 34 | 14244 | 14095 | 13920 | 4.54e-04 | FM2p | 5.59e+01 | FM2p |
| 630 | 34 | 14244 | 14066 | 13888 | 4.08e-04 | FM2p | 5.59e+01 | FM2n |
| 645 | 34 | 14244 | 14039 | 13867 | 3.70e-04 | FM2p | 3.58e+01 | FPhitp2p |
| 660 | 34 | 14244 | 14002 | 13831 | 3.36e-04 | FM2p | 6.04e+01 | FM2n |
| 675 | 34 | 14244 | 13941 | 13798 | 3.02e-04 | FM2p | 2.40e+01 | FM2p |
| 690 | 34 | 14244 | 13916 | 13761 | 3.00e-04 | FM2n | 6.98e+01 | FM2p |
| 705 | 34 | 14244 | 13882 | 13735 | 2.86e-04 | FM2n | 2.16e+01 | FM2p |
| 720 | 34 | 14244 | 13823 | 13687 | 2.63e-04 | FM2n | 1.56e+01 | FM2p |
| 735 | 34 | 14244 | 13764 | 13634 | 2.35e-04 | FM2n | 2.92e+01 | FM2n |
| 750 | 34 | 14244 | 13660 | 13519 | 2.16e-04 | FM2p | 2.93e+01 | FM4p |
| 765 | 34 | 14244 | 13526 | 13407 | 2.01e-04 | FM2p | 1.72e+01 | FM4p |
| 780 | 34 | 14244 | 13416 | 13277 | 1.86e-04 | FM0p | 4.23e+01 | FM2p |
| 795 | 34 | 14244 | 13249 | 13138 | 1.73e-04 | FM0p | 3.48e+01 | FM0p |
| 810 | 34 | 14244 | 13053 | 12931 | 1.58e-04 | FM0p | 4.23e+01 | FM2p |
| 825 | 34 | 14244 | 12852 | 12752 | 1.43e-04 | FM0p | 2.07e+01 | FM2p |
| 840 | 34 | 14244 | 12554 | 12480 | 1.27e-04 | FM0p | 2.87e+01 | FM2p |
| 855 | 34 | 14244 | 12234 | 12137 | 1.11e-04 | FM0p | 2.07e+01 | FM2p |
| 870 | 34 | 14244 | 11854 | 11758 | 9.57e-05 | FM0p | 1.78e+01 | FM4n |
| 885 | 34 | 14244 | 11418 | 11340 | 8.14e-05 | FM0p | 2.51e+01 | FM4n |
| 900 | 34 | 14244 | 10844 | 10770 | 6.81e-05 | FM0p | 1.56e+01 | FM2n |
| 915 | 34 | 14244 | 10364 | 10305 | 5.86e-05 | FM2n | 6.66e+00 | FM4p |
| 930 | 34 | 14244 | 9752 | 9707 | 5.11e-05 | FM2n | 9.52e+00 | FM2n |
| 945 | 34 | 14242 | 9102 | 9063 | 4.37e-05 | FM2n | 6.79e+00 | FSigmapp5p |
| 960 | 34 | 14244 | 8407 | 8357 | 3.67e-05 | FM2n | 9.54e+00 | FM4p |
| 975 | 34 | 14243 | 7711 | 7661 | 3.06e-05 | FM2p | 8.80e+00 | FM4p |
| 990 | 34 | 14244 | 7000 | 6959 | 2.58e-05 | FM2p | 6.98e+00 | FM2n |
| 1005 | 34 | 14241 | 6224 | 6173 | 2.13e-05 | FM2p | 5.11e+00 | FM2p |
| 1020 | 34 | 14242 | 5344 | 5310 | 1.72e-05 | FM2p | 5.48e+00 | FM4n |
| 1035 | 34 | 14240 | 4551 | 4526 | 1.36e-05 | FM2p | 3.44e+00 | FM2n |
| 1050 | 34 | 14237 | 3738 | 3720 | 1.05e-05 | FM2p | 2.70e+00 | FM2p |

###### By part

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 2415 | 1.38e-02 | 4.78e+00 | 252 | 2130 | 0/0 |
| 1b | 22851 | 2.92e-02 | 6.66e+01 | 21659 | 778 | 14/17 |
| 2b | 977314 | 2.47e-02 | 8.02e+02 | 875430 | 94953 | 284/292 |

##### 10 worst elements by relative diff

| file | key | ref | test | value |
|---|---|---|---|---|
| N2LOsat_FPhitp2p_120_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 5, -1),(0, 2, 5, 1),3,4 | 2.497206e-06 | 2.004146e-03 | 8.02e+02 |
| N2LOsat_FM4p_390_2b.snt | 2b:(0, 2, 3, 1),(1, 0, 1, -1),(0, 2, 3, -1),(0, 2, 3, 1),2,3 | -1.731940e-06 | -1.168284e-03 | 6.74e+02 |
| N2LOsat_FSigmap1n_90_2b.snt | 2b:(0, 2, 3, 1),(1, 0, 1, -1),(0, 2, 5, 1),(1, 0, 1, -1),2,2 | 1.221934e-06 | 8.171451e-04 | 6.68e+02 |
| N2LOsat_FSigmap1n_90_2b.snt | 2b:(0, 2, 5, 1),(1, 0, 1, -1),(0, 2, 3, 1),(1, 0, 1, -1),2,2 | 1.221934e-06 | 8.171451e-04 | 6.68e+02 |
| N2LOsat_FSigmap1p_135_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 5, 1),(0, 2, 3, -1),2,3 | 1.204295e-06 | 7.166409e-04 | 5.94e+02 |
| N2LOsat_FM2p_75_2b.snt | 2b:(0, 2, 5, 1),(1, 0, 1, -1),(0, 2, 5, -1),(1, 0, 1, 1),2,2 | -1.162357e-06 | 5.902644e-04 | 5.09e+02 |
| N2LOsat_FM2p_75_2b.snt | 2b:(0, 2, 5, -1),(1, 0, 1, 1),(0, 2, 5, 1),(1, 0, 1, -1),2,2 | -1.162357e-06 | 5.902644e-04 | 5.09e+02 |
| N2LOsat_FM2p_300_2b.snt | 2b:(0, 2, 5, -1),(1, 0, 1, 1),(0, 2, 5, -1),(0, 2, 5, 1),3,4 | 4.621026e-06 | 2.033654e-03 | 4.39e+02 |
| N2LOsat_FPhitp4p_135_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 3, -1),(0, 2, 5, 1),(0, 2, 3, -1),3,3 | -3.090592e-06 | -1.105590e-03 | 3.57e+02 |
| N2LOsat_FSigmapp3p_435_2b.snt | 2b:(0, 2, 3, 1),(1, 0, 1, -1),(0, 2, 3, 1),(1, 0, 1, -1),2,2 | -1.036567e-06 | 3.564422e-04 | 3.45e+02 |

##### 10 worst elements by absolute diff

| file | key | ref | test | value |
|---|---|---|---|---|
| N2LOsat.snt | 0b:- | -1.182895e+02 | -1.191874e+02 | 8.98e-01 |
| N2LOsat.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 4.922652e+00 | 5.765414e+00 | 8.43e-01 |
| N2LOsat.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 1.750397e+00 | 2.476480e+00 | 7.26e-01 |
| N2LOsat.snt | 1b:(1, 0, 1, -1),(1, 0, 1, -1) | -2.372864e-01 | 2.679330e-01 | 5.05e-01 |
| N2LOsat.snt | 1b:(1, 0, 1, 1),(1, 0, 1, 1) | -3.088239e+00 | -2.738257e+00 | 3.50e-01 |
| N2LOsat.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, -1),(0, 2, 3, 1),1 | 5.393245e-01 | 8.063093e-01 | 2.67e-01 |
| N2LOsat.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 5, -1),(0, 2, 5, 1),1 | 1.579394e+00 | 1.387195e+00 | 1.92e-01 |
| N2LOsat.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(1, 0, 1, -1),(1, 0, 1, 1),1 | -3.207878e-01 | -5.037561e-01 | 1.83e-01 |
| N2LOsat.snt | 2b:(1, 0, 1, -1),(1, 0, 1, -1),(1, 0, 1, -1),(1, 0, 1, -1),0 | -4.362193e-02 | 1.307954e-01 | 1.74e-01 |
| N2LOsat.snt | 2b:(0, 2, 3, -1),(0, 2, 5, 1),(0, 2, 3, 1),(0, 2, 5, -1),2 | 2.304076e+00 | 2.443759e+00 | 1.40e-01 |

#### v3 vs v2, DN2LOGO

Absolute |test - ref| for all elements; relative only for |ref| >= 1e-06; flag threshold rel > 0.001.


##### Flow (results.json, flow.txt)

| quantity | ref | test | diff |
|---|---|---|---|
| final E0 (results.json) | -217.0702962989 | -216.9873334514 | 8.30e-02 |
| num_omegas (results.json) | 2 | 40 | 38 |
| num_omegas (flow.txt, last row) | 2 | 40 | 38 |
| stage 1 final E0 | -157.3565744454 | -157.2525908373 | 1.04e-01 |
| stage 1 final s | 11.169789 | 10.430953 | 7.39e-01 |
| stage 1 final step | 43 | 36 | -7 |
| stage 2 final E0 | -217.0702962989 | -216.9873334514 | 8.30e-02 |
| stage 2 final s | 47.305050 | 41.726708 | 5.58e+00 |
| stage 2 final step | 465 | 135 | -330 |

##### Flow directory files (valence H, TCM)

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 1 | 165 | 165 | 6.24e-01 (-) | 1.88e+00 (-) | 163 | 1.88e+00 | 163 of 165 | 0 | 0/0 | 1.16e+02 |
| TCM(flow) | 1 | 162 | 157 | 3.96e-02 (-) | 5.94e+00 (-) | 156 | 5.94e+00 | 155 of 156 | 5 | 9/5 | 7.08e+00 |

H by part:

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 1 | 1.86e-01 | 1.60e-03 | 1 | 0 | 0/0 |
| 1b | 6 | 6.24e-01 | 1.52e+00 | 6 | 0 | 0/0 |
| 2b | 158 | 2.01e-01 | 1.88e+00 | 156 | 0 | 0/0 |

##### Farm: 2415 file pairs compared

* files: ref 2415, test 2415; TCM copies: ref 1, test 1 (first used)

Missing files:

* none

###### Per operator column

| column | files | elements | rel-eligible | max abs (q) | max rel (q) | n rel>0.001 | max rel, \|ref\|>=0.001 | n rel>0.001 of n \|ref\|>=0.001 | n small | only ref/test | max abs ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FM0n | 71 | 11554 | 10610 | 5.60e-03 (165) | 5.78e+01 (330) | 10458 | 1.26e+00 | 3320 of 3418 | 944 | 8/5 | 8.00e+00 |
| FM0p | 71 | 11554 | 10638 | 9.88e-03 (135) | 8.79e+01 (225) | 10504 | 6.70e-01 | 3372 of 3468 | 916 | 1/4 | 8.00e+00 |
| FM2n | 71 | 40391 | 38724 | 7.24e-03 (210) | 2.29e+02 (90) | 38258 | 2.40e+00 | 15704 of 16004 | 1667 | 2/2 | 6.66e-01 |
| FM2p | 71 | 40389 | 38915 | 6.60e-03 (255) | 4.19e+02 (135) | 38507 | 3.43e+00 | 15797 of 16063 | 1474 | 2/3 | 6.57e-01 |
| FM4n | 71 | 31151 | 29217 | 2.05e-03 (360) | 3.03e+02 (360) | 28819 | 9.20e-01 | 11119 of 11285 | 1934 | 3/4 | 4.20e-01 |
| FM4p | 71 | 31151 | 29466 | 3.14e-03 (345) | 7.62e+01 (135) | 29168 | 1.49e+00 | 11207 of 11358 | 1685 | 4/1 | 4.08e-01 |
| FPhipp0n | 71 | 11715 | 9525 | 1.03e-02 (0) | 1.36e+02 (285) | 9406 | 8.22e-01 | 2890 of 2935 | 2190 | 1/9 | 8.05e-01 |
| FPhipp0p | 71 | 11715 | 9494 | 1.12e-02 (90) | 3.22e+01 (330) | 9376 | 1.22e+00 | 2859 of 2901 | 2221 | 9/3 | 8.31e-01 |
| FPhipp2n | 71 | 40963 | 35214 | 8.53e-03 (0) | 3.89e+02 (0) | 34761 | 1.79e+00 | 14587 of 14812 | 5749 | 11/13 | 1.03e+00 |
| FPhipp2p | 71 | 40966 | 35173 | 1.65e-02 (0) | 1.10e+03 (210) | 34789 | 2.55e+00 | 14508 of 14692 | 5793 | 15/9 | 1.03e+00 |
| FPhipp4n | 71 | 31149 | 26712 | 1.60e-03 (210) | 8.44e+01 (300) | 26252 | 7.74e-01 | 8100 of 8248 | 4437 | 10/9 | 1.90e-01 |
| FPhipp4p | 71 | 31151 | 26662 | 2.08e-03 (165) | 1.71e+02 (180) | 26282 | 1.47e+00 | 8063 of 8220 | 4489 | 4/5 | 1.89e-01 |
| FPhitp2n | 71 | 40965 | 35324 | 1.04e-02 (0) | 3.89e+02 (0) | 34898 | 1.82e+00 | 13819 of 14007 | 5641 | 14/11 | 1.26e+00 |
| FPhitp2p | 71 | 40966 | 35963 | 2.02e-02 (0) | 3.24e+02 (90) | 35608 | 2.83e+00 | 13896 of 14055 | 5003 | 16/15 | 1.26e+00 |
| FPhitp4n | 71 | 31150 | 27019 | 9.83e-04 (150) | 1.20e+02 (210) | 26655 | 4.47e-01 | 7400 of 7541 | 4131 | 6/10 | 2.08e-01 |
| FPhitp4p | 71 | 31151 | 27564 | 1.72e-03 (135) | 6.85e+01 (240) | 27198 | 9.13e-01 | 7401 of 7543 | 3587 | 12/10 | 2.07e-01 |
| FSigmap1n | 71 | 32234 | 28940 | 4.17e-03 (0) | 9.30e+01 (195) | 28624 | 9.80e-01 | 8607 of 8716 | 3294 | 5/7 | 6.30e-01 |
| FSigmap1p | 71 | 32234 | 28995 | 5.96e-03 (60) | 1.42e+02 (0) | 28744 | 9.59e-01 | 8760 of 8849 | 3239 | 10/3 | 6.28e-01 |
| FSigmap3n | 71 | 37031 | 33304 | 1.35e-03 (360) | 8.71e+01 (285) | 32923 | 6.80e-01 | 7505 of 7666 | 3727 | 6/5 | 1.61e-01 |
| FSigmap3p | 71 | 37031 | 33439 | 2.22e-03 (345) | 1.12e+02 (375) | 33060 | 5.02e-01 | 7562 of 7699 | 3592 | 11/10 | 1.61e-01 |
| FSigmap5n | 71 | 18128 | 15898 | 5.72e-04 (405) | 2.28e+01 (390) | 15684 | 3.03e-01 | 3210 of 3281 | 2230 | 8/8 | 1.47e-01 |
| FSigmap5p | 71 | 18130 | 15885 | 6.08e-04 (405) | 9.55e+01 (195) | 15679 | 3.43e-01 | 3237 of 3309 | 2245 | 9/3 | 1.46e-01 |
| FSigmapp1n | 71 | 32234 | 29028 | 2.95e-03 (0) | 8.13e+01 (495) | 28747 | 6.91e-01 | 8012 of 8125 | 3206 | 2/3 | 4.45e-01 |
| FSigmapp1p | 71 | 32232 | 29141 | 3.99e-03 (0) | 1.92e+02 (105) | 28867 | 9.50e-01 | 8120 of 8242 | 3091 | 7/6 | 4.44e-01 |
| FSigmapp3n | 71 | 37031 | 33397 | 1.33e-03 (360) | 1.24e+02 (195) | 32980 | 4.92e-01 | 6859 of 7011 | 3634 | 8/6 | 1.54e-01 |
| FSigmapp3p | 71 | 37030 | 33575 | 1.62e-03 (330) | 8.72e+01 (345) | 33231 | 5.40e-01 | 6916 of 7041 | 3455 | 6/4 | 1.54e-01 |
| FSigmapp5n | 71 | 18127 | 15897 | 5.06e-04 (420) | 6.36e+01 (435) | 15647 | 4.53e-01 | 2869 of 2941 | 2230 | 9/8 | 1.32e-01 |
| FSigmapp5p | 71 | 18130 | 15989 | 5.38e-04 (405) | 3.44e+01 (375) | 15777 | 4.07e-01 | 2913 of 2974 | 2141 | 6/5 | 1.31e-01 |
| FDelta1n | 71 | 32204 | 24586 | 3.20e-03 (0) | 2.25e+02 (30) | 24322 | 5.36e-01 | 4789 of 4858 | 7618 | 48/46 | 6.43e-01 |
| FDelta1p | 71 | 32210 | 24447 | 4.47e-03 (105) | 1.12e+02 (150) | 24174 | 7.21e-01 | 4913 of 4974 | 7763 | 48/70 | 6.42e-01 |
| FDelta3n | 71 | 37010 | 28350 | 4.28e-04 (285) | 4.97e+01 (225) | 27947 | 2.06e-01 | 2336 of 2400 | 8660 | 50/50 | 6.15e-02 |
| FDelta3p | 71 | 37006 | 28245 | 4.94e-04 (270) | 3.00e+01 (225) | 27936 | 2.17e-01 | 2433 of 2476 | 8761 | 59/47 | 6.13e-02 |
| FDelta5n | 71 | 18104 | 13407 | 9.41e-05 (285) | 1.24e+01 (225) | 13127 | 6.64e-02 | 363 of 371 | 4697 | 24/19 | 2.32e-03 |
| FDelta5p | 71 | 18106 | 13384 | 9.75e-05 (285) | 1.60e+01 (390) | 13147 | 7.31e-02 | 369 of 378 | 4722 | 27/26 | 2.42e-03 |
| TCM | 1 | 162 | 157 | 3.96e-02 (-) | 5.94e+00 (-) | 156 | 5.94e+00 | 155 of 156 | 5 | 9/5 | 7.08e+00 |

###### Per q

| q | files | elements | rel-eligible | n rel>0.001 | max abs | worst col (abs) | max rel | worst col (rel) |
|---|---|---|---|---|---|---|---|---|
| 0 | 34 | 5390 | 5358 | 5312 | 2.02e-02 | FPhitp2p | 3.89e+02 | FPhipp2n |
| 15 | 34 | 14209 | 11633 | 11547 | 1.99e-02 | FPhitp2p | 3.88e+02 | FPhipp2n |
| 30 | 34 | 14244 | 12985 | 12902 | 1.89e-02 | FPhitp2p | 2.25e+02 | FDelta1n |
| 45 | 34 | 14244 | 13678 | 13566 | 1.75e-02 | FPhitp2p | 1.63e+02 | FPhitp2n |
| 60 | 34 | 14244 | 13975 | 13868 | 1.60e-02 | FPhitp2p | 1.23e+02 | FPhitp2n |
| 75 | 34 | 14244 | 14105 | 14013 | 1.42e-02 | FPhitp2p | 2.02e+02 | FPhitp2p |
| 90 | 34 | 14244 | 14158 | 14036 | 1.24e-02 | FPhitp2p | 3.24e+02 | FPhitp2p |
| 105 | 34 | 14244 | 14178 | 14056 | 1.12e-02 | FPhipp0p | 1.92e+02 | FSigmapp1p |
| 120 | 34 | 14244 | 14187 | 14049 | 1.10e-02 | FPhipp0p | 1.08e+02 | FPhipp2n |
| 135 | 34 | 14244 | 14195 | 14043 | 1.05e-02 | FPhipp0p | 4.19e+02 | FM2p |
| 150 | 34 | 14244 | 14196 | 14046 | 9.87e-03 | FPhipp0p | 1.12e+02 | FDelta1p |
| 165 | 34 | 14244 | 14206 | 14029 | 9.25e-03 | FM0p | 1.75e+02 | FM2n |
| 180 | 34 | 14244 | 14206 | 14049 | 8.46e-03 | FM0p | 1.71e+02 | FPhipp4p |
| 195 | 34 | 14244 | 14207 | 14062 | 7.50e-03 | FM0p | 1.24e+02 | FSigmapp3n |
| 210 | 34 | 14244 | 14200 | 14036 | 7.24e-03 | FM2n | 1.10e+03 | FPhipp2p |
| 225 | 34 | 14244 | 14204 | 14038 | 7.21e-03 | FM2n | 1.38e+02 | FPhipp4p |
| 240 | 34 | 14244 | 14203 | 14042 | 7.02e-03 | FM2n | 2.70e+02 | FM2p |
| 255 | 34 | 14244 | 14202 | 14035 | 6.68e-03 | FM2n | 1.07e+02 | FM2p |
| 270 | 34 | 14244 | 14203 | 14026 | 6.47e-03 | FM2p | 6.90e+01 | FSigmap1p |
| 285 | 34 | 14244 | 14205 | 14021 | 6.18e-03 | FM2p | 1.36e+02 | FPhipp0n |
| 300 | 34 | 14244 | 14198 | 14031 | 5.80e-03 | FM2p | 1.31e+02 | FPhitp2p |
| 315 | 34 | 14244 | 14198 | 14005 | 5.37e-03 | FM2p | 7.20e+01 | FPhipp2n |
| 330 | 34 | 14244 | 14194 | 14008 | 4.93e-03 | FM2p | 8.05e+01 | FPhipp2n |
| 345 | 34 | 14244 | 14199 | 14022 | 4.51e-03 | FM2p | 2.38e+02 | FPhipp2p |
| 360 | 34 | 14244 | 14199 | 14029 | 4.10e-03 | FM2p | 3.03e+02 | FM4n |
| 375 | 34 | 14244 | 14187 | 14017 | 3.71e-03 | FM2p | 1.12e+02 | FSigmap3p |
| 390 | 34 | 14244 | 14191 | 13993 | 3.33e-03 | FM2p | 1.52e+02 | FM2n |
| 405 | 34 | 14244 | 14190 | 14015 | 2.96e-03 | FM2p | 4.28e+01 | FM2n |
| 420 | 34 | 14244 | 14187 | 13992 | 2.61e-03 | FM2n | 8.47e+01 | FSigmap3p |
| 435 | 34 | 14244 | 14179 | 13993 | 2.37e-03 | FM2n | 6.36e+01 | FSigmapp5n |
| 450 | 34 | 14244 | 14179 | 14009 | 2.12e-03 | FM2n | 1.16e+02 | FPhipp4p |
| 465 | 34 | 14244 | 14176 | 14016 | 1.86e-03 | FM2n | 1.09e+02 | FPhipp4p |
| 480 | 34 | 14244 | 14164 | 13975 | 1.65e-03 | FM4p | 4.50e+01 | FPhitp2n |
| 495 | 34 | 14244 | 14148 | 13951 | 1.49e-03 | FM4p | 8.13e+01 | FSigmapp1n |
| 510 | 34 | 14244 | 14150 | 13972 | 1.33e-03 | FM4p | 1.06e+02 | FM2n |
| 525 | 34 | 14244 | 14153 | 13966 | 1.17e-03 | FM4p | 4.09e+01 | FPhipp4p |
| 540 | 34 | 14244 | 14159 | 13962 | 1.02e-03 | FM4p | 6.98e+01 | FM2n |
| 555 | 34 | 14244 | 14123 | 13961 | 8.80e-04 | FM4p | 2.90e+01 | FSigmap1n |
| 570 | 34 | 14244 | 14133 | 13944 | 7.53e-04 | FM4p | 2.45e+01 | FM4p |
| 585 | 34 | 14244 | 14122 | 13919 | 6.40e-04 | FM4p | 3.75e+01 | FM4p |
| 600 | 34 | 14244 | 14110 | 13943 | 5.43e-04 | FM4p | 7.63e+01 | FM4n |
| 615 | 34 | 14244 | 14124 | 13902 | 4.58e-04 | FM4p | 5.97e+01 | FM4n |
| 630 | 34 | 14244 | 14070 | 13869 | 3.97e-04 | FM4p | 3.98e+01 | FM2n |
| 645 | 34 | 14244 | 14055 | 13850 | 3.41e-04 | FM4p | 9.37e+01 | FM4n |
| 660 | 34 | 14244 | 14014 | 13820 | 2.90e-04 | FM4p | 2.22e+01 | FM4p |
| 675 | 34 | 14244 | 13986 | 13787 | 2.44e-04 | FM4p | 2.04e+01 | FM2n |
| 690 | 34 | 14244 | 13939 | 13722 | 2.04e-04 | FM0p | 1.22e+01 | FM2p |
| 705 | 34 | 14244 | 13851 | 13626 | 1.89e-04 | FM2p | 1.84e+01 | FM2n |
| 720 | 34 | 14244 | 13801 | 13573 | 1.71e-04 | FM2p | 3.04e+01 | FM2n |
| 735 | 34 | 14244 | 13695 | 13461 | 1.50e-04 | FM2p | 2.32e+01 | FM2n |
| 750 | 34 | 14244 | 13568 | 13341 | 1.28e-04 | FM0p | 1.75e+01 | FM2n |
| 765 | 34 | 14244 | 13338 | 13164 | 1.13e-04 | FM0p | 1.24e+01 | FM2p |
| 780 | 34 | 14244 | 13146 | 12942 | 9.86e-05 | FM0p | 1.08e+01 | FM2p |
| 795 | 34 | 14244 | 12877 | 12659 | 8.56e-05 | FM0p | 2.23e+01 | FSigmap1n |
| 810 | 34 | 14244 | 12568 | 12420 | 7.36e-05 | FM0p | 1.38e+01 | FM2p |
| 825 | 34 | 14244 | 12161 | 12006 | 6.27e-05 | FM0p | 1.24e+01 | FM2p |
| 840 | 34 | 14244 | 11675 | 11543 | 5.28e-05 | FM0p | 4.66e+00 | FSigmapp1n |
| 855 | 34 | 14244 | 11108 | 10965 | 4.40e-05 | FM0p | 6.75e+00 | FM2p |
| 870 | 34 | 14244 | 10507 | 10400 | 3.61e-05 | FM0p | 7.27e+00 | FM4p |
| 885 | 34 | 14244 | 9850 | 9728 | 3.23e-05 | FM2p | 3.85e+00 | FM4n |
| 900 | 34 | 14243 | 9213 | 9109 | 2.87e-05 | FM2p | 3.40e+00 | FSigmapp1n |
| 915 | 34 | 14242 | 8572 | 8474 | 2.50e-05 | FM2p | 5.52e+00 | FM2n |
| 930 | 34 | 14243 | 7770 | 7709 | 2.14e-05 | FM2p | 2.65e+00 | FM2n |
| 945 | 34 | 14244 | 6888 | 6850 | 1.80e-05 | FM2p | 3.65e+00 | FM2p |
| 960 | 34 | 14241 | 6030 | 5988 | 1.49e-05 | FM2p | 2.13e+00 | FM4p |
| 975 | 34 | 14244 | 5184 | 5143 | 1.21e-05 | FM2p | 1.76e+00 | FM4p |
| 990 | 34 | 14238 | 4292 | 4258 | 9.66e-06 | FM2p | 2.03e+00 | FM2p |
| 1005 | 34 | 14233 | 3466 | 3445 | 7.60e-06 | FM2p | 1.80e+00 | FM2p |
| 1020 | 34 | 14221 | 2754 | 2729 | 5.89e-06 | FM2p | 1.29e+00 | FM2p |
| 1035 | 34 | 14207 | 2077 | 2059 | 4.49e-06 | FM2p | 1.41e+00 | FM2p |
| 1050 | 34 | 14186 | 1525 | 1514 | 3.37e-06 | FM2p | 1.16e+00 | FM4p |

###### By part

| part | elements | max abs | max rel | n rel>flag | n small | only ref/test |
|---|---|---|---|---|---|---|
| 0b | 2415 | 3.96e-02 | 1.27e+01 | 220 | 2136 | 0/0 |
| 1b | 22843 | 1.97e-02 | 3.70e+01 | 20613 | 1389 | 24/18 |
| 2b | 977197 | 2.02e-02 | 1.10e+03 | 840878 | 126646 | 446/426 |

##### 10 worst elements by relative diff

| file | key | ref | test | value |
|---|---|---|---|---|
| DN2LOGO_FPhipp2p_210_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, -1),(0, 2, 3, 1),3,3 | 1.055957e-06 | 1.159545e-03 | 1.10e+03 |
| DN2LOGO_FM2p_135_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 5, -1),(0, 2, 3, -1),2,3 | -1.406393e-06 | -5.904802e-04 | 4.19e+02 |
| DN2LOGO_FPhipp2n_0_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, 1),(1, 0, 1, -1),0,2 | -4.590335e-06 | 1.779849e-03 | 3.89e+02 |
| DN2LOGO_FPhitp2n_0_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, 1),(1, 0, 1, -1),0,2 | -5.621990e-06 | 2.179861e-03 | 3.89e+02 |
| DN2LOGO_FPhipp2n_15_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 3, 1),(0, 2, 5, 1),(0, 2, 3, 1),3,4 | 1.229867e-06 | -4.764998e-04 | 3.88e+02 |
| DN2LOGO_FPhitp2p_90_2b.snt | 2b:(0, 2, 3, 1),(1, 0, 1, 1),(0, 2, 5, 1),(1, 0, 1, 1),2,2 | -1.639264e-06 | 5.297627e-04 | 3.24e+02 |
| DN2LOGO_FPhitp2p_90_2b.snt | 2b:(0, 2, 5, 1),(1, 0, 1, 1),(0, 2, 3, 1),(1, 0, 1, 1),2,2 | -1.639264e-06 | 5.297627e-04 | 3.24e+02 |
| DN2LOGO_FM4n_360_2b.snt | 2b:(0, 2, 3, 1),(0, 2, 3, 1),(0, 2, 5, 1),(0, 2, 3, 1),2,3 | -3.383023e-06 | 1.021470e-03 | 3.03e+02 |
| DN2LOGO_FM2p_240_2b.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, -1),(0, 2, 3, 1),1,3 | 1.098259e-06 | -2.951172e-04 | 2.70e+02 |
| DN2LOGO_FPhipp2p_345_2b.snt | 2b:(0, 2, 5, 1),(0, 2, 3, -1),(0, 2, 3, 1),(1, 0, 1, -1),2,2 | -1.796114e-06 | 4.258694e-04 | 2.38e+02 |

##### 10 worst elements by absolute diff

| file | key | ref | test | value |
|---|---|---|---|---|
| DN2LOGO.snt | 1b:(0, 2, 3, -1),(0, 2, 3, -1) | 4.298564e+00 | 4.922935e+00 | 6.24e-01 |
| DN2LOGO.snt | 1b:(1, 0, 1, -1),(1, 0, 1, -1) | -3.207795e-01 | 1.663758e-01 | 4.87e-01 |
| DN2LOGO.snt | 1b:(0, 2, 3, 1),(0, 2, 3, 1) | 1.175729e+00 | 1.643128e+00 | 4.67e-01 |
| DN2LOGO.snt | 1b:(1, 0, 1, 1),(1, 0, 1, 1) | -3.323340e+00 | -2.949041e+00 | 3.74e-01 |
| DN2LOGO.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, -1),(0, 2, 3, 1),1 | 2.325332e-01 | 4.335341e-01 | 2.01e-01 |
| DN2LOGO.snt | 0b:- | -1.162915e+02 | -1.164770e+02 | 1.86e-01 |
| DN2LOGO.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(1, 0, 1, -1),(1, 0, 1, 1),1 | -4.770100e-01 | -6.519757e-01 | 1.75e-01 |
| DN2LOGO.snt | 2b:(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 3, -1),(0, 2, 3, -1),0 | 8.771005e-01 | 1.002413e+00 | 1.25e-01 |
| DN2LOGO.snt | 2b:(0, 2, 3, -1),(0, 2, 3, 1),(0, 2, 3, -1),(0, 2, 5, 1),3 | -1.735328e+00 | -1.854896e+00 | 1.20e-01 |
| DN2LOGO.snt | 2b:(0, 2, 3, -1),(1, 0, 1, 1),(0, 2, 5, 1),(1, 0, 1, -1),2 | 1.957371e+00 | 2.072079e+00 | 1.15e-01 |
