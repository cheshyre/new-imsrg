# 27Al production runs (design/50 C9; lead D, 2026-11-26)

First production-size runs of the 27Al pipeline, scalar parts only (H and TCM), variant 1
(compat `imsrg_fork_79e9330`, unsplit Magnus). The owner exempted these runs from golden rule 3;
tests and CI stay at emax <= 6.

* Configuration: `tools/al27/n2losat_v1.toml` (translated from the archived SLURM template with
  `imsrg --print-config <legacy args>`, then `run.compat`, checkpoints, `omega_second_format`).
* Scripts: `tools/al27/run_flow.sh` (one step on the `imsrg-c` node, 56 OpenMP threads, `/usr/bin/time -v`),
  `tools/al27/build_release.sh`, `tools/al27/compare_snt.py`.
* Run directories (scratch, not in the repo): `$IMSRG_WORK/runs/al27/v1/<hamiltonian>/`
  (logs `<step>.log`, checkpoints `checkpoint/`, `.snt`, Omegas).
* Machine: one Frontier node (AMD EPYC 7A53, 56 usable cores, 512 GB), `frontier-cpu-release`
  build (PrgEnv-gnu, Kokkos OpenMP), no GPU.

## 1. Scalability fixes found by the first attempt

The first attempt never got past building the Hamiltonian: serial host loops that are
milliseconds at emax <= 6 dominate at emax 14.

| Step | Before | After | Fix |
|---|---|---|---|
| `MoshinskyTable(e2max = 28)` for TCm (1.9e8 double-double brackets) | > 10 min, 1 thread (killed) | < 1 min | one host task per (lambda, cm tuple) row (`angmom/tables.cc`) |
| `detail::AddTwoBody` for TCm (~1e9 two-body elements) | > 8 min, 1 thread (killed) | ~1 min | one host task per (channel, bra row) (`opslib/detail/pair_engine.h`) |
| me2j read (emax 16 file, 2.2 GB gz, 1.7e8 values) | ~2-3 min | | single-threaded gzip text parse (fine) |
| NO2B read (48 GB, single precision) | ~25 min (~35 MB/s) | ~6-10 min | dense state tables instead of two hash lookups per value (`operator/no2b.cc`, branch al27-production-no2b-reader); imsrg++: 379 s |
| `hf::detail::Monopole3B` (~2e8 monopoles of get_pn) | > 15 min, 1 thread (killed) | ~1-2 min | one task per (i, j) pair, lists concatenated in order |
| `hf::detail::Monopole2B`, `AddThreeBodyContribution` (NO2B -> 2B, HF and NAT basis) | serial | parallel | one task per row; density pairs listed once |

All of these are bit-identical to the serial loops (each element is computed by the same calls in
the same order).

## 2. HF, NAT, truncation (N2LOsat)

| Quantity | archive (README 3.3) | ours |
|---|---|---|
| HF iterations | 33 | 33 |
| e1hf / e2hf / e3hf | 443.7565509 / -573.2507671 / 73.0669254 | 443.7565509 / -573.2507671 / 73.0669254 |
| E_HF (= E_Nat) | -56.4272908 | -56.42729079 (NAT: -56.42729079) |
| EMP2 (emax 14) | -165.52711893 | -165.52711893 |
| EMP2 / EMP3 (emax 10) | -164.949189 / -9.715505 | -164.949189 / -9.715505 (E3 pp/hh/ph 13.873787 / -6.152639 / -17.436652, all digits) |

**Mismatch found and fixed:** the first run gave EMP2 = -165.50073912 (0.026 MeV off) with
identical HF numbers. The fork reference build (`tools/al27/fork_hf.sh`, emax 6, e3max 18,
`method=MP3`) localized it: HF-basis EMP2 identical, plain NAT already different
(E_Nat -52.59268 vs -52.51892). Cause: our `Mbpt2DensityMatrix` dropped the PP elements between a
particle and a *partially filled* hole of the same channel (imsrg++ `DensityMatrixPP` tests
1 - n_b, not the particle list; 27Al: proton 1d5/2 x 0d5/2 with weight 1/6). Closed-shell tests
could not see it. Fixed in `mbpt/natural_orbitals.cc` with a transcription test; afterwards emax 6
(EMP2, EMP3, E_Nat) and emax 14 agree with the fork to all printed digits.

## 3. Flow (N2LOsat)

| | archive | ours |
|---|---|---|
| Stage 1 end: steps, s, E0 | 37, 13.32201, -131.499115156 | 37, 13.32201, -131.499115156 |
| Stage 1 end: eta 1b/2b, Omega 1b/2b | 6.10784e-4 / 6.93899e-4, 0.387846026 / 2.492488952 | 6.10784e-4 / 6.93899e-4, 0.387846026 / 2.492488952 |
| Stage 2 step 1: E0 | -134.053288279 | -134.053288282 |
| Stage 2 steps 50 / 78 / 99: E0 | -196.430219541 / -205.902296147 / -209.800918884 | -196.430219541 / -205.902296147 / -209.800918885 |
| Stage 2 end: steps, s, E0 | 272, 57.65518, -213.961459717 | 272, 57.65518, -213.961459718 |
| Stage 2 end: eta 2b, Omega 2b | 9.74e-4, 7.8282 | 9.74269e-4, 7.828197 |
| Zero body of H in `.snt` (16O NO) | -118.293 | -118.292961 |
| TCM zero body, HF / IMSRG (27Al NO) | 9.232568390638519 / 6.956645787278085 | 9.232568390638521 / 6.956645787282553 |

The flow crossed a job boundary: checkpoint 1 (stage 2, step 78) was written on frontier00016,
the job ended at step 99, and the restart on frontier02161 (a later job, after a pause) reproduced
steps 79 and 80 bit for bit and continued to the end.

**Valence Hamiltonian `.snt`** (`tools/al27/compare_snt.py` against the archived
`J00011053_..._INTFILE.snt`, identical in all N2LOsat jobs): all 6 one-body and 158 two-body
elements present, **max |diff| 5.0e-7** (1b 4.5e-7), i.e. the rounding of the archive's 7
significant digits; target 1e-4.

**TCM `.snt`**: the 6 one-body elements agree to 4.7e-8 and the 143 two-body elements we write
agree to 5.0e-9. Our writer (imsrg-devel format) drops two-body elements below 1e-7 times the 2B
norm of TCM in the emax-10 space (threshold ~1e-3 here); the fork writes all of them, so 15
archived elements (|v| <= 1.07e-3) are missing from our file. Lead B's fork number format
(`TokyoNumberFormat::kFork79e9330`, branch feature/al27-tokyo-fork-format, not merged here) writes
them; the driver must select it under `compat = "imsrg_fork_79e9330"` (AL-D3). Isospin averaging
(`--isospin-average-test`) changes nothing for TCM at this precision.

**Omegas** (both formats, `omega_second_format`): `N2LOsat_omega_00{0,1}.opb` (293 MB each,
native float64) and `N2LOsat_me2jp_omega_00{0,1}.me1j/.me2jp.bin` (74 MB each, float32), with
manifests, basis C and occupations, in `$IMSRG_WORK/runs/al27/v1/N2LOsat/`.

## 3b. DeltaN2LOGO(394) (`tools/al27/dn2logo_v1.toml`)

| Quantity | archive (README 3.3) | ours |
|---|---|---|
| HF iterations, e1/e2/e3 | 29, 505.9244222 / -705.5858928 / 88.7038044 | 29, 505.9244222 / -705.5858928 / 88.7038044 |
| E_HF (= E_Nat) | -110.9576662 | -110.95766616 |
| EMP2 (emax 14) | -104.47937337 | -104.47937337 |
| EMP2 / EMP3 (emax 10) | -104.362419 / -9.735131 | -104.362419 / -9.735131 |
| Stage 1 end: steps, s, E0, eta 2b, Omega 2b | 30, 11.30557, -157.357310359, 8.27e-4, 2.2179 | 30, 11.30557, -157.357310359, 8.27073e-4, 2.217947 |
| Stage 2 end: steps, s, E0, eta 2b, Omega 2b | 253, 47.14639, -217.066364762, 9.77e-4, 7.8900 | 253, 47.14639, -217.066364762, 9.76786e-4, 7.889981 |
| Zero body of H in `.snt` | -116.291 | -116.291242 |
| TCM zero body, HF / IMSRG | 10.617215428337673 / 7.083534497103805 | 10.617215428337671 / 7.083534497103864 |
| H `.snt`, max abs deviation (6 1b + 158 2b) | | **5.0e-7** (1b 4.5e-7) |
| TCM `.snt` | | 1b 4.7e-8; 150 written 2b elements 5e-9; 8 below the devel writer threshold missing (AL-D3) |

Job boundary: killed at stage 2 step ~245 (job end), restarted from checkpoint 6 (step 220,
15-minute interval) in a later job; steps 221-245 reproduced bit for bit. Timings: stage 1
12.3 s/step, stage 2 23.7 s/step (27 s/step near the end), restart run 33.7 min in total
(~17 min preparation, 33 steps, operators and writers). Omegas in both formats in
`$IMSRG_WORK/runs/al27/v1/DN2LOGO/`.

## 4. Timings (N2LOsat, one node, 56 threads, release build)

| Step | ours | imsrg++ archive (48 threads, JUWELS) |
|---|---|---|
| me2j read (emax 16 file) + TRel (Moshinsky e2max 28) | ~4 min | |
| NO2B read (48 GB) | ~6-10 min (I/O bound, ~80 MB/s) | 379 s |
| HF (33 iterations) + NAT + NO2B contraction + truncation + MBPT | ~3-5 min | |
| preparation in total (start to flow step 0) | 16-17 min | 2670 s (44.5 min) |
| stage 1, per step (37 steps) | 13.8 s | 21.2 s |
| stage 2, per step (steps 79-272) | 26.4 s (24.6 s at step 80, 27.8 s at the end) | 35.4 s |
| checkpoint write (H, H_saved, 2 Omegas, native, Lustre) | 1.1-2.3 s | - |
| TCM transform + both writers + Omegas in two formats | ~2 min | |
| peak RSS | 65.8 GB | 78.7 GB |
| whole flow (both stages), without the job boundary | ~1 h 36 min + prep | 3.6 h (13843 s incl. 5 operators) |

The restart costs the whole preparation again (AL-D1). GPU not used yet (M14).
