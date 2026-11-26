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

## 4. Timings
