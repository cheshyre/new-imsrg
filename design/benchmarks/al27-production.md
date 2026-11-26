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
| NO2B read (48 GB, single precision) | ~25 min (~35 MB/s) | | open: two hash lookups per value in `No2bThreeBody::set_isospin` (imsrg++: 379 s) |

## 2. HF, NAT, truncation (N2LOsat)

(filled in below as the run proceeds)

## 3. Flow (N2LOsat)

## 4. Timings
