# Open questions for the project owner

Format: `Q<n> (date, source) — question. Options. **Assumed meanwhile:** ...`
Agents append new questions at the end and never block on them: choose the most reversible
option, write it under "Assumed meanwhile", and continue. The owner answers inline (`**Answer:**`).

## From the planning session (2026-11-11)

### Inputs and conventions
* **Q1** — The 2N file `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` has no header, so its
  e2max and lmax cannot be read from it. **Assumed meanwhile:** `file2e2max=28`, `file2lmax=14`
  (the imsrg++ defaults for `file2e1max=14`). The reader validates the total value count, so a
  wrong guess is detected.
* **Q2** — NO2B file `..._ms6_12_18.stream.bin`: single precision (float) or half? **Assumed
  meanwhile:** float (`no2b_precision=single`, the imsrg++ default). Check against the file size
  (130,743,360 bytes) in M5.
* **Q3** — Tolerances in `30-testing-and-validation.md` §2 (e.g. IMSRG(2) E0 to 1e-5 MeV, HF to
  1e-7 MeV, single terms to 1e-10 relative): acceptable as "numerically equivalent within reason"?

### Architecture
* **Q4** — Project and executable name. **Assumed meanwhile:** namespace `imsrg`, executable
  `imsrg`, Python package `imsrg`.
* **Q5** — Tensor operators: store only `ch_bra <= ch_ket` blocks (as imsrg++ does; half the
  memory) or all blocks (simpler kernels)? Decide when tensor operators are implemented.
* **Q6** — May IMSRG(3) 3B blocks use `float` storage to halve memory, if the benchmark shows
  acceptable accuracy (e.g. ΔE0 < 1 keV)? **Assumed meanwhile:** `double` only.
* **Q7** — In imsrg++ the generator's denominator floor (the minimum |denominator|) loses the
  sign of the denominator (`reference/03` §4). Bug (fix) or intended? **Assumed meanwhile:** keep
  the reference behaviour, flagged in code, until answered.
* **Q8** — Is HF DIIS needed (imsrg++'s version has off-by-one errors and dead code), or is plain
  density mixing enough? **Assumed meanwhile:** mixing only; DIIS later if convergence demands it.
* **Q9** — Target clusters and their oldest compilers/toolchains (GCC, CUDA, ROCm versions)? This
  decides the Kokkos pin (4.x vs 5.x). **Assumed meanwhile:** latest stable Kokkos release that
  supports C++20; GCC ≥ 11, CUDA ≥ 12, ROCm ≥ 6.
* **Q10** — Is there a GPU machine (NVIDIA/AMD) where a later session could compile and run the
  CUDA/HIP builds?

### Config and CLI (from `16-config-and-cli.md`)
* **Q11** — `nsteps=1` with a valence space uses only `core_generator` in imsrg++ (contrary to its
  comment). **Assumed meanwhile:** keep it, with a warning.
* **Q12** — `imsrg3_at_end` without `IMSRG3=true` rebuilds H from the normal-ordered input and
  drops the input 3N. **Assumed meanwhile:** keep it, with a warning.
* **Q13** — The new TOML config requires `emax`, `hw` and the file e1max values explicitly (no
  silent defaults); the legacy `key=value` layer keeps the old defaults and warns. OK?
* **Q14** — `denominator_delta` without `denominator_delta_orbit` is silently ignored in imsrg++.
  **Assumed meanwhile:** keep it, with a warning.

### Python (from `18-python-api.md`)
* **Q15** — Python runtime initialization: lazy on first use, or explicit `imsrg.Runtime()` context
  manager? **Assumed meanwhile:** explicit context manager, plus a lazy default for interactive use.
* **Q16** — Host access to device-resident blocks: read-only zero-copy host views that raise
  `StaleHostError` unless `sync_to_host()` was called? **Assumed meanwhile:** yes.
* **Q17** — Minimum Python version. The stable ABI needs ≥ 3.12. **Assumed meanwhile:** 3.10+,
  without the stable ABI.
* **Q18** — Legacy readers not planned: Oslo, OakRidge, "Jason", "Takayuki", Antoine formats. Any of
  them needed?

### Process
* **Q19** — Agent compaction: Claude Code exposes a global `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` and a
  per-agent `autoCompactWindow`, not per-model percentages. The proposed settings
  (`design/claude-settings.proposed.json`) approximate "60 % for Sonnet/Haiku, 80 % for Opus".
  Please review them and install them before the run, if you agree.

## Added during implementation

(append here)
