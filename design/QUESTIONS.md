# Open questions for the project owner

Format: `Q<n> (date, source) — question. Options. **Assumed meanwhile:** ...`
Agents append new questions at the end and never block on them: choose the most reversible
option, write it under "Assumed meanwhile", and continue. The owner answers inline (`**Answer:**`).

## From the planning session (2026-11-11)

### Inputs and conventions
* **Q1** — The 2N file `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` has one header
  comment line, `(*** nn_transforms v0.2.0 -- generated on 03/07/2022, 10:56:07 -- me2j-f2 ***)`,
  that does not contain the truncations. **Answer (verified 2026-11-11):** the file holds exactly
  168,066,656 values after the header line, which is the count the imsrg++ loop structure predicts
  for `file2e1max=14 file2e2max=28 file2lmax=14` (e2max=24 would give 115,792,936; lmax=10 would
  give 107,554,520). Use 28 and 14. Counting script: `tools/reference/me2j_value_count.py`.
* **Q2** — NO2B file `..._ms6_12_18.stream.bin`: single precision (float) or half? **Answer
  (owner):** single precision (float).
* **Q3** — Tolerances. **Answer (owner):** keep the deterministic unit-level checks tight (reading,
  Wigner symbols, single commutator terms ~1e-10 relative). Relax end-to-end acceptance to HF
  1e-4 MeV and IMSRG E0 / valence-space elements 1e-3 MeV, and always log the actual deviation.
  Applied in `30-testing-and-validation.md` §2.

### Architecture
* **Q4** — Project and executable name. **Answer (owner):** keep `imsrg` (namespace, executable,
  Python package).
* **Q5** — Tensor operators: store only `ch_bra <= ch_ket` blocks (as imsrg++ does; half the
  memory) or all blocks (simpler kernels)? Decide when tensor operators are implemented.
* **Q6** — May IMSRG(3) 3B blocks use `float` storage to halve memory, if the benchmark shows
  acceptable accuracy (e.g. ΔE0 < 1 keV)? **Assumed meanwhile:** `double` only.
* **Q7** — In imsrg++ the generator's denominator floor (the minimum |denominator|) loses the
  sign of the denominator (`reference/03` §4). Bug (fix) or intended? **Answer (owner):** bug.
  Fix it (keep the sign), list it in the deviation log, and validate it (e.g. a unit test with a
  negative near-zero denominator, and an end-to-end run where the floor never triggers matches
  imsrg++).
* **Q8** — Is HF DIIS needed (imsrg++'s version has off-by-one errors and dead code)? **Answer
  (owner):** DIIS from the start. Implement a correct DIIS in M6, keep plain density mixing as an
  option (it is the imsrg++ default path, needed for comparisons), and test that both converge to
  the same HF solution.
* **Q9** — Oldest toolchains to support? **Answer (owner):** GCC ≥ 11, CUDA ≥ 12, ROCm ≥ 6. Pin
  the latest stable Kokkos release compatible with these (check its requirements at M0).
* **Q10** — Is there a GPU machine for CUDA/HIP tests? **Answer (owner):** not now. Rely on the
  device-code rules (`12-compute-backend.md` §5) and reviews; GPU testing comes later.
* **Q10b** — After the IMSRG(3) layout benchmark, may the session choose the layout itself?
  **Answer (owner):** yes. Decide from the data, document the reasoning and the numbers, and
  continue with the IMSRG(3) terms.

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
* **Q18** — Which legacy input formats are needed? **Answer (owner):** me2j (2N), NO2B and me3j
  (3N), and the Navratil/Miyagi operator formats (`OperatorsFromFile`, two-body currents). Dropped:
  HDF5 3N, Oslo, OakRidge, "Jason", "Takayuki" and the other older readers.

### Process
* **Q19** — Agent compaction: Claude Code exposes a global `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` and a
  per-agent `autoCompactWindow`, not per-model percentages. The proposed settings
  (`design/claude-settings.proposed.json`) approximate "60 % for Sonnet/Haiku, 80 % for Opus".
  Please review them and install them before the run, if you agree.

## Added during implementation

(append here)

* **M1-Q1** (2026-11-12, M1) — IT-4/IT-5 (2N + `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_...` file)
  give E_HF(O16, emax 4) = -313.17 MeV (e3hf = -201.9) and a diverging Magnus flow; IT-8 (O16,
  emax 2, 2N only) diverges too. **Investigated (M1 lead):** the 3N file is a contact-only (cE)
  component file: all its T = 3/2 matrix elements vanish (`tools/reference/no2b_isospin_scan.py`),
  the reader layout matches, and no LEC scaling applies to no2b input, so the numbers are genuine
  for this Hamiltonian. The O16 emax-2 divergence is a genuine IMSRG(2) instability (same with the
  White generator and MP denominators; He4 emax 2, O16 emax 3 and IMSRG(3) at emax 2 converge).
  Details: `tests/golden/e2e/README.md`. Question for the owner: is a full N2LO 3N file (all LEC
  terms) available, or was the cE component intended? **Assumed meanwhile:** keep this file as a
  code-path test input; IT-4/IT-5 compare HF numbers and the flow stopped at s = 10
  (`IT-4/smax10`, `IT-5/smax10`); IT-8 compares He4 at emax 2 and O16 at emax 3.
