# Overnight run report — 2026-11-12

*Draft written at 07:40 JST while the run continues; the final version replaces this text at
the end of the run. `design/STATUS.md` has the per-merge log, `design/QUESTIONS.md` every open
question.*

## 1. Summary

* Run start 2026-11-11 23:41 JST. The orchestrator (Opus) ran in a desktop-app worktree; milestone
  leads (Opus) ran in their own worktrees and delegated to Sonnet/Haiku workers.
* **Merged into `develop`:** M0 (bootstrap), M1 (reference build, harness, golden data), M2
  (foundations), M3 (basis), M4 (operator), M8a (config and legacy CLI, the first half of M8).
  CI on `develop`: `tools/ci.sh dev-openmp` PASS with 720 unit, 1 integration and 3 MPI tests.
* **In progress at the time of this draft:** M5+M6 (`io`, `opslib`, `hf`; one combined lead), M7
  (IMSRG(2) commutators), M11 phase 1 (Python bindings).
* **Not started:** M8 proper (flow, BCH, Magnus, driver pipeline, integration tests), M9, M10,
  M12, M13.
* **The usage budget, not wall time, bounded the night.** All agents stopped on the 5-hour usage
  limit twice, 01:30–02:30 and 04:15–07:30 JST, and were resumed from their saved transcripts.

## 2. Owner actions needed

1. **Sync the main checkout** (QUESTIONS O1). The session could not run git in the main checkout,
   so `develop` was advanced from the orchestrator's worktree with `git update-ref` after CI
   passed on each merge. The main checkout's index and files still show the old `develop`
   (331b2d4). If you have no local edits there:
   `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 331b2d4 develop`
   (refuses if local edits conflict), or `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard`.
2. **3N test file** (M1-Q1): `NO2B_ThBME_N2LO_cE_...` contains only the cE contact term (all
   T = 3/2 elements vanish). E_HF(O16, emax 4) = −313.17 MeV and the diverging IMSRG flow are
   genuine for this Hamiltonian. Is a full N2LO file available, or was the component intended?
3. Review the questions with "Assumed meanwhile" in `design/QUESTIONS.md` (Q20–Q24, O1–O2,
   M1-Q1..Q2, M2-Q1..Q7, M3-Q1..Q9, M4-Q1..Q2, M8a-Q1..Q7, and later ones).

## 3. Validation against imsrg-devel (actual deviations)

| Area | Comparison | Max deviation |
|------|------------|---------------|
| basis | 14 golden basis dumps (orbits, occupations, kets, 2B/ph/3B channels, subsets) | exact |
| angmom | 571,800 Wigner 3j/6j/9j/CG values vs sympy (all arguments to emax 8) | 1.5e-14 abs |
| angmom | 405,537 Moshinsky brackets (E ≤ 16) vs exact rational construction | 7.7e-31 (double-double) |
| angmom | 3,027 brackets / 2,493 TalmiB vs libIMSRG | 4.4e-13 |
| operator | norms of 37 golden dumps | 4.1e-16 rel |
| operator | normal ordering O16 / Ne20 (fractional occupations) / undo | 5.8e-15 / 2.7e-15 / 3.6e-15 |
| operator | HF-basis transformation; NO2B normal ordering from the real 3N file | 1.8e-15; 4.4e-15 |
| driver | TOML vs legacy `key=value` for the four `16` §2.11 examples | identical resolved configs |
| harness | `refharness flow` vs the imsrg++ executable, IT-2 | 256 steps both, \|ΔE0\| = 3e-10 MeV |

Golden reference numbers (imsrg-devel, O16, hw 16, emax 4, 2N only): E_HF = −147.893576 MeV,
EMP2 = −15.579114 MeV, IMSRG(2) Magnus E0 = −164.648772308 MeV after 256 steps. IMSRG(3)-N7
E0 = −145.489827 (emax 2) and −158.254073 MeV (emax 3); IMSRG(3f2) E0 = −147.389459022 (emax 2)
and −157.777153257 MeV (emax 3).

## 4. Old-code bugs fixed (details in `30-testing-and-validation.md` §6)

* Moshinsky brackets in plain double precision are off by up to 7.8e-11 relative (22 brackets);
  we use double-double arithmetic.
* `basis` M3-1..M3-6 (e.g. a missing swapped ket returned a valid-looking index; jj-shell overrun;
  two-digit n labels).
* M4-1: `AddToTBME` dropped the anti-Hermitian sign of the partner element.

## 5. Deviations from the design

* Stacked milestone branches: each lead tagged an "API-ready" commit (`m2-api-ready`,
  `m3-api-ready`, `m4-api-ready`) so the next milestone could start before the previous one
  finished. M5 and M6 run under one lead; M8 is split into M8a (config/CLI, merged) and M8
  (pipeline).
* Operator namespace `imsrg::op` (`operator` is a keyword); the runtime is split into
  `comm::Environment` and `exec::Runtime`; project headers use quotes (Q21); one Python venv
  shared by all worktrees (Q23); Kokkos 5.3.0 needs NVCC ≥ 12.2 and ROCm ≥ 6.2 (Q20).
* `autoCompactWindow` set to 600000 (Sonnet/Haiku) and 800000 (Opus lead) after checking the docs
  (1M-token windows; O2).

## 6. Performance

Not measured yet (the M7 micro-benchmark is pending at the time of this draft).

## 7. Recommended next steps

Finish and merge M5+M6 and M7, then M8 (flow, BCH, Magnus, driver pipeline) to make IT-2, IT-3,
IT-4, IT-6, IT-7 and IT-12 runnable; then M9 (MBPT, natural orbitals), the M10 layout benchmark,
and M11 phase 2.
