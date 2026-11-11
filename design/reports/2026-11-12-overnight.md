# Overnight run report — 2026-11-12

Run 1 of the unsupervised implementation: 2026-11-11 23:41 → 2026-11-12 ~08:45 JST, ended at the
owner's request. Per-merge details: `design/STATUS.md`. Open questions: `design/QUESTIONS.md`.
Prompts for what comes next: `design/NEXT_SESSION_PROMPT.md` (run 2) and
`design/INTERVIEW_PROMPT.md` (Q&A and design interview with the owner).

## 1. Summary

* **Milestones merged into `develop`:** M0 bootstrap, M1 reference build + golden data, M2
  foundations, M3 basis, M4 operator, M5+M6 io/opslib/hf, M7 IMSRG(2) commutators, M8 (flow,
  Magnus solver, driver pipeline; its config/CLI half merged earlier as M8a), M9 mbpt, M11 phase 1
  (Python bindings). This exceeds the planned overnight target (M0–M8, with M9/M11 started).
* **The `imsrg` executable runs end to end** (TOML config or legacy `key=value`): me2j/NO2B input →
  HF (imsrg++ mixing or DIIS) → normal ordering → IMSRG(2) Magnus, single reference and 1-/2-step
  valence space → JSON results, flow CSV, tokyo `.snt`. It reproduces imsrg-devel for IT-2, IT-3,
  IT-4, IT-6, IT-7 and IT-12 with identical step counts and |ΔE0| ≤ 3e-10 MeV.
* **CI on `develop` (0676e7b and later):** `tools/ci.sh dev-openmp` PASS with 851 unit, 34
  integration, 3 MPI and 23 Python tests, plus the layering and clang-format checks. Every lead
  also ran `dev-serial` before reporting.
* **Not started:** M10 (IMSRG(3); the layout-benchmark lead was stopped right after launch when the
  owner ended the run, and its brief is in the next-session prompt), M11 phase 2, M12, M13.
* **The usage budget, not wall time, bounded the night:** all agents stopped on the 5-hour usage
  limit twice, 01:30–02:30 and 04:15–07:30 JST, and resumed from their saved transcripts.

## 2. Owner actions needed

1. **Sync the main checkout** (QUESTIONS O1). The session could not run git in the main checkout,
   so `develop` was advanced from the orchestrator's worktree with `git update-ref` after CI passed
   on each merge. The main checkout's index and files still show the old `develop` (331b2d4). With
   no local edits there: `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 331b2d4 develop`
   (refuses if local edits conflict), or `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard`.
2. **3N test file** (M1-Q1): `NO2B_ThBME_N2LO_cE_...` contains only the cE contact term (all
   T = 3/2 elements vanish). E_HF(O16, emax 4) = −313.17 MeV and the diverging IMSRG flow are
   genuine for this Hamiltonian; IT-4/IT-5 therefore compare HF plus the flow stopped at s = 10. Is a
   full N2LO file available, or was the component intended?
3. **Answer the open questions**, ideally with the interview session (`design/INTERVIEW_PROMPT.md`):
   about 50 entries with "Assumed meanwhile" (Q5, Q6, Q11–Q17, Q19–Q24, O1–O2, M1-Q1..Q2,
   M2-Q1..Q7, M3-Q1..Q9, M4-Q1..Q2, M5-Q1, M6-Q1..Q4, M7-Q1..Q4, M8-Q1..Q4, M8a-Q1..Q7,
   M9-Q1..Q4, M11-Q1..Q2).

## 3. Validation against imsrg-devel (actual deviations)

| Area | Comparison | Max deviation |
|------|------------|---------------|
| basis | 14 golden basis dumps (orbits, occupations, kets, 2B/ph/3B channels, subsets) | exact |
| angmom | 571,800 Wigner 3j/6j/9j/CG values vs sympy (all arguments to emax 8) | 1.5e-14 abs |
| angmom | 405,537 Moshinsky brackets (E ≤ 16) vs an exact rational construction | 7.7e-31 (double-double) |
| angmom | 3,027 brackets / 2,493 TalmiB vs libIMSRG | 4.4e-13 |
| operator | norms of 37 golden dumps; dump round trips | 4.1e-16 rel; byte-identical |
| operator | normal ordering O16 / Ne20 (fractional occupations) / undo | 5.8e-15 / 2.7e-15 / 3.6e-15 |
| operator | HF-basis transformation; NO2B normal ordering from the real 3N file | 1.8e-15; 4.4e-15 |
| io | V2 from the me2j file at emax 2 and 4; T_int; Rp2 | 0; 1.1e-14; 6e-16 |
| io | tokyo/NuShellX writers (6 files) | numerically identical |
| hf | E_HF / SPEs / C (emax 2 and 4, 2N and 2N+NO2B); iterations 20/22/30/25 | 2e-13 / 1e-13 / 7e-16; identical counts |
| hf | HNO in the HF basis; IT-1 and IT-4 HF from the real files | 1.7e-13; 1.7e-13 |
| mbpt | MP2/MP3 at emax 2; IT-1 EMP2; IT-5 NAT-basis E_HF | ~1e-16; 3.4e-9; 1.9e-10 MeV |
| commutator | every IMSRG(2) term vs goldens; full commutator; fast vs reference (incl. fractional occupations) | 1.5e-15 rel; 2.9e-16; 1.9e-15 |
| flow | generators vs goldens; emax-2 flow vs the imsrg++ flow file | 2e-17; 4.8e-10 (print precision) |
| driver | TOML vs legacy `key=value` (`16` §2.11 examples; IT-12) | identical configs; bitwise identical results |
| e2e | IT-2 O16 emax 4 IMSRG(2): E0, steps, first 10 trajectory steps | ΔE0 −3.0e-10 MeV; 256 = 256; 5.7e-14 |
| e2e | IT-4 (2N+NO2B, smax 10): E0, steps | 1.5e-10 MeV; 29 = 29 |
| e2e | IT-6 ⟨Rp2⟩ after IMSRG | 1.9e-10 fm² |
| e2e | IT-3 / IT-7 valence space (sd-shell, O16 / Ne20 reference): steps, `.snt` elements | 421 = 421, 8e-15 / 431 = 431, 1.2e-14 |

Golden reference numbers (imsrg-devel, O16, hw 16, emax 4, 2N only): E_HF = −147.893576 MeV,
EMP2 = −15.579114 MeV, IMSRG(2) E0 = −164.648772308 MeV after 256 Magnus steps. IMSRG(3)-N7 E0:
−145.489827 (emax 2), −158.254073 MeV (emax 3); IMSRG(3f2) E0: −147.389459022 (emax 2),
−157.777153257 MeV (emax 3); perturbative triples: He4 emax 2 −0.0520971, O16 emax 3 −0.538674 MeV.
These IMSRG(3) goldens wait for M10.

## 4. Performance (dev machine: Apple M5 Pro, 18 cores; details in `design/benchmarks/`)

| Measurement | Ours | imsrg++ |
|-------------|------|---------|
| Full IMSRG(2) commutator, emax 4, 1 / 6 threads | 5.7 / 1.6 ms | 5.2 / 1.4 ms |
| Full IMSRG(2) commutator, emax 6, 1 / 6 threads | 80 / 16.7 ms | 67 / 14.7 ms |
| HF iterations (emax 2/4 goldens): imsrg++ mixing / DIIS | 20–30 / 8–10 | 20–30 |
| me2j read: emax-4 prefix / whole 754 MB file | 3.6 ms / 9 s | — |

The commutator is 1.10–1.20× the imsrg++ time (from 1.9–2.8× before M7's tuning); known
follow-ups: sparse Pandya maps and smaller Workspace (M7-Q4), per-channel norm reductions,
`No2bThreeBody::get_pn`, `Denominators` copying H every step. No end-to-end timing comparison yet
(IT-13 is not implemented).

## 5. Old-code bugs fixed (details in `30-testing-and-validation.md` §6)

* #1 Moshinsky brackets in plain double precision are off by up to 7.8e-11 relative (22 brackets);
  we use double-double arithmetic.
* M3-1..M3-6 (`basis`): an absent swapped ket returned a valid-looking index; inconsistent hole
  threshold; a nucleus that does not fit was silently truncated; jj-shell overrun; two-digit n
  labels; back-pointers in copies.
* M4-1: `AddToTBME` dropped the anti-Hermitian sign of the partner element.
* M6-1, M6-2: imsrg++'s broken DIIS rescue after 100 HF iterations and the `std::rand` mixing are
  replaced (no golden case reaches them).
* M8-1: the generator's denominator floor keeps the sign (D21); no effect in any IT. M8-2: the
  Hartree-Fock generator no longer fills the diagonal. M8-3: a diverging flow stops with
  `NumericalError` instead of filling H with inf.

## 6. Deviations from the design and the plan

* Process: stacked milestone branches via "API-ready" tags (`m2-`, `m3-`, `m4-`, `m7-api-ready`)
  so dependent milestones started early; M5 and M6 under one lead; M8 split into M8a (config/CLI)
  and M8 (pipeline); `develop` advanced from a worktree via `update-ref` (O1).
* Code: operator namespace `imsrg::op` (`operator` is a keyword); runtime split into
  `comm::Environment` and `exec::Runtime`; project headers included with quotes (Q21); one Python
  venv shared by all worktrees (Q23); Kokkos 5.3.0 needs NVCC ≥ 12.2 and ROCm ≥ 6.2 (Q20); `mbpt`
  does not depend on `hf` (its tests do); the fast commutator path rejects operators without a
  definite hermiticity (M7-Q2).
* Kept on purpose for comparability: Rp2 includes the spin-orbit correction; Tcm drops two-body
  entries ≤ 1e-7 (M6-Q1/Q2).
* `autoCompactWindow` set to 600000 (Sonnet/Haiku) and 800000 (Opus lead) after checking the docs
  (1M-token windows; O2).

## 7. What is left and recommended next steps

1. Run the interview session (`design/INTERVIEW_PROMPT.md`) so the open questions and priorities
   are settled.
2. Run 2 (`design/NEXT_SESSION_PROMPT.md`): M8 follow-ups (Opus review of the flow numerics,
   flow-RK4/DP45, `hartree_fock.mixing` key, MP2 column, NAT basis/IT-5 in the driver); then **M10
   IMSRG(3)**: layout benchmark → decision → brute-force m-scheme reference → perturbative triples
   (IT-8) → N7 (IT-9) → 3f2 (IT-10); in parallel M11 phase 2 and M12 (MPI, IT-11, performance).
3. Later: M13 (Makefile, docs), tensor/dagger/0νββ/DM operators, me3j and Navratil/Miyagi inputs.
