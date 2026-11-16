# Owner interview — 2026-11-16

Interactive session with the project owner after run 2 (agenda: `design/INTERVIEW_PROMPT.md`), in
ten rounds of questions plus a specification review. Outcome: **all 57 questions added during run 2
are answered** (low-impact ones confirmed in batches), the design decisions for run 3 and later are
recorded as **D44–D54** in `00-requirements-and-decisions.md`, the license switched to
**GPL-3.0-only** during the session (merged, D46), and the owner's **ME1J/ME2JP/ME3JP formats** are
specified in `design/19-mexj-formats.md` (D54). Run 3 can start without blocking questions; its
prompt is `design/NEXT_SESSION_PROMPT.md`.

## 1. Decisions

| Area | Decision | Ref |
|------|----------|-----|
| Runs | Start in an app worktree; staging procedure (`ci.sh && [ctest -L slow &&] git update-ref`) is the standard; owner re-syncs the main checkout after each run. Main checkout was synced after run 2 | D44 (O3) |
| N7 target | emax 10–12 with **E3max 16, no dE3max** (177 GB per packed operator, 1.06 TB per N7 step); nodes ≤ 256 GB, sometimes 100 GB; 3B ≤ 50 % of a node; **single channels split across ranks, row-wise cost-balanced distribution as baseline**; kernels tiled over row ranges; accepted by cost model (≤ 10 % imbalance, communication < compute at ~25 GB/s); InfiniBand and Slingshot clusters with Slurm | D45 (M10-Q2) |
| License | GPL-3.0-only (was or-later); holder list unchanged; CI step `license headers`; merged 6bcca88 | D46 (MS-Q4) |
| Flow numerics | Stage 2 integrates [s1, s1 + smax] for every method; s continues; divergence guard, flow+HG ConfigError, no atan switch in Magnus confirmed; `eta_criterion`/`ode_tolerance` double; observables stay in the adaptive error norm | D47 (M8-Q40, Q41, Q50) |
| Hunter-gatherer | H(s) becomes the rebuilt H after the final gather (one Ω for E0 and all operators). Measured shift: −13.4 keV IT-10 O16 emax 2, +2.8 keV He4, −43 / −22 keV IT-2 HG O16 emax 4, −8.1 keV He4 HG; steps and gathers unchanged | D48 (M8-Q10) |
| IMSRG(3), 3f2 | 2b intermediates not now; 3f2 correction in all BCH calls; triples split-stack order fixed; **triples at emax ≥ 8** in run 3; Δ3 floor (D21 sign-keeping); residual 3B dropped as imsrg++ plus an IT-9 Rp2 golden; **`occ_nat3_cut` needed**, other E-PLANNED options stay | D49 (M10-Q20..Q23, Q40, Q46, Q60..Q62) |
| Code style | Physics capitals kept, clang-tidy pattern tightened to a physics-symbol list; accessor rule kept; guards outside `src/` kept; **structural pass with priority, first in run 3, alone, time-boxed** (owner override of the "guidelines only" proposal) | D50 (MS-Q1..Q3, Q5) |
| Python | `Auto` default for users, Python suite under `IMSRG_RESIDENCY=strict` in CI; `18` rewritten to the bound module-shaped API; stop flag, generator hook, driver cancel flag as filler | D51 (M11-Q11..Q16) |
| Benchmarks | Timing-only IMSRG(3) kernels at emax 4–6 (≤ ~8 GB per operator) and IMSRG(2) single commutators at emax 10–12, never in CI, no full flows | D52 (M10-Q41/Q42, M12a) |
| Run-3 order | Interview tasks → MS2 structural pass (alone) → N7 performance (row tiling, triples ≥ 8, `occ_nat3_cut`) ∥ M12b phase 1 → M12b phase 2 → M15 tensor operators → M16 formats; M12a follow-ups and Python hooks as filler; M13 last; M14 after M12b (CUDA+HIP builds, NVIDIA tuning first) | D53 |
| Tensor operators | First milestone covers E2/electric multipoles, M1, Gamow–Teller/Fermi and 0νββ; Magnus transformation only (no ODE flow of tensors); validation against the patched imsrg++ and an m-scheme brute force for tensors | D53 |
| ME1J/ME2JP/ME3JP | pn-resolved ME2J/ME3J (verified from the owner's jIMSRG readers); unnormalized antisymmetrized values; readers and writers; text and gzip, binary `.bin` prepared (spec pending); evolved H by default, others on request; main uses: stored Omega with a `transform` task and checkpoint/restart; after tensor operators (owner override: me2jp/me3jp before checkpoint/restart as a separate item) | D54 |
| J-range safety | The new ME2JP/ME3JP getters take each record's J ranges from one function and the lookup table, return exactly 0.0 outside them, check every offset, and are accepted only with sentinel-value tests against an independent oracle, an `asan` run and an Opus review of these rules (after the owner confirmed the jIMSRG `std::max` finding) | D55 (`19` §8) |
| Run-3 schedule | Conservative until 20:00 JST (at most 50 % of each 5-hour usage window), then normal pacing; lead reports by 06:00, wrap-up 07:00 JST | D56 |
| Process | Run-2 process kept: ≤ 3 leads, Sonnet-heavy, usage pacing, out-of-repo preparation during exclusive tasks, Opus review per numerics merge, `slow` suite before numerics merges, scripted goldens, run-2 report format, interview after every run | D53 (D42) |
| Low-impact batches | M7-Q10/Q11, M8-Q11, M8-Q12, M8-Q20..Q22, M8-Q30..Q33, M8-Q51/Q52, M10-Q1, Q3, Q21, Q22, Q43..Q45, Q63..Q67, M11-Q10, Q12, Q16, M12-Q1..Q4 confirmed as assumed; M12-Q5 deferred to the cluster builds | QUESTIONS |

## 2. What changed compared with the plan of run 2's hand-off

* **The 3B distribution became the central design constraint.** With E3max 16 and nodes of
  100–256 GB, a single channel's six operators (about 51 GB) can exceed a node's 3B budget, so the
  Hagen–Nam-type row distribution is the baseline rather than an option to evaluate, and the N7
  kernels must be tiled over row ranges so that phase 2 needs no kernel rewrite. N7 performance and
  M12b phase 1 run in parallel.
* **A structural style pass comes first** in run 3 (the owner wants code quality now), alone, like
  MS in run 2.
* **Hunter-gatherer semantics change** (D48): the measured E0 shifts (up to 43 keV) are far above the
  1e-3 MeV tolerance, so IT-10 and the hunter-gatherer goldens are regenerated from the patched
  reference.
* **Tensor operators** have a concrete first scope (E2, M1, GT/Fermi, 0νββ), Magnus-only.
* **ME2JP/ME3JP** are specified and come right after tensor operators, carrying stored Omegas and
  checkpoint/restart; D39's "checkpoint after M13" is superseded.
* AGENTS.md rule 3 gained two benchmark-only exceptions (D52); runs officially start in worktrees
  (D44).

## 3. Follow-up tasks for run 3 (details and briefs in `NEXT_SESSION_PROMPT.md`)

1. Interview tasks (brief A): D48 (with the D35 patch extended and goldens regenerated), D47 (stage-2
   range, double tolerances), D49 (Δ3 floor, triples order, IT-9 Rp2 golden), D51 (strict residency
   in CI), D50 (clang-tidy physics-symbol pattern); `18` rewrite as a doc task.
2. MS2 structural pass (brief B), alone and time-boxed.
3. N7 performance (brief C) in parallel with M12b phase 1 (brief D).
4. M12b phase 2 with the D45 acceptance criteria; IT-11.
5. M15 tensor operators (brief E); then M16 formats (brief F) if time remains.
6. Filler: M12a follow-ups with emax 10–12 commutator benchmarks (brief G), Python hooks, run-2
   cleanup items.

## 4. Deferred, with the assumption kept meanwhile

| Item | Reason | Meanwhile |
|------|--------|-----------|
| Full N2LO 3N file (D33) | not available yet | cE-only file; IT-4/IT-5 as they are |
| Binary `me2jp.bin`/`me3jp.bin` header | the owner is fixing the specification | text and gzip; record enumeration separate from value I/O |
| GroupedGemm concurrency with OpenBLAS/MKL (M12-Q5) | decide with the cluster builds | sequential loop outside Accelerate |
| 3f2 2b intermediates (M10-Q20), `full`/`mp4`, `at_end`, `only_2b_omega` (M10-Q62) | not needed now | E-PLANNED |
| Order of me3j, Navratil/Miyagi, dagger, DM (D39) | after M16 | M17+ in the roadmap |

## 5. Owner actions

1. Re-sync the main checkout after this session: `develop` moved twice (6bcca88, the license switch,
   and the merge of this interview). With no local edits: `git -C
   /Users/cheshyre/Downloads/new-imsrg reset --hard`; then `tools/reference/gen_golden_generated.sh`
   there if you run the tests in it.
2. Launch run 3 right after a usage reset, in the app's worktree (`NEXT_SESSION_PROMPT.md`). It runs
   conservatively (at most 50 % of each 5-hour window) until 20:00 JST and normally until the 07:00
   JST wrap-up (D56).
3. Later: the binary ME2JP/ME3JP header specification; sample ME1J/ME2JP/ME3JP files from jIMSRG would
   make good goldens; the full N2LO 3N file.
4. Optional, in jIMSRG: `ME2JPFile::Get2BMatrixElement` computes `jj_max` with `std::max` where the
   file layout uses `std::min` (an out-of-range J reads the neighbouring record instead of 0), and the
   ME3JP validator rejects files longer than expected while ME1J/ME2JP accept an emax prefix
   (`19-mexj-formats.md` §3, §5).

## 6. Session notes

* The session ran in an app worktree on `feature/owner-answers-2026-11-16` (cut from `develop`
  c2fef30) and advanced `develop` by the staging procedure (D44); `tools/ci.sh dev-openmp` passed on
  the branch base (192 s from a fresh worktree), on the license merge (105 s) and on the final merge
  (`STATUS.md`).
* Measured during the interview (a Sonnet worker in a throwaway worktree, patch not committed): the
  M8-Q10 E0 shifts of D48 from the integration tests with `h_ = h_saved_` after the final gather.
* The license switch was done by a Sonnet worker on `feature/license-gpl3-only` (419 files, two
  commits, CI PASS), reviewed and merged by the session.
* The owner provided `jIMSRG-develop/jimsrg/files/formats.zip` (MIT-licensed jIMSRG readers by the
  owner); it was unpacked into the session's scratch directory and read only; nothing from it was
  copied into the repository. The ME2JP/ME3JP loops were compared with our ME2J reader and with the
  ME3J loop listing of `reference/04`; the two 6j recoupling cases of the ME3JP getter were
  re-derived and agree.
* At the owner's request the session showed a progress bar of the agenda after each round.
* No file in the main checkout or in another worktree was changed; `develop`, which the main
  checkout has checked out, was advanced with `git update-ref`, so the main checkout's files lag
  behind until it is re-synced (owner action 1).
