# Owner interview — 2026-11-15

Interactive session with the project owner after run 1 (agenda: `design/INTERVIEW_PROMPT.md`).
Outcome: **every open question in `design/QUESTIONS.md` is answered** (73 entries; the low-impact
ones confirmed as a batch), and the design decisions for the next milestones are recorded as
**D30–D42** in `design/00-requirements-and-decisions.md`. Run 2 can start without blocking
questions; its prompt is `design/NEXT_SESSION_PROMPT.md`.

## 1. Decisions

| Area | Decision | Ref |
|------|----------|-----|
| Toolchain | Baseline CUDA ≥ 12.2, ROCm ≥ 6.2 (GCC ≥ 11); Kokkos 5.3.0 stays | D30 (Q20) |
| Runs | Start in the main checkout on `develop`; ≤ 3 leads, Sonnet-heavy; resume after usage resets with `SendMessage`; launch right after a reset | D31 (O1) |
| Compaction | Per-agent `autoCompactWindow` only (600000 Sonnet/Haiku, 800000 Opus lead, 800000 orchestrator via settings); no `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, which would stack on the windows (Claude Code docs) | D31 (Q19, O2) |
| macOS SDK | Keep the CMake workaround; CLT 26.6 itself ships the pre-release MacOSX27.0 SDK its linker cannot read; no agreement involved; update CLT when offered | Q24 |
| IMSRG(3f2) | = IMSRG(2) + hunter-gatherer + `omega_norm_max` 0.1 + factorized correction, 1b intermediates on, 2b off (owner's pyIMSRG recipe); `[imsrg3] factorized = true` sets these defaults; IT-10 goldens regenerated | D32 (M1-Q2) |
| 3N input | Keep the cE-only NO2B file; a full N2LO file later (then new goldens; cE tests stay as code-path tests); O16 emax-2 instability is known physics | D33 (M1-Q1) |
| IMSRG(3) | dE3max/OccNat cuts drop kets at enumeration; imsrg++ IMSRG(3) bugs fixed and validated by brute force; tolerances as D19; production target N7 at emax 10–12 | D34 (M3-Q9) |
| Tensor storage | Only `ch_bra ≤ ch_ket` blocks (the reading of "Yes" confirmed) | D25 (Q5) |
| imsrg++ quirks | Fixed: Tcm keeps all 2B entries; Rp2 without spin-orbit, `Rso` separate; no compatibility switch; e2e goldens from a patched imsrg-devel build | D35 (M6-Q1/Q2) |
| Numerics | Full-precision 1/√2; double-double Moshinsky; DIIS design; M8-1..M8-3 confirmed; general (non-Hermitian) fast-commutator paths in run 2; no NAT options yet | D36 (M2-Q3, M2-Q7, M6-Q3, M7-Q2, M9-Q3) |
| Config/CLI | `hartree_fock.mixing` = DIIS (TOML) / linear (legacy); TOML paths relative to the config file, legacy/CLI to the working directory, absolute paths in `resolved_config.toml`; legacy core ≠ reference stays unsupported; `results.json` `schema_version` | D37 (M8-Q3, M8a-Q4/Q5) |
| Code style | Google C++ Style Guide with clang-format (80 columns), cpplint (filtered, not followed blindly), clang-tidy naming; full Google naming; `.h`/`.cc`, `#define` guards, SPDX GPL-3.0-or-later headers; exceptions: C++ exceptions, `<filesystem>`, Kokkos `[=]`; conversion first in run 2 | D38, `17-code-style.md` |
| License | GPL-3.0-or-later, `Copyright 2026 The imsrg authors` + AUTHORS (imsrg++ sources are GPL-2.0-or-later, compatible) | D17, D38 |
| Priorities | Run 2: style conversion → M8 follow-ups + interview tasks → M10 (3f2 in parallel with the layout benchmark, then triples → N7), M11 phase 2 and M12a alongside; after IMSRG(3): tensor operators first | D39 |
| Outputs | Only tokyo/NuShellX must stay imsrg++-compatible; no HDF5; new formats me2jp/me3jp later (spec from the owner); checkpoint/restart later (after M13) | D39 |
| Performance | M12a single-node first (parity, then explore; occasional emax-8 benchmarks, never in CI) → M12b MPI (3B blocks: evaluate the Hagen–Nam row-wise cost-balanced distribution, arXiv:1203.3765, against by-channel); GPU after M12; hardware: NVIDIA H100/H200/GH200 and A100, AMD MI250X/MI300A, CPU clusters | D40 |
| Python | Users: owner's group (pipelines) and operator algebra; pip wheels (cp312-abi3, cibuildwheel); exit finalization revisited for GPU | D41 (M11-Q1) |
| Testing/process | `ci.sh` < 5 min (139 s now) + `slow` label; scripted golden regeneration with reason, old/new numbers and review; git local only; interview after every run; run-1 orchestration and report format kept | D42 |
| Low-impact batch | Q21–Q23, M2-Q1/Q2/Q4–Q6, M3-Q1..Q8, M4-Q1/Q2, M5-Q1, M6-Q4, M7-Q1/Q3/Q4, M8a-Q1–Q3/Q5 (exit codes)/Q6/Q7, M8-Q1/Q2/Q4, M9-Q1/Q2/Q4, M11-Q2 confirmed as assumed | QUESTIONS |

## 2. What changed compared with the plan of run 1

* **New first task:** the Google-style conversion (MS), done alone before any other branch,
  because it renames every file and most functions. Measured scale on `develop` 61adb16: 304 C++
  files / 41k lines; 80 columns reflow 281 files (~11,100 lines); cpplint 2.0.2 reports 6,334
  findings, of which 5,513 are line length, 304 missing license lines and 186 false positives that
  the cpplint configuration filters; ~490 function names, 193 enumerators and ~44 constants to
  rename (`17-code-style.md` §2).
* **IT-10 is not valid yet:** its goldens used imsrg++ defaults, not the owner's 3f2 recipe; they are
  regenerated, and hunter-gatherer becomes a prerequisite.
* **The e2e goldens change:** after the Tcm and Rp2 fixes, the goldens with T_int or Rp2 come from
  a patched imsrg-devel build.
* **M10 runs in two tracks** (3f2 needs no 3B storage). **M12 is split:** single-node performance
  before MPI, and GPU work only after M12.
* **AGENTS.md rule 3** gained an exception (occasional emax-8 IMSRG(2) benchmarks, never in CI).

## 3. Follow-up tasks for run 2 (in order; details in `NEXT_SESSION_PROMPT.md`)

1. MS: Google-style conversion per `17-code-style.md` §5 (install clang-tidy with `brew install llvm`).
2. M8 follow-ups and interview tasks: Opus flow-numerics review; hunter-gatherer; Tcm/Rp2/`Rso` with
   the patched reference build and regenerated goldens; `hartree_fock.mixing`; config-relative paths
   and absolute paths in the resolved config (adapt `examples/config/` and the equivalence test);
   `results.json` `schema_version`; MP2 column; NAT/IT-5 in the driver; `Denominators`; flow-RK4/DP45;
   non-Hermitian fast-commutator paths; full-precision 1/√2.
3. M10: 3f2 track (refharness flags, IT-10 regeneration, factorized commutator, config keys) and the
   layout track (benchmark judged against N7 at emax 10–12 → brute force → triples → N7).
4. Alongside: M11 phase 2 and M12a.

## 4. Deferred, with the assumption kept meanwhile

| Item | Reason | Meanwhile |
|------|--------|-----------|
| Full N2LO 3N file (M1-Q1) | not available yet | cE-only file; IT-4/IT-5 as they are |
| me2jp/me3jp formats | later; the owner provides the specification | no new formats |
| Checkpoint/restart, Omega scratch storage (M8-Q3) | after M13 | Omega in memory |
| NAT options `use_NAT_occupations`, `NAT_order` (M9-Q3) | not needed now | defaults only |
| Interpreter-exit finalization (M11-Q1) | matters for GPU builds | as implemented |
| Order of the extensions after tensor operators | not ranked | decide at the next interview |
| Cluster names, node memory, interconnect, scheduler; IMSRG(3) E3max/dE3max and memory per node | not given | ask when M12b/M14 start |

## 5. Owner actions

1. Before run 2: sync the main checkout (`develop` moved again from worktrees):
   `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 68c8927 develop`, or `git -C
   /Users/cheshyre/Downloads/new-imsrg reset --hard` if there are no local edits.
2. Install `design/claude-settings.proposed.json` (updated compaction settings) and launch run 2 in
   the main checkout right after a usage reset.
3. Later: the full N2LO 3N file, the me2jp/me3jp specification; the Command Line Tools update when
   Software Update offers it.

## 6. Session notes

* The main checkout had been synced on 2026-11-12 08:58 (its files match 68c8927); this session ran
  in an app worktree and advanced `develop` by the staging procedure of QUESTIONS O1.
* Background checks during the interview: the Claude Code docs on compaction settings; the current
  Google style guide; the INSPIRE record 1094097 (G. Hagen and H. A. Nam, "Computational aspects of
  nuclear coupled-cluster theory", Prog. Theor. Phys. Suppl. 196, 102 (2012), arXiv:1203.3765; it
  describes the distribution for 2B blocks in coupled cluster: rows of the largest blocks are added
  to a process until load and memory criteria are met, with dense per-process sub-blocks for BLAS).
* cpplint 2.0.2 was installed into the shared venv for the style analysis.
* `tools/ci.sh dev-openmp` passed on the branch before the documentation edits (139 s) and on the
  final merge (see `STATUS.md`).
