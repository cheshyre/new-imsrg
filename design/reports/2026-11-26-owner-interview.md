# Owner interview 2026-11-26 (part 1) and the Al27 night

Supervised session on OLCF Frontier, 2026-11-26 from ~05:00 EST (owner in Wako, JST = EST + 14 h).
Orchestrator in the integrator's clone `$IMSRG_WORK/main`. Part 2 of the interview (the plan of the
unsupervised weekend session, until Monday 2026-11-30 10:00 Knoxville time) follows on 2026-11-27
10:00 JST (2026-11-26 20:00 EST). Answers: `design/QUESTIONS.md` (verbatim where given), decisions
D61-D71 in `00`, the weekend task in `design/50-al27-responses.md`.

## 1. Decisions

| Topic | Decision | Row |
|---|---|---|
| Agent isolation (P-F1) | scheme (a) kept; integrator retires clones and deletes merged hub branches | D61 |
| GitHub (P-F2) | as proposed; `main` fast-forwarded by the owner at milestones; no mirroring | D62 |
| Frontier settings (FP-Q1/2/4/6/7) | settings installed; local git identity; `$IMSRG_WORK/golden-generated` current; BLAS lock kept; serial N7 test labelled slow | D63 |
| DIIS (FP-Q5) | linear mixing in every imsrg++ flow comparison AND default DIIS tolerance ~1e-10 | D64 |
| Compute (FP-Q3) | 50,000 node-hours until the end of the weekend, standard partitions, one node per lead, MPI scaling up to 8 nodes at 1 and 8 ranks/node | D65 |
| Run-4 process | shake-down first; M12b → M14 → M16 → M15 → N7 → fillers; D58 cap unchanged, no weekly ceiling | D66 |
| Weekend task | reproduce the 2024 27Al NREFT responses (three variants), Tilde Phi' J = 2, 4 with √(4π) | D67 |
| Fork compat | one `run.compat` profile reproducing the fork's bugs for variant 1; default path correct | D68 |
| Checkpoints (O4) | float32 enough; both native float64 and ME2JP float32 implemented and compared | D69 |
| GPU | rocBLAS vs Kokkos Kernels, vendor only at >= 5x; flow measured at 1 and 8 ranks | D70 |
| Run-3 questions | IT3-Q2 stop as now (owner overrode the recommendation), IT3-Q9 add split-vs-single triples test, the rest as assumed | D71 |

Deferred: none. The full N2LO 3N file (D33) is still not available.

## 2. The Al27 night (2026-11-26 06:30-19:30 EST)

The owner chose "decisions + start coding" and lifted the usage cap for the night. Leads, each in
its own hub clone and on its own node:

| Lead | Capability (design/50 §3) | Result | Merged |
|---|---|---|---|
| C | C4 HF-reference NAT, C5 truncation, C6 Omega I/O (native + ME2JP), C7 checkpoints, `run.compat` | killed flows restart bit for bit (native), E0 within 7e-9 (ME2JP); fork emax-4 H/TCM `.snt` to printed precision with domega 0.2 | 6f7d324 |
| A | C3 tensor operators (M15), fork tensor normal-ordering bug (M15-1), tensor `.snt` writer | all terms vs goldens <= 1.5e-13, m-scheme oracle ranks 0-5; rank-5 emax-10 commutator 63 s / 18 GB on 16 threads; comm122st then 8x faster | cf04957, 58a7540 |
| B | C1 fork reference build (+ phip_fix variant) and goldens, C2 NREFT library, fork tokyo format | fork tables to 1e-12; fork's own emax-14 radial sums off by 8.5e-7 (AL-B-1, fixed with double-double); four fork scalar `.snt` byte for byte | 941b692 |
| F | C10 `transform` farm (cost model, LPT, failure manifest, Omegas in MPI shared memory), tensor NREFT in the driver | farm = in-process transform byte for byte (native), 1e-9 with ME2JP; odd-rank fork mismatch AL-F1 found and fixed as a fork bug (missing (-1)^L for diagonal cross-coupled pairs) | e17f6b2 |
| — | agent_retire safety (keeps fresh/unpushed clones) | | 9fb645e |
| D | C9 production-size flows (both Hamiltonians, variant 1) | see §3 | cfe9a2f |
| E | C11 GPU path (M14) | see §3 | a519728 |

Incidents: three usage-limit stops (07:10, ~10:20, ~15:00; three Opus leads exhaust a 5-hour window
in ~2.5-3 h); the 2-hour batch chains lapsed during the second stop (the upkeep tick does not fire
while the orchestrator is blocked); `agent_retire --all --merged` deleted a fresh lead clone (fixed);
Kokkos Kernels source download blocked for agents (owner action, see §4); the `extended` partition
allows one running job per user.

## 3. State at the start of part 2

`develop` = cfe9a2f (CI PASS frontier-cpu, pushed to the hub and GitHub).

**Production flows reproduce the archive (lead D; `design/benchmarks/al27-production.md`).** Both
Hamiltonians, variant 1 (unsplit Magnus, compat profile), HF/NAT at emax 14 / e3max 24, IMSRG(2) at
emax 10, CPU only: every printed digit of README §3.3 agrees (HF iterations 33 / 29, e1/e2/e3hf,
E_HF, EMP2 at emax 14, EMP2/EMP3 at emax 10, stage ends 37 + 272 and 30 + 253 steps with s, E0,
eta and Omega norms; final E0 -213.961459718 vs -213.961459717 and -217.066364762 exactly). H `.snt`
max deviation 5.0e-7 (the archive's 7-digit rounding); TCM `.snt` elements within 5e-9, with 15/8
small elements (up to 1.1e-3) that the fork writes and the devel threshold drops (AL-D3: fixed by the
fork tokyo format under compat, already in develop; to be verified). Found and fixed: our MBPT(2)
density dropped particle x partially-filled-hole terms (EMP2 off by 0.026 MeV). Timings: setup
17 min (imsrg++ 44), flow steps 13.8 / 26.4 s (imsrg++ 21.2 / 35.4), checkpoint 1-2 s, peak 66 GB;
restarts across 2-hour jobs bit for bit. Omegas of both flows in both formats in
`$IMSRG_WORK/runs/al27/v1/`.

**GPU (lead E; `design/benchmarks/gpu-m14.md`).** rocBLAS device gemms and device IMSRG(2) scalar and
tensor kernels; one GCD vs 56 CPU cores: scalar commutator emax 10 58 vs 339 ms (5.8x), tensor rank 5
emax 10 1.5 vs 7.4 s (4.9x, 23 GB HBM), O16 Magnus flow emax 8 28 vs 81 s. GPU vs CPU <= 3.5e-15
relative; `tools/ci.sh frontier-hip-run` 1352/1357 (five bitwise atan tests of the 3B generator,
device vs host atan 4e-16, AL-E-Q2). One commutator over 2 or 8 GCDs is slower than one GCD
(AL-E-Q4): one flow per GCD. Kokkos Kernels backend written but unmeasured (tarball, AL-E-Q1);
gemms are 14-40 % of GPU time. CPU side also 3x faster for rank-5 tensors; serial libsci plus
`IMSRG_BLAS_CONCURRENT` 6-7x faster on the CPU (AL-E-Q3, relates to FP-Q6).

**Not yet done for the weekend:** the GPU path is not yet used by the flow driver or the farm
end to end; no farm run at emax 10 (one-item timing open); variants 2 and 3 not started; Tilde
Phi' only at the operator level; the isospin-averaging example (C14); the hub-branch clean-up
script (D61); the DIIS/linear-mixing change (D64); AL-D1 (every restart re-runs the 17-min setup).

## 4. Open points for part 2

* Night questions in `QUESTIONS.md` ("Added during the Al27 night"): AL-A-Q6 (imsrg++'s tensor
  one-body norm can be negative/NaN at emax 10 rank 5 — then its BCH runs to 40 nested commutators;
  emulate in variant 1, sensible norm in 2/3?), AL-A-Q5 (imsrg++ stores plain Edmonds reduced
  elements; the M15-Q1 "gauge" is a property of its operator builders), AL-C2 (restart re-runs
  HF/NAT), AL-F2..F4 (farm file names, cost model, resubmission), AL-B2/B3, AL-A-Q1/Q2.
* Kokkos Kernels tarball (owner: `curl` to `$IMSRG_DEPS_PREFIX/src/kokkos-kernels-5.3.0.tar.gz`).
* The 1.7 MB of fork tensor goldens in `tests/golden/tensor/fork-79e9330/` (every file < 1 MB).
* Weekend logistics: usage pauses every ~3 h, self-sustaining allocation chains, farm jobs of
  ~600-1,800 nodes (12 h limit for >= 184 nodes), the order of variants, reporting.
