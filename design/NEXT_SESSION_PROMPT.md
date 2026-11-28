# Prompt for the session after the 27Al weekend run (draft of 2026-11-27, final by 2026-11-30)

The 27Al weekend run (2026-11-26 22:00 to 2026-11-30 10:00 EST) is over; its report is
`design/reports/2026-11-27-weekend.md` and its prompt is in the git history of this file. The next
step is the owner's review of the results; the session below prepares and runs that review as an
interview, then plans the next milestone.

**Owner pre-flight (once):**

1. Start the session (Opus) on a Frontier login node in `$IMSRG_WORK/main`, on `develop`
   (`git pull --ff-only origin develop`).
2. Read the weekend report's Summary and `design/benchmarks/al27-variants.md` §1 first if you can.
3. Then paste everything below the line.

---

You are the orchestrator of the imsrg rewrite (C++20, Kokkos, MPI; reference imsrg++). The owner is
present for an interview. Follow `AGENTS.md`, `CLAUDE.md` and the interview style in your memory
(one topic at a time, recommended option first, progress bar after every round).

## Read first (delegate big reading; keep your context small)

`design/STATUS.md`, `design/reports/2026-11-27-weekend.md` (all), `design/benchmarks/al27-v1-farm.md`,
`design/benchmarks/al27-variants.md` (§1 and the conclusions), `design/benchmarks/al27-farm.md`
§4, `design/benchmarks/al27-restart.md`, `design/benchmarks/al27-jobs.md`, the section "Added during
the 27Al weekend" at the end of `design/QUESTIONS.md`, D61-D78.

## State (develop after the weekend)

* v1 (fork compat) reproduces every archived `.snt` element: ≤ 1e-9 (native Omegas), ≤ 1e-6
  (ME2JP); v2 and v3 complete for both Hamiltonians; results and one tarball per variant in
  `$IMSRG_WORK/runs/al27/` (`al27_v{1,2,3}.tar.zst`, `al27_tarballs.sha256`).
* v3 - v2 (E0 -0.47 / +0.08 MeV; dominant operator elements a median 4-9 %) is the truncation error
  of the unsplit Magnus(2) flow: ODE IMSRG(2) flows from the same H_0 agree with v3 to 15.5 / 2.2 keV.
* GPU production path (flows one per GCD, farm on GCDs), NAT |occupation| default, D72 tensor norm,
  restarts from `prepared/` H_0 with a SIGTERM checkpoint, per-item farm log; ~270 node-hours used.

## Interview agenda (ask, record answers in QUESTIONS.md and decisions as D79+)

1. The v1 acceptance and the archive-gap values (ΔN2LOGO FM2p at q = 975-1050, FPhitp2/4): accepted?
2. The variants report: is the v3-vs-v2 explanation (Magnus(2) truncation, ODE reference) convincing;
   which variant should future production use (v3 with omega_norm_max 0.25 and domega 0.1, or a
   smaller domega; v3's step error is 1-8 keV)?
3. The v2-vs-v1 tensor differences (up to 0.10, attributed to the emulated fork bugs M15-1/AL-F1 by
   their pattern): is a cleaner decomposition (one compat switch at a time) wanted?
4. WE-D-1: should the devel tokyo writer keep TCM two-body elements below its relative threshold?
5. Open weekend questions (WE-A-*, WE-B-*, B-T3-*, WE-C-*, WE-D-*, WE-O-1, WE-R-*) and the items
   deferred to after the weekend (atan device tolerance, CPU preset switch, KK compile check, AL-B3,
   AL-A-Q1, WE-A-1 hip-run CI stopping before the integration tests).
6. The next milestone (`design/40-roadmap.md`): what follows 27Al.

## Rules

As in `AGENTS.md` (golden rules 1-10, emax rule, never touch main or force-push, the integrator
merges after `tools/ci.sh frontier-cpu`). For any unattended stretch: a **recurring** upkeep
scheduled task from the start (not self-chained one-shot reminders), self-resubmitting allocation
chains (`tools/al27/alloc_chain.sh`), and never retire the clone of an agent that may still be resumed.
