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
  **Answer (owner, 2026-11-13):** Yes, read as the first option: store only `ch_bra <= ch_ket` blocks, as imsrg++ does (half the memory). This reading of a one-word answer to an either/or question is to be confirmed before tensor operators are implemented. Recorded as D25.
  **Confirmed (owner, 2026-11-15):** this reading is correct; D25 stands without the caveat.
* **Q6** — May IMSRG(3) 3B blocks use `float` storage to halve memory, if the benchmark shows
  acceptable accuracy (e.g. ΔE0 < 1 keV)? **Assumed meanwhile:** `double` only.
  **Answer (owner, 2026-11-13):** No: IMSRG(3) three-body blocks stay `double`; no `float` storage (D26).
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
  **Answer (owner, 2026-11-13):** Keep the imsrg++ behaviour, with a warning (implemented as `W-NSTEPS1`). Which legacy parameters are actually useful will be re-evaluated later (D27).
* **Q12** — `imsrg3_at_end` without `IMSRG3=true` rebuilds H from the normal-ordered input and
  drops the input 3N. **Assumed meanwhile:** keep it, with a warning.
  **Answer (owner, 2026-11-13):** Keep, with a warning (implemented as `W-Q11` in `driver/config/validate.cpp`) (D27).
* **Q13** — The new TOML config requires `emax`, `hw` and the file e1max values explicitly (no
  silent defaults); the legacy `key=value` layer keeps the old defaults and warns. OK?
  **Answer (owner, 2026-11-13):** OK (implemented: V-03 for `basis.emax`, `basis.hw` and `interaction.*.e1max` in TOML; the legacy layer keeps the old defaults and warns) (D28).
* **Q14** — `denominator_delta` without `denominator_delta_orbit` is silently ignored in imsrg++.
  **Assumed meanwhile:** keep it, with a warning.
  **Answer (owner, 2026-11-13):** Keep, with a warning (implemented as `W-Q3`; the key is dropped) (D27).

### Python (from `18-python-api.md`)
* **Q15** — Python runtime initialization: lazy on first use, or explicit `imsrg.Runtime()` context
  manager? **Assumed meanwhile:** explicit context manager, plus a lazy default for interactive use.
  **Answer (owner, 2026-11-13):** OK: explicit `imsrg.Runtime()` context manager plus a lazy default (implemented in M11 phase 1) (D29).
* **Q16** — Host access to device-resident blocks: read-only zero-copy host views that raise
  `StaleHostError` unless `sync_to_host()` was called? **Assumed meanwhile:** yes.
  **Answer (owner, 2026-11-13):** Yes. `sync_to_host()` exists; raising `StaleHostError` for device-resident data is still to be implemented and matters only for GPU builds (D29).
* **Q17** — Minimum Python version. The stable ABI needs ≥ 3.12. **Assumed meanwhile:** 3.10+,
  without the stable ABI.
  **Answer (owner, 2026-11-13):** 3.12 is the minimum Python version, which allows nanobind's stable ABI (D29). **Done 2026-11-13** (`feature/python-312-stable-abi`): `requires-python >=3.12`, `wheel.py-api = "cp312"`, both CMake lookups ask for 3.12, `_core` is built with `STABLE_ABI` (`_core.abi3.so`; wheel tag `cp312-abi3`).
* **Q18** — Which legacy input formats are needed? **Answer (owner):** me2j (2N), NO2B and me3j
  (3N), and the Navratil/Miyagi operator formats (`OperatorsFromFile`, two-body currents). Dropped:
  HDF5 3N, Oslo, OakRidge, "Jason", "Takayuki" and the other older readers.

### Process
* **Q19** — Agent compaction: Claude Code exposes a global `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` and a
  per-agent `autoCompactWindow`, not per-model percentages. The proposed settings
  (`design/claude-settings.proposed.json`) approximate "60 % for Sonnet/Haiku, 80 % for Opus".
  Please review them and install them before the run, if you agree.
  **Answer (owner, 2026-11-15):** use the per-agent windows only. Drop `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`: per the Claude Code docs it also applies to subagents, on top of their `autoCompactWindow`, so with 80 the Sonnet/Haiku agents would compact at ~480k and the Opus leads at ~640k tokens. Keep `autoCompactWindow` 600000 (Sonnet/Haiku agents) and 800000 (`imsrg-lead`) in the agent files, and give the Opus orchestrator `"autoCompactWindow": 800000` in the session settings (agents with their own frontmatter value do not inherit it). `design/claude-settings.proposed.json` updated (D31).

## Added during implementation

Ids: milestone leads use `M<k>-Q<n>` (e.g. `M2-Q1`) so that parallel branches never collide;
the orchestrator uses `O<n>`. Older entries from the first parallel leads may still read `Q20`,
`Q21`, ...; they are renumbered at merge time if they collide.

* **O1** (2026-11-11, orchestrator) — **How `develop` is advanced in this run.** The desktop app
  started the session in a git worktree (`.claude/worktrees/imsrg-refactor-0e4f11`, branch
  `claude/imsrg-refactor-overnight-800606`), while `develop` stays checked out in the main
  checkout, where this session may not run git (the permission check denied it). Merges therefore
  happen on the worktree branch, which serves as a staging copy of `develop`: `git merge --no-ff
  feature/<x>`, then `tools/ci.sh dev-openmp` on that merge result, and only on PASS is `develop`
  moved to the tested merge commit with `git update-ref refs/heads/develop <new> <old>` (old value
  checked). History is identical to merging on `develop` directly. **Side effect, owner action
  needed:** the main checkout's index and files still show the old `develop` (331b2d4), so `git
  status` there lists the night's work as staged reversals. Do not commit there before syncing.
  If you have no local edits in the main checkout, run
  `git -C /Users/cheshyre/Downloads/new-imsrg read-tree -m -u 331b2d4 develop` (safe two-tree
  fast-forward of index and files; it refuses if local edits conflict) or simply
  `git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` (HEAD is `develop`). **Assumed
  meanwhile:** this is acceptable, since the owner asked for merges into local `develop`.
  **Answer (owner, 2026-11-15):** the main checkout was synced after run 1 (its files match `develop` at 68c8927, written 2026-11-12 08:58), but `develop` has since advanced again from worktree sessions (2026-11-13, 2026-11-15), so it needs one more sync before run 2. Future unsupervised runs start **in the main checkout on `develop`** and merge there directly; the staging procedure above is only the fallback when the app forces a worktree (D31).
  **Answer (owner, 2026-11-26):** Obsolete on Frontier: the integrator's clone follows develop directly.
* **O2** (2026-11-11, orchestrator) — Q19 follow-up: the Claude Code docs say `autoCompactWindow`
  is an absolute token threshold and that Sonnet 5.5, Haiku 5.5 and Opus 5.x default to 1M-token
  windows, so the planned 150000 meant ~15 %. Changed to 600000 (Sonnet/Haiku agents, ~60 %) and
  800000 (`imsrg-lead`, ~80 %); see `41-agent-workflow.md` §5. The values take effect in the next
  session (definitions load at startup). Combining them with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` is
  undocumented. **Assumed meanwhile:** keep the per-agent windows and recommend dropping the
  percentage override from the proposed settings.
  **Answer (owner, 2026-11-15):** see Q19: per-agent windows only, no percentage override; the orchestrator gets 800000 through the settings key (D31).
* **Q20** (2026-11-12, M0) — Kokkos 5.3.0 (pinned as instructed: latest 5.x) requires NVCC ≥ 12.2
  and HIPCC ≥ 6.2 (`extern/kokkos/cmake/kokkos_compiler_id.cmake`), slightly above the baseline of
  Q9/D20 (CUDA ≥ 12, ROCm ≥ 6); its GCC ≥ 10.4 meets GCC ≥ 11. Options: raise the baseline to
  CUDA 12.2 and ROCm 6.2; or pin a Kokkos 4.x release for clusters with older toolkits (its
  minimums would need checking). **Assumed meanwhile:** keep Kokkos 5.3.0; CMake asks for
  `find_package(Kokkos 5)`. Moving the pin is a one-line change (`extern/README.md`).
  **Answer (owner, 2026-11-15):** raise the baseline to CUDA ≥ 12.2 and ROCm ≥ 6.2 and keep Kokkos 5.3.0 (D30, amends D20).
* **Q21** (2026-11-12, M0) — Include style for project headers: `10-architecture.md` §1 said
  `<imsrg/<module>/x.hpp>`, the M0 brief said `"imsrg/<module>/..."`. **Assumed meanwhile:** quotes
  (Google style; `.clang-format` sorts them into the last include group); `10` §1/§2 updated. The
  layering check accepts both forms, so either choice is safe.
  **Answer (owner, 2026-11-15):** confirmed (quotes, as the Google style guide also prescribes; D38).
* **Q22** (2026-11-12, M0) — The `kokkos-serial` deps build enables `Kokkos_ENABLE_DEBUG` and
  `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`, because dev-serial and asan are the debugging builds. So
  dev-serial differs from dev-openmp in more than the backend, and Serial runs are slower.
  **Assumed meanwhile:** keep it; removing the two options in `tools/bootstrap_deps.sh` rebuilds
  Kokkos automatically (the stamp changes).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **Q23** (2026-11-12, M0) — The Python venv is shared by all worktrees at `$IMSRG_DEPS_PREFIX/venv`
  (`IMSRG_VENV`), with a gitignored `.venv` symlink per worktree, instead of one `.venv` per
  checkout. **Assumed meanwhile:** shared, as the M0 brief asked; `IMSRG_VENV` can point anywhere.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **Q24** (2026-11-12, M0) — Owner action, optional: the Command Line Tools on the dev machine make
  `xcrun` pick `MacOSX27.0.sdk`, which `ld-1267` cannot link against (`20` §8). CMake works around
  it automatically; updating or reinstalling the Command Line Tools would make the workaround
  unnecessary. **Assumed meanwhile:** keep the workaround (it switches itself off once the default
  SDK links).
  **Answer (owner, 2026-11-15):** keep the CMake workaround. No agreement or license acceptance is involved: every installed Command Line Tools package is 26.6, and CLT 26.6 itself ships a pre-release `MacOSX27.0.sdk` whose `.tbd` stubs its own `ld-1267` cannot parse, so reinstalling 26.6 would not help. The owner installs the next CLT update when Software Update offers it; the workaround then switches itself off.
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
  **Answer (owner, 2026-11-15):** a full N2LO 3N file is not available yet; keep the cE-only file (IT-4/IT-5 unchanged). Deferred until the owner provides a full file; that file then gets new full-flow goldens, and the cE file stays as a permanent code-path test (D33).
* **M3-Q1** (2026-11-12, M3) — imsrg++ `GetjjSpace` indexes out of range for a proton or neutron count outside the tabulated closures. **Assumed meanwhile:** `define_valence_space("jj-shell", ...)` throws `InputError` for counts outside [2, 258); in-range behaviour is unchanged.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q2** (2026-11-12, M3 golden tests) — tests/golden/README.md says the ph (Pandya) channel field is Tz = |tz2_p - tz2_q|/2 (non-negative), but the dump files contain Tz in {-1, 0, 1} for ph_channel records, and their ch index obeys 6J + 2(Tz+1) + parity (same formula as pp channels). Is the README wording stale? **Assumed meanwhile:** the files are authoritative; the golden parser checks the index formula and does not constrain the sign.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q3** (2026-11-12, M3 ph/3B channels) — imsrg++ `ModelSpace::GetLocalIndex` (Pandya) returns `NumberKets-1` for an absent swapped ket instead of signalling absence. **Assumed meanwhile:** `ParticleHoleChannels::pair_index` and `local_index` return `std::nullopt` for absent kets (the golden files never exercise the buggy case); callers must handle the empty optional.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q4** (2026-11-12, M3 ph/3B channels) — `ThreeBodyChannel::CheckChannel_ket` contains stretched-state rules the reference itself marks "TODO Put these back in" (redundant). **Assumed meanwhile:** `ThreeBodyChannels` copies all of them exactly, so membership matches the golden files; dropping the redundant rules is left to a later cleanup.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q5** (2026-11-12, M3 golden tests) — The 14-file golden set includes an lmax=2 basis with 22 orbits, so the "norb = (emax+1)(emax+2)" assertion in test_golden_basis_dump.cpp is now restricted to files with lmax >= emax. This corrects a test assumption; the golden comparisons are unchanged. **Assumed meanwhile:** as described.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q6** (2026-11-12, M3 lead) — The 6j-weighted Pandya recoupling maps are not built in
  `basis`. `ParticleHoleChannels` provides the cross-coupled channel structure, `pair_index` (the
  2·nkets ordered-pair index) and flat tables; an elementwise precomputed map is
  O(n_cc² × n_J) per channel, far too large at emax 12-14, so its layout has to be designed with
  the comm222_ph kernel (design/12 §4.2). **Assumed meanwhile:** M7 (commutator) builds the maps
  from `ParticleHoleChannels` and the angmom 6j tables.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q7** (2026-11-12, M3 lead) — `ModelSpace` does not own its device mirror: `DeviceTables`
  is a separate object (`DeviceTables(const ModelSpace&)`), so that building a ModelSpace (and
  every host-only test of later modules) never needs an initialized Kokkos runtime, and the
  `m3-api-ready` behaviour of `ModelSpace::create` stays unchanged. **Assumed meanwhile:**
  consumers (operator/commutator) hold a `DeviceTables`; M4 may add a shared cache to the bundle.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q8** (2026-11-12, M3 lead) — Input checks stricter than imsrg++: a nucleus that does not
  fit into emax throws (imsrg++ prints "Trouble!" and continues with a partial filling), a core or
  hole orbit outside the basis throws (imsrg++ exits in `SetLmax`), while valence orbits outside
  the basis are dropped with a warning (as imsrg++ does); orbit labels accept multi-digit n
  (imsrg++ misreads them). **Assumed meanwhile:** keep the checks; none changes a valid run.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M3-Q9** (2026-11-12, M3 lead) — The IMSRG(3) configuration cuts (dE3max, OccNat3Cut) are not
  applied when the 3B kets are enumerated (imsrg++ enumerates with E3max only and applies them in
  the commutators); `ThreeBodyChannels::passes_configuration_cut` evaluates them per ket.
  **Assumed meanwhile:** M10 decides whether its layout drops those kets or masks them.
  **Answer (owner, 2026-11-15):** drop the kets that fail the cuts when the 3B kets are enumerated: the cut defines the 3B space and acts identically in every term (imsrg++'s inconsistent cuts count as bugs); validated against the brute-force m-scheme reference (D34).
* **M2-Q1** (2026-11-12, M2) — Logging needs process-wide output configuration (level, "only rank
  0 prints"), while AGENTS.md forbids mutable globals. `base` provides explicit, thread-safe
  `log::Logger` objects plus one `log::default_logger()` behind `imsrg::log::info(...)` (design/10
  §4). Timer registries and profiler hooks are explicit objects, never global. **Assumed
  meanwhile:** the default logger is the single sanctioned global (it configures output only and
  never affects results).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M2-Q2** (2026-11-12, M2) — design/10 §4 describes one `imsrg::Runtime` "(module exec plus
  comm)", but the DAG forbids exec<->comm edges and `mpi.h` is confined to comm. **Assumed
  meanwhile:** two RAII objects, `comm::Environment` (MPI, create first) and `exec::Runtime`
  (Kokkos, create second); the combined `imsrg::Runtime` is a thin composition in `driver` (M8).
  Alternative: allow comm -> exec (useful later for GPU-aware MPI on `DeviceBuffer`s).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M2-Q3** (2026-11-12, M2) — imsrg++ `PhysConst::INVSQRT2` is `0.7071067811865` (13 digits,
  relative error 6.7e-14), used in Operator.cc, imsrg_util.cc, Atomic.cc and DaggerCommutators.cc.
  `constants::inv_sqrt2` copies it exactly (as instructed). **Assumed meanwhile:** later modules
  use it only where they reproduce those imsrg++ formulas and `1/sqrt(2)` elsewhere; either way the
  effect (~1e-13 relative) is far below the 1e-10 unit tolerances.
  **Answer (owner, 2026-11-15):** use 1/√2 at full double precision everywhere; the 13-digit copy is renamed `imsrgpp_inv_sqrt2` and used only in tests. Rule: physical constants are copied exactly from imsrg++, mathematical constants are full precision (D36).
* **M2-Q4** (2026-11-12, M2) — linalg's non-Accelerate path declares Fortran LAPACK/BLAS symbols
  without the hidden string-length arguments that gfortran-compiled libraries formally expect
  (a known ABI pitfall); it is not compiled on this machine (Accelerate is used). **Assumed
  meanwhile:** fine for OpenBLAS/MKL in practice; on the first Linux build, test it or switch to
  passing the lengths (or LAPACKE).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M2-Q5** (2026-11-12, M2) — CPU CI cannot exercise GPU-only paths flagged by the exec review:
  `DeviceBuffer` with a separate host mirror (sync copies) and `team_for` kernels with team size
  > 1 (`team_barrier`/`single`). **Assumed meanwhile:** reviewers check these by hand; add a
  CUDA/HIP compile-and-run job when a GPU machine is available.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M2-Q6** (2026-11-12, M2) — `angmom::talmi_b` uses the Brody-Moshinsky factorial formula in
  double like `AngMom::TalmiB` (accurate to ~1e-15 relative for n <= 6, l <= 12, checked against
  exact values), while imsrg_util switches to an lgamma/long-double variant for large arguments and
  to quadrature when radial sums cancel badly (pmax >= 20). **Assumed meanwhile:** the factorial
  version suffices for M2; whoever ports radial integrals (opslib) decides on the large-argument
  path.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).

* **M2-Q7** (2026-11-12, M2) — `angmom::moshinsky` evaluates the Buck et al. sum in
  double-double (the plain double algorithm of imsrg++ misses the 1e-12 bound on 22 of the
  405,537 brackets with E <= 16, by up to 7.8e-11 relative, because large prefactors amplify
  the rounding of the CG/9j factors), so our values deviate from imsrg++ by up to ~1e-10
  relative on such brackets (logged in `30` §6). Its double-double tables cover E <= 40
  (`moshinsky_max_energy`, emax 20; NaN beyond, and `MoshinskyTable` checks it). The double
  `wigner` functions are validated on the emax-8 domain; at l ~ 30 the double <l1 0 l2 0|L 0>
  errs by ~1e-11. **Assumed meanwhile:** double-double only for the bracket tables (built once
  per run; ~3 s for all E <= 16 in a Debug build); raise the limits if larger spaces need them.
  **Answer (owner, 2026-11-15):** confirmed: Moshinsky brackets in double-double (D36).
* **M1-Q2** (2026-11-12, M1 flow) - "IMSRG(3f2)" in `refharness flow --factorized` means plain
  IMSRG(2) (no 3B storage, `IMSRG3` off) plus `BCH::SetUseFactorizedCorrection(true)`; imsrg++ has
  no switch for it. The `Elast`/norm divergence guards (|eta| > 1e12, |E| > 1e9) are harness
  additions. **Assumed meanwhile:** this is the intended IT-10 reference; O16 emax 2 converges
  with it, so no He4 substitute was needed (He4 emax 2 kept as extra).
  **Answer (owner, 2026-11-15):** No. IMSRG(3f2) is plain IMSRG(2) plus, in pyIMSRG terms: `imsrgsolver.SetHunterGatherer(True)`, `imsrgsolver.SetOmegaNormMax(0.1)`, `BCH.SetUseFactorizedCorrection(True)`, `Commutator.FactorizedDoubleCommutator.SetUse_1b_Intermediates(True)`, `Commutator.FactorizedDoubleCommutator.SetUse_2b_Intermediates(False)`. The IT-10 goldens were made with hunter-gatherer off, `omega_norm_max` 0.25 and 2b intermediates on (imsrg++ defaults, `FactorizedDoubleCommutator.cc:15-16`) and must be regenerated with these settings (new refharness flags); hunter-gatherer becomes a prerequisite of IT-10. `[imsrg3] factorized = true` makes these settings the defaults (D32).
* **M4-Q1** (2026-11-12, M4 lead) — The `operator` module cannot use the namespace
  `imsrg::operator` (C++ keyword). **Assumed meanwhile:** `imsrg::op`; the CMake target and the
  include path stay `imsrg::operator` / `imsrg/operator/...` (design/10 §3.3 updated).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M4-Q2** (2026-11-12, M4 lead) — An OperatorDump records occupations but no valence space.
  **Assumed meanwhile:** `model_space_for_dump` builds the partition with the holes
  (n >= OCC_CUT) as core and every other orbit as q-space; the reader rejects occupations that
  differ from the target model space unless `check_occupations = false`.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M7-Q1** (2026-11-12, M7 lead) - The golden README called the omitted
  `Z_comm111ss__eta_white__H` rounding noise (~1e-16), but the term is ~1.4e-8 (the one-body
  parts of eta_white and the HF H nearly commute, so the relative hermiticity check of the
  reader, 1e-12 * max entry, rejects the dump). **Resolved in M7:** README wording corrected;
  the tests compare comm111ss for that pair with `Z_all` minus the other golden terms (3.9e-16).
  **Answer (owner, 2026-11-15):** confirmed (already resolved in M7).
* **M7-Q2** (2026-11-12, M7 lead) - The fast IMSRG(2) kernels support X and Y that are each
  Hermitian or anti-Hermitian; a NonHermitian input throws NotImplementedError (the ladder and
  Pandya paths use X^T = hX X; imsrg++ silently returns wrong numbers, reference/02 P-9).
  `commutator_ref` handles any hermiticity. **Assumed meanwhile:** no phase-1 flow needs
  non-Hermitian operators; add general paths (explicit Y P X products) if one does.
  **Answer (owner, 2026-11-15):** support general operators now: run 2 adds non-Hermitian paths to the fast IMSRG(2) kernels, tested against `commutator_ref` (D36).
* **M7-Q3** (2026-11-12, M7 lead) - design/30 section 1 lists the Jacobi identity as a property
  test, but the IMSRG(2)-truncated commutator satisfies it only for one-body operators (the
  dropped 3B part of [Y2, W2] feeds back into 2B through comm132-type terms). **Assumed
  meanwhile:** the Jacobi test uses particle_rank-1 operators; antisymmetry, linearity and the
  hermiticity of [A, H] are tested for the full IMSRG(2) commutator.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M7-Q4** (2026-11-12, M7 lead) - Workspace memory: the ladder intermediates (Mpp, Mhh and the
  gathered operands, about 4x a two-body operator) and the Pandya scratch (XK, YK, W, ZB, about
  6x) are allocated eagerly; at emax 12 that would be ~11-13 GB (reviewer estimate). Fine for
  emax <= 6 (all tests and runs, AGENTS.md rule 3). **Assumed meanwhile:** keep the eager
  allocation; M12 (performance) can share ZB with W, gather in place, and stream channels.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
  **Done in M12a (2026-11-16):** ZB folded into W, YK stored for one pair order, closed-shell
  ladder operands read in place: Workspace scratch at O16 emax 6 from 10.4x to 5.5x a two-body
  operator (`design/benchmarks/imsrg2-performance.md`).
* **M6-Q1** (2026-11-12, opslib) - The golden `Rp2.dump` includes the spin-orbit term
  (`RpSpinOrbitCorrection`), contradicting the brief. **Assumed meanwhile:** `rp2` includes it by
  default (matches the golden to 6e-16); `include_spin_orbit = false` excludes it.
  **Answer (owner, 2026-11-15):** fix: `rp2` excludes the spin-orbit correction, which becomes its own catalog operator (added explicitly for charge radii); no `include_spin_orbit` option and no imsrg++ compatibility switch (D35).
* **M6-Q2** (2026-11-12, opslib) - imsrg++ quirks reproduced: `t_cm`'s two-body part drops
  entries with |v| <= 1e-7 (R2CM and the radius terms keep all); the E2max skip in the
  `Calculate_p1p2`/`r1r2` loops is moot because the two-body ket basis is already E2max-limited.
  **Assumed meanwhile:** reproduced as is; not silently fixed.
  **Answer (owner, 2026-11-15):** fix: `t_cm` keeps all two-body entries (no 1e-7 threshold), so T_int and H change by up to 1e-7 MeV per element. No compatibility switch; instead the e2e goldens are regenerated from a patched imsrg-devel build (threshold removed, Rp2 without spin-orbit; patch kept in `tools/reference/`), so the ITs keep ~1e-10 MeV agreement and identical step counts (D35).
* **M5-Q1** (2026-11-12, M5/M6 lead) - Input format detection: me2j text is read through zlib
  `gzread`, which detects gzip by content (any extension works); the me2j `.bin` variant (raw
  floats after a 255-byte header) is not implemented. NO2B is binary when the file NAME (last path
  component) contains `stream.bin` (imsrg++ tests the whole path), else plain or gzip text with a
  header line. **Assumed meanwhile:** sufficient for the owner's files; add me2j `.bin` on demand.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M6-Q3** (2026-11-12, M5/M6 lead) - HF DIIS design (owner: "a correct DIIS"): Pulay
  extrapolation of the Fock matrix with error F rho - rho F, the B matrix in the m-scheme metric
  (rows weighted by 2j+1), first extrapolation at iteration 2, subspace 8, history reset when
  the occupations change; the same convergence test as imsrg++ so iteration counts compare
  (golden cases: 8-10 DIIS iterations vs 20-30 linear). imsrg++'s own DIIS rescue (after 100
  linear iterations) is replaced by this one (`30` §6 M6-1). **Assumed meanwhile:** these
  defaults; revisit if open-shell or natural-orbital references converge poorly.
  **Answer (owner, 2026-11-15):** confirmed (D36).
* **M6-Q4** (2026-11-12, M5/M6 lead) - `op::No2bThreeBody::get_pn` scans the isospin orbits
  linearly and recomputes the isospin CG coefficients on every call (noted by M4). HF with 3N
  at emax 4 still solves in < 0.1 s. **Assumed meanwhile:** fine for emax <= 6; M12 adds an
  orbit-index cache and a CG table if profiles show it.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
  **Done in M12a (2026-11-16):** dense orbit table and tabulated isospin CGs, bitwise identical
  (M12-Q1 for the remaining hash lookups).
* **M8a-Q1** (2026-11-12, M8a config) - `Generator` and `Method` enums (with their canonical TOML
  spellings) live in `driver/config/schema.hpp`, since `flow` does not exist yet. **Assumed
  meanwhile:** M8 makes `flow` reuse them or maps them 1:1 in the driver; the spellings are fixed.
  **Answer (owner, 2026-11-15):** confirmed (resolved by M8-Q1).
* **M8a-Q2** (2026-11-12, M8a config) - design/16 2.7 marks methods `flow` (DP45) and `flow_rk4`
  as P1, but V-46 says "methods other than magnus are PL". **Assumed meanwhile:** the table wins
  (roadmap M8 lists flow-RK4/DP45): `magnus`, `flow`, `flow_rk4` are P1, the rest `E-PLANNED`.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M8a-Q3** (2026-11-12, M8a config) - With `imsrg.steps = 1` the stage-2 generator is inert:
  `resolve` resets `generator.valence` to its default and `resolved_config.toml` omits it, so V-41
  (explicit `generator.valence` + one stage + valence space = TOML error) and the round trip both
  hold. **Assumed meanwhile:** fine; revisit with the Q1 HUMAN decision of design/16 3.6.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M8a-Q4** (2026-11-12, M8a legacy) - Legacy `valence_space=<nucleus>` with an explicit,
  different `reference=<nucleus>` (imsrg++: core filled as the first nucleus, normal ordering
  w.r.t. the second, no valence orbits) cannot be expressed in the TOML schema. **Assumed
  meanwhile:** `E-LEGACY-NS`; add a schema key if anyone needs it.
  **Answer (owner, 2026-11-15):** not needed; stays `E-LEGACY-NS` (D37).
* **M8a-Q5** (2026-11-12, M8a CLI) - Exit codes: `--print-config` runs the full validation
  including the V-14 input-file preflight (exit 3 if inputs are missing); exit 3 only if every
  error is V-14 or E-FILE; the pipeline stub `driver::run` throws `ConfigError("E-PLANNED ...")`
  (exit 2) after writing `resolved_config.toml`. Relative input paths are relative to the working
  directory (as imsrg++), not to the config file. **Assumed meanwhile:** as described.
  **Answer (owner, 2026-11-15):** exit codes confirmed. Paths change: relative paths in a TOML config are resolved against the config file's directory; legacy `key=value` and command-line values stay relative to the working directory; `resolved_config.toml` therefore records absolute paths (D37).
* **M8a-Q6** (2026-11-12, M8a operators) - The particle ranks of the P1 operator catalog are best
  guesses, and orbit parameters (`OneOcc_<orb>`, `AllOcc_<orb>`, `OBD_<orb>_<orb>`) use imsrg++
  orbit labels such as `p0s1`, while R05 1.2 reads `OneOcc` via a ModelSpace string (e.g. `1p3`)
  and `AllOcc` via l, j, tz. **Assumed meanwhile:** labels as in design/16 4.1; opslib (M5/M6)
  and M8 confirm ranks and the old spellings, and the catalog follows.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M8a-Q7** (2026-11-12, M8a legacy) - `dE3max` not given maps to "absent" (no cut) instead of
  the old 99; with emax <= 6 every 3-body energy is far below 99, so the numerics agree.
  **Assumed meanwhile:** M8/M10 treat an absent `imsrg3.de3max` as no cut (e.g. in the
  `new_E3max` rule of `imsrg3_at_end`). `valence.custom` is stored as a string; its orbit-list
  format is for M8 to define and check.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M11-Q1** (2026-11-12, M11 python) - Operators alive at interpreter exit block `Runtime`
  finalization; the atexit hook runs `gc.collect()` first, but reference cycles holding operators
  in user code could still keep one alive. **Assumed meanwhile:** `_atexit()` finalizes only if no
  operators remain, otherwise leaves Kokkos to process exit; a C++ `Runtime` API to force
  finalize would be cleaner.
  **Answer (owner, 2026-11-15):** keep; revisit for GPU builds: when M11 phase 2 adds `StaleHostError`, add a force-finalize that frees the device data of live operators (later access raises) (D41).
* **M11-Q2** (2026-11-12, M11 python) - `tests/golden/basis/*.txt` header documents `channel idx
  2J parity 2Tz nkets` but stores J and Tz. **Assumed meanwhile:** tests compare with `two_j/2`,
  `two_tz/2`; the harness comment should be corrected.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M8-Q1** (2026-11-12, M8 lead) - M8a-Q1 resolved by a 1:1 mapping: `flow::GeneratorKind`
  mirrors `driver::Generator` (incl. ShellModelAtanNpnh and OnePa, which throw
  NotImplementedError), and `driver/run.cpp` maps the enums with a switch. **Assumed
  meanwhile:** the driver enum stays the schema enum; flow has no dependency on driver.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M8-Q2** (2026-11-12, M8 lead) - The flow CSV has no MP2 column (design/13 section 7 lists
  one) because `mbpt` (M9) does not exist yet; results.json has no EMP2 either.
  **Assumed meanwhile:** M9 adds the column (cheap, per step) and the IT-1 MBPT numbers.
  **Answer (owner, 2026-11-15):** confirmed: run 2 adds the MP2 column and EMP2 (mbpt exists now).
* **M8-Q3** (2026-11-12, M8 lead) - The Omega stack is always kept in memory:
  `imsrg.omega.storage = "scratch"`/`"discard"` are accepted but behave like "memory" (at
  emax <= 6 an Omega is a few MB). HF mixing is always linear (the imsrg++ default): the schema
  has no key to select DIIS yet. **Assumed meanwhile:** add `hartree_fock.mixing` and real
  scratch storage in a follow-up; both are numerics-neutral for the IT goldens.
  **Answer (owner, 2026-11-15):** `hartree_fock.mixing` defaults to `diis` in TOML configs and to linear mixing in the legacy layer (imsrg++ reproduction); IT configs set it explicitly. Omega stays in memory; real scratch storage comes with checkpoint/restart, which is deferred until after M13 (D37, D39).
* **M8-Q4** (2026-11-12, M8 lead) - imsrg++ prints "pushing back another Omega" via stdout;
  ours logs it at info level, and the flow lines go to the log as well as to the CSV. Output
  file names differ from imsrg++ (`<stem>.snt`, `<stem>_<op>.snt`, `results.json`).
  **Assumed meanwhile:** fine (design/16 output section); the ITs compare numerically.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M9-Q1** (2026-11-12, M9 mbpt) - mbpt may not depend on hf, but the NAT-basis Hamiltonian needs
  hf's normal-ordering machinery for a density other than the HF one. **Assumed meanwhile:**
  mbpt exposes only `mbpt2_density_matrix`, `natural_orbitals`, `reference_density_ho`, `matmul`
  and `to_nat_basis`; the tests (`tests/unit/mbpt/mbpt_test_support.hpp`) assemble the NAT H by
  passing a modified `HFResult` (C = C_ho2hf C_hf2nat, rho, F, E_HF, reference occupations) to
  `hf::normal_ordered_hamiltonian`, using `hf::detail::Monopole2B/3B`. hf should expose a public
  "normal-ordered H for a given orbital basis and density" API (or the driver, M8, owns this).
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M9-Q2** (2026-11-12, M9 mbpt) - `natural_orbitals` checks sum (2j+1) n = A with 1e-6; imsrg++
  uses 1e-8 and `exit()`s. **Assumed meanwhile:** 1e-6 and `InvariantError`; measured deviations at
  emax 4 are far below both.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).
* **M9-Q3** (2026-11-12, M9 mbpt) - Not implemented: `use_NAT_occupations` and the `NAT_order`
  "energy"/"mp2" options (imsrg++ default occupations = reference, order = descending occupation
  within a channel). **Assumed meanwhile:** default behaviour only; add when a job needs them.
  **Answer (owner, 2026-11-15):** not now; add the options when a job needs them (D36).
* **M9-Q4** (2026-11-12, M9 mbpt) - IT-5 `Hbare_0b` / `mbpt.EMP2` in `result.json` are the NAT-basis
  E_HF and E_MP2 (the `hf.EHF` entry is the HF-basis value). Reproduced to 2e-10 (E_HF NAT) and
  4e-9 (E_MP2 NAT); worth noting in design/30 that the keys have this meaning.
  **Answer (owner, 2026-11-15):** confirmed as assumed (low-impact batch).

## Owner interview 2026-11-15

Every entry above now has an answer; the low-impact ones were confirmed as a batch. The interview also settled
further design decisions (D30–D42 in `00-requirements-and-decisions.md`; report
`design/reports/2026-11-15-owner-interview.md`). Run 2 appends its new questions below.

## Added during run 2

* **MS-Q1** (2026-11-15, MS Google style) - Google style names variables in snake_case, but the code
  uses physics notation with capitals in about 900 variable, parameter and member names (`X`, `Y`,
  `Z` for commutator operands, `H`, `A`/`Z`, `J`, `Jp`, `Tz`, `hw_MeV`, `twoJ`). Lower-casing them
  would merge distinct quantities (`j` and `J` often live in the same scope) and lose the
  correspondence with imsrg++ and the literature. **Assumed meanwhile:** they keep their capitals;
  documented as exception 6 in design/17 section 3; `.clang-tidy` accepts any variable name with a
  capital. Reversible by a clang-tidy run if the owner prefers strict snake_case.
  **Answer (owner, 2026-11-16):** keep the physics capitals (exception 6), but tighten the
  clang-tidy check: the ignore pattern (today any name containing a capital, which also lets
  camelCase names through) becomes a curated physics-symbol list (X, Y, Z, H, J, Jp, Tz, hw_MeV,
  twoJ, ... as names or name parts). A small run-3 task, no renames (D50).
* **MS-Q2** (2026-11-15, MS Google style) - The guide lets accessors and mutators keep variable-style
  names. **Assumed meanwhile:** an accessor is a const method returning a property or part of the
  object, possibly selected by an index or key (`num_orbits()`, `orbit(i)`, `spec()`,
  `get_normalized(...)`); `set_*` mutators and STL analogs also stay snake_case; everything else is
  PascalCase (static factories too: `ModelSpace::FromNames`). The 199-name allowlist is in
  `.clang-tidy` (design/17 section 3 item 7). Alternative: PascalCase for all methods (would collide
  with type names such as `Orbit`, `Channel`, `Reference`, `Partition`, `ModelSpace`).
  **Answer (owner, 2026-11-16):** keep the rule as applied in run 2: accessors and `set_*` mutators
  snake_case, every other method PascalCase, allowlist in `.clang-tidy` (D50).
* **MS-Q3** (2026-11-15, MS Google style) - Header guards outside `src/` use cpplint's fallback, the
  path from the repository root without the `IMSRG_` prefix (`TESTS_UNIT_HF_HF_TEST_SUPPORT_H_`,
  `PYTHON_SRC_BINDINGS_H_`). **Assumed meanwhile:** fine (unique, and what the blocking cpplint step
  accepts); an `IMSRG_` prefix there would need NOLINTs or extra CPPLINT.cfg files.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **MS-Q4** (2026-11-15, MS Google style) - `AUTHORS` lists the owner (taken from the git history) as
  the copyright holder named by "Copyright 2026 The imsrg authors"; the SPDX line says
  `GPL-3.0-or-later` as written in design/17 section 2.A (LICENSE is the GPL-3.0 text). **Assumed
  meanwhile:** as written; please confirm the holder list and "or later".
  **Answer (owner, 2026-11-16):** GPL-3.0-only, not "or later"; the holder list stays as written
  (AUTHORS). Every SPDX line becomes `// SPDX-License-Identifier: GPL-3.0-only` (D46, amends
  D17/D38); still compatible with imsrg++ (GPL-2.0-or-later). Implemented during the interview at
  the owner's request: `feature/license-gpl3-only` merged into `develop` (6bcca88; `tools/ci.sh
  dev-openmp` PASS, new step `license headers`).
* **MS-Q5** (2026-11-15, MS Google style) - Structural items not converted (design/17 section 6):
  inline functions over ~10 lines in headers (device-code lookups, templates, header-only test
  helpers), the 36 structs with member functions, functions over ~40 lines, unsigned types.
  **Assumed meanwhile:** guidelines for new code, not worth a churn of the existing code; device
  lookups stay inline for performance. Schedule a cleanup task if wanted.
  **Answer (owner, 2026-11-16):** scheduled task with priority: a full structural pass in run 3,
  ahead of tensor operators. Structs with member functions that hold invariants become classes,
  unsigned types become signed except sizes and bit patterns, functions over about 40 lines are
  split, long inline functions in headers move to `.cc` files (device-code lookups and templates
  excepted); results bitwise identical (D50). Placement (owner, same interview): first in run 3,
  right after the small interview tasks, alone and time-boxed, with the N7 and MPI work prepared
  outside the repository meanwhile (D53).
* **O3** (2026-11-15, orchestrator) — **Run 2 started in an app worktree again, not in the main
  checkout (D31 not met).** The desktop app opened the session in
  `.claude/worktrees/imsrg-rewrite-run-2-36094d` (branch `claude/imsrg-rewrite-run-2-36094d`, at
  `main` = 71d5917), and the permission check denied every git command against the main checkout
  (`git -C /Users/cheshyre/Downloads/new-imsrg status ...`) as well as `git reset --hard` in the
  worktree. The preflight was therefore adapted: the worktree branch was fast-forwarded to
  `develop` (c01d766) and serves as the staging copy of `develop` (O1 fallback); `feature/design-plan`
  is an ancestor of `develop`; the worktree had no tracked changes; `tools/ci.sh dev-openmp` passed
  (137 s, 13:55 JST). The checks `git branch --show-current` = `develop` and the status of the main
  checkout could not be run. `develop` is advanced only with `tools/ci.sh dev-openmp && git
  update-ref refs/heads/develop <new> <old>` on the tested merge commit. **Side effect (owner
  action):** the main checkout's index and files keep showing the `develop` it had before run 2;
  sync it before committing there (`git -C /Users/cheshyre/Downloads/new-imsrg reset --hard` if it
  has no local edits, else `read-tree -m -u <old> develop`). **Assumed meanwhile:** acceptable, as in
  run 1. To start a run in the main checkout, the session has to be opened there without a
  worktree (app setting).
  **Answer (owner, 2026-11-16):** the main checkout was synced and `build/golden-generated`
  regenerated after run 2 ("Already done"). Runs start in an app worktree from now on: the staging
  procedure (merge on the worktree branch, then `tools/ci.sh dev-openmp && git update-ref
  refs/heads/develop <new> <old>`, chained to the exit codes, with `ctest -L slow` for numerics
  merges) is the standard, not a fallback (D44, amends D31). The owner re-syncs the main checkout
  after each run (owner action in every run report).
  **Answer (owner, 2026-11-26):** Obsolete on Frontier: the integrator's clone follows develop directly.

### M8-Q10 (hunter-gatherer, `H(s)` after the final gather)
imsrg++ `GatherOmega` rebuilds `H_saved` from `H_0` through the gathered Omega but leaves `H(s)`
untouched, so after a hunter-gatherer solve `H(s)` and `Transform(H_0)` differ at the truncation level
of the BCH product (up to 2e-3 in the O16 emax 2 unit test (4 gathers; E0 1e-4, see its printout)); stage 2's
first eta comes from the pre-gather `H(s)`. **Assumed:** copy imsrg++ (orchestrator decision), because
the IT-10 and `IT-2/hunter_gatherer` goldens depend on it. A cleaner variant (re-evaluate `H(s)` from
the gatherer, or gather before the next stage) would be an option with a deviation entry.

**Answer (owner, 2026-11-16):** after the final gather of a hunter-gatherer solve, H(s) becomes the
H rebuilt from H0 with the gathered Omega (`h_ = h_saved_`), so E0 and every transformed operator
come from the one gathered Omega, which is what hunter-gatherer exists for in 3f2 (D32). Measured
during the interview (patch on `develop` c2fef30): ΔE0 = -13.4 keV (IT-10 O16 emax 2, 24 gathers),
+2.8 keV (IT-10 He4 emax 2, 8), -43 keV (IT-2 hunter-gatherer O16 emax 4, omega_norm_max 0.25, 8
gathers), -22 keV (0.1, 21 gathers), -8.1 keV (He4 emax 2, 7 gathers); step and gather counts and
the trajectories are unchanged. The rebuilt H is not flowed further; stage 2 starts from it. The D35
patch of imsrg-devel gets the same change, the IT-10 and hunter-gatherer goldens are regenerated
through `tools/reference` (D42), and the deviation goes into `30` section 6 (D48).

### M8-Q11 (hunter-gatherer, gather count and `set_generator`)
`SolveResult::num_gathers` counts the loop gathers plus the final gather (as the `gathering Omega.`
log lines of imsrg++; the goldens' `n_gathers`). A gather inside `set_generator` (hunter non-zero,
which cannot happen after a hunter-gatherer `Solve`) is counted in `MagnusSolver::num_gathers()` only.
`results.json` has `imsrg.num_gathers` (solver total) and `imsrg.stages[].num_gathers`.
`imsrg.omega.hunter_gatherer` is rejected nowhere any more; `imsrg.omega.storage = scratch/discard`
with hunter-gatherer is unchanged (not implemented either way).

**Answer (owner, 2026-11-16):** confirmed as assumed (config/Python batch).

### M8-Q12 (flow fixes: ds with `magnus_adaptive = false`, divergence on NaN)
(1) imsrg++'s member `ds` is set by `SetDs(ds_0)` and capped by `SetDsmax`; with `magnus_adaptive = false`
the Magnus loop uses `ds` unchanged (then `min(ds, smax - s)`). **Assumed/implemented:** the solver
starts each `Solve` with `ds = min(ds_0, dsmax)`; adaptive mode is untouched (golden flows identical).
(2) A NaN/inf ||eta|| or E0 is `StopReason::kDiverged` (the driver then throws `NumericalError`, M8-3);
unlike imsrg++ (`FlowingOps[0] *= 1.0/0.0`) H is left as it is. (3) `denominator_cutoff <= 0` is
rejected by `ComputeGenerator` with `InputError` (0 would disable the floor and give inf/NaN).
(4) `imsrg.omega.storage = scratch|discard` is `Phase::kPlanned` in the enum table (E-PLANNED at
validation); the round-trip tests use `memory`.

**Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M8-Q20** (2026-11-15, M8 D35 task) - D35 asks that "wherever the code computes a charge radius" the
  Rso contribution is added. The code has no charge-radius computation yet (design/16 section 6 assigns
  `PROTON_RCH2`, `NEUTRON_RCH2`, `DARWIN_FOLDY` to the output stage), so only the catalog operator `Rso`
  (`PrepareOperator`) was added and the contract is documented (`opslib::Rp2` docstring, design/13,
  design/30 D35-2): charge radius = Rp2 + Rso/Z + nucleon-size terms. **Assumed meanwhile:** the
  output-stage task implements it that way. Not blocking.
  **Resolved (run 2, M8 charge-radius task):** implemented in `driver/pipeline/charge_radius.h`, written to
  `results.json` as `charge_radius` (schema version 3), see design/13 section 8 and M8-Q50..Q52.
  **Answer (owner, 2026-11-16):** confirmed as implemented (see M8-Q50..Q52).
* **M8-Q21** (2026-11-15, M8 D35 task) - The gitignored emax-4 dumps in `build/golden-generated` (read
  through `IMSRG_GENERATED_GOLDEN_DIR` by `opslib_golden`, `hf_*`, `io_*` emax-4 tests) must come from the
  patched build too; with dumps from the pristine build `Emax4TrelAndRp2` fails (Rp2 differs by 0.063).
  **Assumed meanwhile:** after merging, run `REFHARNESS=<ref root>/harness-build-d35/refharness
  tools/reference/gen_golden_generated.sh` once per checkout (also done by `regen_golden_d35.sh` into
  `<ref root>/golden-generated-d35`). Alternative: make the script default to the patched harness.
  **Resolved (2026-11-15, M8 review follow-up):** every `gen_golden_*.sh` now defaults to the patched
  harness, so `tools/reference/gen_golden_generated.sh` alone regenerates `build/golden-generated` with
  the right build; each checkout runs it once (`tests/golden/README.md`).
  **Answer (owner, 2026-11-16):** confirmed (resolved).
* **M8-Q22** (2026-11-15, M8 D35 task) - `tools/reference/refharness.cpp` and `gen_golden_flow.sh` were
  edited (a few lines each: `IMSRG_REFERENCE_SHA_FILE`, `IMSRG_GOLDEN_E2E_OUT`, `FLOW_JOBS`); the
  hunter-gatherer task edits the same two files, so the merge may need a manual resolution of
  neighbouring hunks. **Assumed meanwhile:** the orchestrator resolves it (both changes are independent). **Resolved at the merge (orchestrator, 2026-11-15):** `gen_golden_flow.sh` taken from the hunter-gatherer branch, which already contains the D35 overrides; `refharness.cpp` merged automatically.
  **Answer (owner, 2026-11-16):** confirmed (resolved).
* **M8-Q30** (2026-11-15, M8 config/driver task) - Relative paths (D37): a TOML *file*'s values resolve
  against the file's directory, but the **default** `output.directory = "output"` (no key in the file)
  and `--set` values resolve against the working directory. Alternative: default relative to the config
  directory too. **Assumed meanwhile:** working directory (an output default is not a path "in the
  config"); implemented in `LoadConfig` and documented in design/16 1.8. The TOML examples name their
  inputs without a directory, so they run from a directory that holds the data (`examples/config/README.md`).
  **Answer (owner, 2026-11-16):** confirmed as assumed (config/Python batch).
* **M8-Q31** (2026-11-15, M8 config/driver task) - `results.json` schema version: the layout was
  already new in run 2 (nothing released), so `schema_version = 1` covers the table as documented,
  including `e_mp2`, `imsrg.num_gathers` and the NAT-basis meaning of `e_reference`/`e_mp2`. **Assumed:**
  bump to 2 only for the first change after a release; the table test fails on any undocumented key.
  **Answer (owner, 2026-11-16):** confirmed with the policy as practised: `schema_version` is bumped
  on every documented schema change (1 to 2 to 3 in run 2), each with a changelog entry in `16`; the
  table test fails on any undocumented key (config/Python batch).
* **M8-Q32** (2026-11-15, M8 config/driver task) - `hartree_fock.mixing = "diis"` (the new TOML default)
  stops at the same HF tolerance with fewer iterations but changes E_HF at the 1e-8 MeV level and may
  change flow step counts relative to imsrg++. **Assumed:** the integration tests that reproduce
  imsrg++ step counts (IT-2..IT-7, IT-12, hunter-gatherer) set `mixing = "linear"` explicitly; legacy
  command lines get `linear` automatically.
  **Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M8-Q33** (2026-11-15, M8 config/driver task) - NAT basis in the driver covers `natural_orbitals`
  defaults only: `use_occupations = true` and `order = "energy"` are `E-PLANNED` (M9-Q3, D36). The NAT
  reference occupations are those of the HF solution (closed shells; ensemble references untested).
  New `hf::SolutionInBasis` is the public API replacing the test-only `hf::detail` coupling (M9-Q1).
  **Answer (owner, 2026-11-16):** confirmed as assumed (config/Python batch).

### M8-Q40 (F2b: deviations of the ODE flow methods from imsrg++)
(1) imsrg++'s adaptive `flow` method integrates a COPY of the solver (`auto system = *this;`,
IMSRGSolver.cc:826), so the member `s` (and `ds`) never advance during stage 1; a second stage then
restarts at s = 0 and integrates [0, smax] again (RK4 does not have this). **Assumed/implemented:** the
flow parameter continues across stages for both methods; the results agree whenever stage 1 has
converged, the flow log shows continued s. (2) Non-finite or runaway (> 1e12 / 1e9) ||eta|| or E0 ends
`flow` and `flow_rk4` as `kDiverged` (imsrg++: no guard in these paths; its Magnus path has one). (3)
imsrg++'s ODE monitor (`monitor.report()`) prints times with 3 decimals from a stale `Eta`; not
reproduced. (4) `hunter_gatherer = true` with a flow method is a ConfigError (imsrg++ gathers an Omega
that is always zero). (5) Tensor operators and operators that are not number conserving cannot flow
(`OdeSolver` throws InputError; `PrepareOperator` supplies scalars only today). (6) `flow_euler`,
`restore_4th_order`, `magnus_backoff`, ... stay E-PLANNED. (7) The Magnus path lacks imsrg++'s
`shell-model-atan` denominator-cutoff switch (1e-6 after ||eta|| < 1): a no-op there because the
default cutoff is already 1e-6; `flow_rk4` reproduces the 1.0 -> 1e-6 switch.

**Answer (owner, 2026-11-16):** items (1) continuing s, (2) the divergence guard, (4) the
ConfigError for hunter-gatherer with a flow method and (7) no shell-model-atan switch in the Magnus
path are confirmed. In addition, stage 2 integrates [s1, s1 + smax] for every method (Magnus,
`flow_rk4`, `flow`): this replaces imsrg++'s doubled Magnus smax and fixes the empty `flow_rk4`
stage 2 when stage 1 reached smax (D47). Goldens stay where stage 2 converges before its end; a case
that stops at smax (check IT-15 RK4) is regenerated from the patched reference with the reason and
old/new numbers (D42).

### M8-Q41 (F2b: `ode_tolerance` as float)
The tolerance is rounded to float (`static_cast<double>(static_cast<float>(tol))`) like imsrg++'s
`SetODETolerance(float)`, so 1e-6 is 9.99999997e-7; the goldens' step counts depend on it only at
round-off, but design/16 Q14 ("fix: double") is NOT applied to the ODE path, as for `eta_criterion`
(quirk P7). One line to change if the owner prefers double.

**Answer (owner, 2026-11-16):** double for both `eta_criterion` and `ode_tolerance`, as `16` Q14
designed (quirk P7 dropped). The thresholds move by 3e-9 relative; if a golden step count flips, the
D35 patch of imsrg-devel gets the same change and the golden is regenerated with the reason and
old/new numbers (D42) (D47).
* **M10-Q20** (2026-11-15, M10 3f2) - Only the topologies with one-body intermediates of the
  factorized double commutator are implemented (the D32 recipe). `FactorizedOptions::
  two_body_intermediates` / `[imsrg3] factorized_2b_intermediates = true` throw NotImplementedError /
  E-PLANNED. **Assumed meanwhile:** the 2b topologies (imsrg++ `comm223_231_chi2b`,
  `comm223_232_chi2b`, Pandya-transformed intermediates, about 25 two-body operators of memory) are
  needed only if the owner wants the imsrg++ default (both on). Say if they should be scheduled.
  **Answer (owner, 2026-11-16):** not needed now. The 2b topologies stay E-PLANNED; the D32 recipe
  (1b intermediates on, 2b off) remains the production 3f2 definition (D49).
* **M10-Q21** (2026-11-15, M10 3f2) - `[imsrg3] factorized` together with `in_flow`/`at_end` is an
  error (V-57: the correction would double count the explicit 3-body terms); imsrg++ does not forbid
  it. Also V-56 (sub-switches without `factorized`). **Assumed meanwhile:** errors; reversible.
  **Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M10-Q22** (2026-11-15, M10 3f2) - The factorized correction requires an anti-Hermitian eta
  (NotImplementedError otherwise): imsrg++'s `Eta_matrix_nnnn` sign is commented "SRS this is a guess
  to fix bug for herm,herm case", so a Hermitian eta is not validated there either. Every BCH use
  has an anti-Hermitian Omega. **Assumed meanwhile:** fine.
  **Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M10-Q23** (2026-11-15, M10 3f2) - imsrg++ applies the 3f2 correction to every BCH call once
  `SetUseFactorizedCorrection` is on, including the transformation of other operators (`Rp2` etc.)
  and the hunter-gatherer `H_saved` rebuild; `BchOptions::factorized` does the same (the solver passes
  one `BchOptions` to all its BCH calls). **Assumed meanwhile:** intended (it is what the IT-10
  goldens contain for H); say if operators should be transformed without the correction.
  **Answer (owner, 2026-11-16):** keep as implemented: the 3f2 correction applies to every BCH call,
  the transformation of observables and the hunter-gatherer rebuild included (D49).
* **M10-Q1** (2026-11-15, M10 layout benchmark) - The emax rule (IMSRG(3) only at emax 2 and 3) was
  read as covering operators and kernels, not basis bookkeeping: the layout memory at the production
  target (emax 10-12) comes from `imsrg3_layout_bench --count-only`, which enumerates kets and triples
  only (no operator, no 6j table; 30-50 ms at emax 12). **Assumed meanwhile:** allowed, like the me2j
  value count at emax 14. Alternative: extrapolate the emax-2/3 counts with a fitted power law.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M10-Q2** (2026-11-15, M10 layout benchmark) - The memory gate of the layout decision assumes 4
  nodes x 256 GB with at most 50 % for the 3B operators of an N7 Magnus step (about six operators plus
  scratch). Which E3max/dE3max does the owner use at emax 10-12, and how much node memory? At E3max
  14 layout A needs 38-40 GB per operator (packed), at E3max 16 without a dE3max cut 157-177 GB,
  with dE3max 8 only 4.4 GB (`design/benchmarks/imsrg3-layout.md` section 4). **Assumed meanwhile:**
  the gate above; the layout choice (A) holds for every case, only the node count changes.
  **Answer (owner, 2026-11-16):** "E3max 16, no dE3max is a good starting point, but for the 3-body
  part we want to scale to large numbers of nodes." Node memory: "256 GB or less (sometimes only 100
  GB). Three-body operators will be very heavily distributed." Restated and confirmed ("Correct, and
  keep 50% headroom"): production starts at emax 10-12 with E3max 16 and no dE3max (about 177 GB per
  packed layout-A operator at emax 12, about 1.06 TB for the six operators of an N7 step);
  three-body operators may use at most 50 % of a node's memory, so an N7 step at emax 12 needs at
  least 22 nodes of 100 GB (9 of 256 GB). MPI phase 2 must split single channels across ranks (the
  largest channel, 46,188 kets, holds about 8.5 GB packed per operator, about 51 GB for six), so a
  row-wise, cost-balanced (Hagen-Nam-type) distribution is the baseline and whole-channel ownership
  only the comparison; the N7 kernels are tiled over row ranges of distributed blocks; scaling is
  judged by cost model at 22 to 100+ nodes (D45).
* **M10-Q3** (2026-11-15, M10 layout benchmark) - Layout A is chosen with **packed** channel blocks
  (upper triangle, as imsrg++), which halves the memory; unpacking a channel for GEMM-type terms cost
  4 % of a [2,2]->3 evaluation at emax 3 (0.40 of 9 ms). **Assumed meanwhile:** packed storage, dense
  scratch per channel inside kernels. Alternative: dense blocks (simpler kernels, twice the memory).
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M7-Q10** (2026-11-15, M7 general paths) - The general (NonHermitian) paths are chosen by the
  hermiticity FLAG of X or Y, not by inspecting the data: an operator flagged NonHermitian but
  holding Hermitian data gives the same numbers as the symmetric paths (tested, 4e-16) at higher
  cost (about 2x ladder flops and 4x particle-hole flops; the Pandya scratch grows by at most
  4/3, allocated on first use; the ladders need no extra memory). **Assumed meanwhile:** callers who know the symmetry set the flag (the flow does).
  **Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M7-Q11** (2026-11-15, M7 general paths) - The brief asked to test `commutator_ref` for general
  operators against a dense (m-scheme) brute force if the reference did not support them. It does:
  every term is bilinear, the reference matches imsrg++ goldens and the symmetry-based fast kernels
  for all (anti-)Hermitian pairs, which span every operator, and
  `commutator_ref_properties.general_by_general_bilinearity` shows the reference respects the
  bilinear decomposition for general X and Y. **Assumed meanwhile:** that is sufficient; a true
  m-scheme brute force stays on the M10 list (needed there for 3B anyway, design/30 section 1).
  **Answer (owner, 2026-11-16):** confirmed; resolved since by the m-scheme brute force of M10
  (`commutator_ref::mscheme`), which checks the IMSRG(2) reference terms to 1.2e-15 (low-impact
  numerics batch).
* **M11-Q10** (2026-11-15, M11 python phase 2) - The brief asks for a commutator that accepts general
  (non-Hermitian) operators, but `commutator::Commutator` of this core still throws
  `NotImplementedError` for a `NonHermitian` X or Y (the symmetry tricks of the kernels need Hermitian
  or anti-Hermitian input; `commutator_ref` handles any hermiticity). **Assumed meanwhile:** the binding
  passes everything through and surfaces the core's `NotImplementedError` (tested); it will accept
  general operators as soon as the core does, no binding change needed. **Resolved at the merge
  (orchestrator, 2026-11-15):** the core gained the general paths (`feature/m7-general-commutator`);
  `test_non_hermitian_operands` now checks a general operand end to end (general Z, antisymmetry).
  **Answer (owner, 2026-11-16):** confirmed (resolved at the merge).
* **M11-Q11** (2026-11-15, M11 python phase 2) - D29 says a stale host view raises `StaleHostError`.
  On Serial/OpenMP builds the host mirror aliases the device data, and 23 existing tests (names fixed)
  read views of library results directly. **Assumed meanwhile:** the state machine is always tracked,
  but the default policy `Auto` raises only where the memory is not host accessible (GPU); `Strict`
  (`with imsrg.strict_residency():` or `IMSRG_RESIDENCY=strict`) raises everywhere, so scripts can be
  tested for portability to GPU nodes; `Implicit` never raises. Say if the default should be `Strict`.
  **Answer (owner, 2026-11-16):** `Auto` stays the user default (CPU scripts need no sync calls; GPU
  builds raise as D29 says), but the library and the tests must stay GPU-portable: the 23 tests and
  the examples gain explicit `sync_to_host()` calls, and `tools/ci.sh` runs the Python suite under
  `IMSRG_RESIDENCY=strict` (run-3 task) (D51).
* **M11-Q12** (2026-11-15, M11 python phase 2) - Force-finalize (answer to M11-Q1): `Runtime.finalize(force=True)`
  and the exit hook now free the device data of every live operator and solver (later use raises
  `StaleHostError`), but refuse (all or nothing) while numpy views of an operator are alive, because
  freeing under a view would leave it dangling. **Assumed meanwhile:** at exit a global view then blocks
  finalization as before (Kokkos is left to process exit).
  **Answer (owner, 2026-11-16):** confirmed as assumed (config/Python batch).
* **M11-Q13** (2026-11-15, M11 python phase 2) - Solver observers: the C++ solvers have no stop request
  from a status callback, so `solve(on_status)` cannot stop on a returned `True` (design/18 2.7); an
  exception raised by the callback aborts the solve at a step boundary and is re-raised. Python
  generator callbacks (`GeneratorFn`) are not bound (needs a C++ hook in `flow`). **Assumed meanwhile:**
  exception-based early exit is enough for the owner's group; a stop flag/generator hook would be a
  `flow` API addition.
  **Answer (owner, 2026-11-16):** run-3 filler task: a stop request from status callbacks (returning
  True stops at a step boundary) and a C++ generator hook in `flow` so that Python generator
  callbacks can be bound (D51).
* **M11-Q14** (2026-11-15, M11 python phase 2) - API shape: the bound surface follows the C++ modules
  (`imsrg.io.read_me2j`, `hf.solve(h, v3, mixing=...)`, `HFResult`, `flow.MagnusSolver`/`OdeSolver`,
  `commutators.commutator(terms=...)`, `pipeline.run(config) -> RunResult` over `results.json`) rather
  than the proposal's dataclass tree (`Solver`, `SolverOptions(method=...)`, `ModelSpace.from_reference`),
  which the existing phase-1 names (`ModelSpace(emax, reference, ...)`) already departed from. Names stay
  snake_case, enum members CamelCase as in phase 1. **Assumed meanwhile:** fine; also bound `opslib`
  (`t_rel`, `rp2`, `h_cm`, ...) because a scripted pipeline needs them.
  **Answer (owner, 2026-11-16):** keep the module-shaped API (and the `opslib` bindings); design/18
  is rewritten to describe the bound surface. No binding churn (D51).
* **M10-Q40** (2026-11-15, M10 triples driver) - imsrg++ `CalculatePerturbativeTriples`
  (IMSRGSolver.cc:1186) combines a split Omega stack as `omega = BCH_Product(omega, Omega[n])`, i.e.
  e^omega = e^{Omega_0} e^{Omega_n}, while the transformation of H is e^{Omega_n} ... e^{Omega_0} H
  e^{-Omega_0} ... e^{-Omega_n} (the oldest piece acts first). For a single Omega (no split, as in
  IT-8 He4 emax 2, O16 emax 3) the order is irrelevant; for a split stack the combined omega differs
  from the one that generated H at the commutator level. **Assumed meanwhile:** reproduce imsrg++
  (`flow::PerturbativeTriples` uses `BchProduct(omega, Omega_n)`). Alternative: `BchProduct(Omega_n,
  omega)`, consistent with the transform order; say if wanted (a fix would be an old-code bug entry
  in design/30 section 6).
  **Answer (owner, 2026-11-16):** fix it: combine the stack in transformation order, `omega =
  BchProduct(Omega_n, omega)` (e^omega = e^{Omega_n} ... e^{Omega_0}, the unitary that generated H).
  No change for IT-8 (single Omega); old-code bug entry in `30` section 6 and a split-stack
  consistency test (D49).
* **M10-Q41** (2026-11-15, M10 T5 production comm223) - The production `commutator::Comm223`
  gathers the two-body legs per tile (no global gather tables, as the layout review asked) and is
  1.7x slower than the prototype K1 at O16 emax 3 E3max 9 (15.3 vs 9.0 ms, 18 threads): W sums 6.8 ms
  (about K1's), ket-leg gathers 3.3, bra-leg gathers 1.6, epilogue 1.7, set-up 1.5 ms. The tiles at
  emax 3 are tiny (m <= 12 states, groups <= 6 orbits), so per-leg overhead dominates; at emax 12
  (m ~ 40-80) the gathers should cost about (thirds + cache misses) / m' ~ 10-20 % of W, not
  measured (IMSRG(3) only at emax 2-3). If it matters: one team per block of bra triples sharing
  thirds (ket legs reused; deviates from "one team per bra triple"), or a ket cache keyed by
  (pair, J). **Assumed meanwhile:** one team per bra triple, as briefed.
  **Answer (owner, 2026-11-16):** allowed: timing-only kernel benchmarks (Comm223, comm232 tiles) at
  emax 4-6, with an E3max that keeps one operator under about 8 GB (checked first with
  `--count-only`), never in CI and no full flows; an exception to AGENTS.md rule 3 like D40's emax-8
  IMSRG(2) benchmarks (D52). Team layout (one team per bra triple vs blocks sharing thirds) is
  decided from those measurements in the N7 task.
* **M10-Q42** (2026-11-15, M10 T5) - Locality at emax 12, not measurable today: (1) the dense
  `angmom::SixJ3Table` holds nj^3 nJ^2 nJ3 doubles, 451 MB at emax 12 (360 KB at emax 3), and W
  reads one 6j per j block at stride c2 (2.7 MB at emax 12), a cache line per value; a per-tile
  transposed 6j slice in scratch (or a symmetry-reduced table) would fix both. (2) Bra legs are rows
  of column-major two-body blocks (stride n per element; cached per third w, 4 slots). **Assumed
  meanwhile:** as is; worth a timing-only run at emax 4-6 if the owner allows one.
  **Answer (owner, 2026-11-16):** the locality measurements (per-tile transposed 6j slices or a
  symmetry-reduced table, bra-leg rows) go into the timing-only emax 4-6 runs allowed under M10-Q41
  (D52).
* **M10-Q43** (2026-11-15, M10 T5) - imsrg++ `comm223ss` returns early (Z3 unchanged) when
  ||X2|| ||Y2|| < 1e-6 outside the first `scalar3b_transform` pass (IMSRG3Commutators.cc:5020);
  `Comm223` always computes (documented in three_body.h). Results differ only when that product is
  tiny but nonzero. **Assumed meanwhile:** not reproduced; one line to add if wanted.
  **Answer (owner, 2026-11-16):** confirmed as assumed (low-impact numerics batch).
* **M10-Q44** (2026-11-15, M10 T5) - kernels/comm223.cc uses Kokkos directly (TeamPolicy with
  dynamic schedule, chunk 1, AUTO team size and vector length, level-0 or level-1 scratch, nested
  TeamThreadRange/ThreadVectorRange reductions, a Sum reducer): `exec::TeamFor` has no schedule,
  scratch level, or reducer (design/12 section 2). **Assumed meanwhile:** direct Kokkos inside the
  kernel TU (public headers stay Kokkos-free); add `exec::TeamFor`/`TeamReduce` options when the
  next team kernels ([2,3]->3, [3,3]->2) arrive.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M10-Q45** (2026-11-15, M10 T5) - Additive API beyond the brief: `ThreeBodyContext::
  ScratchBytesPerTeam()` (reported by the benchmark) and `impl()` (the internal tables, a
  `detail::` type for tests and the benchmark). **Assumed meanwhile:** kept; both removable. The
  context also holds `op::PairThirdSpace`'s E map (1.1 MB of its 4.2 MB at O16 emax 3 E3max 9),
  which these kernels never read; PairThirdSpace has no option to skip it.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M10-Q46** (2026-11-15, M10 triples lead) - The perturbative triples use E3max = 3 emax, as
  imsrg++ (`IMSRGSolver.cc:1201-1209`), so without a dE3max cut the triples' kept space, its R map
  and the (unused) E map grow like the full canonical space: harmless at emax <= 3 (He4 emax 2:
  6.5 s for the whole IT-8 run in Debug), but at the owner's emax 10-12 target the tables alone
  would need tens of GB. **Assumed meanwhile:** as imsrg++ (the triples are a post-processing step
  and `imsrg3.de3max` bounds the space). Follow-up if triples at emax >= 8 are wanted: build R per
  tile inside the kernel (it is a few 6j products per entry) and skip E for the triples context.
  **Answer (owner, 2026-11-16):** yes, perturbative triples are wanted at emax 8 and above, in run 3
  alongside the N7 performance work: build R per tile inside the kernel, skip the E map for the
  triples context, and touch only the ppp/hhh blocks the triples need; memory judged by counting
  (emax rule) (D49).
* **M10-Q60** (2026-11-16, M10 step 3) - The three-body generator copies imsrg++ exactly: Delta3 is
  the one-body diagonal only (no Epstein-Nesbet terms) and, unlike the one- and two-body
  denominators, it has NO floor (`Get3bDenominator`, Generator.cc:231-239) and no
  `denominator_delta`; a vanishing Delta3 gives inf/NaN with white. **Assumed meanwhile:** as
  imsrg++ (the 2B floor deviation D21 is not extended to 3B). Floor it if a flow hits it.
  **Answer (owner, 2026-11-16):** floor Delta3 like the one- and two-body denominators: the
  sign-keeping floor with `denominator_cutoff` (D21); no `denominator_delta` in the three-body part.
  No golden changes (IT-9's ppp-hhh Delta3 are tens of MeV); deviation entry in `30` section 6 and a
  unit test with a vanishing Delta3 (D49).
* **M10-Q61** (2026-11-16, M10 step 3) - With IMSRG(3) the final H and transformed operators have
  their residual three-body part dropped before the valence-space/operator output (imsrg++.cc:1177,
  1256 do the same after renormal ordering). Operators are transformed with the IMSRG(3) commutator
  as in imsrg++ (rank-3 result, 3B part dropped), which for a rank-2 operator adds the terms that
  need Omega_3. **Assumed meanwhile:** as imsrg++; not validated against an imsrg++ operator golden
  (IT-9 has none).
  **Answer (owner, 2026-11-16):** keep the imsrg++ semantics (residual 3B part dropped before output
  after renormal ordering; rank-2 operators transformed with the IMSRG(3) commutator, Omega_3 terms
  included) and add an Rp2 operator golden for IT-9 (He4 and O16, emax 2) from the patched reference
  to validate the operator path (D49).
* **M10-Q62** (2026-11-16, M10 step 3) - `imsrg3.occ_nat3_cut` with IMSRG(3), `variant = "full"` and
  `"mp4"`, `at_end`, `only_2b_omega` stay E-PLANNED: the cut needs the NAT occupations in the
  three-body space (`ThreeBodyCuts::natural_occupations`) which the driver does not pass yet.
  **Answer (owner, 2026-11-16):** production needs `imsrg3.occ_nat3_cut`: the driver passes the NAT
  occupations into the three-body space, the cut drops kets at enumeration (D34), and the cut path
  is validated against the m-scheme brute force with the cut. `variant = "full"`/`"mp4"`, `at_end`
  and `only_2b_omega` stay E-PLANNED (D49).
* **M10-Q63** (2026-11-16, M10 step 3 lead) - `comm232ss` processes its pair-third intermediate
  blocks in batches of channels under a 512 MiB budget, but never splits a channel (one block set
  is ~33 GB at emax 12 / E3max 14) and every batch visits every Z2 element (Opus review findings
  2-3). **Assumed meanwhile:** fine at emax <= 3 (60 ms at O16 emax 3); tile by column ranges and
  clamp J3 per batch before production runs (design/14 section 1.1, next steps).
  **Answer (owner, 2026-11-16):** confirmed; resolved by the comm232 follow-up 47aa8b2 (column tiles
  split channels, tables in `ThreeBodyContext`) (engineering batch).
* **M10-Q64** (2026-11-16, M10 step 3 lead) - `comm232ss` rebuilds its sorted pair-third tables on
  every call (0.2 ms at emax 2, 3 ms at emax 3); they depend only on the model space. **Assumed
  meanwhile:** per call; move them into `ThreeBodyContext` with the next kernel pass.
  **Answer (owner, 2026-11-16):** confirmed; resolved by the comm232 follow-up 47aa8b2 (column tiles
  split channels, tables in `ThreeBodyContext`) (engineering batch).
* **M10-Q65** (2026-11-16, M10 step 3 lead) - The comm232ss kernels (like comm223) call Kokkos
  policies (dynamic range, team) directly instead of the `exec` facade. **Assumed meanwhile:**
  accepted in kernel TUs (design/12 section 5 allows Kokkos there); extend the facade when a second
  backend needs it.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M10-Q66** (2026-11-16, M10 step 3 lead) - A `commutator::Workspace` on a model space with a
  three-body space always builds a `ThreeBodyContext` (6j table, pair-third maps), also when only
  IMSRG(2) terms are used on it. **Assumed meanwhile:** fine (no lazy caches, D10; cheap at emax
  <= 3); an explicit `Workspace` option could skip it if an IMSRG(2) flow on a three-body space
  appears.
  **Answer (owner, 2026-11-16):** confirmed; the explicit lazy-context option goes into the run-3 N7
  performance task (engineering batch).
* **M10-Q67** (2026-11-16, M10 step 3 lead; Python, not N7) - `tests/python/test_flow.py::
  test_solve_releases_the_gil` failed on `tools/ci.sh dev-serial` (ticks 7 for a 45 ms solve; it
  passes on dev-openmp): the emax-1 Magnus solve is short on the Serial backend and part of it runs
  with the GIL held (setup, and since the M11 hardening the signal poll at each step). **Assumed
  meanwhile:** the bound now scales with the measured solve time (`ticks > max(2, min(20,
  elapsed_ms / 10))`, still failing if the GIL is held throughout); the M11 owner may prefer a
  longer solve or a measurement of the GIL-free fraction.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).
* **M8-Q50** (2026-11-15, M8 charge radius / two-stage ODE task) - (1) The charge radius needs `Rso`; if it was
  not requested the driver adds it as one more operator that flows alongside H (ODE methods) or is replayed
  with the Omega stack (Magnus). For the adaptive `flow` method the error norm of the step control includes
  every flowing operator, so the step sequence differs from imsrg++'s (which flowed a single Rp2 that
  contained the spin-orbit term): the end results agree (O16 emax 3: E0 within 4e-8, radius within 5e-8 fm) but
  the flow file of the driver-level adaptive tests is no longer line-by-line comparable (`He4Emax2AdaptiveDriver`
  and `O16Emax3AdaptiveDriver` compare final quantities; the solver-level test still compares the trajectory).
  **Assumed meanwhile:** fine. Alternative: let `OdeSolver` exclude "passive" operators from the error norm (a
  `flow/` change). (2) imsrg++'s adaptive `flow` restarts s at 0 in stage 2 (M8-Q40), integrating [0, smax]
  again even if stage 1 ran to smax; with a continuing s our stage 2 would be empty then (He4 p-shell, stage 1
  reaches smax = 200). **Assumed/implemented:** for `imsrg.method = "flow"` stage 2 gets `smax_2 = s_after_stage_1
  + smax` (the equation is autonomous). `flow_rk4` keeps imsrg++'s behaviour (continuing s, same smax). Matches
  the golden `IT-15/He4_emax2_pshell_flow` (E0 3.5e-9, `.snt` 4e-9).
  **Answer (owner, 2026-11-16):** keep every flowing operator in the error norm of the adaptive
  `flow` method, as imsrg++ (the ODE tolerance also controls the accuracy of the observables). Item
  (2) is superseded by D47: stage 2 integrates [s1, s1 + smax] for every method (D47).
* **M8-Q51** (2026-11-15, M8 charge radius) - The charge radius is produced only for the plain `Rp2` operator,
  not for `Rp2Z_<Z>`/`Rp2AZ_<A>_<Z>` (those would need their own Z, A and an Rso with that Z) and not for
  `run.task = "hf"` (no operators are evaluated there). `Rch` is `null` when `Rch2 < 0`; imsrg++'s HF printout
  returns 0 for `|Rp2| < 1e-6` (not copied). N/Z uses the mass number of the Rp2 operator (`basis.mass_number`),
  imsrg++ uses the target mass of the model space (the same number unless `A` is overridden). **Assumed meanwhile:**
  as stated; trivial to extend.
  **Answer (owner, 2026-11-16):** confirmed as assumed.
* **M8-Q52** (2026-11-15, M8 charge radius) - The pristine imsrg++ prints a charge radius only for non-Magnus
  methods ("HF charge radius" and "IMSRG charge radius", `imsrg++.cc:845,1367`), so the IT-6 (Magnus) golden has no
  imsrg++ radius to compare with. The validation uses the pristine build on `flow_RK4`/`flow` runs (He4 emax 2,
  O16 emax 3: agreement 2e-9 fm HF, within the 6 printed decimals after the flow); the Magnus radius uses the same
  code path with the replayed operators. **Assumed meanwhile:** sufficient; no Magnus-path radius number from
  imsrg++ exists.
  **Answer (owner, 2026-11-16):** confirmed: sufficient.
* **M11-Q15** (2026-11-16, M11 python hardening) - Ctrl-C: the solvers poll Python signals at every status
  line (a step boundary) and raise `KeyboardInterrupt`; the pipeline driver (`driver::Run`) has no
  cancellation hook, so `pipeline.run` lets Ctrl-C stop the process (SIGINT default action) while the run
  is in progress instead of ignoring it. **Assumed meanwhile:** fine for scripts; a cancel flag in
  `driver::Run` would allow a clean `KeyboardInterrupt`.
  **Answer (owner, 2026-11-16):** run-3 filler task: a cancel flag in `driver::Run`, so that
  `pipeline.run` raises `KeyboardInterrupt` cleanly at a step boundary (D51).
* **M11-Q16** (2026-11-16, M11 python hardening) - `SymmetryError` and `ConvergenceError` exist as classes
  (design/18): `ConvergenceError` is raised by `pipeline.run` on divergence and by
  `hf.solve(raise_on_failure=True)` (default False, so existing results stay reportable); `SymmetryError` is
  reserved until tensor operators exist. A missing file raises `InputFileNotFoundError` (an `InputError`
  and a `FileNotFoundError`), malformed content `FileFormatError`, operands on different model spaces
  `ModelSpaceError`. **Assumed meanwhile:** as described.
  **Answer (owner, 2026-11-16):** confirmed as assumed (config/Python batch).

* **M12-Q1** (2026-11-15, M12a get_pn) - M6-Q4 is resolved for the orbit scan and the CGs
  (dense (n, l, 2j) table, tabulated isospin CGs, bitwise identical). In a release build `get_pn`
  on calls passing the cheap filters drops from 183 to 65 ns; the remainder is the six
  `channel.states.find` hash lookups in `get_isospin`. **Assumed:** left as is (no layout change
  was allowed); a dense per-channel state table would remove it if HF-with-3N profiles still show it.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).

* **M12-Q2** (2026-11-15, M12a norms) - `TwoBodyNorm` now relies on `BlockSparseMatrix` blocks
  being contiguous and in storage order (IMSRG_CHECKed). **Assumed:** that stays the layout. Per call
  the kernel allocates two small device buffers (offsets, weights); at 1 thread the single
  reduction is about as fast as 84 small ones (0.73 vs 0.68 ms at emax 6), the gain is at > 1
  thread (1.73 -> 0.17 ms at 6 threads). Caching the buffers on the operator would remove the
  allocation if it matters.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).

* **M12-Q3** (2026-11-15, M12a ladder) - `tests/python/test_flow.py::test_solve_releases_the_gil`
  fails in dev-serial already at 7f92c7e (before the ladder change; repeatably, 5-6 ticks vs the
  required 20): the O16 solve finishes in about 5 ms, shorter than the test's 20 ms tick window, so
  the test is wall-time fragile and gets worse as the commutator gets faster. **Assumed:** not
  loosened here; the test needs a longer solve (e.g. larger smax or a bigger fixture) or a
  deterministic hook (event set inside a callback) instead of counting ticks.
  **Resolved by the M12a lead:** the test now solves to smax 20 (35 ms serial) and requires at least
  a quarter of the ticks the measured solve time allows (and at least 5); a GIL-holding solve gives
  about one. Same check, independent of the solver speed. (A time-stamp variant was rejected: the
  GIL hand-off at a callback's entry also produces stamps inside the windows.)
  **Revised (2026-11-16):** tick counting still failed in dev-serial at load ~38 (another lead's
  slow tests; the main thread woke only every ~14 ms). The test now passes the builtin
  `list.append` as the status callback and has the main thread poll the number of calls: it must
  see a count strictly between 0 and the final one. A builtin runs no bytecode, so a GIL-holding
  solve could not hand the GIL over between its first and last callback (negative control: a
  GIL-holding C loop calling `list.append` 3e6 times gives no intermediate count; a GIL-releasing
  loop gives ~47 of 50). The check needs a single main-thread wake-up during the solve (smax 40,
  88 steps): 25/25 passes at normal load and 25/25 with 40-64 busy processes.
  **Answer (owner, 2026-11-16):** confirmed (resolved by the list.append test).

* **M12-Q4** (2026-11-16, M12a ladder-direct) - Direct channels (contiguous listed kets, all weights
  exactly 1.0) skip the gather and the gemm reads X and Y in place with leading dimension N. The
  hh projector of a closed shell is direct only for channels where the hole kets happen to be
  contiguous (e.g. a single listed ket); the check is generic. Assumption: exact `== 1.0` on the
  weight, so a channel with a fractionally occupied orbit in a listed ket never takes the direct path
  (fractional references can still have direct channels whose pp kets use only empty orbits, which is
  exact; review note). Benchmark checksums (emax 2, 4;
  T 1, 4) are bitwise identical to before; the ladder scratch of O16 shrank from 3.75x to 2.0x of
  a two-body operator at emax 6. No open decision.
  **Answer (owner, 2026-11-16):** confirmed as assumed (engineering batch).

* **M12-Q5** (2026-11-16, M12a lead) - `linalg::GroupedGemm` runs its problems concurrently on the
  host threads only with Accelerate (one BLAS call per item; measured faster at emax 8 and
  schedule-independent). With OpenBLAS (pthreads) or MKL a call inside the parallel region would use
  the library's own threads (oversubscription) and MKL's rounding may depend on its thread count.
  **Assumed meanwhile:** other BLAS libraries keep the sequential loop; enabling them needs
  single-threaded calls per item (MKL `mkl_set_num_threads_local(1)`, OpenBLAS built with OpenMP or
  `OPENBLAS_NUM_THREADS=1`) and a determinism check, to be decided with the cluster builds (M12b/M14).
  **Answer (owner, 2026-11-16):** deferred to the cluster builds (M12b/M14): concurrent
  `GroupedGemm` with OpenBLAS/MKL needs single-threaded calls per item and a determinism check;
  until then they keep the sequential loop.

## Owner interview 2026-11-16

Every entry above now has an answer (the low-impact ones confirmed in batches). The interview also
settled the decisions for run 3 and later: D44–D54 in `00-requirements-and-decisions.md`, report
`design/reports/2026-11-16-owner-interview.md`. Answers given outside the question ids:

* **Run-3 order and design topics** (multiple choice; D50, D52, D53): the structural style pass
  first in run 3, alone and time-boxed; then N7 performance with distribution-ready row tiling,
  triples at emax ≥ 8 and `occ_nat3_cut`, in parallel with M12b phase 1; M12b phase 2; tensor
  operators (E2 and other electric multipoles, M1, Gamow–Teller/Fermi, 0νββ; Magnus transformation
  only); M12a follow-ups and the Python hooks as filler; M13 last. 3B distribution accepted by cost
  model at ≤ 10 % imbalance with communication below compute at ~25 GB/s per node; clusters with
  InfiniBand + Slurm and Slingshot + Slurm (Cray). GPU: CUDA and HIP builds, tuning on NVIDIA first.
  IMSRG(2) single-commutator benchmarks at emax 10–12 allowed. Process and report format kept. The
  full N2LO 3N file is not available yet (D33 unchanged).
* **Extension after tensor operators:** me2jp/me3jp (not checkpoint/restart as proposed). Verbatim:
  "I provide sample code in a second." and "Wait until I have provided the me2jp and me3jp files and
  you have worked through the spec. THen write it up."
* **ME2JP/ME3JP specification** (the owner sent `jIMSRG-develop/jimsrg/files/formats.zip`; worked out
  in `19-mexj-formats.md`, D54). Verbatim: "The ME2JP and ME3JP formats are implemented in this
  directory. Check that the ME2JP is a straight forward extension of ME2J from NLJJ to NLJJT. ME3JP is
  analogous but for full three-body angular momentum coupling." Normalization: unnormalized, as
  ME2J/ME3J (confirmed). Scope: "Readers and writers. Keep in mind as a todo that there is also a
  binary version of these files where the header is stored as a fixed set of characters. I am
  figuring out the exact specification here. For now, prepare for a binary implementation
  (me2jp.bin, me3jp.bin)." File variants: "See previous response." Contents: "The evolved H by
  default, other operators optionally. me2jp and me3jp will be most useful for checkpointing the code
  and storing the Magnus operators for efficient evaluation of a large amount of extra operators."
  On the restated summary: "Correct except that gzip should be implemented for me2jp and me3jp as
  well. This goes after tensor ops."
* **ME2JP getter J range (follow-up):** after the explanation of the jIMSRG reader's `std::max` upper
  J bound (`19-mexj-formats.md` §5), verbatim: "Your understanding is correct. Make absolutely sure
  this is implemented properly in the new ME2JP and ME3JP implementations." Recorded as D55: the
  rules and mandatory tests of `19` §8.
* **Run-3 usage schedule:** verbatim: "Modify the next session prompt to run conservatively (not
  consuming more than 50% of usage) until 8pm and then running as normal until 7am." Recorded as D56
  (at most 50 % of each 5-hour usage window until 20:00 JST, normal pacing until the 07:00 JST
  wrap-up) in `NEXT_SESSION_PROMPT.md`.

## Added during run 3

Run 3 appends its questions below (ids per brief, e.g. `M10-Q70..Q79`, `M12b-Q1..Q9`, `MS2-Q1..Q9`;
the orchestrator uses `O4` and up).

* **IT3-Q4** (2026-11-16, run 3 brief A part 2, D51) - Strict residency in CI. `tools/ci.sh` now runs
  `tests/python` with `IMSRG_RESIDENCY=strict`. Only four tests needed changes (the 23 of M11-Q11 had
  been fixed in phase 2): `test_non_hermitian_operands` and `test_numpy_view_round_trip` gained
  `sync_to_host()`; the two policy tests (`test_default_policy_matches_the_build`,
  `test_implicit_policy_syncs_by_itself`) set the policy they test inside a `residency_policy_scope`
  (Auto stays tested), and a new test checks the `IMSRG_RESIDENCY` values in subprocesses.
  **Assumed:** nothing else; `examples/python/he4_pipeline.py` already syncs and its test passes under
  strict. A GPU build still needs a real GPU run to confirm portability.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **IT3-Q5** (2026-11-16, run 3 brief A part 2, D50) - Physics-symbol list. `.clang-tidy` ignore
  patterns for variables, parameters, members and local constants are generated from
  `tools/physics_symbols.txt` (159 symbols) by `tools/gen_physics_symbols.py`; names pass if their
  `_`-parts are lower-case words or listed symbols. **Assumed:** the one-letter-prefixed forms (`iJ`,
  `nA`, `tJ`, `dZ`) and labelled matrices (`Hh`, `Fk`, `Ct`, `EB`, `Lam`, `PL`, `Tde`) count as physics
  notation; 18 non-physics locals were renamed (separate commit); 12 accessors missing from the
  method allowlist were added so that `tools/lint.sh` is clean (it was not on `develop`). Say if the
  prefixed forms should be rejected (the list would shrink by about 20 names and those locals be renamed).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **IT3-Q6** (2026-11-16, run 3 brief A part 2, D51) - `design/18` rewritten to the bound API (modules
  `imsrg`, `basis` types in `imsrg`, `operators`, `io`, `hf`, `mbpt`, `commutators`, `flow`,
  `pipeline`). **Assumed:** the old proposal's unbound parts (dataclass `SolverOptions(method=...)`,
  `ModelSpace.from_reference`, term-by-term `terms.*`, `angmom`, stubs, callbacks of generators) are
  listed once in a "not implemented" section instead of kept as design text; the old old-`pyIMSRG`
  mapping table was dropped (it lives in git history). Say if the table should be kept for the
  term-by-term milestone.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **IT3-Q7** (2026-11-16, run 3 brief A part 3, D49 / M10-Q60) - Delta3 floor. `Denominators::three_body`
  now uses `FloorDenominator` (sign-keeping, an exact zero gives +cutoff, as the 1B/2B code) with
  `denominator_cutoff`; no `denominator_delta`. **Assumed:** the cutoff is the same value for 1B, 2B
  and 3B; a separate 3B cutoff can be added if wanted. IT-9 unchanged.
  **Answer (owner, 2026-11-26):** One cutoff for 1B/2B/3B, no separate 3B key.
* **IT3-Q8** (2026-11-16, run 3 brief A part 3, D49 / M10-Q40) - Split-stack order. Added the public
  helper `flow::CombineOmegaStack` (triples.h) so the order is unit-testable; it uses
  `BchProduct(Omega_n, omega)`. **Assumed:** Magnus `Transform` applies Omega_0 first (checked in
  magnus.cc), so this is the transformation order. The `BchProduct` itself keeps the imsrg++ truncation
  (no -1/24 term), so the combined omega is correct to the BCH truncation only.
* **IT3-Q9** (2026-11-16, run 3 brief A part 3) - No IT golden has a split Omega stack in the
  perturbative triples (IT-8 has split Omega stacks; He4 emax 2 has 3 Omegas), so the fix is validated by the BCH-level consistency
  test only, not by an end-to-end imsrg++ comparison (imsrg++ gives the old, inconsistent order).
  Say if a split-flow end-to-end triples test (split vs single Omega energy) is wanted.
  **Correction to the owner's premise "no change for IT-8 (single Omega)":** IT-8's Magnus flows
  have split Omega stacks, so the fix shifts the triples dE by -3.9e-4 MeV (He4 emax 2) and
  -2.0e-3 MeV (O16 emax 3). Applied per D49 (old-code bug fix, reversible); no compatibility switch
  (D35). **Done:** the D35 patch gained the same fix in both `CalculatePerturbativeTriples` overloads
  (stamp `3b0ff5eed6806a72`), and the IT-8 goldens were regenerated through the scripts (D42): dE He4
  emax 2 -0.05209710 -> -0.05249001, O16 emax 3 -0.53867396 -> -0.54068122 MeV; `driver_triples`
  compares against them at the old tolerances.
  **Answer (owner, 2026-11-26):** Yes, add an end-to-end test: the He4 emax 2 triples energy with a split Omega stack vs a single Omega must agree to the BCH truncation level (expected ~1e-5 MeV) (run-4 filler).
* **IT3-Q1** (2026-11-16, run 3, brief A part 1) - The D35 patch of imsrg-devel now carries the D47 and D48 hunks
  in the same file (version stamps `0c7af4dd0112f621`, after the review fix `a908abab2eded439`); goldens whose
  numbers did not change keep their old stamps on purpose, to avoid churn: `cb9d11ba30f579d4` (IT-1..IT-7, IT-14,
  hamiltonian, hf, ...) and `5fdd2b73265417f2` (old IT-15, imsrg3, n7; IT-8/He4_emax2 and IT-8/emax3 now carry `3b0ff5eed6806a72`, IT3-Q9). All are listed in
  `tests/golden/README.md` as equivalent. **Assumed meanwhile:** fine. Alternative: regenerate every golden once so
  that all carry the newest stamp.
  **Answer (owner, 2026-11-26):** Accepted (old stamps kept).
* **IT3-Q2** (2026-11-16, run 3, D48; extended after the review) - After the final gather `MagnusSolver` replaces H(s)
  and now also recomputes eta(s) from it (the patched imsrg++ leaves Eta untouched, which no output shows). The
  flow does NOT continue: measured ||eta(rebuilt H)|| = 1.8e-3 (He4 emax 2, omax 0.1), 4.8e-3 (O16 emax 4, omax
  0.25), 2.5e-3 (O16 emax 4, omax 0.1), i.e. 2000-5000 times eta_criterion = 1e-6, where the loop had stopped on
  the hunter-step H (9.6e-7); E0 of the rebuilt H is 8-43 keV away from the last hunter-step line. A warning is
  logged, `norm_eta` of the stage and `eta()` describe the rebuilt H, and the flow log gets one extra row for it
  (istep = steps + 1, same s). **Owner to decide** whether the flow should instead continue until the rebuilt H
  converges (changes step counts and the goldens).
  **Answer (owner, 2026-11-26):** The flow stops as now (warning, eta/norm_eta of the rebuilt H, one extra row); the owner chose this over continuing until the rebuilt H converges.
* **IT3-Q3** (2026-11-16, run 3, D47) - The adaptive `flow` method of the patched imsrg++ still restarts s at 0 in
  stage 2 (it integrates a copy of the solver), so for `flow` the reference flow file ends at s = smax while ours
  ends at s1 + smax; the range LENGTH is the same, only the final quantities are compared
  (`FlowMethods.StageTwoRangeAdaptiveSmax10`). Also: `tools/ci.sh` started from a background job of a
  non-interactive shell fails `test_ctrl_c_*` of `tests/python/test_threading.py` (background jobs ignore SIGINT);
  run it in the foreground. **Assumed meanwhile:** no change.
  **Answer (owner, 2026-11-26):** Fine as is.
* **O4** (2026-11-16, orchestrator, binary ME2JP/ME3JP, D57) - The binary variant stores 4-byte
  floats after the 255-byte header (A. Tichai's reader, imsrg++ `.bin`), so a `.bin` file keeps
  about 7 digits and does not round-trip our doubles; D54 wants checkpoints and stored Omegas to
  restart bit for bit (`19` §3: `%.17g`). **Assumed meanwhile:** `.bin` (float32) is for
  interactions and other inputs; checkpoints and stored Omegas are written as text or gzip.
  Alternative: an additional double-precision binary variant for checkpoints (for example a
  `.bin64` suffix or a header token `prec=64`, which jIMSRG would not read). Say which.
  **Answer (owner, 2026-11-26):** "The float32 binary is sufficient for the checkpointing. The difference between float32 and float64 is numerically much smaller than other method uncertainties in our calculations." (verbatim; float32 is accepted for checkpoints). Follow-up confirmed: bit-exact restart (D54) is dropped; checkpoints and stored Omegas may be float32 `.bin`, a restart must match the uninterrupted run within float32 precision (~1e-7 relative, E0 ~1e-6 MeV, step counts may differ by a few); text/gzip (%.17g) stays available and gives bit-exact restarts when wanted (tests). D54 amended.
* **IT3-Q10** (2026-11-16, run 3 follow-ups) - `ode_tolerance` has no separately testable double behaviour (it only enters the
  dopri5 step-size controller), so `OdeFlow.EtaCriterionIsADouble` covers `eta_criterion` for rk4 and dopri5 only.
  **Assumed:** enough; D47-2 states ode_tolerance as double by construction.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **IT3-Q11** (2026-11-16, run 3 follow-ups) - The IT-8/emax2 golden carries stamp `cb9d11ba30f579d4` (patched build), so the
  `tests/golden/README.md` "pristine" list no longer names it. **Assumed:** the divergent runs of IT-8/emax2 need no separate entry.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **IT3-Q12** (2026-11-16, run 3 follow-ups) - The Ctrl-C children restore the default SIGINT handler (a `&` job of a
  non-interactive shell starts them with SIGINT ignored). **Assumed:** this matches a real terminal Ctrl-C, which is what the tests mean to cover.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **MS2-Q1** (2026-11-16, run 3, MS2 item 1, D50) - Structs with member functions: 12 became classes
  (Python wrappers `PyModelSpace`, `PyOperator`, `PyMagnus`, `PyOde`; the driver helpers
  `LineWriter`, `KeyCollector`, `SchemaLists`; the angmom views `SixJView`, `NineJLSjjView`,
  `MoshinskyView`, `SixJ3View`, still trivially copyable; the bench's `TriplesBody`). **Kept as
  structs:** passive aggregates whose functions are comparisons or derived values (`*Config`,
  `Orbit`, `Nucleus`, `OperatorSpec`, `Ket`, ...), pimpl `Impl` structs (the owning class holds the
  invariant), kernel functors and data-less guards, and - deliberately - the device views and kernel
  data bundles that are aggregate-initialized at many sites or filled field by field by one builder
  (`linalg::MatrixView`, `commutator::detail::TwoBodyView`, `basis::TwoBodyTableView`,
  `KetSubsetTableView`, `OrbitGroupsView`, `PandyaData`, `LadderData`, `LiftedData`,
  `Comm232Tables`): converting them touches ~150 sites, most in kernels that the N7 row tiling (brief
  C) and M12b are about to restructure. **Assumed:** they stay structs for now; alternative: convert
  them inside the N7/M12b work (private data, a builder that is a static member).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **MS2-Q2** (2026-11-16, run 3, MS2 item 2) - Unsigned -> signed: the 14 scoped enums with
  `std::uint8_t` underlying types now use `std::int8_t` (no arithmetic on them; done so that unsigned
  types appear only where the rule allows them), the NO2B monopole orbit indices are `std::int16_t`.
  **Kept unsigned:** bit patterns (`base/half.h`, `TermMask`, m-scheme masks, hash keys,
  `bit_cast` checks), RNG seeds and LCG/SplitMix state (modular arithmetic), zlib and `isspace`
  interfaces, file sizes, `size_t` sizes, the `DeviceBuffer<std::uint8_t/uint16_t>` instantiations
  and the `unsigned __int128` factorial of `moshinsky_test.cc` (needs the full 128-bit range).
  **Assumed:** enums count as "not a bit pattern"; say if unsigned underlying types should be allowed
  again (a one-line revert per enum).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **MS2-Q3** (2026-11-16, run 3, MS2 baseline) - The perturbative-triples energy is not bitwise
  reproducible from run to run at a fixed thread count: IT-8 He4 emax 2 gave dE =
  -0.052490010312560093 and ...086 in two runs of the same binary (1-2 ulp), so its reduction order
  is nondeterministic (all other integration outputs were bitwise reproducible). The MS2 comparison
  checks it to 13 digits. **Assumed:** harmless (far below every tolerance); the N7 lead may want a
  deterministic reduction when the triples are tiled (D45 distribution).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **MS2-Q4** (2026-11-16, run 3, MS2 baseline) - Two sources of cross-build noise in test output:
  (a) 53 parameterized tests (`basis_reference.Setups/PartitionGolden...`) embed the raw bytes of
  their parameter object, including heap pointers, in the test name, so `ctest -N` changes with every
  rebuild (a `PrintTo` or a name generator would fix it); (b) the generated emax-4 goldens
  (`tools/reference/gen_golden_generated.sh`) differ in the last bits between generations
  (`HfHnoReference.Emax4TwoPlusThreeNucleon` prints max|dHNO| 1.4e-13 in one worktree, 8.5e-14 in
  another), i.e. the reference harness's emax-4 HF/HNO output is not bitwise reproducible.
  **Assumed:** both harmless; not changed by MS2.
  **Answer (owner, 2026-11-26):** Accepted as assumed.

### M15 reference preparation (run 3, phase 0; tensor operators, D53)

* **M15-Q1** (2026-11-16) - Phase/gauge of the reduced matrix elements. imsrg++'s one-body closed forms (E2, GamowTeller)
  equal `(-1)^{j_bra-j_ket}` times the Edmonds reduced element (`(-1)^{j_bra-j_ket+λ}` for E-multipoles); the
  hermiticity relation `(-1)^{j_a-j_b}` is the same in both. The factor telescopes in `[scalar, tensor]` commutators.
  **Assumed:** the new code may use the Edmonds convention internally only if it converts at the edges (inputs,
  dumps, 6j/9j recoupling formulas of `TensorCommutators.cc`, whose phases were derived for the imsrg++ gauge);
  the safest choice is to keep imsrg++'s gauge. Decision needed before the tensor `OperatorDump` v2.
  **Answer (owner, 2026-11-26):** imsrg++ gauge internally and in dumps; Edmonds values only through an explicit getter/converter at the edges.
* **M15-Q2** (2026-11-16) - Tensor `OperatorDump` layout. The raw format stores only `ch_bra <= ch_ket` blocks and
  lists labelled blocks (`block ...` + `2b p q r s v`). **Assumed:** v2 should store the same minimal set plus the
  rule `<hi||O||lo> = (-1)^{J_lo-J_hi} <lo||O||hi>` for Hermitian, and say how ΔTz = ±1, ±2 are labelled (imsrg++
  stores `O + O†` and calls it Hermitian; `delta_Tz` means `|ΔTz|`). Whether the new code stores ΔTz > 0 operators as
  `O + O†` like imsrg++ or as the one-directional operator is open (it changes the Hermiticity checks).
  **Answer (owner, 2026-11-26):** Store ΔTz > 0 operators as O + O† like imsrg++.
* **M15-Q3** (2026-11-16) - Reducedness of J=0 isospin-changing scalars (Fermi, 0νββ, DGT): imsrg++ builds them
  reduced but makes them not reduced before the HF transformation (`imsrg++.cc:815`) and leaves them so; the
  `Commutator` dispatch treats them as scalars. The dumps are all reduced (`reduced 1`). **Assumed:** the new
  code stores J=0 channel-changing operators in the same convention as J>0 (reduced); the scalar-scalar
  commutator with them (not covered by these goldens) needs a separate reference.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M15-Q4** (2026-11-16) - A one-body Y (E2, M1, GT, Fermi) exercises only `comm111st`, `comm121st`, `comm122st`;
  the 222/221 terms are zero. The term-by-term goldens for them use seeded random tensors at **emax 1** (He4) because
  emax-2 files are about 85 KB per 2B-dense dump (limit 1 MB total). **Assumed:** enough for the first tensor
  milestone; random tensors at emax 2 (He4, O16) and tensor Y with 2B parts from 0νββ-like operators can be added
  as gitignored `generated/` data when needed. No E1/E3/M1 commutator goldens (only E2, GT) and no commutator with
  the J=0, ΔTz = 2 operators.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M15-Q5** (2026-11-16) - 0νββ operators and the Magnus golden. imsrg++ provides `M0nu_{GT,F,T}`, `M0nu_C`,
  `M0nuHeavy_*`, `DGT` (all J=0, ΔTz=2, 2B only; the HO operators depend on the model-space target mass through
  `R0 A^{1/3}`). Only `Ec = 7.72`, `src = none`, and the default form factors are dumped, in HO and HF bases, with
  ΔTz = 2 stored as the pp (bra) - nn (ket) block. The Magnus golden is one BCH step with a random anti-Hermitian
  Omega of norm 0.3, not a flow-produced Omega. **Assumed:** acceptable; a flow-produced Omega
  (`refharness flow` writes none) needs a harness hook. The imsrg++ `BCH_Transform` stops at the default thresholds,
  so the golden contains truncation error of that order.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
### MPI (M12b, brief D) and N7 performance (brief C), run 3

* **M12b-Q1** (2026-11-17, run 3 brief D) - Where the communicator lives: `WorkspaceOptions::comm` (phase 1) and the model space's `ThreeBodyDistribution` (phase 2, adding the edge `basis -> comm`; comm is layer 0). **Assumed:** yes; phase 1 is done with `WorkspaceOptions::comm`; the alternative (`operator -> comm`) would let operators of one model space disagree.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q2** (2026-11-17, run 3 brief D) - Combine by reduce + broadcast (identical bits by construction) instead of `Allreduce`. **Assumed:** yes, implemented as `ReduceBcastSum`; `Allreduce` only after a measured gain and a bitwise check.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q3** (2026-11-17, run 3 brief D) - Row blocks of big 2B/cc channels (whole channels fail beyond ~8 ranks). **Assumed:** the ladder is split by a contiguous cost-balanced cut of the global row sequence (done); the particle-hole term goes by whole cc channels for now (imbalance 5/151/903 % at 22/64/256 ranks at emax 12); ph row panels are a follow-up.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q4** (2026-11-17, run 3 brief D) - Cheap terms (110/220/111/121) on rank 0; 3f2 correction replicated in phase 1. **Assumed:** yes, implemented.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q5** (2026-11-17, run 3 brief D) - N7 on rank 0 with Z3 broadcast until the line tiles land. **Assumed:** yes, implemented.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q6** (2026-11-17, run 3 brief D) - Every rank reads the interaction files. **Assumed:** yes; root read + broadcast only if I/O contention shows up on a cluster.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q7** (2026-11-17, run 3 brief D) - Output and info logs only on rank 0; warnings from every rank with a rank prefix. **Assumed:** as implemented, `run.cc` calls `Logger::set_rank(comm.rank())` without `all_ranks`, so only rank 0 prints; non-root ranks print nothing (their warnings are lost). Only rank 0 writes files. Rank-prefixed warnings from all ranks need `all_ranks` and are not enabled.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q8** (2026-11-17, run 3 brief D) - Distributed 3B parts are packed only (dense/non-Hermitian -> NotImplementedError). **Assumed:** yes (phase 2).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q9** (2026-11-17, run 3 brief D) - Lockstep `Alltoallv` rounds instead of one-sided `MPI_Get`. **Assumed:** rounds (deterministic, portable to InfiniBand and Slingshot); RMA as a later optimization (phase 2).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q10** (2026-11-17, run 3 brief D) - Partition weights from the cost model (formulas) rather than runtime timing. **Assumed:** formulas (ladder row cost 2 N_c (k_pp + k_hh); channel costs for LPT), calibrated later. Update 02:40: the joint ladder cut uses measured term weights w_122 = 59 and w_ph = 2.9 ladder units (one-rank release bench at emax 8, load 12-50, about 15 %); recalibrate on an idle machine.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q11** (2026-11-17, run 3 brief D) - One rank per node with OpenMP threads. **Assumed:** yes; one rank per GPU in M14. (Measured: on one node more ranks at fixed ranks x threads are slower, see Q17.)
  **Answer (owner, 2026-11-26):** Accepted as assumed. Placement answer (owner, verbatim): "IMSRG(2) commutators will not scale to that many nodes. Only go up to 8 nodes and test both 1 and 8 ranks/node." (cap at 8 nodes; test 1 and 8 ranks per node).
* **M12b-Q12** (2026-11-17, run 3 brief D) - Device-aware MPI. **Assumed:** M14; host staging now.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q13** (2026-11-17, run 3 brief D) - Python bindings stay serial. **Assumed:** yes.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q14** (2026-11-17, run 3 brief D) - Config key `[parallel] three_body_distribution` ("rows" default with more than one rank, "replicated" = step A, for tests). **Assumed:** phase 2; not in phase 1, which is always replicated.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q15** (2026-11-17, run 3 brief D) - Rank-invariance tolerance (IT-11). **Assumed:** max|d| <= 1e-12 max(1, max|ref|) on E0, norms and `.snt` files (the `30` section 2 form, not element-wise relative), step counts identical; a differing step count is investigated, never accepted.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q16** (2026-11-17, run 3 brief D) - `results.json` records no rank or thread count. **Assumed:** fine for now.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q17** (2026-11-17, run 3 brief D) - One-node scaling: at fixed ranks x threads = 16, emax 8 full commutator 30.5 ms (1x16), 43.5 (2x8), 61.8 (4x4); combine 3.7 ms (2x8) and 12.3 ms (4x4), see `design/benchmarks/mpi-scaling.md`; ph term 15.0/21.9/31.0 ms. **Assumed:** MPI only across nodes; no warning for several ranks on one node.
  **Answer (owner, 2026-11-26):** Accepted as assumed. Placement answer (owner, verbatim): "IMSRG(2) commutators will not scale to that many nodes. Only go up to 8 nodes and test both 1 and 8 ranks/node." (cap at 8 nodes; test 1 and 8 ranks per node).
* **M12b-Q18** (2026-11-17, run 3 brief D) - The ph inverse transform is only partly distributed: every rank loops over all pair x J x J' (only the zbar reads are masked); with the pair-skip and J'-trim in distributed mode the cost drops but is not split. **Assumed:** acceptable; ph row panels with a per-(J', panel) inverse mask later (plan review note m6).
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M12b-Q19** (2026-11-17, run 3 brief D) - `CheckReplicatedSetup` requires equal thread counts on all ranks, so heterogeneous nodes fail. **Assumed:** keep; bitwise identity across ranks needs the same reduction order.
  **Answer (owner, 2026-11-26):** Accepted as assumed.
* **M10-Q75** (2026-11-17, C12 triples) - `TriplesResult::kept_rows` / `total_kets` after the move of the perturbative
  triples to the reduced `commutator::TriplesContext` (M10-Q46). The full three-body space is no longer built, so the
  old meanings (all kept rows of the space; all channel kets before the cuts) cannot be counted cheaply. Nothing
  compares them (only the driver log line "3-body states kept: X out of Y"). **Assumed:** `kept_rows` = kets of the
  triples the context holds (hhh/ppp triples of the channels with an hhh triple, after the dE3max/OccNat3 cuts),
  `total_kets` = the same triples' kets before those cuts. Both are documented in `flow/triples.h`; the log text
  in `driver/run.cc` (not touched, another worker) still reads "out of", which stays correct. Also: the leg/group
  table builders of `ThreeBodyContext::Impl` (three_body.cc) are duplicated in `commutator/triples.cc` because
  three_body.cc was outside this task's files; a later cleanup can factor them into one internal header.
  **Answer (owner, 2026-11-26):** Accepted; semantics kept, fix the "out of" log text.
* **M10-Q70..Q74, Q76..Q79** (2026-11-17, brief C, N7 performance; proposals of the plan
  `new-imsrg-reference/plans/m10-n7-perf-plan.md` section 8, taken as proposed unless noted):
  * **Q70** Row = today's stored segment z(m >= k, k); the packed storage, dumps and Python views are unchanged.
  * **Q71** Line attribution (a batch owns whole matrix rows; transpose parts read through the views) for every
    term; the zipper `RowPartition::Balanced` balances compute and memory together.
  * **Q72** 133, 331, 231, 132 rewritten on the pair-third intermediates H and W; the element-wise kernels stay as
    `detail::Comm<term>ssReference` (tests; dense-Z path of 133). Measured deviations <= 3e-14 in the M3 form.
  * **Q73** One fused batch loop (`detail::RunN7`) in `Commutator()`; the `terms::` functions stay single-term and
    rebuild H/W per call.
  * **Q74** Batch budget `detail::kN7BatchBytes` = 512 MiB; it bounds H + W of one batch (allocation max(H) +
    max(W) over the batches, at most twice the budget, plus TransposeOut and the 133 slot tables).
    `CommutatorOptions::imsrg3_batch_bytes` is **not** added yet: passing it to `ThreeBodyTerms()` changes a
    signature in `commutator.cc` that the M12b lead edits at the same time. **Assumed:** add the option after the
    M12b merge (one-line plumbing).
  * **Q76** OccNat3Cut occupations: the MBPT(2) natural occupations (the NAT step's values for `basis.kind = "nat"`,
    computed in an HF-basis run too, as imsrg++.cc:570), for the IMSRG(3) flow and the perturbative triples.
  * **Q77** Facade (C4) as proposed: `exec::TeamOptions` (schedule, team/vector size, scratch level with
    kAuto = level 0 if the bytes fit, else level 1), `TeamFor`, `TeamReduce`; the comm223/triples launches use
    them; nested `TeamThreadRange`/`ThreadVectorRange`/`single` stay Kokkos inside kernel TUs.
  * **Q78** `WorkspaceOptions::three_body_context` (kAuto/kSkip) plus an explicit
    `Workspace::BuildThreeBodyContext()`; the exception type for a missing context stays `InputError`.
  * **Q79** Additions per output part keep imsrg++'s order (Z3 133 -> 223, Z2 232 -> 132, Z1 331 -> 231). The fused
    driver equals the per-term sequence bitwise when a commutator fits one batch (all emax <= 2 cases) and differs
    at 8e-17 in Z1/Z2 at emax 3 (two batches); IT-9 step counts and E0 are unchanged at emax 2 and 3.
  **Answer (owner, 2026-11-26):** Accepted. Q74: add `CommutatorOptions::imsrg3_batch_bytes` in the N7 follow-up. Q75: semantics kept, fix the "out of" log text.

### Orchestrator, end of run 3

* **O5** (2026-11-17, orchestrator, N7 row-cost model) - `commutator::kRowOverheadMadd = 1.3e5`, the
  per-row term of `N7LineWeights`, was fitted on this machine at O16 emax 5 with 18 threads in
  `n7_bench`'s simulated single-process partition, on a loaded machine (Opus review of
  `feature/n7-row-cost-model`). beta/alpha may depend on the thread count and the problem size (at
  emax 12 a row averages 1.9e7 multiply-adds, so the term is about 0.7 % of a row's weight), and the
  fit exists only as prose. **Assumed meanwhile:** the constant stays a documented machine number;
  no production code uses it yet (M12b's `ThreeBodyDistribution` takes a ready-made partition).
  Run 4: decide whether it becomes a runtime option; refit at another emax and thread count with a
  committed fit script; validate `n7_bench --row-overhead` (silently ignored without `--partition`
  or with `--partition-contiguous`) and add it to the usage text; P = 64 still has max/mean 1.16.

## Added during the Frontier port

Supervised session 2026-11-26 (OLCF Frontier, nph123). Proposals P-F* need an owner decision
(D-rows) in the interview after the port; until then the session works as assumed.

* **P-F1** (2026-11-26, isolation scheme) - **Proposal: scheme (a)**, Claude Code's own isolation:
  `isolation: worktree` in `imsrg-lead`, `imsrg-implementer`, `imsrg-mechanic`, with the committed
  `.claude/settings.json` hooks `WorktreeCreate` (clone the hub into `$IMSRG_WORK/agents/<name>` on
  `feature/<name>`; workers of a lead start from the lead's pushed branch), `WorktreeRemove`
  (`tools/agent_retire.sh`) and the `PreToolUse` guard `tools/hooks/agent_guard.py`; read-only
  agents in the integrator's clone. Tested and working (design/41 §3); fallback (b),
  `tools/agent_clone.sh` plus `cd <clone> &&` briefs, exists but was not needed. **Assumed
  meanwhile:** (a) is in use. Open points: Claude Code never calls `WorktreeRemove` when a
  subagent finishes, so clones accumulate until `tools/agent_retire.sh --all --merged`; the hub
  keeps every pushed feature branch, including the test branches `feature/agent-aed5510638e6f6d22`,
  `feature/agent-ab7ad3f93c28d0a1e` and `feature/selftest` of this session: delete merged or test
  branches on the hub (who, when), or keep them all?
  **Answer (owner, 2026-11-26):** Scheme (a) is adopted (becomes a D-row). The integrator deletes hub feature branches that are fully merged into develop, plus the test branches `feature/agent-aed5510638e6f6d22`, `feature/agent-ab7ad3f93c28d0a1e`, `feature/selftest`, at the end of each run or interview (through an integrator-only script, a run-4 task, since raw `git push origin --delete` stays denied); unmerged branches stay. The integrator retires merged clones (`tools/agent_retire.sh --all --merged`).
* **P-F2** (2026-11-26, GitHub policy) - **Proposal:** only the integrator talks to GitHub
  (`github` remote of `$IMSRG_WORK/main`); GitHub changes come in with `--ff-only` merges;
  `develop` is pushed to the hub first and to GitHub only after `tools/ci.sh frontier-cpu` passed
  on it; never force-push, never delete or rename a remote branch, never rewrite published
  history; `main` is left alone; feature branches stay on the hub. Agents never touch GitHub (their
  clones have no GitHub remote; `agent_guard.py` denies pushes elsewhere than `origin`).
  **Assumed meanwhile:** as proposed. Question: should `main` follow `develop` at milestones
  (fast-forward by the owner), and should feature branches be mirrored to GitHub for review?
  **Answer (owner, 2026-11-26):** Adopted as proposed. `main` is fast-forwarded to `develop` by the owner at milestones; agents and the integrator never touch it. Feature branches are not mirrored to GitHub.
* **FP-Q1** (2026-11-26, settings) - `design/claude-settings.proposed.json` now proposes allow rules
  for hub pushes (`git push origin *`), `git fetch`, Slurm and module commands, and denies force
  pushes and remote branch deletion. **Assumed meanwhile:** the current
  `.claude/settings.local.json` (Mac era) stays; the owner installs the proposal if wanted.
  **Answer (owner, 2026-11-26):** Installed during the interview (2026-11-26) as `.claude/settings.local.json`; the Mac-era file is backed up in `$IMSRG_WORK/trash/settings.local.json.mac-2026-11-26`.
* **FP-Q2** (2026-11-26, git identity) - the integrator's clone had no git identity, so commits
  made by agents in clones were attributed to `mheinz@login10...`. **Assumed meanwhile:** set
  `user.name`/`user.email` locally in `$IMSRG_WORK/main` to the identity of the existing history;
  `tools/agent_clone.sh` copies it into each clone. A global `~/.gitconfig` would also work.
  **Answer (owner, 2026-11-26):** Local identity in the integrator's clone, copied into clones, as now.
* **FP-Q3** (2026-11-26, compute allocations) - builds and tests need an allocation; the session
  keeps one 1-node `batch` job named `imsrg` (2 h limit) and queues the next with
  `--dependency=afterany`. **Assumed meanwhile:** fine for supervised sessions; for long
  unattended runs, is a chain of 2-hour jobs acceptable for the nph123 allocation, or should runs
  use the `extended` partition / a reservation?
  **Answer (owner, 2026-11-26):** "For the budget, until the end of the weekend knoxville time, you can use up to 50,000 node hours. Do what you need to with those resources but plan such that they are sufficient for the whole plan. Standard partitions are appropriate." (verbatim; the budget is 50,000 node hours until the end of the weekend, Knoxville time, standard partitions, planned to last for the whole plan). Also (owner, verbatim): "one node per lead is good. Request what is needed for the computations." (one node per lead, requesting what the computations need).
* **FP-Q4** (2026-11-26, generated goldens) - the emax-4 dumps are shared by every clone in
  `$IMSRG_WORK/golden-generated` (copied from the integrator's `build/golden-generated`, which
  differs file by file from `$IMSRG_REFERENCE_ROOT/golden-generated-d35`; which one is newer is
  not recorded). **Assumed meanwhile:** the presets point at `$IMSRG_WORK/golden-generated`.
  Which of the two sets is current, and should the other be removed?
  **Answer (owner, 2026-11-26):** `$IMSRG_WORK/golden-generated` is the current set; `$IMSRG_REFERENCE_ROOT/golden-generated-d35` is obsolete (to be moved to `$IMSRG_WORK/trash`).
* **FP-Q5** (2026-11-26, adaptive flow vs IT-14) - `FlowMethods.He4Emax2AdaptiveMatchesImsrgFlowFile`
  failed on Frontier with max |Δs| = 1.27e-5 > 1e-5 (E0, |H|, eta within 3e-9). Cause: the test's
  TOML used the DIIS HF default, imsrg++ uses linear mixing. DIIS stops after 8 iterations at an H0
  that differs from imsrg++'s at the HF tolerance (|H| by 2.5e-8 at s = 0) and moves with round-off
  (an `-O2 -ffp-contract=fast` build changed e1 by 1.2e-6 and max |Δs| to 5.1e-5); the adaptive
  step control amplifies that in the converged tail (s > 80, ||eta|| ~ eta_criterion), where s
  carries no physics. With `mixing = "linear"` (as the other imsrg++ comparisons, D37): HF 26
  iterations and e1 equal to imsrg++, every printed s equal to imsrg++ run on Frontier (0 of 572
  rows differ), E0/|H|/eta within 8e-10/2.7e-9/5.4e-10, FMA and non-FMA builds within 5e-9 in s;
  vs the Mac golden max |Δs| = 7.6e-6, the same 25 rows whose printed s differ between imsrg++ on
  the Mac and on Frontier (imsrg++'s own platform spread). Tolerance unchanged. **Open:** the
  driver-level flow tests (and O16 under `slow`) still run DIIS and keep 1e-5 with less margin;
  should every imsrg++ flow comparison use linear mixing, and is DIIS's round-off sensitivity at
  the 1e-8 HF tolerance acceptable for production (it changes nothing physical)?
  **Answer (owner, 2026-11-26):** Every imsrg++ flow golden comparison (also driver-level and O16 slow) uses `mixing = "linear"`, AND the default DIIS HF convergence tolerance is tightened to about 1e-10 so production results stop moving with the build; run 4 records old vs new numbers of any of our own DIIS-based expectations that move.
* **FP-Q6** (2026-11-26, BLAS reentrancy) - `test_two_threads_on_distinct_solvers_and_operators`
  segfaulted on Frontier in `dgemm_kernel_loop_mnk_a1b1_mt_naples` (`libsci_gnu_mp.so.6`, the
  OpenMP cray-libsci that `CC -fopenmp` links). gdb: one Python thread in libsci `dgemm` with its
  OpenMP team, the other in a Kokkos `parallel_for` (Comm121ss); `OMP_NUM_THREADS=1` passes. A
  20-line C++ program without Kokkos or imsrg code (two `std::thread`s calling `dgemm_` on their
  own matrices, n = 100 or 400) crashes with `libsci_gnu_mp` at 4 and at 1 OpenMP thread; it runs
  clean when the calls are under a mutex, when one thread runs plain `omp parallel for` loops
  instead, and with the serial `libsci_gnu`. `CRAYBLAS_ALLOC_TYPE=2` and
  `CRAYBLAS_FORCE_HEAP_ALLOC=1` do not help; `man intro_libsci` says nothing about reentrancy.
  So Kokkos OpenMP entered from two host threads is fine; the threaded libsci is not reentrant.
  **Assumed meanwhile:** `linalg` holds a process-wide mutex around every BLAS/LAPACK call
  (`IMSRG_BLAS_SERIALIZE`, default ON except Accelerate); single-threaded hosts pay one uncontended
  lock per call, and threaded BLAS stays available for large gemms. The alternative, linking the
  serial `libsci_gnu` (reentrant, would also allow concurrent `GroupedGemm` items, M12-Q5), loses
  BLAS threading for the large ladder gemms; decide with the Frontier benchmarks.
  **Answer (owner, 2026-11-26):** Keep the BLAS lock (`IMSRG_BLAS_SERIALIZE`); revisit after the Frontier benchmarks.
* **FP-Q7** (2026-11-26, frontier-serial) - with the debug Serial Kokkos (`Kokkos_ENABLE_DEBUG`,
  bounds checks) GCC 14 at `-O1` reports `-Wmaybe-uninitialized` in `Kokkos_View.hpp:1143` (View
  copy, `from`/`to`), inlined into our units, so the `-isystem` treatment of the installed Kokkos
  does not hide it (21 diagnostics; none with the OpenMP Kokkos or Apple clang). **Assumed
  meanwhile:** the `frontier-serial` preset adds `-Wno-error=maybe-uninitialized` (warning still
  printed, not fatal); no other preset changes. Alternative: report upstream / build kokkos-serial
  without `Kokkos_ENABLE_DEBUG`.
  Also (frontier-serial, 2026-11-26): `driver_imsrg3_n7.Imsrg3N7.He4Emax2OccNat3Cut` exceeds its
  300 s timeout with the single-threaded debug Serial Kokkos (it takes ~113 s under frontier-cpu);
  every other unit and integration test of the preset passes up to that point. **Assumed
  meanwhile:** `frontier-serial` is not the merge gate (frontier-cpu is); raise the test's timeout
  or label it `slow` for serial builds? Owner to decide.
  **Answer (owner, 2026-11-26):** Label `Imsrg3N7.He4Emax2OccNat3Cut` `slow` under serial builds; the maybe-uninitialized warning stays non-fatal in frontier-serial; frontier-serial is not a merge gate.

## Added during the Al27 night (2026-11-26)

* **AL-C1** (lead C, C5 `[imsrg.space]`) - The two-body channels have no e2max cut, so
  `imsrg.space.e2max < 2*imsrg.space.emax` is E-PLANNED (the 27Al runs use e2max = 2 emax = 20).
  `e3max_imsrg`/`emax_3body_imsrg` are accepted and unused for IMSRG(2). **Assumed:** enough for the
  task; an e2max cut belongs to `basis::TwoBodyChannels` later.
* **AL-C2** (lead C, C7) - A restart re-runs HF/NAT (deterministic) instead of storing H_0; the flow
  file of the restarted run starts at the restored step (not appended); the config hash covers every
  key except `[output]`, `[checkpoint]` and `run.log_level`, so adding operators between kill and
  restart is refused. **Assumed:** acceptable (HF at emax 14 is minutes against hours of flow); say if
  H_0 should be stored or operators excluded from the hash.
* **AL-C3** (lead C, C4) - The fork re-sorts NAT orbitals with `std::sort` on |occupation| (not
  stable); ties keep the signed order here (`std::stable_sort`). Exact ties do not occur in practice.
  The C4 comparison against lead B's fork reference dumps is pending (only a test-side
  re-implementation of the fork loops so far).
* **AL-C4** (lead C, C8) - The fork's `Parameters.hh` default is `domega = 0.2` (imsrg-devel 0.1);
  the archive's template does not set it. With 0.1 our run of lead B's fork golden case
  (`tests/golden/nreft/flow/fork-79e9330/nat_emax4_imsrg3`, branch feature/al27-nreft-operators)
  differs by 5e-4 MeV in the .snt; with 0.2 H agrees to the printed precision (2B max 4.9e-7, 1B max
  2.0e-6 on values ~10, zero body -339.425981 vs -339.426; TCM 1.4e-7). **Done:** compat option
  `fork_step_defaults` sets domega = 0.2 unless given. The fork also has the keys
  `only_2b_omega_at_end`, `reduced_232_impl`, `imsrg3_commutator_depth` (in the template); the legacy
  layer now accepts them at their defaults (dropped) and rejects other values.
* **AL-C5** (lead C, ME2JP io) - (a) design/19 §2's closed index formula
  `e(e+1)+o_l+o_j+[proton]` is wrong (d3/2 lands at 10 instead of 8); the implementation uses the
  file order `2*nlj + [proton]` (the doc's "equivalently" sentence). (b) Text ME2JP round trips are
  bit exact except for entries with exactly one identical-orbit pair (x*sqrt2 then /sqrt2, within
  2 ulp); native files are the exact format. (c) A `.bin` header without a tag is accepted. The
  `asan` preset was not run (Mac-only preset). **Assumed:** fine; fix the formula in the doc.
### Lead A (M15 tensor operators, capability C3)

* **AL-A-Q1** (2026-11-26) - Seeded random operators depend on the C++ standard library. The Frontier
  harness (libstdc++) gives other `random_A` / `rand_*` values than the Mac (libc++), so every tensor
  golden built from them changed when `gen_golden_tensor.sh` was rerun on Frontier (deterministic ones
  agree to <= 1.5e-14). **Assumed:** the tensor goldens were regenerated on Frontier and made
  self-contained (their random inputs are stored); the scalar `commutator/emax2_O16/random_A.dump`
  (Mac) stays as it is. Other golden scripts that build random operators will show the same effect
  when rerun on Frontier.
* **AL-A-Q2** (2026-11-26) - Memory of the tensor particle-hole term. The rewrite keeps Zbar for all
  cross-coupled channel pairs over ALL ordered pairs (rows and columns), about 4x imsrg++'s Zbar
  (which uses c3 <= c4 and unordered rows with phase bookkeeping). Fine up to emax 6; estimated
  ~10-20 GB at emax 10 with rank 5. The 9j table (one dense box per (A, B, D, E)) is 35 MB at emax 6
  and ~1.1 GB at emax 10, built serially with Wigner9j (minutes at emax 10). **Assumed:** acceptable
  for the first milestone; before the farm (C10/C11): use the Zbar symmetries (c3 <= c4, unordered
  rows), tile the channel pairs, and cache/parallelize the 9j table (or evaluate 9j from 6j tables on
  the device).
* **AL-A-Q3** (2026-11-26) - imsrg-devel's fix of the tensor normal ordering (`ME *= 2` for a == b,
  J_bra != J_ket, Operator.cc:555-558) is right only for Hermitian tensors; for an anti-Hermitian
  tensor the two orientations cancel (and devel's mirror `O(b,a) = herm * O(a,b)` flips the diagonal
  sign on every update). **Assumed:** the rewrite sums both orientations explicitly (correct for both
  hermiticities); the fork's bug is the option `TensorNormalOrderOptions::fork79e9330_diagonal_bug`.
  All 27Al operators are Hermitian, so this only matters for anti-Hermitian tensors.
* **AL-A-Q4** (2026-11-26) - `[scalar, tensor]` with a one-body Y: imsrg++ gives Z Y's particle rank,
  the rewrite gives Z max(rank X, rank Y) (so comm122st is kept for a one-body Y). BCH promotes Y to
  rank 2 first in both codes, so transformed operators agree. **Assumed:** keep the rewrite's rule.
* **AL-A-Q5** (2026-11-26) - Storage convention of reduced elements (follow-up to M15-Q1). The m-scheme
  oracle (`commutator_ref::mscheme::ExpandTensor`) reproduces every emax-1 tensor golden (all terms, four
  hermiticity pairs, 1.7e-15) only with PLAIN Edmonds reduction, <j'm'|T_mu|jm> = <j m L mu|j' m'> R /
  sqrt(2j'+1), for one-body elements and J-coupled pairs alike; extra phases (-1)^(j_bra-j_ket),
  (-1)^(J_bra-J_ket) or (-1)^L deviate by O(1). So imsrg++'s storage IS Edmonds; the "gauge" of
  design/15 is a property of the closed forms of its operator builders (E2, GT), i.e. of which physical
  operator a name denotes. **Assumed:** the rewrite stores what imsrg++ stores (owner answer to M15-Q1
  holds unchanged); the NREFT builders (lead B) must reproduce the fork's closed forms, and an Edmonds
  converter at the edges is then the identity on storage. Worth a line in design/15.
* **AL-A-Q6** (2026-11-26, reviewer finding) - imsrg++'s tensor one-body norm uses the integer
  degeneracy `(2j_p+1) * ((min(2j_q, 2j_p + L) - max(-2j_q, 2j_p - L)) / 2 + 1)`, which mixes 2j with
  the undoubled rank and can be NEGATIVE (2j_p = 21, 2j_q = 11, L = 5 gives -22; first possible at
  emax 10 with rank 5, i.e. in the 27Al Sigma'/Sigma''/Delta J=5 operators). A negative weighted sum
  makes the norm NaN, and imsrg++'s BCH then never meets `||nested|| < epsilon` and runs to 40 nested
  commutators, while the rewrite's BCH stops at the first non-finite norm. **Assumed:** the rewrite
  keeps imsrg++'s formula (it drives the BCH thresholds the archive was produced with); whether the
  fork runs hit the NaN case is to be checked when the J=5 operators are reproduced (C8). A
  physically sensible norm ((2j_p+1) per element, or (2j_p+1)(2j_q+1)/(2L+1)) can be a named option.
### Lead B (NREFT library, C1/C2)

* **AL-B1** (lead B, FPhitp normalization) - `FPhitp<L><p|n>_<q>` (L = 2, 4) uses sqrt(4 pi) x
  `Phitp(L, q)`, Hermitian, like FPhipp (reference/06 section 2.4; orchestrator's notes
  `$IMSRG_WORK/inputs/al27/analysis/phitp-normalization-notes.md`). The constant lives only in
  `opslib::nreft::PhitpShorthandPrefactor()`. **Assumed meanwhile:** sqrt(4 pi); owner to confirm.
  **Answer (owner, 2026-11-26, via the orchestrator):** "The prefactor is good for now." (design/50 section 4).
* **AL-B2** (lead B, NREFT table tolerance) - the fork's one-body tables are reproduced element by
  element to 1e-12 relative, except near-zero elements: tables that vanish analytically (Delta'_J at
  emax 2) or elements that are 1e-12 of the table at q = 1050 MeV carry only the rounding noise of
  the alternating radial sums, in the fork as here (the fork's own FM0p_15 <0||10> and <10||0> differ
  by 6e-14 relative). **Assumed meanwhile:** pass if |d| <= 1e-12 |ref| or |d| <= 1e-14 S_q, S_q =
  the largest element of all fork tables at that q (the Wigner-symbol rule of design/30 section 2;
  observed max |d|/S_q = 7.8e-15). Alternative: per-element error estimates from the magnitude of the
  summed terms.
  Update (later the same night): at emax 14 the fork itself is not exact (design/30 section 6,
  row AL-B-1: up to 8.5e-7 relative, 1.5e-9 S_q). The port now sums in double-double and is
  checked against a 50-digit mpmath reference (1e-12 rule, max 5.5e-13); the emax-14 fork
  comparison only bounds the fork's error (1e-8 S_q).
* **AL-B3** (lead B, fork NaNs at q = 0) - in the fork, `jdmho`/`jdpho` with multipole L = 0 contain
  y^(-1/2) and give inf/NaN at q = 0; this reaches Phi'_1, Phi''_1, tilde-Phi'_1, Delta'_1 and
  Mg(J, 0) at q = 0 (none of the archived shorthands; FPhitp2/4 are finite). The port reproduces the
  fork (NaN), and the golden test skips non-finite fork values. **Assumed meanwhile:** keep NaN (only
  reachable through raw blocks, not through any catalog name). Alternative: the analytic q -> 0 limit.
* **AL-B4** (lead B, rank L > 0 NREFT operators) - the tensor storage of lead A
  (`feature/m15-tensor-operators`) is not on develop yet; the L > 0 shorthands are planned catalog rows
  and their tables are plain `OneBodyTable`s. **Assumed meanwhile:** wire them as tensor operators
  (reduced one-body part, imsrg++ gauge) on a separate branch on top of A's branch, merged after it.
* **AL-F1** (lead F ops, odd-rank NREFT tensors vs the fork goldens) - Against `tests/golden/nreft/flow/
  fork-79e9330/{nat_emax2,nat_emax4_imsrg3}` (driver_al27_fork_golden) H, TCM, the scalar and the EVEN-rank
  tensors (FM2p, FPhipp2n) agree to 1e-10 (printed precision), including the fork's tensor normal-ordering
  bug (needed: without it 1e-2 off). The ODD-rank tensors (FSigmap1p/3n, FSigmapp1n, FDelta1p) deviate by up
  to 1.6e-4 (emax 2) / 5.7e-4 (emax 4) in the final `.snt`, growing with the flow time; the first Magnus step
  (only Omega x one-body Y) is exact, so the cause is in terms with a two-body Y. Source comparison fork
  Commutator.cc vs devel TensorCommutators.cc: the fork's `comm222_phst` lacks (-1)^L at (a) the flipped
  ket orientation of Ybar (`flipphaseY` in PhaseMatYJ1J2, fork 12696 vs devel phase(Jbra+Jket+Lambda)) and
  (b) the reads of Zbar through swapped pairs (fork 12399, 12404 vs devel phase(Lambda+...)). Both are
  emulated (compat `tensor_fork_ph_missing_rank_phase`, odd L only); this reduced the deviation 189 -> 25
  rows at s = 0.1 and 3.6e-4 -> 5.6e-5 at the end, but not to zero. comm121st (fork form checked
  numerically), comm122st, comm222_pp_hh_221st and the normal ordering were compared and are equivalent.
  **Resolved (alf1, 2026-11-26):** the remaining cause was the fork's doubled Zbar columns: it stores
  each cross-coupled ket pair twice (k <= j, and flipped k >= j with the Ybar part missing (-1)^L), and
  the inverse transform reads the flipped half also for a DIAGONAL pair (`k>j?0:nkets` with k == j,
  Commutator.cc:12399/12404/12472/12477), so for odd L the Xbar*Ybar part of those elements flips sign.
  The emulation negated only k > j columns; comm222_phst now also keeps Xbar*YA in fork mode and reads
  such elements as Zbar - 2 Xbar*YA. Per-term fork goldens (`tests/golden/tensor/fork-79e9330`,
  `tools/reference/gen_golden_tensor_fork.sh`, forkharness commutator-tensor) agree to 1e-13 for ranks
  1, 3 (both parities) and 2; driver_al27_fork_golden passes with the strict tolerance (odd-rank max dev
  1.6e-4/5.7e-4 -> <= 1e-10, the printed precision). A fork bug, emulated only under the compat option
  (design/30 section 6, row AL-F1); the default path stays imsrg-devel's (which our default matches for
  ranks 1 and 2, commutator_tensor_golden).
* **AL-F2** (lead F, C10 file names) - The transform farm writes the fork's names `<stem><name>.snt` /
  `<stem><name>_2b.snt` (with `output.stem = "int"` exactly the archive's), the in-process driver keeps
  `<stem>_<name>.snt` / `<stem>_<name>_2b.snt`. **Assumed:** keep both (no change for existing runs);
  say if the in-process names should follow the fork under `run.compat`.
* **AL-F3** (lead F, C10 cost model) - The static assignment weighs an item by num_omegas x the
  block-gemm work of one [Omega, O] for its rank L (design/50 §6.1), assuming the same number of BCH
  terms for every operator of a stack. Not yet timed at emax 10 (one item per rank makes it irrelevant
  for the planned 2415-rank jobs; it matters only with fewer ranks than items). **Assumed:** fine until a
  timing of one item at emax 10 exists; refit then.
* **AL-F4** (lead F, C10 farm resubmission) - A killed farm job leaves `failed.txt` listing every item
  (written first, rewritten at the end), and outputs are renamed into place; resubmission uses either
  the failure manifest as the work list or the same list with `transform.skip_existing = true`.
  **Assumed:** enough; a per-item progress log is not written.

### Lead E (GPU, M14/C11/C12)

* **AL-E-Q1** (lead E, M14/C11) - The Kokkos Kernels backend of the device BLAS
  (`IMSRG_DEVICE_BLAS=KOKKOSKERNELS`) is written but neither built nor measured: the 5.3.0 source
  tarball cannot be downloaded from an agent session (the worktree hook rejects commands with the
  GitHub URL), compute nodes have no internet, and the Frontier `kokkos-kernels` modules (3.x-4.6)
  do not match Kokkos 5.3. **Assumed:** rocBLAS stays the hip default until someone puts
  `kokkos-kernels-5.3.0.tar.gz` into `$IMSRG_DEPS_PREFIX/src/` and runs
  `tools/bootstrap_deps.sh --backends hip`; the GEMMs are only 11-14 % of the GPU time, so the
  choice matters little (design/benchmarks/gpu-m14.md, proposed D-row there).
* **AL-E-Q2** (lead E, M14) - `flow_generator3_kernel` Bitwise* (5 tests) fail on the GCD: the
  atan generator kinds use device `Kokkos::atan` against host `std::atan` in the reference loop
  (22-281 of 74497 elements differ, max relative 4e-16; not FMA, `-ffp-contract=off` is set for
  frontier-hip). **Assumed:** left failing (no tolerance loosened); options: compare with a few-ulp
  tolerance on device backends only, or evaluate the atan on the host into a table. Owner to choose.
* **AL-E-Q3** (lead E, M14, follows FP-Q6) - With the default Frontier CPU BLAS (threaded
  `libsci_gnu_mp` behind the `IMSRG_BLAS_SERIALIZE` lock, sequential GroupedGemm) the scalar
  commutator is 6-7x slower than with the serial `libsci_gnu`, `IMSRG_BLAS_SERIALIZE=OFF` and the
  new `IMSRG_BLAS_CONCURRENT=ON` (emax 8, 56 cores: 430 vs 65 ms). **Assumed:** presets unchanged
  (the owner kept the lock, FP-Q6); proposal: switch the `frontier-cpu*` presets to the serial
  libsci with concurrent items after the Python two-thread test passes with it.
* **AL-E-Q4** (lead E, M14/C12) - Distributing one commutator over GCDs (M12b phase 1) is much
  slower than one GCD (emax 10: 78 ms on 1 GCD, 454 / 656 ms on 2 / 8 GCDs): the combine copies Z
  to the host and reduces with host MPI every commutator, and the owned-column paths keep their
  host shapes. **Assumed:** flows run one rank per GCD (8 independent flows per node); GPU-aware
  MPI and device shapes for the distributed paths are later work.
* **AL-E-Q5** (lead E, M14) - The HIP Kokkos build (`kokkos-hip`) has a Serial host space, so
  every host-parallel part (`HostParallelForDynamic`, host loops of the generator and normal
  ordering, the tensor table builds) runs on one core in the GPU build: the IMSRG(3) N7 driver
  test takes ~155 s, CPU-bound, on a GCD (113 s on frontier-cpu). **Assumed:** unchanged (the
  IMSRG(2) commutators run on the device); adding `Kokkos_ENABLE_OPENMP` to kokkos-hip (HIP + OpenMP
  host) would give each rank its 7 cores. Measured with such a Kokkos (built in the agent clone,
build/kokkos-hip-omp, OMP_NUM_THREADS=7): O16 emax 8 Magnus flow on one GCD 28.0 -> 23.9 s;
commutators and the tensor first call unchanged. Owner/integrator to decide with the C12 layout.
* **AL-D1** (lead D, C7 at production size) - A restart re-runs the whole preparation (me2j and
  48 GB NO2B read, HF, NAT, NO2B contraction): ~16 min of every 2-hour job at emax 14 (13 %), and
  the 30-minute checkpoint interval loses up to 30 min more per job boundary (the N2LOsat run lost
  21 stage-2 steps). Restarts are bit-identical across nodes (checked at stage 2, steps 79/80).
  **Assumed:** acceptable tonight; for the farm and longer chains, store H_0 (emax 10, ~0.3 GB)
  and the NAT C in the checkpoint so a restart skips the readers and HF (answers AL-C2), and write a
  checkpoint on SIGTERM / before the job's end (Slurm `--signal`).
* **AL-D2** (lead D, NAT) - Our MBPT(2) density matrix dropped the PP elements between a particle
  and a partially filled hole (imsrg++ tests 1 - n_b); fixed (commit "Include partially filled
  orbits ..."). This was a bug of the port, not of imsrg++, so it is not in design/30 §6. AL-C3's
  pending comparison against fork dumps: done indirectly, emax 6 and 14 EMP2/EMP3/E_Nat agree with
  the fork build to all printed digits.
* **AL-D3** (lead D, writer) - Under `compat = "imsrg_fork_79e9330"` the driver still writes the
  imsrg-devel tokyo format: relative two-body threshold 1e-7 x norm2b drops 15 of the 158 archived
  TCM two-body elements (|v| <= 1.07e-3; the 143 written agree to 5e-9). H is unaffected (all 158
  written, max deviation 5e-7). **Assumed:** lead B's `TokyoNumberFormat::kFork79e9330`
  (feature/al27-tokyo-fork-format) gets selected by the compat profile when that branch merges
  (one line in driver `TokyoOptions`); not done here to keep the branches disjoint.

### Answers (owner interview part 2, 2026-11-27; D72-D78)

The owner chose the recommended option unless a verbatim answer is given. Asked whether to confirm
all remaining assumptions in one go, the owner answered (verbatim): "Interview me on all questions."

* **AL-A-Q6:** emulate imsrg++'s tensor one-body norm under `run.compat` only (variant 1); default
  (2j_p+1)(2j_q+1)/(2L+1) per element; Phase 0 checks which archived J=5 logs hit the NaN (D72).
* **AL-NAT (new, owner, verbatim):** "To be clear NAT orbitals should be sorted by |occupation| not
  signed occupation. Verify that this is what is done in v2 and v3." Checked: it was not (default
  `mbpt::NatSorting::kSignedDescending`, |occupation| only under compat). To do in Phase 0:
  |occupation| becomes the default for every run (D72, design/30 §6 row AL-NAT).
* **GPU BLAS (D70 follow-up):** rocBLAS stays the hip default (D73).
* **AL-E-Q1:** Kokkos Kernels gets a compile check in `tools/ci.sh frontier-hip` (later; D73).
* **AL-E-Q2:** 4-ulp relative tolerance in the five Bitwise tests on device builds only (later).
* **AL-E-Q3:** switch the `frontier-cpu*` presets to serial libsci + concurrent items after the
  Python two-thread test passes (D74; after the weekend).
* **AL-E-Q4:** one flow per GCD; GPU-aware distribution later (M12b).
* **AL-E-Q5:** kokkos-hip with OpenMP host space (Phase 0; D73).
* **Fork tensor goldens (1.7 MB):** keep.
* **AL-C1:** e2max cut later, in `basis::TwoBodyChannels`.
* **AL-C2 / AL-D1:** store H_0 and the NAT coefficients in checkpoints; checkpoint on SIGTERM;
  operators stay in the config hash (Phase 0, before the v2/v3 flows; D75).
* **AL-C3:** keep `std::stable_sort`.
* **AL-C4:** variants 2/3 use the imsrg-devel default domega 0.1; 0.2 only under compat.
* **AL-C5:** design/19 formula fixed (done 2026-11-27); 2-ulp text round trips and tag-less headers
  accepted.
* **AL-A-Q1:** every golden script that builds random operators stores its inputs (later).
* **AL-A-Q2:** memory is enough for the farm; unordered rows and tiling later.
* **AL-A-Q3:** explicit sum of both orientations (design/30 §6 row M15-1).
* **AL-A-Q4:** keep the rewrite's particle-rank rule.
* **AL-A-Q5:** storage is plain Edmonds; the gauge belongs to the builders (design/15, confirmed).
* **AL-B2:** keep the rule with the mpmath reference (design/30 §2).
* **AL-B3:** the default path returns the analytic q → 0 limit, compat keeps the fork's NaN (later).
* **AL-B4, AL-D2:** confirmed.
* **AL-D3:** confirmed; the TCM `.snt` of both v1 flows is re-checked in Phase 0 (158 archived
  two-body elements to 1e-4).
* **AL-F2:** fork file names under `run.compat` in the in-process driver and in the farm; devel names
  otherwise (Phase 0).
* **AL-F3:** refit the cost model after the Phase-0 timing.
* **AL-F4:** add a per-item farm log (name, start, end, seconds, BCH terms, Omega norm) (Phase 0).
* **Difference reports (owner, verbatim):** "When you compare the differences between v1 v2 and v3,
  note both absolute and relative differences (ignoring matrix elements in the .snt that are very
  small, smaller than 1e-6)." (D76)
* **Scale (owner, verbatim):** "Not a must this weekend. Run each variant as its own ~600-1,200-node
  job; document how 19,320 ranks would be reached. Use the GPU resources efficiently with what you
  have available." (D77)
* "later" = after the weekend: the owner chose "27Al only, no fillers" for the weekend run (D77).

## Added during the 27Al weekend

* **WE-A-1** (lead A, Brief A) - `tools/ci.sh frontier-hip-run` stops after the unit step, which
  always fails on the five known atan Bitwise tests (AL-E-Q2, 4-ulp fix decided for "later"), so
  the integration step never runs there. **Done meanwhile:** the integration tests were run by hand
  on a GCD (`ctest -L integration -LE 'slow|mpi' -j 3`, `OMP_NUM_THREADS=2`): 87/87 pass with the
  OpenMP host space. The script is unchanged (the AL-E-Q2 fix removes the need).
* **WE-A-2** (lead A, farm sizing) - At the measured emax-10 GPU cost a whole farm of one flow
  (2414 NREFT items + TCM) is ~42 GCD-hours (~5.3 node-hours, design/benchmarks/al27-farm.md
  section 4); the longest item takes ~2 min (rank 5, q = 150 MeV; TCM ~13 min, host-bound). A
  600-1,200-node job (4,800-9,600 GCDs) has more GCDs than items, so most idle and the job ends
  after the longest item. **Assumed:** recommend ~38-76 nodes per farm job with the refit cost model
  (longest first, 1-2 items per GCD-minute balance), or 302 nodes for one item per GCD; the
  orchestrator decides (D77 says "use the GPU resources efficiently").
* **WE-A-3** (lead A, TCM in the farm) - TCM is the farm's longest item (~790 s on 7 host threads,
  preparation at basis.emax 14) and is already written by every flow (same Omegas, same
  `BchTransform`). **Assumed:** keep it in the default work list (the cost model now puts it alone
  on a rank); `make_work_list.py --no-tcm` drops it.
* **WE-A-4** (lead A, checkpoint hash) - `FlowConfigHash` hashes `ToToml` of the whole configuration
  (minus output and checkpoint), so any new schema key (also `transform.dump_native` of this
  branch, which never affects a flow) changes the hash of every configuration: a checkpoint written
  by an older binary cannot be restarted by a newer one (seen: the v1 N2LOsat checkpoint refused by
  the GPU build, job 6006593). **Assumed:** unchanged here (Brief B owns the checkpoint code);
  proposal: hash only the flow-relevant tables, or keys whose value differs from the default, and
  pin the binary for a flow's whole lifetime meanwhile (flows_gpu.sh copies it per run).
  **Resolved (lead B, feature/al27-defaults-restart):** `FlowConfigHash` and
  `PreparationHash` now hash only explicit tables (`pipeline::ConfigTablesHash`): basis,
  reference, valence, hamiltonian, interaction.*, hartree_fock, natural_orbitals, imsrg.space,
  imsrg3, run.compat (+ imsrg.*, operators, run.task for the flow). Checkpoints written before
  that change are refused once by the new hash.

## 27Al weekend, Lead B task T3 (restarts without re-preparation, D75), 2026-11-27

Assumptions taken (most reversible option; open for the owner or the lead):
* **B-T3-1:** `checkpoint.restart = true` together with `checkpoint.prepared_from` is allowed (not
  an error): the own `prepared/` wins when present, else `prepared_from` is used. So a v3 run
  resubmitted with `restart = true` needs no config edit.
* **B-T3-2:** a restart whose checkpoint directory has `prepared/` but no `state.txt` (killed before
  the first flow checkpoint) starts the flow from the stored H_0 instead of failing.
* **B-T3-3:** `PreparationHash` keeps `[imsrg3]` (conservative; it does not change H_0 today).
* **B-T3-4:** the occupation and C-matrix file helpers of `transform/omega_store` were not moved
  down to `pipeline` (another agent edits the transform code this weekend); `prepared_state.cc` has
  its own small reader/writer for the same C format. Merge them later.
* **B-T3-5:** the ODE methods (`flow`, `flow_rk4`) do not check the SIGTERM flag (no step callback);
  only Magnus flows checkpoint on SIGTERM.
* **WE-O-1** (orchestrator, Opus review of feature/al27-defaults-restart, verdict merge) - Minor
  follow-ups, none blocking the weekend: (1) `PreparationHash` (`prepared_state.cc:257`) hashes the
  interaction file paths, not their contents, so a file regenerated at the same path reuses a stale
  H_0; (2) hashing the `imsrg3` table blocks sharing one H_0 between IMSRG(2) and IMSRG(3) runs
  (B-T3-3), and a new defaulted key in a hashed table invalidates stored states; (3) with
  `abs_occupation`, the HF-basis `occ_nat3_cut` (`run.cc:576`) differs from imsrg-devel at emax >= 4,
  so devel comparisons must set `sort = "signed_occupation"` (follows D72); (4) SIGTERM is acted on
  only between Magnus steps (not during preparation, ODE methods, the farm or FinishRun), and the
  `#SBATCH --signal` line of `run_flow.sh` has no effect inside a `salloc --no-shell` allocation
  (lead D's `flows_chain.sh` sends TERM to the process itself); (5) `Run` never clears the
  termination flag (a second run in one process, e.g. from Python, stops at once), and
  `prepared_from` pointing at the run's own directory races with root's `remove_all` under MPI;
  (6) `Manifest` does not declare its copy and move operations. **Assumed:** fix after the weekend.
* **WE-D-1** (lead D, devel writer threshold for TCM) - The devel tokyo writer (`CollectTokyoTwoBody`,
  imsrg-devel rule) drops scalar valence two-body elements below 1e-7 x ||O_2b|| of the whole
  operator. For TCM (||TCM_2b|| ~ 1e4 at emax 10) that is ~1.2e-3: the v2/v3 TCM files keep
  140-143 two-body elements, the fork writer (absolute 1e-7) 158; e.g. the N2LOsat pp
  (0d5/2)^2 -> (0d3/2)^2 J = 2 element (1.9e-3 in v1) is absent in v2. The NREFT operators have
  small norms (threshold <= 1e-7), so only TCM and H are affected (H: 1e-7 x ||H_2b||).
  **Assumed:** keep the devel rule (v2/v3 use the devel writer by decision of Brief B); the
  difference reports count such elements as "only in ref". Owner: an absolute threshold for
  operators in the default writer?
* **WE-D-2** (lead D, ODE stage end) - With `imsrg.method = "flow"` (dopri5) the stage-1 flow of
  the 27Al scan reached |eta| = 1e-3 at s ~ 14 but integrated on to smax = 100 at constant E0
  (|eta| hovering at 1.0e-3 +- 3e-6): the criterion appears to be checked so that values just
  above 1e-3 continue the stage. Harmless for E0 (stage-1 E0 constant to 1e-6) but costs time and
  shifts stage 2 to start at s = 100. **Assumed:** no change during the weekend; check the
  observer/stop logic of `flow::OdeSolver` against imsrg++ afterwards.

* **WE-R-1 (restart rewrites flow.txt).** In the restart check (design/benchmarks/al27-restart.md)
  the restarted link opened `output.flow_file` fresh: the killed v3 N2LOsat run's `flow.txt` holds
  only the rows from the restart step on (143 of 192 data rows), and `flow_summary.py` then shows
  one stage. All numerical results (E0, 42 Omegas, .snt) are bit-identical to an uninterrupted run
  and the existing rows equal the straight ones. **Assumed:** no src/ change during the weekend; the
  fix is append mode for the flow file on `checkpoint.restart` (src/imsrg/driver/run.cc, FlowLog
  construction near line 1030), with a test; until then the full table is in `flow-gpu.log`.
  **Fixed** (feature/al27-investigate-flowfile-restart): `FlowLog` is now built after the checkpoint is
  read; on a restart from a flow checkpoint it keeps the existing rows with (stage, istep) <= the
  checkpoint's, drops later and torn rows, and appends (other header: warn, fresh file). Tests
  `Checkpoint.RestartedFlowFileIsContinuous` / `FlowFileTruncatedUnlessRestart`.
