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
* **MS-Q2** (2026-11-15, MS Google style) - The guide lets accessors and mutators keep variable-style
  names. **Assumed meanwhile:** an accessor is a const method returning a property or part of the
  object, possibly selected by an index or key (`num_orbits()`, `orbit(i)`, `spec()`,
  `get_normalized(...)`); `set_*` mutators and STL analogs also stay snake_case; everything else is
  PascalCase (static factories too: `ModelSpace::FromNames`). The 199-name allowlist is in
  `.clang-tidy` (design/17 section 3 item 7). Alternative: PascalCase for all methods (would collide
  with type names such as `Orbit`, `Channel`, `Reference`, `Partition`, `ModelSpace`).
* **MS-Q3** (2026-11-15, MS Google style) - Header guards outside `src/` use cpplint's fallback, the
  path from the repository root without the `IMSRG_` prefix (`TESTS_UNIT_HF_HF_TEST_SUPPORT_H_`,
  `PYTHON_SRC_BINDINGS_H_`). **Assumed meanwhile:** fine (unique, and what the blocking cpplint step
  accepts); an `IMSRG_` prefix there would need NOLINTs or extra CPPLINT.cfg files.
* **MS-Q4** (2026-11-15, MS Google style) - `AUTHORS` lists the owner (taken from the git history) as
  the copyright holder named by "Copyright 2026 The imsrg authors"; the SPDX line says
  `GPL-3.0-or-later` as written in design/17 section 2.A (LICENSE is the GPL-3.0 text). **Assumed
  meanwhile:** as written; please confirm the holder list and "or later".
* **MS-Q5** (2026-11-15, MS Google style) - Structural items not converted (design/17 section 6):
  inline functions over ~10 lines in headers (device-code lookups, templates, header-only test
  helpers), the 36 structs with member functions, functions over ~40 lines, unsigned types.
  **Assumed meanwhile:** guidelines for new code, not worth a churn of the existing code; device
  lookups stay inline for performance. Schedule a cleanup task if wanted.
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

### M8-Q10 (hunter-gatherer, `H(s)` after the final gather)
imsrg++ `GatherOmega` rebuilds `H_saved` from `H_0` through the gathered Omega but leaves `H(s)`
untouched, so after a hunter-gatherer solve `H(s)` and `Transform(H_0)` differ at the truncation level
of the BCH product (up to 2e-3 in the O16 emax 2 unit test (4 gathers; E0 1e-4, see its printout)); stage 2's
first eta comes from the pre-gather `H(s)`. **Assumed:** copy imsrg++ (orchestrator decision), because
the IT-10 and `IT-2/hunter_gatherer` goldens depend on it. A cleaner variant (re-evaluate `H(s)` from
the gatherer, or gather before the next stage) would be an option with a deviation entry.

### M8-Q11 (hunter-gatherer, gather count and `set_generator`)
`SolveResult::num_gathers` counts the loop gathers plus the final gather (as the `gathering Omega.`
log lines of imsrg++; the goldens' `n_gathers`). A gather inside `set_generator` (hunter non-zero,
which cannot happen after a hunter-gatherer `Solve`) is counted in `MagnusSolver::num_gathers()` only.
`results.json` has `imsrg.num_gathers` (solver total) and `imsrg.stages[].num_gathers`.
`imsrg.omega.hunter_gatherer` is rejected nowhere any more; `imsrg.omega.storage = scratch/discard`
with hunter-gatherer is unchanged (not implemented either way).
* **M8-Q20** (2026-11-15, M8 D35 task) - D35 asks that "wherever the code computes a charge radius" the
  Rso contribution is added. The code has no charge-radius computation yet (design/16 section 6 assigns
  `PROTON_RCH2`, `NEUTRON_RCH2`, `DARWIN_FOLDY` to the output stage), so only the catalog operator `Rso`
  (`PrepareOperator`) was added and the contract is documented (`opslib::Rp2` docstring, design/13,
  design/30 D35-2): charge radius = Rp2 + Rso/Z + nucleon-size terms. **Assumed meanwhile:** the
  output-stage task implements it that way. Not blocking.
* **M8-Q21** (2026-11-15, M8 D35 task) - The gitignored emax-4 dumps in `build/golden-generated` (read
  through `IMSRG_GENERATED_GOLDEN_DIR` by `opslib_golden`, `hf_*`, `io_*` emax-4 tests) must come from the
  patched build too; with dumps from the pristine build `Emax4TrelAndRp2` fails (Rp2 differs by 0.063).
  **Assumed meanwhile:** after merging, run `REFHARNESS=<ref root>/harness-build-d35/refharness
  tools/reference/gen_golden_generated.sh` once per checkout (also done by `regen_golden_d35.sh` into
  `<ref root>/golden-generated-d35`). Alternative: make the script default to the patched harness.
* **M8-Q22** (2026-11-15, M8 D35 task) - `tools/reference/refharness.cpp` and `gen_golden_flow.sh` were
  edited (a few lines each: `IMSRG_REFERENCE_SHA_FILE`, `IMSRG_GOLDEN_E2E_OUT`, `FLOW_JOBS`); the
  hunter-gatherer task edits the same two files, so the merge may need a manual resolution of
  neighbouring hunks. **Assumed meanwhile:** the orchestrator resolves it (both changes are independent). **Resolved at the merge (orchestrator, 2026-11-15):** `gen_golden_flow.sh` taken from the hunter-gatherer branch, which already contains the D35 overrides; `refharness.cpp` merged automatically.
