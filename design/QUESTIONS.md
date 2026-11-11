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
* **O2** (2026-11-11, orchestrator) — Q19 follow-up: the Claude Code docs say `autoCompactWindow`
  is an absolute token threshold and that Sonnet 5.5, Haiku 5.5 and Opus 5.x default to 1M-token
  windows, so the planned 150000 meant ~15 %. Changed to 600000 (Sonnet/Haiku agents, ~60 %) and
  800000 (`imsrg-lead`, ~80 %); see `41-agent-workflow.md` §5. The values take effect in the next
  session (definitions load at startup). Combining them with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` is
  undocumented. **Assumed meanwhile:** keep the per-agent windows and recommend dropping the
  percentage override from the proposed settings.
* **Q20** (2026-11-12, M0) — Kokkos 5.3.0 (pinned as instructed: latest 5.x) requires NVCC ≥ 12.2
  and HIPCC ≥ 6.2 (`extern/kokkos/cmake/kokkos_compiler_id.cmake`), slightly above the baseline of
  Q9/D20 (CUDA ≥ 12, ROCm ≥ 6); its GCC ≥ 10.4 meets GCC ≥ 11. Options: raise the baseline to
  CUDA 12.2 and ROCm 6.2; or pin a Kokkos 4.x release for clusters with older toolkits (its
  minimums would need checking). **Assumed meanwhile:** keep Kokkos 5.3.0; CMake asks for
  `find_package(Kokkos 5)`. Moving the pin is a one-line change (`extern/README.md`).
* **Q21** (2026-11-12, M0) — Include style for project headers: `10-architecture.md` §1 said
  `<imsrg/<module>/x.hpp>`, the M0 brief said `"imsrg/<module>/..."`. **Assumed meanwhile:** quotes
  (Google style; `.clang-format` sorts them into the last include group); `10` §1/§2 updated. The
  layering check accepts both forms, so either choice is safe.
* **Q22** (2026-11-12, M0) — The `kokkos-serial` deps build enables `Kokkos_ENABLE_DEBUG` and
  `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`, because dev-serial and asan are the debugging builds. So
  dev-serial differs from dev-openmp in more than the backend, and Serial runs are slower.
  **Assumed meanwhile:** keep it; removing the two options in `tools/bootstrap_deps.sh` rebuilds
  Kokkos automatically (the stamp changes).
* **Q23** (2026-11-12, M0) — The Python venv is shared by all worktrees at `$IMSRG_DEPS_PREFIX/venv`
  (`IMSRG_VENV`), with a gitignored `.venv` symlink per worktree, instead of one `.venv` per
  checkout. **Assumed meanwhile:** shared, as the M0 brief asked; `IMSRG_VENV` can point anywhere.
* **Q24** (2026-11-12, M0) — Owner action, optional: the Command Line Tools on the dev machine make
  `xcrun` pick `MacOSX27.0.sdk`, which `ld-1267` cannot link against (`20` §8). CMake works around
  it automatically; updating or reinstalling the Command Line Tools would make the workaround
  unnecessary. **Assumed meanwhile:** keep the workaround (it switches itself off once the default
  SDK links).
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
* **M3-Q1** (2026-11-12, M3) — imsrg++ `GetjjSpace` indexes out of range for a proton or neutron count outside the tabulated closures. **Assumed meanwhile:** `define_valence_space("jj-shell", ...)` throws `InputError` for counts outside [2, 258); in-range behaviour is unchanged.
* **M3-Q2** (2026-11-12, M3 golden tests) — tests/golden/README.md says the ph (Pandya) channel field is Tz = |tz2_p - tz2_q|/2 (non-negative), but the dump files contain Tz in {-1, 0, 1} for ph_channel records, and their ch index obeys 6J + 2(Tz+1) + parity (same formula as pp channels). Is the README wording stale? **Assumed meanwhile:** the files are authoritative; the golden parser checks the index formula and does not constrain the sign.
* **M3-Q3** (2026-11-12, M3 ph/3B channels) — imsrg++ `ModelSpace::GetLocalIndex` (Pandya) returns `NumberKets-1` for an absent swapped ket instead of signalling absence. **Assumed meanwhile:** `ParticleHoleChannels::pair_index` and `local_index` return `std::nullopt` for absent kets (the golden files never exercise the buggy case); callers must handle the empty optional.
* **M3-Q4** (2026-11-12, M3 ph/3B channels) — `ThreeBodyChannel::CheckChannel_ket` contains stretched-state rules the reference itself marks "TODO Put these back in" (redundant). **Assumed meanwhile:** `ThreeBodyChannels` copies all of them exactly, so membership matches the golden files; dropping the redundant rules is left to a later cleanup.
* **M3-Q5** (2026-11-12, M3 golden tests) — The 14-file golden set includes an lmax=2 basis with 22 orbits, so the "norb = (emax+1)(emax+2)" assertion in test_golden_basis_dump.cpp is now restricted to files with lmax >= emax. This corrects a test assumption; the golden comparisons are unchanged. **Assumed meanwhile:** as described.
* **M3-Q6** (2026-11-12, M3 lead) — The 6j-weighted Pandya recoupling maps are not built in
  `basis`. `ParticleHoleChannels` provides the cross-coupled channel structure, `pair_index` (the
  2·nkets ordered-pair index) and flat tables; an elementwise precomputed map is
  O(n_cc² × n_J) per channel, far too large at emax 12-14, so its layout has to be designed with
  the comm222_ph kernel (design/12 §4.2). **Assumed meanwhile:** M7 (commutator) builds the maps
  from `ParticleHoleChannels` and the angmom 6j tables.
* **M3-Q7** (2026-11-12, M3 lead) — `ModelSpace` does not own its device mirror: `DeviceTables`
  is a separate object (`DeviceTables(const ModelSpace&)`), so that building a ModelSpace (and
  every host-only test of later modules) never needs an initialized Kokkos runtime, and the
  `m3-api-ready` behaviour of `ModelSpace::create` stays unchanged. **Assumed meanwhile:**
  consumers (operator/commutator) hold a `DeviceTables`; M4 may add a shared cache to the bundle.
* **M3-Q8** (2026-11-12, M3 lead) — Input checks stricter than imsrg++: a nucleus that does not
  fit into emax throws (imsrg++ prints "Trouble!" and continues with a partial filling), a core or
  hole orbit outside the basis throws (imsrg++ exits in `SetLmax`), while valence orbits outside
  the basis are dropped with a warning (as imsrg++ does); orbit labels accept multi-digit n
  (imsrg++ misreads them). **Assumed meanwhile:** keep the checks; none changes a valid run.
* **M3-Q9** (2026-11-12, M3 lead) — The IMSRG(3) configuration cuts (dE3max, OccNat3Cut) are not
  applied when the 3B kets are enumerated (imsrg++ enumerates with E3max only and applies them in
  the commutators); `ThreeBodyChannels::passes_configuration_cut` evaluates them per ket.
  **Assumed meanwhile:** M10 decides whether its layout drops those kets or masks them.
* **M4-Q1** (2026-11-12, M4 lead) — The `operator` module cannot use the namespace
  `imsrg::operator` (C++ keyword). **Assumed meanwhile:** `imsrg::op`; the CMake target and the
  include path stay `imsrg::operator` / `imsrg/operator/...` (design/10 §3.3 updated).
* **M4-Q2** (2026-11-12, M4 lead) — An OperatorDump records occupations but no valence space.
  **Assumed meanwhile:** `model_space_for_dump` builds the partition with the holes
  (n >= OCC_CUT) as core and every other orbit as q-space; the reader rejects occupations that
  differ from the target model space unless `check_occupations = false`.
* **M2-Q1** (2026-11-12, M2) — Logging needs process-wide output configuration (level, "only rank
  0 prints"), while AGENTS.md forbids mutable globals. `base` provides explicit, thread-safe
  `log::Logger` objects plus one `log::default_logger()` behind `imsrg::log::info(...)` (design/10
  §4). Timer registries and profiler hooks are explicit objects, never global. **Assumed
  meanwhile:** the default logger is the single sanctioned global (it configures output only and
  never affects results).
* **M2-Q2** (2026-11-12, M2) — design/10 §4 describes one `imsrg::Runtime` "(module exec plus
  comm)", but the DAG forbids exec<->comm edges and `mpi.h` is confined to comm. **Assumed
  meanwhile:** two RAII objects, `comm::Environment` (MPI, create first) and `exec::Runtime`
  (Kokkos, create second); the combined `imsrg::Runtime` is a thin composition in `driver` (M8).
  Alternative: allow comm -> exec (useful later for GPU-aware MPI on `DeviceBuffer`s).
* **M2-Q3** (2026-11-12, M2) — imsrg++ `PhysConst::INVSQRT2` is `0.7071067811865` (13 digits,
  relative error 6.7e-14), used in Operator.cc, imsrg_util.cc, Atomic.cc and DaggerCommutators.cc.
  `constants::inv_sqrt2` copies it exactly (as instructed). **Assumed meanwhile:** later modules
  use it only where they reproduce those imsrg++ formulas and `1/sqrt(2)` elsewhere; either way the
  effect (~1e-13 relative) is far below the 1e-10 unit tolerances.
* **M2-Q4** (2026-11-12, M2) — linalg's non-Accelerate path declares Fortran LAPACK/BLAS symbols
  without the hidden string-length arguments that gfortran-compiled libraries formally expect
  (a known ABI pitfall); it is not compiled on this machine (Accelerate is used). **Assumed
  meanwhile:** fine for OpenBLAS/MKL in practice; on the first Linux build, test it or switch to
  passing the lengths (or LAPACKE).
* **M2-Q5** (2026-11-12, M2) — CPU CI cannot exercise GPU-only paths flagged by the exec review:
  `DeviceBuffer` with a separate host mirror (sync copies) and `team_for` kernels with team size
  > 1 (`team_barrier`/`single`). **Assumed meanwhile:** reviewers check these by hand; add a
  CUDA/HIP compile-and-run job when a GPU machine is available.
* **M2-Q6** (2026-11-12, M2) — `angmom::talmi_b` uses the Brody-Moshinsky factorial formula in
  double like `AngMom::TalmiB` (accurate to ~1e-15 relative for n <= 6, l <= 12, checked against
  exact values), while imsrg_util switches to an lgamma/long-double variant for large arguments and
  to quadrature when radial sums cancel badly (pmax >= 20). **Assumed meanwhile:** the factorial
  version suffices for M2; whoever ports radial integrals (opslib) decides on the large-argument
  path.

* **M2-Q7** (2026-11-12, M2) — `angmom::moshinsky` evaluates the Buck et al. sum in
  double-double (the plain double algorithm of imsrg++ misses the 1e-12 bound on 22 of the
  405,537 brackets with E <= 16, by up to 7.8e-11 relative, because large prefactors amplify
  the rounding of the CG/9j factors), so our values deviate from imsrg++ by up to ~1e-10
  relative on such brackets (logged in `30` §6). Its double-double tables cover E <= 40
  (`moshinsky_max_energy`, emax 20; NaN beyond, and `MoshinskyTable` checks it). The double
  `wigner` functions are validated on the emax-8 domain; at l ~ 30 the double <l1 0 l2 0|L 0>
  errs by ~1e-11. **Assumed meanwhile:** double-double only for the bracket tables (built once
  per run; ~3 s for all E <= 16 in a Debug build); raise the limits if larger spaces need them.
* **M1-Q2** (2026-11-12, M1 flow) - "IMSRG(3f2)" in `refharness flow --factorized` means plain
  IMSRG(2) (no 3B storage, `IMSRG3` off) plus `BCH::SetUseFactorizedCorrection(true)`; imsrg++ has
  no switch for it. The `Elast`/norm divergence guards (|eta| > 1e12, |E| > 1e9) are harness
  additions. **Assumed meanwhile:** this is the intended IT-10 reference; O16 emax 2 converges
  with it, so no He4 substitute was needed (He4 emax 2 kept as extra).
