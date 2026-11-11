# AGENTS.md — instructions for coding agents (and humans)

## What this is

A ground-up rewrite of **imsrg++** (In-Medium Similarity Renormalization Group for nuclear
structure) in modern C++20. It uses Kokkos for CPU/GPU portability, MPI for distribution, and
nanobind Python bindings. The old code is the reference for features and for numerical checks;
we do not port it line by line.

* Design docs: `design/` (start with `design/README.md`). Current progress: `design/STATUS.md`.
  Open questions: `design/QUESTIONS.md`.
* Reference code (READ-ONLY, never modify, never build in place):
  `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel`. Analyses of it, with file:line citations:
  `design/reference/01..05-*.md`.
* Big test inputs (never copy them into the repo), in `/Users/cheshyre/Downloads/new-imsrg/`:
  * 2N: `VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz` → `file2e1max=14 file2e2max=28 file2lmax=14 hw=16`
  * 3N: `NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin` → `3bme_type=no2b file3e1max=6 file3e2max=12 file3e3max=18 hw=16`

## Golden rules

1. **Git is local only.** Never `git push`. Never commit to `main`. Work on
   `feature/<milestone>-<topic>` branches cut from `develop`, and merge back with `--no-ff` only
   after `tools/ci.sh` passes. Never use bare `git stash` (the stash is shared between worktrees);
   use WIP commits.
2. **Tests gate everything.** No merge without unit tests. Numerics must match imsrg-devel (or the
   brute-force reference) within the tolerances of `design/30-testing-and-validation.md` §2. Never
   loosen a tolerance silently.
3. **Problem sizes:** emax ≤ 6 in every test and run; IMSRG(3) only at emax 2 and 3.
4. **Never block on a question.** Append it to `design/QUESTIONS.md`, choose the most reversible
   option, note the assumption, and continue.
5. **Respect the module DAG** (`design/10-architecture.md` §2). `tools/check_layering.py`
   enforces it. A new module edge needs a design-doc update in the same commit.
6. **Old-code bugs:** fix them, document each in `design/30-testing-and-validation.md` §6, and
   validate the fixed path another way (brute force, a patched reference copy, or a limit case).
7. Never commit large files (> 1 MB) or generated data. Golden data in `tests/golden/` stays small.

## Build and test (available after milestone M0)

```bash
tools/bootstrap_deps.sh            # once per machine/backend: builds Kokkos into ~/.cache/imsrg-deps
cmake --preset dev-openmp          # or dev-serial, release-openmp, asan
cmake --build --preset dev-openmp
ctest --preset dev-openmp -L unit  # labels: unit, integration, mpi, slow, reference
tools/ci.sh dev-openmp             # the full pre-merge check; prints PASS/FAIL
```
Python: `.venv/bin/pip install -e .` then `.venv/bin/pytest tests/python`.

## Code conventions

* C++20, no compiler extensions. Formatting: `.clang-format` (100 columns).
* Naming: types `PascalCase`; functions and variables `snake_case`; private members
  `snake_case_`; macros `IMSRG_UPPER_CASE`; files `snake_case.hpp/.cpp`; namespaces
  `imsrg::<module>`.
* **Strong types** for indices and quantum numbers (`OrbitIndex`, `ChannelIndex`, `TwoJ`,
  `TwoTz`, `Parity`, ...). No raw `int` for these in public APIs. Store twice-j as `TwoJ`.
* **Value semantics**, composition, `std::variant` or enums for closed sets. No inheritance
  hierarchies for data types, no virtual functions in hot paths, no object slicing.
* **No mutable globals or singletons.** Options are passed in explicit structs. Angular-momentum
  data comes from immutable precomputed tables.
* Library code never calls `exit()`/`abort()`. Input errors throw typed exceptions; invariants
  use `IMSRG_ASSERT` (debug) or `IMSRG_CHECK` (always).
* Public headers must not include Kokkos, MPI, toml++ or CLI11. Kernels live in
  `src/imsrg/<module>/kernels/*.cpp`.
* Device-code rules (`design/12-compute-backend.md` §5): `IMSRG_LAMBDA` with value captures only;
  no containers, exceptions, virtual calls or allocation in kernels; trivially copyable kernel
  types.
* Every public function or class has a `///` docstring stating what it computes, its conventions
  (normalization, phases, units), preconditions, and, for kernels, its complexity.

## Physics conventions (same as imsrg++, so results compare element by element)

* Orbit index order: shell N ascending, l descending, j descending, then proton (tz2 = −1) before
  neutron. Proton indices are even; neutron = proton + 1.
* 2B kets `|pq; J⟩` with `p ≤ q`, lexicographic. Matrix elements are stored **normalized** and
  antisymmetrized. Getters for unnormalized values are named explicitly.
* Operator norms: 1B weighted by (2j+1), 2B by (2J+1)². These drive step sizes and BCH
  thresholds; do not change them.
* Units: MeV and fm. Physical constants are copied exactly from imsrg++ `PhysicalConstants.hh`.

## Commits

Subject line ≤ 72 characters in the imperative mood, a blank line, then a body explaining what
and why (design decisions!). End with a `Co-Authored-By:` trailer naming the model that wrote
the change. One logical change per commit.
