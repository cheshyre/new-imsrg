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
3. **Problem sizes:** emax ≤ 6 in every test and run; IMSRG(3) only at emax 2 and 3. Exception
   (owner, D40): occasional emax-8 IMSRG(2) performance benchmarks against imsrg++, never in CI.
4. **Never block on a question.** Append it to `design/QUESTIONS.md`, choose the most reversible
   option, note the assumption, and continue.
5. **Respect the module DAG** (`design/10-architecture.md` §2). `tools/check_layering.py`
   enforces it. A new module edge needs a design-doc update in the same commit.
6. **Old-code bugs:** fix them, document each in `design/30-testing-and-validation.md` §6, and
   validate the fixed path another way (brute force, a patched reference copy, or a limit case).
7. Never commit large files (> 1 MB) or generated data. Golden data in `tests/golden/` stays small
   and is regenerated only through the `tools/reference` scripts, with the reason and the old vs
   new numbers in the commit message (D42).
8. Never run two builds in the same build directory at once. Tasks sharing a checkout use their own
   build directory (`cmake --preset dev-openmp -B build/<task>`) or build one after another.
9. Scripts and globs work on `git ls-files`: the main checkout root also holds untracked
   `imsrg-devel/` and `.claude/worktrees/`.

## Build and test (since milestone M0; details in `design/20-build-and-tooling.md`)

```bash
tools/bootstrap_deps.sh            # once per machine (idempotent): Kokkos openmp+serial and the venv
cmake --preset dev-openmp          # or dev-serial, release-openmp, asan
cmake --build --preset dev-openmp
ctest --preset dev-openmp -L unit  # labels: unit, integration, mpi, slow, reference
tools/ci.sh dev-openmp             # the full pre-merge check; last line: CI PASS/FAIL <preset>
```
* **Shared per-machine state (one rule):** `IMSRG_DEPS_PREFIX` (default `~/.cache/imsrg-deps`)
  holds the Kokkos installs `kokkos-<backend>/`; `IMSRG_VENV` (default `$IMSRG_DEPS_PREFIX/venv`)
  is the one Python venv (numpy, pytest, sympy, scikit-build-core, cpplint) for all worktrees.
  `tools/bootstrap_deps.sh` and `tools/ci.sh` link it as `.venv` in each worktree (gitignored).
  Never create a venv per worktree.
* **Several tasks in one checkout:** give each its own build tree:
  `cmake --preset dev-openmp -B build/<task>`, `cmake --build build/<task>`,
  `ctest --test-dir build/<task> -L unit`, and `IMSRG_BUILD_DIR=build/<task> tools/ci.sh dev-openmp`.
* **New module:** add `src/imsrg/<m>/CMakeLists.txt` with `imsrg_add_module(NAME <m> SOURCES ...
  DEPS ...)` and `tests/unit/<m>/CMakeLists.txt` with `imsrg_add_test(NAME <m>_<topic> SOURCES ...
  DEPS <m> LABELS unit)` (test names are global, hence the module prefix). Both directories are
  discovered automatically; no shared CMake file changes. Include project headers as
  `"imsrg/<module>/x.h"`. Kernels in tests go into free functions, never into a `TEST` body
  (`design/12-compute-backend.md` §5).
* macOS: the default SDK of this machine does not link; CMake works around it by itself
  (`cmake/ImsrgAppleSdk.cmake`). For compiles outside CMake, export
  `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`.

Python (from M11): `CMAKE_ARGS="-DKokkos_ROOT=$HOME/.cache/imsrg-deps/kokkos-openmp -DOpenMP_ROOT=$(brew --prefix libomp)" .venv/bin/pip install -e . --no-build-isolation`,
then `.venv/bin/pytest tests/python`.

## Code conventions

* **Style: the Google C++ Style Guide** (owner, D38), with the exceptions and tooling in
  `design/17-code-style.md` (exceptions D16, `<filesystem>` outside kernels, Kokkos `[=]` device
  lambdas, physics notation in variable names). `tools/ci.sh` fails on clang-format or cpplint
  findings; run `tools/lint.sh` (clang-tidy naming check, needs `brew install llvm`) after adding
  or renaming names. cpplint is not followed blindly: a justified exception is a
  `// NOLINT(<category>): <reason>` on the line.
* C++20, no compiler extensions. Formatting: `.clang-format` (Google base, **80 columns**,
  include order: related header, C system, C++ standard, other libraries, project headers).
* Files `snake_case.h` / `snake_case.cc`, each starting with
  `// SPDX-License-Identifier: GPL-3.0-or-later` and `// Copyright 2026 The imsrg authors`;
  headers use `#define` guards `IMSRG_<PATH>_H_` (path below `src/`; outside `src/` the path
  from the repository root, e.g. `TESTS_UNIT_HF_HF_TEST_SUPPORT_H_`). Include project headers as
  `"imsrg/<module>/x.h"`, test- and binding-local headers by their path from the repository root
  (`"tests/support/commutator_test_support.h"`).
* Naming: types, concepts and **functions/methods `PascalCase`** (`hf::Solve`, `Mp2Energy`);
  accessors (const, return a property or part of the object, possibly selected by an index or
  key: `num_orbits()`, `orbit(i)`, `spec()`), mutators `set_x()` and STL analogs (`begin`,
  `size`, `find`, `value`) stay `snake_case` (the allowlist is in `.clang-tidy`); variables
  `snake_case`, but physics symbols keep their capitals (`X`, `Y`, `Z`, `J`, `Tz`, `hw_MeV`);
  class data members `snake_case_`, struct members without the `_`; constants with static
  storage and enumerators `kName` (`kInvSqrt2`, `Mixing::kLinear`); macros `IMSRG_UPPER_CASE`;
  namespaces `imsrg::<module>`, internals in `detail`.
* No using-directives (put tests inside the namespace they test; aliases and using-declarations
  in `.cc` files); `std::int64_t` etc. instead of `long`; C++ casts; every class declares its
  copy and move operations; objects with static storage are trivially destructible (use
  `static const auto* const kX = new T(...);`); floating-point literals with digits on both
  sides of the point (`1.0e-10`).
* **Strong types** for indices and quantum numbers (`OrbitIndex`, `ChannelIndex`, `TwoJ`,
  `TwoTz`, `Parity`, ...). No raw `int` for these in public APIs. Store twice-j as `TwoJ`.
* **Value semantics**, composition, `std::variant` or enums for closed sets. No inheritance
  hierarchies for data types, no virtual functions in hot paths, no object slicing.
* **No mutable globals or singletons.** Options are passed in explicit structs. Angular-momentum
  data comes from immutable precomputed tables.
* Library code never calls `exit()`/`abort()`. Input errors throw typed exceptions; invariants
  use `IMSRG_ASSERT` (debug) or `IMSRG_CHECK` (always).
* Public headers must not include Kokkos, MPI, toml++ or CLI11. Kernels live in
  `src/imsrg/<module>/kernels/*.cc`.
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
* Operator norms: 1B weighted by (2j+1), 2B by (2J+1)², 3B as imsrg++'s canonical pn Frobenius norm
  (for any layout); the zero-body part is excluded from the total. These drive step sizes and BCH
  thresholds; do not change them.
* Units: MeV and fm. Physical constants are copied exactly from imsrg++ `PhysicalConstants.hh`;
  mathematical constants (e.g. 1/√2, which imsrg++ truncates to 13 digits) use full double
  precision (D36).

## Commits

Subject line ≤ 72 characters in the imperative mood, a blank line, then a body explaining what
and why (design decisions!). End with a `Co-Authored-By:` trailer naming the model that wrote
the change. One logical change per commit.
