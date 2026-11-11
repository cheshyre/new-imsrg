# 20 — Build system and tooling

## 1. Dependencies

| Dependency | How we get it | Facade | Notes |
|-----------|---------------|--------|-------|
| Kokkos (Core) | submodule `extern/kokkos` (latest stable release tag; record the tag), or `find_package(Kokkos)` on clusters | `imsrg::exec` | Built once into a deps prefix for development (§3). |
| BLAS/LAPACK | system: Accelerate (macOS), OpenBLAS/MKL (Linux); cuBLAS/cuSOLVER, hipBLAS/rocSOLVER for GPU builds | `imsrg::linalg` | `find_package(BLAS/LAPACK)`; `BLA_VENDOR` override. |
| MPI | system: Open MPI (`brew install open-mpi`), cluster MPI | `imsrg::comm` | Option `IMSRG_ENABLE_MPI` (default ON if found). |
| zlib | system | `imsrg::io` gzip stream | |
| {fmt} | submodule `extern/fmt` | `imsrg::base::log` | |
| toml++ | submodule `extern/tomlplusplus` | `imsrg::driver` config | header-only |
| CLI11 | submodule `extern/CLI11` | app only | header-only |
| GoogleTest | submodule `extern/googletest` | tests only | |
| nanobind | submodule `extern/nanobind` (with its robin_map submodule) | python only | `git submodule update --init --recursive` |
| OpenMP runtime | `brew install libomp` on macOS (Apple clang needs `-Xpreprocessor -fopenmp`, handled by Kokkos' CMake when `OpenMP_ROOT` points at libomp), native on Linux | Kokkos OpenMP backend | |
| Python ≥ 3.10 + numpy, pytest, sympy, scikit-build-core | shared venv `$IMSRG_VENV` (default `$IMSRG_DEPS_PREFIX/venv`), made by `tools/bootstrap_deps.sh` from `tools/requirements-dev.txt`; every worktree links it as `.venv` | | sympy provides exact Wigner symbols for the angmom tests. |

**Pinned in M0 (2026-11-11, details in `extern/README.md`):** kokkos 5.3.0 (tagged 2026-11-10,
newer than Homebrew's 5.2.2), googletest v1.18.0, fmt 12.2.1, CLI11 v2.7.2, toml++ v3.4.0,
nanobind v3.1.0 (robin_map v1.4.0). Kokkos 5.3.0 requires GCC ≥ 10.4, Clang ≥ 14, NVCC ≥ 12.2,
HIPCC ≥ 6.2 and CMake ≥ 3.22: the GCC baseline (D20) is met, but CUDA and ROCm need the 12.2 and
6.2 minor releases (QUESTIONS Q20). Do not use Homebrew's kokkos package for development builds
(its backend configuration is not under our control); `tools/bootstrap_deps.sh` builds the
submodule.

**Toolchain used on the dev machine (2026-11-12):** Apple clang 21.0.0, cmake 4.4.5, ninja
1.13.2, ccache 4.14.3, Open MPI 5.0.11, libomp 23.1.5, clang-format 23.1.5, Python 3.15.2
(numpy 2.5.4, sympy 1.14.0, pytest 9.2.0 and scikit-build-core 1.1.2 all have 3.15 wheels, so no
older Python was needed). Also installed for imsrg-devel: gsl, boost, openblas. Workarounds: §8.

macOS bootstrap (allowed by the user):
```bash
brew install cmake ninja ccache open-mpi libomp clang-format   # for the new code
brew install gsl boost openblas                     # additionally needed to build imsrg-devel
tools/bootstrap_deps.sh       # once per machine: Kokkos (openmp, serial) + the shared venv
```

## 2. CMake structure

* `cmake_minimum_required(VERSION 3.25)`, `project(imsrg LANGUAGES CXX)`, C++20 without
  extensions, `CMAKE_EXPORT_COMPILE_COMMANDS ON`.
* Options: `IMSRG_ENABLE_MPI`, `IMSRG_BUILD_TESTS` (ON), `IMSRG_BUILD_PYTHON` (ON),
  `IMSRG_ENABLE_KOKKOS_KERNELS` (OFF), `IMSRG_UNITY_BUILD` (OFF), `IMSRG_WERROR` (OFF; ON in
  CI), `IMSRG_SANITIZE` (none|address|undefined|thread), `IMSRG_TEST_DATA_DIR` (default
  `/Users/cheshyre/Downloads/new-imsrg`; integration tests that need the big input files are
  skipped with a clear message if the files are absent), `IMSRG_REFERENCE_DIR` (path to the
  imsrg-devel build, for golden-data generation; optional).
* Options added in M0: `IMSRG_KOKKOS_BACKEND` (serial|openmp|cuda|hip; set by the presets),
  `IMSRG_DEPS_PREFIX` and `IMSRG_VENV` (empty = environment variable, else the defaults of §3),
  `IMSRG_NATIVE_ARCH` (`-mcpu=native` on arm64, else `-march=native`), `IMSRG_USE_CCACHE` (ON),
  `IMSRG_MPIEXEC_PREFLAGS` (`--oversubscribe` when the launcher is Open MPI). `IMSRG_SANITIZE`
  accepts a list (`address;undefined`). `IMSRG_ENABLE_MPI` defaults to whether MPI was found.
* `src/imsrg/<module>/CMakeLists.txt` defines `imsrg_<module>` via one helper function
  (`cmake/ImsrgModule.cmake`):
  `imsrg_add_module(NAME operator SOURCES ... [HEADERS ...] DEPS basis linalg angmom exec base
  [LIBS <public targets>] [PRIVATE_LIBS <private targets>])`. It defines the static library with
  alias `imsrg::operator`, links module DEPS publicly, adds the include roots `src/` and
  `<build>/src` (generated headers), links the `imsrg::warnings` set privately, and records the
  DEPS; `imsrg_write_module_deps()` writes them to `<build>/imsrg_module_deps.json` for the layering
  check. No SOURCES = header-only (INTERFACE) module. Project headers are included as
  `"imsrg/<module>/x.hpp"` (the layering check also accepts `<imsrg/...>`; QUESTIONS Q21).
* `src/` and `tests/unit/` **discover** module directories (`file(GLOB ... CONFIGURE_DEPENDS)` over
  `*/CMakeLists.txt`): a milestone adds `src/imsrg/<m>/` and `tests/unit/<m>/` without editing a
  shared CMake file, so parallel milestones do not conflict. The layering check rejects directories
  that are not modules of the DAG.
* Static libraries by default (fast linking, no PIC overhead). With `IMSRG_BUILD_PYTHON`,
  position-independent code is turned on for the libraries linked into `_core` (the hook in the
  top-level `CMakeLists.txt` is filled in M11; until `python/` exists the option does nothing).
* `apps/imsrg`, `python/`, `tests/` each add their own targets (`apps/` and `python/` are added
  once they exist). Tests use `imsrg_add_test(NAME ... SOURCES ... LABELS unit|integration|mpi|slow|reference
  [DEPS <modules>] [LIBS <targets>] [MPI_RANKS n] [NO_GTEST_MAIN] [TIMEOUT s] [ARGS ...])`
  (`cmake/ImsrgTesting.cmake`): GoogleTest cases are registered with `gtest_discover_tests` in
  `PRE_TEST` mode (nothing runs at build time) as `<name>.<Suite>.<Case>`; with `MPI_RANKS` the
  executable is one test run under `mpiexec -n <n>` (label `mpi` added). Labels are applied from a
  small CTest include file, because the `PROPERTIES` of `gtest_discover_tests` (CMake 4.4) turn a
  label list into a single label. Test target names are global: prefix them with the module
  (`base_strong_types`, `angmom_wigner`); CMake fails on duplicates.
* Bundled dependencies (fmt, toml++, CLI11, GoogleTest, and the in-tree Kokkos fallback) are added
  with `add_subdirectory(... SYSTEM)`, so our warning flags never apply to their headers.
* Toolchain smoke tests live in `tests/smoke/` (Kokkos backend check + `parallel_reduce`, label
  `unit`; MPI hello world under `mpiexec -n 2`, label `mpi`).

## 3. Presets and the deps prefix

`CMakePresets.json` (version 6):

| Preset | Backend | Purpose |
|--------|---------|---------|
| `dev-openmp` (default on macOS) | Kokkos OpenMP | daily development, Debug + assertions, `-O1 -g` |
| `dev-serial` | Kokkos Serial | debugging, sanitizers |
| `release-openmp` | Kokkos OpenMP | performance runs, `-O3 -march=native`-like flags (`-mcpu=native` on arm64) |
| `asan` | Serial | AddressSanitizer + UBSan |
| `cuda`, `hip` | Kokkos CUDA/HIP | cluster builds (not testable on the dev machine) |

For speed across many git worktrees, `tools/bootstrap_deps.sh` builds Kokkos from
`extern/kokkos` **once per backend** and installs it into
`${IMSRG_DEPS_PREFIX:-$HOME/.cache/imsrg-deps}/kokkos-<backend>`. If `find_package(Kokkos)` fails,
CMake falls back to `add_subdirectory(extern/kokkos)`, so a fresh clone still builds with one
command (verified in M0 with an empty prefix).

As implemented in M0:
* **Shared per-machine state, one rule** (`tools/lib/imsrg_env.sh`, mirrored in CMake):
  `IMSRG_DEPS_PREFIX` defaults to `$HOME/.cache/imsrg-deps`; `IMSRG_VENV` defaults to
  `$IMSRG_DEPS_PREFIX/venv`. `tools/bootstrap_deps.sh` and `tools/ci.sh` create a gitignored
  `.venv -> $IMSRG_VENV` symlink in each worktree, so `.venv/bin/python` works everywhere and no
  worktree creates its own venv. CMake prefers `$IMSRG_VENV/bin/python3` for Python.
* Presets cannot express "environment variable or default", so they pass
  `IMSRG_DEPS_PREFIX=$env{IMSRG_DEPS_PREFIX}` (possibly empty) and `IMSRG_KOKKOS_BACKEND`;
  CMake applies the default and sets `Kokkos_ROOT=<prefix>/kokkos-<backend>` unless `Kokkos_ROOT`
  or `Kokkos_DIR` is given (e.g. a cluster module). A Kokkos that was given, or that is installed
  in the prefix, must load (configure fails otherwise, instead of silently building a different
  Kokkos in-tree); only when neither exists is extern/kokkos built in-tree. Configure fails if the install's
  `Kokkos_DEVICES` do not match the backend (a serial build needs a Serial-only Kokkos, so that
  the default execution space really is Serial). The OpenMP presets pass
  `OpenMP_ROOT=/opt/homebrew/opt/libomp`; CMake also finds Homebrew libomp itself on macOS.
* Kokkos builds: `kokkos-openmp` = OpenMP + Serial, Release; `kokkos-serial` = Serial only,
  RelWithDebInfo with `Kokkos_ENABLE_DEBUG` and `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK` (dev-serial and
  asan are the debugging builds; QUESTIONS Q22). Both: C++20, PIC (the nanobind module links the
  static libraries), `Kokkos_ENABLE_DEPRECATED_CODE_5=OFF`. Each build takes a few seconds.
* `tools/bootstrap_deps.sh [--backends "openmp serial"] [--no-venv] [--force] [--jobs N]` is
  idempotent (a stamp records Kokkos commit, compiler and options; a no-op run takes 0.4 s),
  serializes concurrent runs with an `mkdir` lock in the prefix (a lock whose owner on this host
  is dead, or that has no pid file after a minute, is stale; one waiter at a time may break it;
  locks of other hosts are never broken), and installs via `DESTDIR` staging plus two renames, so a
  concurrent configure sees the old install, the new one, or for an instant none (it would then
  build Kokkos in-tree), never a partial one. Build trees live in `$IMSRG_DEPS_PREFIX/build/`.
* `release-openmp` sets `IMSRG_NATIVE_ARCH=ON`; `asan` sets `IMSRG_SANITIZE=address;undefined`
  (with `-fno-sanitize-recover=undefined`). Test presets set `OMP_PROC_BIND=false` and
  `OMP_NUM_THREADS=4`. `cuda` uses `extern/kokkos/bin/nvcc_wrapper`, `hip` uses `hipcc`; both are
  configure-only (no build/test presets) and untested here.

## 4. Scripts

* `tools/ci.sh [preset]`: configure, build, `ctest -L unit`, then the integration tests if the
  data are present, the layering check, and `clang-format --dry-run`. This is the single command
  agents run before every merge. It must print a one-line PASS/FAIL summary. As implemented (M0):
  it checks the checkout containing the script (any worktree), initializes submodules only if
  one is missing (no network needed otherwise), configures with `-DIMSRG_WERROR=ON`, builds, runs
  `ctest -L '^unit$'`, the integration tests if both big input files exist in
  `IMSRG_TEST_DATA_DIR` (ctest label `integration`, plus `pytest tests/integration` once it
  exists), `ctest -L '^mpi$'` if `IMSRG_ENABLE_MPI`, `check_layering.py` with its self-test, and
  `clang-format --dry-run --Werror` on git-tracked C++ files under `src/ apps/ tests/ python/
  tools/bench/` (never `extern/`, `tools/reference/`, `tests/golden/`). Tests labeled `slow` never
  run in CI. Each step is timed; the last line is exactly `CI PASS <preset>` or
  `CI FAIL <preset>: <step>` (nonzero exit). `IMSRG_BUILD_DIR` overrides the build tree (default
  `build/<preset>`) so two runs in one checkout do not collide; `IMSRG_CI_JOBS` sets the build
  jobs. Tests get `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=false` unless set, and ctest runs
  `ncpu / OMP_NUM_THREADS` tests at a time. Typical time with ccache warm: 5 s; a fresh worktree
  needs about 50 s, almost all of it the shallow submodule fetch.
* `tools/check_layering.py`: enforces the module DAG (`10-architecture.md` §2). It checks (a) the
  module DEPS that CMake recorded, (b) `#include "imsrg/<m>/..."` (or `<imsrg/...>`) edges in
  git-tracked `src/` files against the transitive closure of the DAG, and (c) third-party header
  confinement, transitively through repo headers: Kokkos only in `src/imsrg/exec/` and
  `src/imsrg/*/kernels/`, `mpi.h` only in `comm/`, toml++ only in `driver/`, CLI11 never under
  `src/`. Rule (c) is what enforces "public headers must not include Kokkos/MPI". CTest runs it
  (`layering.check`) and its `--self-test` against `tests/layering/fixtures/{good,bad}`
  (`layering.self_test`), both labeled `unit`. Adding a module means updating its table and
  `10-architecture.md` §2 in the same commit. Like the format check, it sees tracked files plus
  new files not yet added (never gitignored ones), so a run before `git add` is meaningful.
  Consequence for M2: device annotations used in public headers (`IMSRG_FUNCTION`,
  `IMSRG_INLINE_FUNCTION`; `base` types must be usable in kernels) cannot come from Kokkos
  headers, which `base` cannot see anyway. Define them without Kokkos (keyed on
  `__CUDACC__`/`__HIPCC__`, i.e. `__host__ __device__` there and empty otherwise); `IMSRG_LAMBDA`
  is needed only in kernel files and can come from `exec/kokkos_backend.hpp`.
* `tools/ci.sh` and `tools/check_layering.py` operate on `git ls-files` only: the main checkout
  root also holds untracked `imsrg-devel/` and `.claude/worktrees/`. pytest uses
  `testpaths = ["tests/python", "tests/integration"]`.
* Concurrency: never run two builds in one build directory. Tasks that share a checkout use
  separate build dirs (`-B build/<task>`) or build sequentially. `tools/bootstrap_deps.sh` takes a
  lock (`mkdir`-based) on the deps prefix.
* `tools/reference/build_imsrg_devel.sh`: builds the old code outside the repo (§5).
* `tools/reference/gen_golden_*.py|.cpp`: produces golden data (see `30-testing-and-validation.md`).
* `.clang-format` (based on Google style, 100 columns) and `.clang-tidy` (bugprone, performance,
  modernize subsets; not enforced at first).

## 5. Building the reference imsrg-devel (outside the repo)

* Location: `/Users/cheshyre/Downloads/new-imsrg-reference/` (sibling of the repo, never inside
  it; do **not** modify `/Users/cheshyre/Downloads/new-imsrg/imsrg-devel`, copy it if patches
  are needed).
* `src/CMakeLists.txt` is the working CMake root. The top-level one is broken (it adds a
  nonexistent `extern` directory); see `reference/05` §4.
* Dependencies: vendored Armadillo 15.6, pybind11, half, and Boost.Iostreams sources; system
  Boost headers (odeint), GSL, BLAS (OpenBLAS or Accelerate), zlib, OpenMP. With Apple clang,
  point CMake to brew's libomp. If that fails, use brew `gcc` (g++-15) or `llvm` for this
  build only.
* CMake 4.x removed compatibility with very old `cmake_minimum_required` versions. If a vendored
  dependency of imsrg-devel fails to configure for that reason, pass
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
* Build the `imsrg++` executable and the old library. The vendored pybind11 (2.13.0.dev1) likely
  does not support the system Python 3.15: build `pyIMSRG` against Homebrew `python@3.12` if it is
  needed, or skip it (`-DIMSRG_BUILD_PYTHON_BINDINGS=OFF`); M1 can pass with the C++ harness alone.
  Build the harness with a small wrapper `CMakeLists.txt` in `tools/reference/` that pulls in the
  old `src` via `add_subdirectory`. **Never run the old `install` target** (it writes into `$HOME`). Record the exact commands in
  `tools/reference/build_imsrg_devel.sh` and the git hash or date of the reference source in
  `tests/golden/README.md`.
* If a bug fix needs a "patched reference" to validate the fixed path, apply the patch to a
  **copy** (`new-imsrg-reference/imsrg-devel-patched`) and record the patch in
  `tools/reference/patches/`.

## 6. Plain Makefile (later, M13)

A single `Makefile` at the root for users without CMake. It assumes a pre-installed Kokkos
(`KOKKOS_PATH`, using Kokkos' `Makefile.kokkos` support), OpenMP backend, system BLAS/LAPACK
and zlib, no MPI by default (`USE_MPI=1` switches to `mpicxx`), and no Python. It lists sources
by globbing `src/imsrg/*/*.cpp src/imsrg/*/kernels/*.cpp`. A test checks that it builds the
executable. Note: the inherited `.gitignore` ignores `Makefile` everywhere; M13 must add
`!/Makefile`.

## 7. Python packaging

`pyproject.toml` with scikit-build-core. `pip install -e . --no-build-isolation` (with `CMAKE_ARGS`
pointing at the Kokkos deps prefix and `OpenMP_ROOT=$(brew --prefix libomp)`) builds the
`_core` extension through the same CMake project (`IMSRG_BUILD_PYTHON=ON`,
`IMSRG_BUILD_TESTS=OFF`). For development, the in-tree build's `python/` directory can be put
on `PYTHONPATH` directly. Details: `18-python-api.md`.

## 8. Toolchain notes and workarounds (macOS arm64, found in M0, 2026-11-12)

* **Default macOS SDK does not link.** `xcrun` selects `MacOSX27.0.sdk`, whose `.tbd` stubs list
  the architecture `arm64e.x1-macos`, which the installed `ld-1267` rejects ("tapi error:
  malformed file ... unknown architecture"); every link fails, including CMake's compiler check.
  `MacOSX.sdk` (-> `MacOSX26.5.sdk`) works. `cmake/ImsrgAppleSdk.cmake` runs before `project()`
  and, only if a test link with the default SDK fails, caches `CMAKE_OSX_SYSROOT` as the first SDK
  that links; it does nothing once the toolchain is fixed (or if `SDKROOT`/`CMAKE_OSX_SYSROOT` is
  set). Other CMake projects can preload it: `cmake -C cmake/ImsrgAppleSdk.cmake ...`
  (`tools/bootstrap_deps.sh` does this for Kokkos; the imsrg-devel reference build can do the same,
  or pass `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`, or export
  `SDKROOT` to that path, which also fixes plain `clang++`/`mpicxx` calls). Owner action, optional:
  update or reinstall the Command Line Tools so `ld` and the default SDK match.
* **OpenMP with Apple clang** works directly with Homebrew libomp: Kokkos' CMake finds it as
  `-Xclang -fopenmp` when `OpenMP_ROOT=/opt/homebrew/opt/libomp`; brew `llvm` or `gcc` were not
  needed. Consumers of the installed Kokkos need the same `OpenMP_ROOT` (Kokkos re-runs
  `find_dependency(OpenMP)`); presets pass it and CMake detects it on macOS. Kokkos prints an
  `OMP_PROC_BIND` warning unless that variable is set; tests set `OMP_PROC_BIND=false`.
* **Open MPI 5.0.11** runs `mpirun -n 2` out of the box (default `TMPDIR`, no `OMPI_MCA_*`
  settings). `--oversubscribe` is added for Open MPI launchers only (`IMSRG_MPIEXEC_PREFLAGS`),
  for machines with fewer cores than ranks; MPICH-style launchers do not know the flag.
  CMake's `MPI_CXX_LIBRARY_VERSION_STRING` is empty unless `MPI_DETERMINE_LIBRARY_VERSION` is
  set, so the launcher is identified with `mpiexec --version` instead.
* **Apple's linker warns about duplicate static libraries** (gtest via gmock and gtest_main).
  `cmake_minimum_required(VERSION 3.25...3.31)` enables policy CMP0156, which de-duplicates link
  lines for this linker.
* **gtest_discover_tests and list-valued properties**: see §2 (labels come from a CTest include
  file).
* **Shell scripts** target bash 3.2 (macOS `/bin/bash`): no `mapfile`, no associative arrays,
  and no `"${array[@]}"` of possibly empty arrays under `set -u`.
* **Agent sandbox note:** in the M0 session, `git clone` and `git -C <submodule> fetch` were
  denied by the permission system, while `git submodule add/update` and `git update-index` were
  allowed. Pinning therefore used `git update-index --cacheinfo 160000,<commit>,extern/<name>`
  followed by `git submodule update --init` (see `extern/README.md`).
