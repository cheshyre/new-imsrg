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
| Python ≥ 3.12 (D29) + numpy, pytest, sympy, scikit-build-core | shared venv `$IMSRG_VENV` (default `$IMSRG_DEPS_PREFIX/venv`), made by `tools/bootstrap_deps.sh` from `tools/requirements-dev.txt`; every worktree links it as `.venv` | | sympy provides exact Wigner symbols for the angmom tests. |

**Pinned in M0 (2026-11-11, details in `extern/README.md`):** kokkos 5.3.0 (tagged 2026-11-10,
newer than Homebrew's 5.2.2), googletest v1.18.0, fmt 12.2.1, CLI11 v2.7.2, toml++ v3.4.0,
nanobind v3.1.0 (robin_map v1.4.0). Kokkos 5.3.0 requires GCC ≥ 10.4, Clang ≥ 14, NVCC ≥ 12.2,
HIPCC ≥ 6.2 and CMake ≥ 3.22: the GCC baseline (D20) is met, and the owner raised the CUDA and
ROCm baseline to 12.2 and 6.2 (QUESTIONS Q20, D30). Do not use Homebrew's kokkos package for development builds
(its backend configuration is not under our control); `tools/bootstrap_deps.sh` builds the
submodule.

**Toolchain used on the dev machine (2026-11-12):** Apple clang 21.0.0, cmake 4.4.5, ninja
1.13.2, ccache 4.14.3, Open MPI 5.0.11, libomp 23.1.5, clang-format 23.1.5, Python 3.15.2
(numpy 2.5.4, sympy 1.14.0, pytest 9.2.0 and scikit-build-core 1.1.2 all have 3.15 wheels, so no
older Python was needed). Also installed for imsrg-devel: gsl, boost, openblas. Workarounds: §8.
Added in run 2 (Google style, D38): cpplint 2.0.2 in the shared venv (tools/requirements-dev.txt)
and Homebrew LLVM 23.1.5 for clang-tidy (`tools/lint.sh`).

macOS bootstrap (allowed by the user):
```bash
brew install cmake ninja ccache open-mpi libomp clang-format   # for the new code
brew install llvm                                   # clang-tidy for tools/lint.sh (optional)
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
  `"imsrg/<module>/x.h"` (the layering check also accepts `<imsrg/...>`; QUESTIONS Q21).
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
  exists), `ctest -L '^mpi$'` if `IMSRG_ENABLE_MPI`, the Python binding tests if the module was
  built, `check_layering.py` with its self-test, `clang-format --dry-run --Werror` and cpplint
  (since run 2, D38; it prints every finding and fails on any) on git-tracked C++ files under
  `src/ apps/ tests/ python/ tools/bench/` (never `extern/`, `tools/reference/`,
  `tests/golden/`), and since 2026-11-16 the `license headers` step (D46: every tracked `.h`/`.cc`
  of ours carries `SPDX-License-Identifier: GPL-3.0-only` in its first 5 lines). Tests labeled
  `slow` never run in CI. Each step is timed; the last line is exactly `CI PASS <preset>` or
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
  is needed only in kernel files and can come from `exec/kokkos_backend.h`.
* `tools/ci.sh` and `tools/check_layering.py` operate on `git ls-files` only: the main checkout
  root also holds untracked `imsrg-devel/` and `.claude/worktrees/`. pytest uses
  `testpaths = ["tests/python", "tests/integration"]`.
* Concurrency: never run two builds in one build directory. Tasks that share a checkout use
  separate build dirs (`-B build/<task>`) or build sequentially. `tools/bootstrap_deps.sh` takes a
  lock (`mkdir`-based) on the deps prefix.
* `tools/reference/build_imsrg_devel.sh`: builds the old code outside the repo (§5).
* `tools/reference/gen_golden_*.py|.cpp`: produces golden data (see `30-testing-and-validation.md`).
* Style tooling (Google C++ Style Guide, owner D38; converted in run 2, details in
  `17-code-style.md`): `.clang-format` (Google base, 80 columns, the guide's include order),
  `CPPLINT.cfg` (cpplint 2.0, blocking in `ci.sh`; each filter has its reason in the file),
  `.clang-tidy` (`readability-identifier-naming` with the Google naming and the accessor
  allowlist; the bugprone, performance and modernize subsets stay advisory).
* `tools/lint.sh [file...]`: the naming check. Runs clang-tidy (Homebrew LLVM if none is on the
  PATH) with only `readability-identifier-naming` on the given units or on every unit of the
  compile database of `IMSRG_BUILD_DIR` (default `build/dev-openmp`) under `src/ apps/ tests/
  python/ tools/bench/`, prints each finding once and exits nonzero on any (about 25 s for all
  units). Not part of `ci.sh`, which stays under 5 minutes (D42); run it after adding or renaming
  names. Code the build does not compile (`communicator_serial.cc` without MPI, `tools/bench`
  without `IMSRG_BUILD_BENCH`, CUDA/HIP branches) needs a second build tree, e.g.
  `cmake --preset dev-openmp -B build/style-cov -DIMSRG_ENABLE_MPI=OFF -DIMSRG_BUILD_BENCH=ON`.
* `.git-blame-ignore-revs` lists pure reformat commits (`git config blame.ignoreRevsFile
  .git-blame-ignore-revs`).

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

**What worked (M1, 2026-11-11):** `tools/reference/build_imsrg_devel.sh [all|imsrg|harness]`.
* The script rsyncs a pristine copy of `src/` to `new-imsrg-reference/imsrg-devel-src/src`
  (checksum verified against the original; the reference has no `.git`, so its version is the
  sha256 over `src/`, recorded in `tests/golden/README.md`) and builds from the copy, so nothing
  is ever written into the read-only tree.
* Toolchain: Apple clang 21 + Homebrew libomp via `-DOpenMP_ROOT=$(brew --prefix)/opt/libomp`
  (FindOpenMP then adds `-Xclang -fopenmp`), OpenBLAS/GSL/Boost from `CMAKE_PREFIX_PATH`,
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, Ninja, ccache, `-DIMSRG_BUILD_PYTHON_BINDINGS=OFF`. No
  source patches were needed; the old code compiles warning-free. Full build ≈ 1–2 min on 18 cores.
* **macOS SDK pitfall (affects M0 too):** xcrun's default SDK on the dev machine is
  `MacOSX27.0.sdk`, whose `libSystem.tbd` lists the architecture `arm64e.x1-macos` that the
  installed linker (ld-1267) rejects ("tapi error: malformed file"); even a C hello-world fails
  to link, so CMake reports a broken compiler. Fix: `-DCMAKE_OSX_SYSROOT=` the 26.5 SDK
  (`/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`) or `SDKROOT=` the same. The script
  probes the default SDK and falls back automatically.
* The old `src/CMakeLists.txt` sets `CMAKE_CURRENT_BINARY_DIR` to `"../build"` for its generated
  `version.cc`, so its binary directory must be named `build` (then `../build` is itself):
  `new-imsrg-reference/build` for the standalone build, and `add_subdirectory(<src> build)` in the
  wrapper `tools/reference/CMakeLists.txt` (binary dir `new-imsrg-reference/harness-build`).
  The wrapper also needs its own `find_package(OpenMP)` (imported targets are directory-scoped).
* **Frontier (2026-11-26).** All paths come from `tools/reference/ref_env.sh` (sourced by every
  script): `REF_ROOT` = `$IMSRG_REFERENCE_ROOT` (SHA files, read-only), `REF_BUILD` =
  `$IMSRG_REFERENCE_BUILD`, else `$IMSRG_WORK/reference-build` (all copies, builds, runs; the SHA
  files it writes there are used only when `REF_ROOT` has none), `REF_SRC` = `$REF_ROOT/imsrg-devel`,
  `DATA` = `$IMSRG_TEST_DATA_DIR`, else `$IMSRG_WORK/inputs`; without `IMSRG_WORK` the Mac
  defaults apply. Build on a compute node (no internet there, nothing to download):
  `source tools/lib/frontier_env.sh; imsrg_compute tools/reference/build_imsrg_devel.sh patched`
  (then `all` for the pristine pair; about 1 min each on 56 cores). The script loads the CPU modules of
  section 9 plus `gsl/2.8 boost/1.88.0` (the login node's system Boost is 1.66, the compute nodes have
  none) and uses the Cray wrappers `cc`/`CC` (gcc 14.2, cray-libsci as BLAS). What differed from
  the Mac: (1) the hard-coded `BLA_VENDOR OpenBLAS` finds nothing, so `-DBLAS_LIBRARIES=sci_gnu_mp`
  is preset (the wrapper links libsci itself); (2) gcc 14 needs `<cstdint>` in
  `imsrg_util/Pwd.hh`, supplied by `-DCMAKE_CXX_FLAGS="-include cstdint"` instead of a source patch;
  (3) the wrappers default to static linking, so `CRAYPE_LINK_TYPE=dynamic` and
  `CMAKE_POSITION_INDEPENDENT_CODE=ON` for `libIMSRG.so`. The source checksum equals the Mac one
  (`ddcbf1af...`). Regenerated `golden-generated` agrees with the Mac files to about 1e-15 relative
  in the physics numbers (E_HF, MP2, norms) and 1e-13 absolute in the dumps; the entries below
  about 1e-17 are rounding noise that the dumps write or omit differently.


## 6. Plain Makefile (later, M13)

A single `Makefile` at the root for users without CMake. It assumes a pre-installed Kokkos
(`KOKKOS_PATH`, using Kokkos' `Makefile.kokkos` support), OpenMP backend, system BLAS/LAPACK
and zlib, no MPI by default (`USE_MPI=1` switches to `mpicxx`), and no Python. It lists sources
by globbing `src/imsrg/*/*.cc src/imsrg/*/kernels/*.cc`. A test checks that it builds the
executable. Note: the inherited `.gitignore` ignores `Makefile` everywhere; M13 must add
`!/Makefile`.

## 7. Python packaging

`pyproject.toml` with scikit-build-core. `pip install -e . --no-build-isolation` (with `CMAKE_ARGS`
pointing at the Kokkos deps prefix and `OpenMP_ROOT=$(brew --prefix libomp)`) builds the
`_core` extension through the same CMake project (`IMSRG_BUILD_PYTHON=ON`,
`IMSRG_BUILD_TESTS=OFF`). For development, the in-tree build's `python/` directory can be put
on `PYTHONPATH` directly. Python ≥ 3.12 (D29); the extension is a stable-ABI module
(`_core.abi3.so`, CMake ≥ 3.26) and `wheel.py-api = "cp312"` makes `pip wheel` produce one
`cp312-abi3` wheel per platform. Details: `18-python-api.md` §8. Distribution target (owner,
2026-11-15, D41): pip wheels built with cibuildwheel (Linux x86_64/aarch64, macOS arm64), Kokkos
OpenMP inside; GPU/MPI builds from source.

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
  update or reinstall the Command Line Tools so `ld` and the default SDK match. *Cause (checked
  2026-11-15):* every installed CLT package is 26.6, and CLT 26.6 itself ships the pre-release
  `MacOSX27.0.sdk`, so reinstalling 26.6 does not help; no license agreement is involved. The owner
  keeps the workaround and installs the next CLT update when Software Update offers it (Q24).
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

Implemented in M11 phase 1: `IMSRG_BUILD_PYTHON` adds `python/`, which builds `_core` with
`nanobind_add_module(NB_STATIC)` into `<build>/python/imsrg/` (with the pure-Python files copied
next to it), so `PYTHONPATH=<build>/python` works. `tools/ci.sh` runs `pytest tests/python` after
the integration step when the extension exists, and prints a skip message otherwise. Planned
(owner, 2026-11-16, D51): the Python step runs with `IMSRG_RESIDENCY=strict`, so the tests and
examples stay GPU-portable (they call `sync_to_host()` explicitly).

## 9. OLCF Frontier (since 2026-11-26)

The project moved from the Mac to Frontier (project nph123). The Mac sections above stay valid for
the Mac presets; this section is the Frontier recipe. Paths come from `~/.bashrc`: `IMSRG_HUB`,
`IMSRG_WORK` (`$IMSRG_WORK/main` = the integrator's clone, `agents/` = agent clones, `inputs/`,
`golden-generated/`, `logs/`), `IMSRG_REFERENCE_ROOT`, `IMSRG_DEPS_PREFIX` (`$IMSRG_WORK/deps`),
`CCACHE_DIR`, `CCACHE_BASEDIR`.

**Toolchain** (`tools/lib/frontier_env.sh`, `imsrg_frontier_modules cpu|hip`): PrgEnv-gnu 8.7.0
with gcc-native 14.2 through the Cray wrapper `CC` (it links cray-mpich 9.1.0 and cray-libsci
26.03.0 implicitly: FindMPI and FindBLAS/FindLAPACK report "implicitly linked", the BLAS backend
name is "All"), cmake 4.1.5, ninja 1.13.2, ccache 4.12.1, Python 3.14.3 (spack module; the
`cray-python` modules are older). `hip` adds rocm 7.0.2 and uses `hipcc`, with `MPI_HOME=$MPICH_DIR`
(hipcc is not a Cray wrapper). Kokkos 5.3.0 per backend in `$IMSRG_DEPS_PREFIX`: `kokkos-openmp`
and `kokkos-serial` built with `CC`, `kokkos-hip` with hipcc (`Kokkos_ENABLE_HIP`,
`Kokkos_ARCH_AMD_GFX90A`, `Kokkos_ARCH_ZEN3`, Serial host space). Lint tools come pinned from pip
into the venv (`tools/requirements-dev.txt`): cpplint 2.0.2, clang-format 23.1.5 (same as the
Mac's Homebrew one), clang-tidy 23.1.4 (no 23.1.5 wheel; only `readability-identifier-naming` is
used). `ci.sh` and `lint.sh` prefer the venv's binaries.

**Compute node rule.** The login node is shared: editing, git, short commands, job submission.
Configure, builds and tests run on a compute node of a running allocation, normally one named
`imsrg`: `salloc -A nph123 -N 1 -t 02:00:00 -J imsrg --no-shell` (batch partition, 2 h limit for
1-node jobs; renew when it ends). `imsrg_frontier_jobid` finds it (`IMSRG_SLURM_JOBID`, else
`SLURM_JOB_ID`, else the newest running job of the user named `$IMSRG_SLURM_JOBNAME`, default
`imsrg`); `imsrg_compute <cmd>` runs `srun --jobid=<job> --overlap -N1 -n1 -c 56 --cpu-bind=none
--export=ALL <cmd>`. `--overlap` lets several agents share the node (56 cores, 112 hardware threads,
512 GB). Compute nodes have no internet access: anything that downloads (pip, submodule fetches)
runs on the login node.

**Presets.** `frontier-cpu` (CI; Debug `-O1 -g`, Kokkos OpenMP, `CMAKE_CXX_COMPILER=CC`), `frontier-serial` (the same with the debug Serial Kokkos, the Mac `dev-serial` equivalent),
`frontier-cpu-release` (Release, `IMSRG_NATIVE_ARCH`, i.e. `-march=native` = znver3), `frontier-hip`
(Release, hipcc, `IMSRG_BUILD_PYTHON=OFF`; compile check). All set
`IMSRG_TEST_DATA_DIR=$IMSRG_WORK/inputs`, `IMSRG_GENERATED_GOLDEN_DIR=$IMSRG_WORK/golden-generated`
(shared by every clone, so agent clones need no copy), and the MPI test launcher
`MPIEXEC_EXECUTABLE=/usr/bin/srun`, `MPIEXEC_NUMPROC_FLAG=-n`,
`IMSRG_MPIEXEC_PREFLAGS=--overlap --cpu-bind=none -c 4`: inside the compute step, ctest starts each
MPI test as a nested `srun -n <ranks>` step.

**`tools/ci.sh frontier-*`** loads the modules, finds the allocation, and runs configure, build,
unit, integration, MPI and Python tests through `imsrg_compute` (56 build jobs, 14 test slots of 4
threads); layering, clang-format, cpplint and the license check run on the login node.
`frontier-hip` stops after the build. **`tools/bootstrap_deps.sh`** loads the modules itself and
builds `openmp serial hip` by default on Frontier; run it on a compute node
(`srun --jobid=<job> --overlap -n1 -c 56 tools/bootstrap_deps.sh`), which skips the venv, then
`tools/bootstrap_deps.sh --backends ""` on the login node for the venv.

**Pitfalls found during the port.**
* pip on a compute node hangs (no network): the venv is made on the login node.
* `.gitmodules` asks for shallow submodules; local clones ignore `--depth` with a warning. The
  nested `extern/nanobind/ext/robin_map` must be initialized in the integrator's clone
  (`git submodule update --init --recursive`) before agent clones can borrow it.
* The integrator's clone has its own git identity (`git config user.name/user.email`, local);
  `tools/agent_clone.sh` copies it into each agent clone.
