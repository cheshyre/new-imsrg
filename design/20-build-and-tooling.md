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
| Python ≥ 3.10 + numpy, pytest, sympy, scikit-build-core | project venv `.venv` (`python3 -m venv .venv`) | | sympy provides exact Wigner symbols for the angmom tests. |

Homebrew versions on 2026-11-11 (checked with `brew info`, nothing installed yet): cmake 4.4.5,
ninja 1.13.2, ccache 4.14.3, open-mpi 5.0.11, libomp 23.1.5, gsl 2.8, boost 1.92.0, openblas
0.3.34, llvm 23.1.5, gcc 16.2.0, **kokkos 5.2.2**. Pin the Kokkos submodule to the latest 5.x
release tag after checking its minimum toolchain against the owner's baseline (GCC ≥ 11,
CUDA ≥ 12, ROCm ≥ 6; D20). If Kokkos 5 needs newer CUDA/ROCm minor versions than the baseline,
record that in QUESTIONS.md and keep 5.x anyway, unless 4.x support turns out to be essential.
Do not use Homebrew's kokkos package for development builds (its backend configuration is not
under our control); build from the submodule via `tools/bootstrap_deps.sh`.

macOS bootstrap (allowed by the user):
```bash
brew install cmake ninja ccache open-mpi libomp clang-format   # for the new code
brew install gsl boost openblas                     # additionally needed to build imsrg-devel
python3 -m venv .venv && .venv/bin/pip install numpy pytest sympy scikit-build-core
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
* `src/imsrg/<module>/CMakeLists.txt` defines `imsrg_<module>` via one helper function:
  `imsrg_add_module(NAME operator SOURCES ... DEPS basis linalg angmom exec base)`. The helper
  sets include dirs, warnings, and the alias `imsrg::operator`, and registers the module's
  dependencies for the layering check.
* Static libraries by default (fast linking, no PIC overhead). With `IMSRG_BUILD_PYTHON`,
  position-independent code is turned on for the libraries linked into `_core`.
* `apps/imsrg`, `python/`, `tests/` each add their own targets. Every unit-test executable is
  registered with `gtest_discover_tests` and labeled (`unit`, `integration`, `mpi`, `slow`,
  `reference`).

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
`${IMSRG_DEPS_PREFIX:-$HOME/.cache/imsrg-deps}/kokkos-<backend>`. The presets point
`Kokkos_ROOT` there. If `find_package(Kokkos)` fails, CMake falls back to
`add_subdirectory(extern/kokkos)`, so a fresh clone still builds with one command.

## 4. Scripts

* `tools/ci.sh [preset]`: configure, build, `ctest -L unit`, then the integration tests if the
  data are present, the layering check, and `clang-format --dry-run`. This is the single command
  agents run before every merge. It must print a one-line PASS/FAIL summary.
* `tools/check_layering.py`: enforces the module DAG (`10-architecture.md` §2).
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

## 6. Plain Makefile (later, M13)

A single `Makefile` at the root for users without CMake. It assumes a pre-installed Kokkos
(`KOKKOS_PATH`, using Kokkos' `Makefile.kokkos` support), OpenMP backend, system BLAS/LAPACK
and zlib, no MPI by default (`USE_MPI=1` switches to `mpicxx`), and no Python. It lists sources
by globbing `src/imsrg/*/*.cpp src/imsrg/*/kernels/*.cpp`. A test checks that it builds the
executable.

## 7. Python packaging

`pyproject.toml` with scikit-build-core. `pip install -e . --no-build-isolation` (with `CMAKE_ARGS`
pointing at the Kokkos deps prefix and `OpenMP_ROOT=$(brew --prefix libomp)`) builds the
`_core` extension through the same CMake project (`IMSRG_BUILD_PYTHON=ON`,
`IMSRG_BUILD_TESTS=OFF`). For development, the in-tree build's `python/` directory can be put
on `PYTHONPATH` directly. Details: `18-python-api.md`.
