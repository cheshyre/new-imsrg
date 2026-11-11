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

macOS bootstrap (allowed by the user):
```bash
brew install cmake ninja ccache open-mpi libomp     # for the new code
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
* Build both the `imsrg++` executable and the `pyIMSRG` module. Record the exact commands in
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
executable.

## 7. Python packaging

`pyproject.toml` with scikit-build-core. `pip install -e .` (or `pip install .`) builds the
`_core` extension through the same CMake project (`IMSRG_BUILD_PYTHON=ON`,
`IMSRG_BUILD_TESTS=OFF`). For development, the in-tree build's `python/` directory can be put
on `PYTHONPATH` directly. Details: `18-python-api.md`.
