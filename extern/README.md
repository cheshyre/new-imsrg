# extern/ — third-party dependencies (git submodules)

Every dependency is a git submodule pinned to a **release tag**. `.gitmodules` sets
`shallow = true`, so each checkout (every git worktree checks out its own submodules) fetches
only the pinned commit. `tools/ci.sh` initializes missing submodules itself; by hand:

```bash
git submodule update --init --recursive    # needs network only when a submodule is missing
```

| Submodule | Upstream | Tag | Commit | Commit date | License | Used by |
|-----------|----------|-----|--------|-------------|---------|---------|
| `extern/kokkos` | https://github.com/kokkos/kokkos | `5.3.0` | `dc826b0fad122b57fee60223e86154aecb8e60f4` | 2026-11-10 | Apache-2.0 WITH LLVM-exception | `imsrg::exec` (built once per backend by `tools/bootstrap_deps.sh`; in-tree fallback) |
| `extern/googletest` | https://github.com/google/googletest | `v1.18.0` | `063de7e9578f82b369302001269680b4b1553359` | 2026-08-10 | BSD-3-Clause | tests only |
| `extern/fmt` | https://github.com/fmtlib/fmt | `12.2.1` | `208219daff58f2a7438f6931cc2ddc358acf33ef` | 2026-11-07 | MIT | `imsrg::base` logging (M2) |
| `extern/CLI11` | https://github.com/CLIUtils/CLI11 | `v2.7.2` | `cbd58a3696887b34c70949aef21a71735a0c2ad5` | 2026-08-02 | BSD-3-Clause | `apps/imsrg` only (M8) |
| `extern/tomlplusplus` | https://github.com/marzer/tomlplusplus | `v3.4.0` | `30172438cee64926dc41fdd9c11fb3ba5b2ba9de` | 2023-10-13 | MIT | `imsrg::driver` config (M8) |
| `extern/nanobind` | https://github.com/wjakob/nanobind | `v3.1.0` | `68480a9e6883bb1bf65e1925423321e2a89d07f9` | 2026-09-18 | BSD-3-Clause | Python bindings (M11) |
| `extern/nanobind/ext/robin_map` | https://github.com/Tessil/robin-map | `v1.4.0` (pinned by nanobind) | `4ec1bf19c6a96125ea22062f38c2cf5b958e448e` | 2025-03-30 | MIT | nanobind |

Pinned on 2026-11-11 (each was the latest release tag of its project on that day; the commit is
the peeled tag commit). Commit dates are the dates of the tagged commits.

## Kokkos toolchain requirements (checked for 5.3.0)

From `extern/kokkos/cmake/kokkos_compiler_id.cmake`: GCC ≥ 10.4, Clang (CPU) ≥ 14, Clang (CUDA)
≥ 15, NVCC ≥ 12.2, HIPCC ≥ 6.2, CMake ≥ 3.22, C++20. The owner baseline (D20: GCC ≥ 11,
CUDA ≥ 12, ROCm ≥ 6) is met for GCC; CUDA and ROCm need at least the 12.2 and 6.2 minor
releases (recorded in `design/QUESTIONS.md`).

## Updating a pin

1. Find the tag's commit: `git ls-remote <url> 'refs/tags/<tag>*'` (use the `^{}` line for
   annotated tags).
2. Point the gitlink at it and fetch only that commit:
   `git update-index --cacheinfo 160000,<commit>,extern/<name>` then
   `git submodule update --init --recursive extern/<name>`.
3. Update this table, rebuild the Kokkos installs if Kokkos changed
   (`tools/bootstrap_deps.sh` detects the new commit and rebuilds), run `tools/ci.sh`, and commit
   with the old and new tags in the message.

Do not edit code inside `extern/`. Patches to a dependency go upstream, or into
`cmake/patches/` with a note here.
