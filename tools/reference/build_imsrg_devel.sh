#!/usr/bin/env bash
# Reproducible out-of-tree build of the reference code imsrg-devel (imsrg++) and of the
# golden-data harness `refharness` (tools/reference/refharness.cpp).
#
# Design: design/20-build-and-tooling.md section 5, design/30-testing-and-validation.md section 3.
#
# The reference source is READ-ONLY. This script never builds inside it: it first makes a
# pristine copy of `src/` (rsync --delete, no patches) under the reference root and verifies
# that the copy has the same checksum as the original. Every build directory lives under the
# reference root. The old `install` target (which writes into $HOME) is never run.
#
# Usage:
#   tools/reference/build_imsrg_devel.sh [all|imsrg|harness|checksum]   (default: all)
#
#   imsrg     configure + build the old library (libIMSRG.so) and the imsrg++ executable,
#             with the old src/CMakeLists.txt as the CMake root (build dir: $REF_ROOT/build)
#   harness   configure + build refharness through tools/reference/CMakeLists.txt, which pulls
#             the old src in via add_subdirectory (build dir: $REF_ROOT/harness-build)
#   checksum  print the reference version string (sha256 over src/, the tree has no .git)
#
# Environment overrides:
#   IMSRG_REFERENCE_SOURCE  reference checkout   (default /Users/cheshyre/Downloads/new-imsrg/imsrg-devel)
#   IMSRG_REFERENCE_ROOT    build root           (default /Users/cheshyre/Downloads/new-imsrg-reference)
#   BUILD_JOBS              parallel jobs        (default: number of cores)
#   FRESH=1                 reconfigure from scratch (cmake --fresh), e.g. after a toolchain change
#   IMSRG_REFERENCE_SDK     macOS SDK path (default: probed, see pick_sdk below)
#
# Toolchain that worked (2026-11-11, macOS 26 arm64): Apple clang 21, cmake 4.4.5, ninja,
# Homebrew libomp (OpenMP via OpenMP_ROOT), openblas, gsl, boost 1.92 (headers), system zlib.
set -euo pipefail

stage="${1:-all}"
REF_SRC="${IMSRG_REFERENCE_SOURCE:-/Users/cheshyre/Downloads/new-imsrg/imsrg-devel}"
REF_ROOT="${IMSRG_REFERENCE_ROOT:-/Users/cheshyre/Downloads/new-imsrg-reference}"
JOBS="${BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREW="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

COPY="${REF_ROOT}/imsrg-devel-src"   # pristine copy; COPY/src is the CMake source dir

# sha256 over all files of <dir>/src (relative paths, sorted, .DS_Store ignored).
src_checksum() {
  local base="$1"
  (cd "$base" && find src -type f ! -name '.DS_Store' -print0 | LC_ALL=C sort -z \
     | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
}

make_copy() {
  mkdir -p "${COPY}"
  rsync -a --delete --exclude '.DS_Store' "${REF_SRC}/src/" "${COPY}/src/"
  cp -p "${REF_SRC}/LICENSE" "${REF_SRC}/README.md" "${COPY}/" 2>/dev/null || true
  local a b
  a="$(src_checksum "${REF_SRC}")"
  b="$(src_checksum "${COPY}")"
  if [[ "$a" != "$b" ]]; then
    echo "ERROR: copy of the reference src/ differs from the original ($a vs $b)" >&2
    exit 1
  fi
  echo "reference src/ sha256 (tree checksum): $a"
  echo "$a" > "${REF_ROOT}/REFERENCE_SRC_SHA256"
}

# Common cache settings for both builds.
#  * CMAKE_POLICY_VERSION_MINIMUM=3.5: cmake 4.x refuses very old cmake_minimum_required()
#    calls in vendored projects; harmless otherwise.
#  * OpenMP_ROOT: Apple clang has no bundled OpenMP runtime; CMake's FindOpenMP finds brew's
#    libomp from this hint and adds -Xclang -fopenmp.
#  * BLA_VENDOR=OpenBLAS is hard-coded in the old CMakeLists; CMAKE_PREFIX_PATH finds brew's.
common_args=(
  ${FRESH:+--fresh}
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_C_COMPILER=clang
  -DCMAKE_CXX_COMPILER=clang++
  -DOpenMP_ROOT="${BREW}/opt/libomp"
  -DCMAKE_PREFIX_PATH="${BREW}/opt/openblas;${BREW}/opt/gsl;${BREW}/opt/boost;${BREW}/opt/libomp;${BREW}"
  -DIMSRG_BUILD_PYTHON_BINDINGS=OFF
)
if command -v ccache >/dev/null 2>&1; then
  common_args+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

# macOS SDK: on the dev machine (2026-11-11) xcrun's default SDK is MacOSX27.0.sdk, whose
# libSystem.tbd lists an architecture ("arm64e.x1-macos") that the installed linker (ld-1267)
# rejects ("tapi error: malformed file"), so even a hello-world fails to link. Probe the default
# SDK and fall back to the newest SDK that links. Override with IMSRG_REFERENCE_SDK=<path>.
pick_sdk() {
  if [[ -n "${IMSRG_REFERENCE_SDK:-}" ]]; then echo "${IMSRG_REFERENCE_SDK}"; return; fi
  [[ "$(uname -s)" == "Darwin" ]] || return 0
  local probe; probe="$(mktemp -d)"
  printf 'int main(void){return 0;}\n' > "${probe}/p.c"
  if clang "${probe}/p.c" -o "${probe}/p" >/dev/null 2>&1; then rm -rf "${probe}"; return 0; fi
  local sdk
  for sdk in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX[0-9]*.sdk 2>/dev/null | sort -rV); do
    if clang -isysroot "${sdk}" "${probe}/p.c" -o "${probe}/p" >/dev/null 2>&1; then
      rm -rf "${probe}"; echo "${sdk}"; return 0
    fi
  done
  rm -rf "${probe}"
}
SDK="$(pick_sdk)"
if [[ -n "${SDK}" ]]; then
  echo "using macOS SDK ${SDK}"
  common_args+=(-DCMAKE_OSX_SYSROOT="${SDK}")
fi

build_imsrg() {
  make_copy
  # The old src/CMakeLists.txt overrides CMAKE_CURRENT_BINARY_DIR to "../build" and generates
  # version.cc there, so the binary dir must itself be called "build" (then ../build == .).
  local bdir="${REF_ROOT}/build"
  cmake -S "${COPY}/src" -B "${bdir}" "${common_args[@]}"
  cmake --build "${bdir}" --target imsrg++ IMSRG -j "${JOBS}"
  ls -l "${bdir}/imsrg++" "${bdir}/libIMSRG.so"
}

build_harness() {
  make_copy
  local bdir="${REF_ROOT}/harness-build"
  cmake -S "${HERE}" -B "${bdir}" "${common_args[@]}" \
        -DIMSRG_REFERENCE_SRC="${COPY}/src"
  cmake --build "${bdir}" --target refharness -j "${JOBS}"
  ls -l "${bdir}/refharness"
}

case "${stage}" in
  all)      build_imsrg; build_harness ;;
  imsrg)    build_imsrg ;;
  harness)  build_harness ;;
  checksum) echo "$(src_checksum "${REF_SRC}")" ;;
  *) echo "usage: $0 [all|imsrg|harness|checksum]" >&2; exit 2 ;;
esac
