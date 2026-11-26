#!/usr/bin/env bash
# Reproducible out-of-tree build of the owner's imsrg++ fork at bleeding_edge 79e9330 (the code
# of the archived 27Al NREFT responses, design/50-al27-responses.md capability C1) and of its
# golden-data harness `forkharness` (tools/reference/fork/).
#
# The fork sources are READ-ONLY ($IMSRG_REFERENCE_ROOT/imsrg-fork-79e9330/src79/src, extracted
# with `git archive` from fork.git). This script copies them (rsync --delete), verifies the copy's
# tree checksum against the original, applies the patches below and builds in directories of its
# own under $REF_BUILD (tools/reference/ref_env.sh). It never builds inside the original.
#
# Variants (each a separate copy and build directory; one never touches the other):
#   plain    79e9330 + patches/fork-79e9330-cxx14.patch (CMAKE_CXX_STANDARD 11 -> 14, the next
#            commit 25c88ede on bleeding_edge; the provenance README section 4 names it the most
#            likely content of the "+" in the archived build string). This is the reference for
#            the archived calculations.
#            copy  $REF_BUILD/fork-79e9330/src      builds  $REF_BUILD/fork-79e9330/build{,-harness}
#   phipfix  plain + patches/fork-79e9330-phip-fix.patch: the Hermiticity part of the fork's
#            phip_fix branch (81cf5ea1): Phip loses its (wrong) anti-Hermitian flag and Phitp is
#            flagged Hermitian. Not master's anti-Hermitian Delta flag (owner: a bug). Needed for
#            any evolved tilde-Phi' (reference/06 section 2.4); bare one-body tables are identical.
#            copy  $REF_BUILD/fork-79e9330-phipfix/src   builds  .../build{,-harness}
#
# Version files (read by tools/reference/gen_golden_nreft.sh and written into every golden):
#   $REF_BUILD/FORK_79E9330_SRC_SHA256          "<tree sha256 of src79/src>+cxx14:<patch sha16>"
#   $REF_BUILD/FORK_79E9330_PHIPFIX_SRC_SHA256  "...+cxx14:<sha16>+phipfix:<sha16>"
# The expected tree checksum of the pristine 79e9330 src/ is committed in
# tools/reference/FORK_79E9330_SRC_SHA256; the script refuses to build from a different tree.
#
# Usage:  tools/reference/build_fork.sh [all|plain|phipfix|checksum]     (default: all)
#         (all = plain and phipfix, each the imsrg++ executable plus forkharness)
#
# Frontier: run it on a compute node (`imsrg_compute tools/reference/build_fork.sh`), like
# build_imsrg_devel.sh: PrgEnv-gnu through the Cray CC wrapper, cray-libsci as BLAS, modules
# gsl/2.8 and boost/1.88.0. Environment: BUILD_JOBS (default: nproc), FRESH=1 (cmake --fresh),
# IMSRG_FORK_SOURCE (default $REF_ROOT/imsrg-fork-79e9330/src79).
set -euo pipefail

stage="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/ref_env.sh"   # REF_ROOT, REF_BUILD
JOBS="${BUILD_JOBS:-$(nproc)}"
FORK_SRC="${IMSRG_FORK_SOURCE:-${REF_ROOT}/imsrg-fork-79e9330/src79}"
EXPECTED_SHA="$(cat "${HERE}/FORK_79E9330_SRC_SHA256")"
P_CXX14="${HERE}/patches/fork-79e9330-cxx14.patch"
P_PHIP="${HERE}/patches/fork-79e9330-phip-fix.patch"

src_checksum() {   # sha256 over all files of <dir>/src (relative paths, sorted)
  (cd "$1" && find src -type f ! -name '.DS_Store' -print0 | LC_ALL=C sort -z \
     | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
}
psha() { shasum -a 256 "$1" | cut -c1-16; }

if [[ "${stage}" == checksum ]]; then src_checksum "${FORK_SRC}"; exit 0; fi

if [[ "${LMOD_SYSTEM_NAME:-}" == frontier ]]; then
  # shellcheck disable=SC1091
  . "${HERE}/../lib/frontier_env.sh"
  imsrg_frontier_modules cpu
  module load gsl/2.8 boost/1.88.0 >/dev/null 2>&1 || {
    echo "ERROR: module load gsl/2.8 boost/1.88.0 failed" >&2; exit 1; }
  export CRAYPE_LINK_TYPE=dynamic
  tool_args=(-DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DBLA_VENDOR=All
             -DBLAS_LIBRARIES="sci_gnu_mp" "-DCMAKE_CXX_FLAGS=-include cstdint"
             -DCMAKE_POLICY_DEFAULT_CMP0144=NEW -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
else
  tool_args=()   # only Frontier is supported for the fork build (2026-11-26)
fi
common_args=(${FRESH:+--fresh} -G Ninja -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DIMSRG_BUILD_PYTHON_BINDINGS=OFF
             "${tool_args[@]}")
if command -v ccache >/dev/null 2>&1; then
  common_args+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

# make_copy <dest dir> <version string> <patch>...: pristine copy + patches, idempotent.
make_copy() {
  local dest="$1" version="$2"; shift 2
  local stamp="${dest}/.patches-applied"
  if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${version}" ]]; then
    echo "copy ${dest} up to date: ${version}"; return 0
  fi
  local a; a="$(src_checksum "${FORK_SRC}")"
  if [[ "$a" != "${EXPECTED_SHA}" ]]; then
    echo "ERROR: ${FORK_SRC}/src has checksum $a, expected ${EXPECTED_SHA}" >&2; exit 1
  fi
  mkdir -p "${dest}"
  rsync -a --delete --exclude '.DS_Store' "${FORK_SRC}/src/" "${dest}/src/"
  chmod -R u+w "${dest}/src"
  local b; b="$(src_checksum "${dest}")"
  [[ "$a" == "$b" ]] || { echo "ERROR: copy differs from the original ($a vs $b)" >&2; exit 1; }
  local p
  for p in "$@"; do (cd "${dest}" && patch -p1 --no-backup-if-mismatch < "$p"); done
  echo "${version}" > "${stamp}"
}

build_variant() {   # build_variant <name> <sha file> <patch>...
  local name="$1" shafile="$2"; shift 2
  local dest="${REF_BUILD}/${name}" version="${EXPECTED_SHA}+cxx14:$(psha "${P_CXX14}")"
  [[ $# -gt 1 ]] && version="${version}+phipfix:$(psha "${P_PHIP}")"
  make_copy "${dest}" "${version}" "$@"
  echo "${version}" > "${REF_BUILD}/${shafile}"
  echo "fork version: ${version}"
  # The fork's src/CMakeLists.txt generates version.cc in its binary dir (no "../build" redirect
  # as in devel). git_version.cmake runs git in the build dir, which is no repository: the build
  # string becomes "N/A_N/A".
  cmake -S "${dest}/src" -B "${dest}/build" "${common_args[@]}"
  cmake --build "${dest}/build" --target imsrg++ -j "${JOBS}"
  cmake -S "${HERE}/fork" -B "${dest}/build-harness" "${common_args[@]}" \
        -DIMSRG_FORK_SRC="${dest}/src"
  cmake --build "${dest}/build-harness" --target forkharness -j "${JOBS}"
  ls -l "${dest}/build/imsrg++" "${dest}/build-harness/forkharness"
}

case "${stage}" in
  all)     build_variant fork-79e9330 FORK_79E9330_SRC_SHA256 "${P_CXX14}"
           build_variant fork-79e9330-phipfix FORK_79E9330_PHIPFIX_SRC_SHA256 \
                         "${P_CXX14}" "${P_PHIP}" ;;
  plain)   build_variant fork-79e9330 FORK_79E9330_SRC_SHA256 "${P_CXX14}" ;;
  phipfix) build_variant fork-79e9330-phipfix FORK_79E9330_PHIPFIX_SRC_SHA256 \
                         "${P_CXX14}" "${P_PHIP}" ;;
  *) echo "usage: $0 [all|plain|phipfix|checksum]" >&2; exit 2 ;;
esac
