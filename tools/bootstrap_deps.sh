#!/usr/bin/env bash
# tools/bootstrap_deps.sh -- build the per-machine development dependencies once.
#
# Usage: tools/bootstrap_deps.sh [--backends "openmp serial"] [--no-venv] [--force] [--jobs N]
#
# Installs into ${IMSRG_DEPS_PREFIX:-$HOME/.cache/imsrg-deps}, shared by all git worktrees:
#   kokkos-openmp/  Kokkos from extern/kokkos with the OpenMP (+ Serial) backend, Release
#   kokkos-serial/  Kokkos with the Serial backend, RelWithDebInfo, debug checks + bounds checking
#   venv/           Python venv ($IMSRG_VENV) with tools/requirements-dev.txt
# and links <this worktree>/.venv -> $IMSRG_VENV (gitignored).
#
# Idempotent: a backend is rebuilt only if its stamp (Kokkos commit, compiler, options) changed;
# the venv only if a package is missing. Safe to run from several worktrees at once: an mkdir lock
# on the prefix serializes runs, and installs are staged and then moved into place, so a concurrent
# configure never sees a half-written Kokkos. Build trees live in $IMSRG_DEPS_PREFIX/build/.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/imsrg_env.sh
. "$script_dir/lib/imsrg_env.sh"
repo_root=$(imsrg_repo_root "$script_dir")

backends="openmp serial"
do_venv=1
force=0
jobs=$(imsrg_ncpu)
while [ $# -gt 0 ]; do
  case $1 in
    --backends) backends=$2; shift 2 ;;
    --no-venv) do_venv=0; shift ;;
    --force) force=1; shift ;;
    --jobs|-j) jobs=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "bootstrap_deps: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

log() { echo "bootstrap_deps: $*"; }
mkdir -p "$IMSRG_DEPS_PREFIX/build"

# ---- lock ------------------------------------------------------------------------------------
# mkdir is atomic. The owner writes its pid and host into the lock directory right away. A lock is
# stale if its owner (on this host) is dead, or if it has no pid file and is older than a minute
# (the owner died between mkdir and writing the pid). Only one waiter at a time may break a stale
# lock (it holds $lock_dir.break and re-checks inside), so two waiters cannot both break it and
# then remove each other's fresh lock. Locks of other hosts are never broken.
lock_dir="$IMSRG_DEPS_PREFIX/.bootstrap.lock"
this_host=$(hostname)

lock_is_stale() {
  local pid host
  pid=$(cat "$lock_dir/pid" 2>/dev/null || true)
  host=$(cat "$lock_dir/host" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    [ "$host" = "$this_host" ] && ! kill -0 "$pid" 2>/dev/null
  else
    [ -n "$(find "$lock_dir" -maxdepth 0 -mmin +1 2>/dev/null)" ]
  fi
}

waited=0
while ! mkdir "$lock_dir" 2>/dev/null; do
  if [ -n "$(find "$lock_dir.break" -maxdepth 0 -mmin +1 2>/dev/null)" ]; then
    rmdir "$lock_dir.break" 2>/dev/null || true  # a breaker died; its marker is empty
  fi
  if lock_is_stale && mkdir "$lock_dir.break" 2>/dev/null; then
    if lock_is_stale; then
      log "removing stale lock $lock_dir (pid $(cat "$lock_dir/pid" 2>/dev/null || echo none))"
      rm -rf "$lock_dir"
    fi
    rmdir "$lock_dir.break"
    continue
  fi
  if [ $((waited % 60)) -eq 0 ]; then
    log "waiting for the lock $lock_dir (held by pid $(cat "$lock_dir/pid" 2>/dev/null ||
      echo '?') on $(cat "$lock_dir/host" 2>/dev/null || echo '?'))"
  fi
  if [ "$waited" -ge 3600 ]; then
    log "gave up waiting for the lock after ${waited}s; remove $lock_dir if it is stale" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
trap 'rm -rf "$lock_dir"' EXIT
echo $$ > "$lock_dir/pid"
echo "$this_host" > "$lock_dir/host"

# ---- Kokkos ----------------------------------------------------------------------------------
sm_status=$(git -C "$repo_root" submodule status extern/kokkos)
case $sm_status in
  -*)
    log "initializing extern/kokkos"
    git -C "$repo_root" submodule update --init extern/kokkos
    sm_status=$(git -C "$repo_root" submodule status extern/kokkos) ;;
  +*) log "warning: extern/kokkos is checked out at a commit other than the pinned one" ;;
esac
kokkos_commit=$(echo "$sm_status" | sed 's/^[-+ U]//' | awk '{print $1}')

cxx=${CXX:-c++}
cxx_path=$(command -v "$cxx")
cxx_version=$("$cxx" --version 2>/dev/null | head -n 1)
generator_args=""
if command -v ninja >/dev/null 2>&1; then generator_args="-G Ninja"; fi
launcher_args=""
if command -v ccache >/dev/null 2>&1; then launcher_args="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"; fi
omp_root=""
if command -v brew >/dev/null 2>&1; then
  omp_root=$(brew --prefix libomp 2>/dev/null || true)
  if [ ! -d "$omp_root" ]; then omp_root=""; fi
fi

# Prints the Kokkos CMake options for backend $1, one per line (part of the stamp).
kokkos_options() {
  echo "-DCMAKE_CXX_STANDARD=20"
  echo "-DCMAKE_CXX_EXTENSIONS=OFF"
  echo "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  echo "-DKokkos_ENABLE_SERIAL=ON"
  echo "-DKokkos_ENABLE_TESTS=OFF"
  echo "-DKokkos_ENABLE_EXAMPLES=OFF"
  echo "-DKokkos_ENABLE_BENCHMARKS=OFF"
  echo "-DKokkos_ENABLE_DEPRECATED_CODE_5=OFF"
  case $1 in
    openmp)
      echo "-DCMAKE_BUILD_TYPE=Release"
      echo "-DKokkos_ENABLE_OPENMP=ON"
      if [ -n "$omp_root" ]; then echo "-DOpenMP_ROOT=$omp_root"; fi ;;
    serial)
      echo "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
      echo "-DKokkos_ENABLE_DEBUG=ON"
      echo "-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON" ;;
    *) echo "bootstrap_deps: unsupported backend '$1' (openmp, serial)" >&2; return 1 ;;
  esac
}

build_kokkos() {
  local backend=$1
  local install_dir="$IMSRG_DEPS_PREFIX/kokkos-$backend"
  local build_dir="$IMSRG_DEPS_PREFIX/build/kokkos-$backend"
  local stage_dir="$IMSRG_DEPS_PREFIX/build/kokkos-$backend-stage"
  local options stamp t0
  options=$(kokkos_options "$backend")
  stamp=$(printf 'kokkos_commit=%s\nbackend=%s\ncxx=%s\ncxx_version=%s\noptions:\n%s\n' \
    "$kokkos_commit" "$backend" "$cxx_path" "$cxx_version" "$options")
  if [ "$force" -eq 0 ] && [ -f "$install_dir/.imsrg-stamp" ] \
     && [ "$(cat "$install_dir/.imsrg-stamp")" = "$stamp" ] \
     && ls "$install_dir"/lib*/cmake/Kokkos/KokkosConfig.cmake >/dev/null 2>&1; then
    log "kokkos-$backend is up to date (commit ${kokkos_commit:0:10})"
    return 0
  fi
  log "building kokkos-$backend (commit ${kokkos_commit:0:10}) with $cxx_version"
  t0=$(date +%s)
  # shellcheck disable=SC2086  # option lists are intentionally word-split (no spaces inside)
  cmake -S "$repo_root/extern/kokkos" -B "$build_dir" $generator_args \
    -C "$repo_root/cmake/ImsrgAppleSdk.cmake" \
    -DCMAKE_INSTALL_PREFIX="$install_dir" $launcher_args $options
  cmake --build "$build_dir" --parallel "$jobs"
  rm -rf "$stage_dir"
  DESTDIR="$stage_dir" cmake --install "$build_dir" >/dev/null
  printf '%s\n' "$stamp" > "$stage_dir$install_dir/.imsrg-stamp"
  # Two renames: a concurrent configure sees the old install, the new one, or (for an instant)
  # none, never a partial one. If the second rename fails, the old install is put back.
  rm -rf "$install_dir.old"
  if [ -e "$install_dir" ]; then mv "$install_dir" "$install_dir.old"; fi
  if ! mv "$stage_dir$install_dir" "$install_dir"; then
    if [ -e "$install_dir.old" ]; then mv "$install_dir.old" "$install_dir"; fi
    log "could not move the new kokkos-$backend into place" >&2
    return 1
  fi
  rm -rf "$install_dir.old" "$stage_dir"
  log "installed kokkos-$backend into $install_dir in $(( $(date +%s) - t0 ))s"
}

for backend in $backends; do
  build_kokkos "$backend"
done

# ---- Python venv -----------------------------------------------------------------------------
if [ "$do_venv" -eq 1 ]; then
  check_imports='import numpy, pytest, sympy, scikit_build_core'
  if [ -x "$IMSRG_VENV/bin/python" ] && "$IMSRG_VENV/bin/python" -c "$check_imports" 2>/dev/null
  then
    log "venv $IMSRG_VENV is up to date"
  else
    python=${IMSRG_PYTHON:-python3}
    if [ ! -x "$IMSRG_VENV/bin/python" ]; then
      log "creating venv $IMSRG_VENV with $("$python" --version 2>&1)"
      "$python" -m venv "$IMSRG_VENV"
    fi
    "$IMSRG_VENV/bin/python" -m pip install --quiet --upgrade pip
    "$IMSRG_VENV/bin/python" -m pip install --quiet --only-binary=numpy \
      -r "$repo_root/tools/requirements-dev.txt"
    "$IMSRG_VENV/bin/python" -c "$check_imports"
    log "venv ready: $("$IMSRG_VENV/bin/python" --version 2>&1)"
  fi
  imsrg_link_venv "$repo_root"
fi

log "done (prefix $IMSRG_DEPS_PREFIX)"
