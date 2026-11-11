# tools/lib/imsrg_env.sh -- shared settings for tools/*.sh. Source it; do not execute it.
# Written for bash 3.2 (macOS /bin/bash): no mapfile, no associative arrays.
#
# The single rule for per-machine state shared by all git worktrees (AGENTS.md, design/20 section 3):
#   IMSRG_DEPS_PREFIX  default $HOME/.cache/imsrg-deps  Kokkos installs: kokkos-<backend>/
#   IMSRG_VENV         default $IMSRG_DEPS_PREFIX/venv  the Python venv (numpy, pytest, sympy, ...)
# Every worktree gets a gitignored convenience symlink .venv -> $IMSRG_VENV (made by
# tools/bootstrap_deps.sh and tools/ci.sh), so `.venv/bin/pytest` works in any worktree.

: "${IMSRG_DEPS_PREFIX:=$HOME/.cache/imsrg-deps}"
: "${IMSRG_VENV:=$IMSRG_DEPS_PREFIX/venv}"
export IMSRG_DEPS_PREFIX IMSRG_VENV

# Prints the root of the checkout (worktree) that contains directory $1.
imsrg_repo_root() {
  git -C "$1" rev-parse --show-toplevel
}

# Prints the number of logical CPUs.
imsrg_ncpu() {
  sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4
}

# Creates <repo>/.venv -> $IMSRG_VENV if the shared venv exists and <repo>/.venv does not.
imsrg_link_venv() {
  if [ -x "$IMSRG_VENV/bin/python" ] && [ ! -e "$1/.venv" ] && [ ! -L "$1/.venv" ]; then
    ln -s "$IMSRG_VENV" "$1/.venv"
  fi
}
