#!/usr/bin/env bash
# tools/ci.sh -- the pre-merge check (design/20 section 4).   Usage: tools/ci.sh [preset]
#
# Runs, timing each step: submodules (init if missing) -> configure (preset, -DIMSRG_WERROR=ON)
# -> build -> unit tests -> integration tests (only if the input data exist) -> MPI tests (only
# if MPI is enabled) -> layering check -> clang-format (our sources only). Tests labeled `slow`
# never run here. The last line is exactly `CI PASS <preset>` or `CI FAIL <preset>: <step>`, and
# the exit status is nonzero on failure. Works from any worktree: it checks the checkout that
# contains this script. Needs no network once the submodules are checked out. The layering and
# format checks see tracked files and new files not yet added (never gitignored ones).
#
# Environment:
#   IMSRG_BUILD_DIR    build tree (default <repo>/build/<preset>); give two CI runs in one checkout
#                      different build trees
#   IMSRG_DEPS_PREFIX  Kokkos installs and venv (default $HOME/.cache/imsrg-deps; see
#                      tools/lib/imsrg_env.sh); without them Kokkos is built in-tree (slow)
#   IMSRG_CI_JOBS      parallel build jobs (default: number of CPUs)
#   OMP_NUM_THREADS    threads per test process (default 4); OMP_PROC_BIND (default false)
set -uo pipefail

preset=${1:-dev-openmp}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/imsrg_env.sh
. "$script_dir/lib/imsrg_env.sh"

summary=""
ci_start=$SECONDS
current_step=setup
finished=0

finish() {  # $1 = PASS or the name of the failed step
  finished=1
  printf '\n[ci] summary for preset %s:\n%b' "$preset" "$summary"
  printf '[ci] total %ds\n' $((SECONDS - ci_start))
  if [ "$1" = PASS ]; then
    echo "CI PASS $preset"
    exit 0
  fi
  echo "CI FAIL $preset: $1"
  exit 1
}

# Any unexpected exit (an unset variable, a signal) still ends with the CI FAIL line.
on_exit() {
  local rc=$?
  if [ "$finished" -ne 1 ]; then
    echo "CI FAIL $preset: $current_step (aborted, status $rc)"
    if [ "$rc" -eq 0 ]; then exit 1; fi
  fi
}
trap on_exit EXIT

# run_step <name> <command...>: runs the command, records its time, and fails CI if it fails.
run_step() {
  local name=$1 t0=$SECONDS rc
  shift
  current_step=$name
  printf '\n[ci] ==== %s ====\n' "$name"
  "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    summary="$summary  FAIL  $(printf '%-18s' "$name") $((SECONDS - t0))s\n"
    finish "$name"
  fi
  summary="$summary  ok    $(printf '%-18s' "$name") $((SECONDS - t0))s\n"
}

skip_step() {  # <name> <reason>
  printf '\n[ci] ==== %s ==== skipped: %s\n' "$1" "$2"
  summary="$summary  skip  $(printf '%-18s' "$1") $2\n"
}

if ! repo_root=$(imsrg_repo_root "$script_dir"); then
  finish "setup (not inside a git checkout)"
fi
build_dir=${IMSRG_BUILD_DIR:-$repo_root/build/$preset}
case $build_dir in
  /*) ;;
  *) build_dir="$PWD/$build_dir" ;;
esac
cd "$repo_root" || finish "setup (cannot cd to $repo_root)"

ncpu=$(imsrg_ncpu)
jobs=${IMSRG_CI_JOBS:-$ncpu}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
threads=${OMP_NUM_THREADS%%,*}  # "4,2" (nested levels) -> 4
case $threads in
  ''|*[!0-9]*|0) threads=4 ;;
esac
test_jobs=$((ncpu / threads))
if [ "$test_jobs" -lt 1 ]; then test_jobs=1; fi
python=python3
if [ -x "$IMSRG_VENV/bin/python" ]; then python="$IMSRG_VENV/bin/python"; fi
imsrg_link_venv "$repo_root"

echo "[ci] preset $preset, checkout $repo_root"
echo "[ci] build dir $build_dir, $jobs build jobs, $test_jobs test jobs x $threads threads"
echo "[ci] deps prefix $IMSRG_DEPS_PREFIX; $(cmake --version | head -n 1); python $python"

# ---- steps -----------------------------------------------------------------------------------
step_submodules() {
  local status
  status=$(git submodule status --recursive) || return 1
  if echo "$status" | grep -q '^U'; then
    echo "submodule merge conflict:"; echo "$status" | grep '^U'; return 1
  fi
  if echo "$status" | grep -q '^-'; then
    echo "initializing missing submodules"
    git submodule update --init --recursive || return 1
  else
    echo "all submodules present"
  fi
  if echo "$status" | grep -q '^+'; then
    echo "warning: submodules not at their pinned commits:"; echo "$status" | grep '^+'
  fi
}

step_configure() {
  cmake --preset "$preset" -B "$build_dir" -DIMSRG_WERROR=ON
}

step_build() {
  cmake --build "$build_dir" --parallel "$jobs"
}

cache_value() {  # <variable>: value from the build tree's CMakeCache.txt
  sed -n "s/^$1:[A-Z]*=//p" "$build_dir/CMakeCache.txt" | head -n 1
}

step_unit() {
  ctest --test-dir "$build_dir" -L '^unit$' -LE '^slow$' --output-on-failure --no-tests=error \
    -j "$test_jobs"
}

step_integration() {
  ctest --test-dir "$build_dir" -L '^integration$' -LE '^(slow|mpi)$' --output-on-failure \
    --no-tests=ignore -j "$test_jobs" || return 1
  if ls tests/integration/test_*.py >/dev/null 2>&1; then
    IMSRG_BUILD_DIR="$build_dir" "$python" -m pytest -q tests/integration -m "not slow"
  fi
}

step_mpi() {
  ctest --test-dir "$build_dir" -L '^mpi$' -LE '^slow$' --output-on-failure --no-tests=error \
    -j "$test_jobs"
}

step_python() {
  PYTHONPATH="$build_dir/python" "$python" -m pytest -q tests/python
}

step_layering() {
  "$python" tools/check_layering.py --repo-root "$repo_root" \
    --module-deps "$build_dir/imsrg_module_deps.json" &&
    "$python" tools/check_layering.py --repo-root "$repo_root" --self-test
}

step_format() {
  if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format not found (brew install clang-format)"; return 1
  fi
  echo "$(clang-format --version)"
  # Tracked files plus new files not yet added (but never ignored ones), that still exist.
  local files f
  files=$(git ls-files --cached --others --exclude-standard -- src apps tests python tools/bench \
    | grep -E '\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx|inl|ipp|tpp)$' | grep -v '^tests/golden/' \
    | sort -u | while IFS= read -r f; do if [ -f "$f" ]; then echo "$f"; fi; done)
  if [ -z "$files" ]; then echo "no C++ sources"; return 0; fi
  echo "checking $(echo "$files" | wc -l | tr -d ' ') files"
  echo "$files" | tr '\n' '\0' | xargs -0 clang-format --dry-run --Werror
}

run_step submodules step_submodules
if ! ls "$IMSRG_DEPS_PREFIX"/kokkos-*/lib*/cmake/Kokkos/KokkosConfig.cmake >/dev/null 2>&1; then
  echo "[ci] note: no Kokkos install in $IMSRG_DEPS_PREFIX; CMake builds extern/kokkos in-tree" \
       "(run tools/bootstrap_deps.sh once to speed this up)"
fi
run_step configure step_configure
run_step build step_build
run_step "unit tests" step_unit

data_dir=$(cache_value IMSRG_TEST_DATA_DIR)
if [ -n "$data_dir" ] && [ -f "$data_dir/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz" ] \
   && [ -f "$data_dir/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin" ]; then
  run_step "integration tests" step_integration
else
  skip_step "integration tests" "input data not found in '${data_dir:-unset}'"
fi

case $(cache_value IMSRG_ENABLE_MPI) in
  ON|on|TRUE|true|YES|yes|1|Y|y) run_step "mpi tests" step_mpi ;;
  *) skip_step "mpi tests" "IMSRG_ENABLE_MPI is off" ;;
esac

case $(cache_value IMSRG_BUILD_PYTHON) in
  ON|on|TRUE|true|YES|yes|1|Y|y)
    if ls "$build_dir"/python/imsrg/_core*.so >/dev/null 2>&1; then
      run_step "python tests" step_python
    else
      skip_step "python tests" "extension imsrg._core not found in $build_dir/python/imsrg"
    fi ;;
  *) skip_step "python tests" "IMSRG_BUILD_PYTHON is off" ;;
esac

run_step layering step_layering
run_step clang-format step_format
finish PASS
