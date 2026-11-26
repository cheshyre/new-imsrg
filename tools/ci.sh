#!/usr/bin/env bash
# tools/ci.sh -- the pre-merge check (design/20 section 4).   Usage: tools/ci.sh [preset]
#
# Runs, timing each step: submodules (init if missing) -> configure (preset, -DIMSRG_WERROR=ON)
# -> build -> unit tests -> integration tests (only if the input data exist) -> MPI tests (only
# if MPI is enabled) -> layering check -> clang-format, cpplint and the license-header check (our sources only). Tests
# labeled `slow` never run here. The last line is exactly `CI PASS <preset>` or `CI FAIL <preset>: <step>`, and
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
#
# OLCF Frontier (presets frontier-*; tools/lib/frontier_env.sh): the script loads the pinned
# modules (frontier-hip: plus rocm) and runs configure, build and the tests on a compute node of a
# running allocation through `srun --overlap` (IMSRG_SLURM_JOBID, SLURM_JOB_ID, or the newest
# running job named `imsrg`); MPI tests start their ranks with a nested `srun -n`. The lint steps
# stay on the login node. frontier-hip stops after the build (compile check); frontier-hip-run
# builds the same tree (preset frontier-hip, build/frontier-hip) and runs the unit and
# integration tests on one GCD (`imsrg_compute_gpu`: srun --gpus=1 -c 7), M14. MPI tests are
# skipped there (one GCD).
#   IMSRG_COMPUTE_CPUS cores of the compute step (default 56; build jobs and test slots follow)
set -uo pipefail

preset=${1:-dev-openmp}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/imsrg_env.sh
. "$script_dir/lib/imsrg_env.sh"
# shellcheck source=lib/frontier_env.sh
. "$script_dir/lib/frontier_env.sh"

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
# The CMake preset: frontier-hip-run tests the tree of frontier-hip.
cmake_preset=$preset
case $preset in frontier-hip-run) cmake_preset=frontier-hip ;; esac
build_dir=${IMSRG_BUILD_DIR:-$repo_root/build/$cmake_preset}
case $build_dir in
  /*) ;;
  *) build_dir="$PWD/$build_dir" ;;
esac
cd "$repo_root" || finish "setup (cannot cd to $repo_root)"

# On Frontier the heavy steps run on a compute node (`on_compute`); elsewhere `on_compute` is empty.
on_compute=""
on_test=""   # runs the test steps (frontier-hip-run: on one GCD)
build_only=0
gpu_tests=0
case $preset in
  frontier-*)
    if ! imsrg_on_frontier; then finish "setup (preset $preset needs OLCF Frontier)"; fi
    toolchain=cpu
    case $preset in
      frontier-hip-run) toolchain=hip; gpu_tests=1 ;;
      frontier-hip*) toolchain=hip; build_only=1 ;;
    esac
    imsrg_frontier_modules "$toolchain" || finish "setup (module load)"
    jobid=$(imsrg_frontier_jobid) || finish "setup (no compute allocation)"
    on_compute=imsrg_compute
    on_test=imsrg_compute
    if [ "$gpu_tests" -eq 1 ]; then on_test=imsrg_compute_gpu; fi
    ncpu=$IMSRG_COMPUTE_CPUS
    echo "[ci] Frontier: toolchain $toolchain, compute steps in job $jobid ($ncpu cores)" ;;
  *) ncpu=$(imsrg_ncpu) ;;
esac
jobs=${IMSRG_CI_JOBS:-$ncpu}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
threads=${OMP_NUM_THREADS%%,*}  # "4,2" (nested levels) -> 4
case $threads in
  ''|*[!0-9]*|0) threads=4 ;;
esac
test_jobs=$((ncpu / threads))
# One GCD with 7 cores; the HIP build's host space is Serial: 6 test processes share the GCD.
if [ "$gpu_tests" -eq 1 ]; then test_jobs=6; fi
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
  $on_compute cmake --preset "$cmake_preset" -B "$build_dir" -DIMSRG_WERROR=ON
}

step_build() {
  $on_compute cmake --build "$build_dir" --parallel "$jobs"
}

cache_value() {  # <variable>: value from the build tree's CMakeCache.txt
  sed -n "s/^$1:[A-Z]*=//p" "$build_dir/CMakeCache.txt" | head -n 1
}

step_unit() {
  $on_test ctest --test-dir "$build_dir" -L '^unit$' -LE '^slow$' --output-on-failure --no-tests=error \
    -j "$test_jobs"
}

step_integration() {
  # On one GCD the long integration tests (IMSRG(3) N7, ~155 s alone, host part single-threaded
  # in the HIP build) exceed their 300 s timeout when six share the device: three at a time.
  local jobs_it=$test_jobs
  if [ "$gpu_tests" -eq 1 ]; then jobs_it=3; fi
  $on_test ctest --test-dir "$build_dir" -L '^integration$' -LE '^(slow|mpi)$' --output-on-failure \
    --no-tests=ignore -j "$jobs_it" || return 1
  if ls tests/integration/test_*.py >/dev/null 2>&1; then
    IMSRG_BUILD_DIR="$build_dir" $on_test "$python" -m pytest -q tests/integration -m "not slow"
  fi
}

step_mpi() {
  $on_compute ctest --test-dir "$build_dir" -L '^mpi$' -LE '^slow$' --output-on-failure --no-tests=error \
    -j "$test_jobs"
}

step_python() {
  # The extension was built (the step only runs then): a missing _core must fail, not skip.
  # The whole suite runs under strict residency (D51): a test that reads a library result
  # without sync_to_host() fails here although it would pass on the CPU, so the scripts stay
  # portable to GPU builds. The user default (Auto) is exercised by tests that set it explicitly.
  IMSRG_RESIDENCY=strict IMSRG_REQUIRE_PYTHON_EXT=1 IMSRG_TEST_DATA_DIR="$(cache_value IMSRG_TEST_DATA_DIR)" \
    PYTHONPATH="$build_dir/python" $on_compute "$python" -m pytest -q tests/python
}

step_layering() {
  "$python" tools/check_layering.py --repo-root "$repo_root" \
    --module-deps "$build_dir/imsrg_module_deps.json" &&
    "$python" tools/check_layering.py --repo-root "$repo_root" --self-test
}

# Our C++ sources: tracked files plus new files not yet added (never ignored ones) that still
# exist, under src apps tests python tools/bench (never extern/, tools/reference/, tests/golden/).
cxx_sources() {
  local f
  git ls-files --cached --others --exclude-standard -- src apps tests python tools/bench \
    | grep -E '\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx|inl|ipp|tpp)$' | grep -v '^tests/golden/' \
    | sort -u | while IFS= read -r f; do if [ -f "$f" ]; then echo "$f"; fi; done
}

step_format() {
  # The pinned clang-format of the venv (tools/requirements-dev.txt), else the one on the PATH.
  local clang_format=$IMSRG_VENV/bin/clang-format
  if [ ! -x "$clang_format" ]; then
    clang_format=$(command -v clang-format) || {
      echo "clang-format not found (pip install -r tools/requirements-dev.txt)"; return 1; }
  fi
  echo "$("$clang_format" --version)"
  local files
  files=$(cxx_sources)
  if [ -z "$files" ]; then echo "no C++ sources"; return 0; fi
  echo "checking $(echo "$files" | wc -l | tr -d ' ') files"
  echo "$files" | tr '\n' '\0' | xargs -0 "$clang_format" --dry-run --Werror
}

# cpplint with the repository's CPPLINT.cfg (design/17 section 4); fails on any finding.
# Justified exceptions carry `// NOLINT(<category>): <reason>` in the code.
step_cpplint() {
  local cpplint=$IMSRG_VENV/bin/cpplint files out n
  if [ ! -x "$cpplint" ]; then
    cpplint=$(command -v cpplint) || {
      echo "cpplint not found (pip install -r tools/requirements-dev.txt)"; return 1; }
  fi
  echo "$("$cpplint" --version | head -n 2 | tail -n 1)"
  files=$(cxx_sources | grep -E '\.(h|cc)$')
  if [ -z "$files" ]; then echo "no .h/.cc sources"; return 0; fi
  echo "checking $(echo "$files" | wc -l | tr -d ' ') files"
  out=$(echo "$files" | tr '\n' '\0' | xargs -0 "$cpplint" --quiet 2>&1)
  if echo "$out" | grep -q 'Traceback'; then
    echo "$out"; echo "cpplint crashed"; return 1
  fi
  n=$(echo "$out" | grep -c -E '\[[a-z_+0-9/]+\] \[[0-9]\]$')
  if [ "$n" -gt 0 ]; then
    echo "$out" | grep -E '\[[a-z_+0-9/]+\] \[[0-9]\]$'
    echo "cpplint: $n findings"
    return 1
  fi
  echo "cpplint: no findings"
}

# Every tracked .h/.cc of ours (not extern/, not tests/golden/) carries the SPDX identifier
# `GPL-3.0-only` in its first 5 lines (owner decision D46; design/17 section 2.A).
step_license() {
  local bad n=0 f
  bad=$(git -C "$repo_root" ls-files -z -- '*.h' '*.cc' ':!extern' ':!tests/golden' \
    | while IFS= read -r -d '' f; do
        head -n 5 "$repo_root/$f" | grep -q 'SPDX-License-Identifier: GPL-3.0-only' || echo "$f"
      done)
  if [ -n "$bad" ]; then
    echo "$bad"; n=$(echo "$bad" | wc -l | tr -d ' ')
    echo "license headers: $n files lack 'SPDX-License-Identifier: GPL-3.0-only' in the first 5 lines"
    return 1
  fi
  echo "license headers: all present"
}

run_step submodules step_submodules
if ! ls "$IMSRG_DEPS_PREFIX"/kokkos-*/lib*/cmake/Kokkos/KokkosConfig.cmake >/dev/null 2>&1; then
  echo "[ci] note: no Kokkos install in $IMSRG_DEPS_PREFIX; CMake builds extern/kokkos in-tree" \
       "(run tools/bootstrap_deps.sh once to speed this up)"
fi
run_step configure step_configure
run_step build step_build
if [ "$build_only" -eq 1 ]; then
  skip_step "tests" "preset $preset is a compile check"
else
run_step "unit tests" step_unit

data_dir=$(cache_value IMSRG_TEST_DATA_DIR)
if [ -n "$data_dir" ] && [ -f "$data_dir/VNN_N3LO_EM500_lambda_1.80_hw16.00_emax_14.me2j.gz" ] \
   && [ -f "$data_dir/NO2B_ThBME_N2LO_cE_NonLocal4_394_IS_hw16_ms6_12_18.stream.bin" ]; then
  run_step "integration tests" step_integration
else
  skip_step "integration tests" "input data not found in '${data_dir:-unset}'"
fi

case $gpu_tests:$(cache_value IMSRG_ENABLE_MPI) in
  1:*) skip_step "mpi tests" "one GCD (frontier-hip-run)" ;;
  0:ON|0:on|0:TRUE|0:true|0:YES|0:yes|0:1|0:Y|0:y) run_step "mpi tests" step_mpi ;;
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
fi  # build_only

run_step layering step_layering
run_step clang-format step_format
run_step cpplint step_cpplint
run_step "license headers" step_license
finish PASS
