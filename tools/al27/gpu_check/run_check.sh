#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# GPU-vs-CPU check of the 27Al production path (Brief A, acceptance item 4):
# the emax-4 variant-1 flow of al27_emax4_v1.toml (writes Omegas), then the
# farm (run.task = "transform") over its native Omegas for one item per rank
# L = 0, 1, 5 plus TCM, with transform.dump_native = true. Run it once with
# the CPU build and once with the GPU build, then compare the two directories
# with compare_runs.py.
#
# usage: tools/al27/gpu_check/run_check.sh cpu|gpu OUT_DIR
#   cpu: build/frontier-cpu-release, 56 OpenMP threads on the node
#   gpu: build/frontier-hip, one GCD and 7 OpenMP threads
#   Environment: IMSRG_SLURM_JOBNAME (default imsrg-a), IMSRG_BIN (override).
# Runs on the compute node of the running job (srun --overlap); test inputs
# from $IMSRG_WORK/inputs (emax 4: within the test-size rule).
set -euo pipefail
kind=${1:?usage: run_check.sh cpu|gpu OUT_DIR}
out=$(mkdir -p "${2:?usage: run_check.sh cpu|gpu OUT_DIR}" && cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
export IMSRG_SLURM_JOBNAME=${IMSRG_SLURM_JOBNAME:-imsrg-a}
# shellcheck source=/dev/null
source "$root/tools/lib/frontier_env.sh"
case $kind in
  cpu)
    imsrg_frontier_modules cpu >/dev/null 2>&1
    bin=${IMSRG_BIN:-$root/build/frontier-cpu-release/apps/imsrg/imsrg}
    export OMP_NUM_THREADS=56 OMP_PROC_BIND=spread OMP_PLACES=threads
    launch() { IMSRG_COMPUTE_CPUS=56 imsrg_compute "$@"; } ;;
  gpu)
    imsrg_frontier_modules hip >/dev/null 2>&1
    bin=${IMSRG_BIN:-$root/build/frontier-hip/apps/imsrg/imsrg}
    export OMP_NUM_THREADS=7 OMP_PROC_BIND=spread OMP_PLACES=threads
    launch() { imsrg_compute_gpu "$@"; } ;;
  *) echo "run_check.sh: kind must be cpu or gpu" >&2; exit 2 ;;
esac
[[ -x $bin ]] || { echo "run_check.sh: no binary $bin" >&2; exit 2; }

sed -e "s|@INPUTS@|$IMSRG_WORK/inputs|g" -e "s|@OUT@|$out|g" \
  "$here/al27_emax4_v1.toml" > "$out/config.toml"
printf '%s\n' FM0p_150 FSigmap1p_150 FSigmap5n_150 FDelta5p_300 TCM \
  > "$out/work.txt"

echo "flow ($kind) -> $out/flow.log"
launch "$bin" --config "$out/config.toml" > "$out/flow.log" 2>&1
echo "farm ($kind) -> $out/farm.log"
launch "$bin" --config "$out/resolved_config.toml" \
  --set run.task=transform \
  --set "operators.list=[]" \
  --set "output.directory=$out/farm" \
  --set "transform.manifest=$out/al27e4_omegas.toml" \
  --set "transform.work_list=$out/work.txt" \
  --set transform.dump_native=true > "$out/farm.log" 2>&1
echo "done: $out"
