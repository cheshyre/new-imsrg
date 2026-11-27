#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Runs one 27Al production step on the compute node of the running Slurm job
# named $IMSRG_SLURM_JOBNAME (default imsrg-c), with 56 OpenMP threads, and
# records wall time and peak memory (/usr/bin/time -v).
#
# usage: tools/al27/run_flow.sh <config.toml> <run_dir> <log_name> [--set k=v]...
# Output (stdout+stderr of imsrg, then the time -v report) goes to
# <run_dir>/<log_name>.log. The binary is build/frontier-cpu-release.
#
# SIGTERM (design/00 D75): as the body of a batch job, ask Slurm for SIGTERM
# 300 s before the end; the script forwards it to srun, which forwards it to
# imsrg, which writes a checkpoint (with checkpoint.directory set) and exits
# with status 75 (continue with --set checkpoint.restart=true). /usr/bin/time
# ignores TERM so that it outlives imsrg and the step is not torn down early.
#SBATCH --signal=B:TERM@300
set -euo pipefail
cfg=$(realpath "$1"); run=$2; name=$3; shift 3
root=$(cd "$(dirname "$0")/../.." && pwd)
bin=${IMSRG_BIN:-$root/build/frontier-cpu-release/apps/imsrg/imsrg}
export IMSRG_SLURM_JOBNAME=${IMSRG_SLURM_JOBNAME:-imsrg-c}
# shellcheck source=/dev/null
source "$root/tools/lib/frontier_env.sh"
imsrg_frontier_modules cpu >/dev/null 2>&1
mkdir -p "$run"
cp "$bin" "$run/imsrg.$name"   # the binary of this step (rebuilds leave it)
export OMP_NUM_THREADS=56 OMP_PROC_BIND=spread OMP_PLACES=threads
cd "$run"
echo "start $(date -Is) job $(imsrg_frontier_jobid 2>/dev/null || true)" > "$name.log"
# shellcheck disable=SC2016
imsrg_compute bash -c 'trap "" TERM; exec /usr/bin/time -v "$@"' time \
  "$run/imsrg.$name" --config "$cfg" \
  --set output.directory="$run" "$@" >> "$name.log" 2>&1 &
child=$!
# The background job is a subshell running srun: TERM goes to its children.
trap 'echo "SIGTERM $(date -Is): forwarded" >> "$name.log";
      pkill -TERM -P "$child" || kill -TERM "$child"' TERM
st=0
wait "$child" || st=$?
while kill -0 "$child" 2>/dev/null; do wait "$child" || st=$?; done
trap - TERM
echo "end $(date -Is) status $st" >> "$name.log"
