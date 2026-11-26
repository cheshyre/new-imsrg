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
st=0
imsrg_compute /usr/bin/time -v "$run/imsrg.$name" --config "$cfg" \
  --set output.directory="$run" "$@" >> "$name.log" 2>&1 || st=$?
echo "end $(date -Is) status $st" >> "$name.log"
