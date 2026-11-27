#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Small transform-farm test on the development node of $IMSRG_SLURM_JOBNAME
# (default imsrg-c): RANKS MPI ranks (default 1) with THREADS OpenMP threads
# each (default 7, the per-GCD share of a farm rank), over the items of a work
# list, from the stored Omegas of one v1 flow. Records time -v per run.
#
# usage: tools/al27/farm_test.sh FLOW_DIR MANIFEST OUT_DIR WORK_LIST
#   FLOW_DIR  flow output directory (resolved_config.toml)
#   MANIFEST  Omega manifest file name inside FLOW_DIR (e.g. N2LOsat_omegas.toml)
#   OUT_DIR   output directory of the .snt files (created)
#   WORK_LIST one operator name per line
# The binary is IMSRG_BIN (default build/frontier-cpu-release); it is copied to
# OUT_DIR so later rebuilds do not affect a running test. EXTRA_SET=key=value
# adds one more --set (e.g. output.isospin_average_operators=false).
set -euo pipefail
flow=$(realpath "$1"); manifest=$2; out=$3; work=$(realpath "$4")
root=$(cd "$(dirname "$0")/../.." && pwd)
bin=${IMSRG_BIN:-$root/build/frontier-cpu-release/apps/imsrg/imsrg}
ranks=${RANKS:-1}; threads=${THREADS:-7}
export IMSRG_SLURM_JOBNAME=${IMSRG_SLURM_JOBNAME:-imsrg-c}
# shellcheck source=/dev/null
source "$root/tools/lib/frontier_env.sh"
imsrg_frontier_modules cpu >/dev/null 2>&1
mkdir -p "$out"
out=$(realpath "$out")
cp "$bin" "$out/imsrg.farm"
export OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread OMP_PLACES=threads
job=$(imsrg_frontier_jobid)
echo "start $(date -Is) job $job ranks $ranks threads $threads"
st=0
srun --jobid="$job" --overlap -N 1 -n "$ranks" -c "$threads" --cpu-bind=none \
  --export=ALL /usr/bin/time -v "$out/imsrg.farm" \
  --config "$flow/resolved_config.toml" --set run.task=transform \
  --set "operators.list=[]" --set "output.directory=$out" \
  --set "transform.manifest=$flow/$manifest" \
  --set "transform.work_list=$work" \
  --set transform.skip_existing=true ${EXTRA_SET:+--set "$EXTRA_SET"} || st=$?
echo "end $(date -Is) status $st"
