#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Times farm items (run.task = "transform") on ONE GCD of the running job
# $IMSRG_SLURM_JOBNAME (default imsrg-a): one rank, 7 OpenMP host threads,
# frontier-hip build. Default work list: one item per rank L = 0..5 at
# q = 150 MeV plus TCM (Brief A item 5). Records the node load before and
# after (uptime, GPU use) in OUT_DIR/load.txt; per-item times are the
# "transform: <name> done in <s> s" lines of OUT_DIR/farm.log.
#
# usage: tools/al27/time_farm_gpu.sh FLOW_DIR MANIFEST OUT_DIR [ITEM ...]
#   FLOW_DIR  flow output directory (resolved_config.toml)
#   MANIFEST  Omega manifest (e.g. FLOW_DIR/N2LOsat_omegas_no.toml)
set -euo pipefail
flow=$(cd "${1:?usage}" && pwd)
manifest=$(realpath "${2:?usage}")
out=$(mkdir -p "${3:?usage}" && cd "$3" && pwd)
shift 3
items=("$@")
if (( ${#items[@]} == 0 )); then
  items=(FM0p_150 FSigmap1p_150 FM2p_150 FSigmap3p_150 FM4p_150 FSigmap5p_150
         TCM)
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
export IMSRG_SLURM_JOBNAME=${IMSRG_SLURM_JOBNAME:-imsrg-a}
# shellcheck source=/dev/null
source "$root/tools/lib/frontier_env.sh"
imsrg_frontier_modules hip >/dev/null 2>&1
bin=${IMSRG_BIN:-$root/build/frontier-hip/apps/imsrg/imsrg}
cp "$bin" "$out/imsrg.timing"
printf '%s\n' "${items[@]}" > "$out/work.txt"
export OMP_NUM_THREADS=7 OMP_PROC_BIND=spread OMP_PLACES=threads
load() {
  imsrg_compute bash -c 'echo "$(date -Is) $(hostname) $(uptime)"; \
    rocm-smi --showuse 2>/dev/null | grep -E "^(GPU|card)" || true' \
    >> "$out/load.txt" 2>&1 || true
}
load
imsrg_compute_gpu /usr/bin/time -v "$out/imsrg.timing" \
  --config "$flow/resolved_config.toml" \
  --set run.task=transform \
  --set "operators.list=[]" \
  --set "output.directory=$out" \
  --set "transform.manifest=$manifest" \
  --set "transform.work_list=$out/work.txt" > "$out/farm.log" 2>&1
load
grep "done in\|FAILED\|finished" "$out/farm.log"
