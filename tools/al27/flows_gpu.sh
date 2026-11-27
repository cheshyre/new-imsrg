#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Submits ONE single-node Slurm job that runs up to 4 27Al production flows
# side by side, one flow per MI250X GCD (D73, AL-E-Q4): every flow is its own
# `srun -n1 --gpus=1` step with 14 cores (OpenMP host threads for the host
# preparation at emax 14: 66 GB host peak per flow, 4 x 66 GB fit in 512 GB).
# The IMSRG(2) commutators, BCH and norms run on the GCD (frontier-hip build).
#
# usage: tools/al27/flows_gpu.sh CONFIG:RUN_DIR [CONFIG:RUN_DIR ...] [-- --set k=v ...]
#   Every flow writes <RUN_DIR>/flow-gpu.log (stdout+stderr, /usr/bin/time -v);
#   output.directory is set to RUN_DIR. Arguments after "--" go to every flow.
#   Environment: WALLTIME (default 02:00:00, the batch limit for 1-91 nodes;
#   flows longer than that continue from their checkpoint in a new job), ACCOUNT (nph123), PARTITION
#   (batch), IMSRG_BIN (default build/frontier-hip/apps/imsrg/imsrg, copied
#   into each run directory first), CORES (per flow, default 14), DRY_RUN=1
#   (print the job instead of submitting it).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
pairs=()
while [[ $# -gt 0 && $1 != "--" ]]; do pairs+=("$1"); shift; done
[[ ${1:-} == "--" ]] && shift
extra=("$@")
(( ${#pairs[@]} >= 1 && ${#pairs[@]} <= 4 )) || {
  echo "usage: flows_gpu.sh CONFIG:RUN_DIR (1-4 of them) [-- --set k=v ...]" >&2
  exit 2
}
bin=${IMSRG_BIN:-$root/build/frontier-hip/apps/imsrg/imsrg}
[[ -x $bin ]] || { echo "flows_gpu.sh: no binary $bin" >&2; exit 2; }
cores=${CORES:-14}
stamp=$(date +%Y%m%d-%H%M%S)

steps=""
for p in "${pairs[@]}"; do
  cfg=$(realpath "${p%%:*}")
  run=${p#*:}
  mkdir -p "$run"
  run=$(cd "$run" && pwd)
  cp "$bin" "$run/imsrg.gpu-$stamp"
  quoted=""
  for e in "${extra[@]}"; do quoted+=" $(printf '%q' "$e")"; done
  # Paths are %q-quoted: the generated script must survive spaces or shell
  # characters in them.
  qrun=$(printf '%q' "$run")
  qcfg=$(printf '%q' "$cfg")
  qbin=$(printf '%q' "$run/imsrg.gpu-$stamp")
  qout=$(printf '%q' "output.directory=$run")
  steps+="( cd $qrun && echo \"start \$(date -Is) job \$SLURM_JOB_ID\" > flow-gpu.log && \\
  srun -N1 -n1 -c $cores --gpus=1 --gpu-bind=closest --exact \\
    bash -c 'trap \"\" TERM; exec /usr/bin/time -v \"\$@\"' time \\
    $qbin --config $qcfg \\
    --set $qout$quoted >> flow-gpu.log 2>&1; \\
  echo \"end \$(date -Is) status \$?\" >> flow-gpu.log ) &
"
done

job=$(cat <<EOF
#!/bin/bash
#SBATCH -A ${ACCOUNT:-nph123}
#SBATCH -J al27-flows
#SBATCH -N 1
#SBATCH -t ${WALLTIME:-02:00:00}
#SBATCH -p ${PARTITION:-batch}
#SBATCH -o ${pairs[0]#*:}/slurm-flows-%j.out
#SBATCH --gpus-per-node=8
#SBATCH --signal=B:TERM@300
set -uo pipefail
source "$root/tools/lib/frontier_env.sh"
imsrg_frontier_modules hip
export OMP_NUM_THREADS=$cores OMP_PROC_BIND=spread OMP_PLACES=threads
# D75: 300 s before the end Slurm sends TERM to this script; forward it to
# the srun steps (whole node is ours), which pass it to imsrg: it writes a
# checkpoint after the current step and exits with 75 (continue with
# --set checkpoint.restart=true). /usr/bin/time ignores TERM so the step
# lives until imsrg is done.
trap 'echo "SIGTERM \$(date -Is): forwarded to the flows"; pkill -TERM -x srun' TERM
$steps
wait
while [[ -n "\$(jobs -rp)" ]]; do wait; done
EOF
)
if [[ -n ${DRY_RUN:-} ]]; then echo "$job"; else echo "$job" | sbatch; fi
