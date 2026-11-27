#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Submission of the 27Al response farm (design/50 C10, C13 and section 6):
# run.task = "transform" over the stored Omegas of one or several flows. ONE
# sbatch job holds one srun step per SPEC (concurrent, disjoint nodes); one
# MPI rank per GCD-sized share (8 per node); the binary assigns the work-list
# items to the ranks itself (static, cost-weighted, farm.cc AssignStatic).
#
# Usage (login node; submits one job, or prints it with DRY_RUN=1):
#   tools/al27/farm_sweep.sh SPEC [SPEC ...]
#   SPEC = FLOW_DIR:MANIFEST[:TAG[:WORK_LIST]]
#     FLOW_DIR   flow output directory (has resolved_config.toml)
#     MANIFEST   Omega manifest file name inside FLOW_DIR, e.g.
#                N2LOsat_omegas_no.toml
#     TAG        output label (default: the manifest's `format`); results go
#                to $FLOW_DIR/farm-$TAG
#     WORK_LIST  default $FLOW_DIR/farm-$TAG/work.txt, created with
#                make_work_list.py if absent (FAMILIES=FM0,FSigmap1 passes
#                --families, NO_TCM=1 passes --no-tcm)
#   (Old form `FLOW_DIR` alone still works: manifest = first *_omegas.toml.)
#   Paths must not contain ':'.
#
# Environment:
#   BACKEND      cpu (default) | hip
#   NODES_<i>    nodes of the i-th SPEC (1-based); else TARGET_HOURS (default
#                1.0) sizes it with tools/al27/farm_plan.py (farm_costs.toml)
#   MAX_NODES    cap of the planner (default 604)
#   WALLTIME     override of the job walltime (default: planner, max over
#                the specs)
#   ACCOUNT (nph123), PARTITION (batch), JOB_NAME (al27-farm)
#   IMSRG_BIN    default build/frontier-cpu-release/apps/imsrg/imsrg (cpu),
#                build/frontier-hip/apps/imsrg/imsrg (hip); it is COPIED to
#                $OUT/imsrg.<jobtag> at submission, so rebuilds do not affect
#                a queued job
#   DRY_RUN=1    print the plan and the job script; submit nothing, create
#                nothing
#
# Resubmission: step k writes $FLOW_DIR/farm-$TAG/failed-<jobtag>-s<k>.txt (items that
# did not finish, also after a kill; lines `name<TAB>reason` are a valid work
# list). Rerun the failures with
#   SPEC=FLOW:MANIFEST:TAG:$FLOW/farm-$TAG/failed-<jobtag>-s<k>.txt
# Several steps may share one TAG (output directory), e.g. one step per
# operator rank L: the binary's cost weights do not follow the CPU timings
# (L=1 items take 14x an L=0 item, not 2x), so homogeneous steps balance.
# transform.skip_existing=true skips items whose files are already complete.
#
# A sweep = 2 Hamiltonians x 3 variants (x 2 Omega formats for the format
# comparison of variant 1).
set -euo pipefail

(( $# >= 1 )) || { sed -n '8,12p' "$0" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BACKEND=${BACKEND:-cpu}
case $BACKEND in cpu|hip) ;; *) echo "BACKEND must be cpu or hip" >&2; exit 2;; esac
PY=${IMSRG_DEPS_PREFIX:-}/venv/bin/python
[[ -x $PY ]] || PY=python3
if [[ -z ${IMSRG_BIN:-} ]]; then
  if [[ $BACKEND == hip ]]; then
    IMSRG_BIN=$REPO/build/frontier-hip/apps/imsrg/imsrg
  else
    IMSRG_BIN=$REPO/build/frontier-cpu-release/apps/imsrg/imsrg
  fi
fi
DRY_RUN=${DRY_RUN:-}
if [[ ! -x $IMSRG_BIN ]]; then
  [[ -n $DRY_RUN ]] || { echo "no binary $IMSRG_BIN" >&2; exit 2; }
  echo "warning: no binary $IMSRG_BIN (dry run)" >&2
fi
JOBTAG=$(date +%Y%m%d-%H%M%S)
TARGET_HOURS=${TARGET_HOURS:-1.0}
MAX_NODES=${MAX_NODES:-604}
RPN=8
THREADS=7

declare -a FLOWS MANIFESTS TAGS OUTS WLISTS NODES RANKS
total_nodes=0
max_wall_s=0
summary=""
i=0
for spec in "$@"; do
  i=$((i + 1))
  IFS=: read -r flow manifest tag wl <<< "$spec"
  [[ -d $flow ]] || { echo "no flow directory $flow" >&2; exit 2; }
  flow=$(cd "$flow" && pwd)
  [[ -f $flow/resolved_config.toml ]] \
    || { echo "no $flow/resolved_config.toml" >&2; exit 2; }
  if [[ -z $manifest ]]; then  # old usage: FLOW_DIR alone
    manifest=$(basename "$(ls "$flow"/*_omegas.toml | head -n 1)")
  fi
  [[ -f $flow/$manifest ]] || { echo "no $flow/$manifest" >&2; exit 2; }
  if [[ -z $tag ]]; then
    tag=$(sed -n 's/^format *= *"\(.*\)"/\1/p' "$flow/$manifest" | head -n 1)
  fi
  [[ -n $tag ]] || { echo "no tag for spec $spec" >&2; exit 2; }
  out=$flow/farm-$tag
  [[ -n $wl ]] || wl=$out/work.txt
  if [[ ! -f $wl ]]; then
    gen=()
    [[ -z ${FAMILIES:-} ]] || gen+=(--families "$FAMILIES")
    [[ -z ${NO_TCM:-} ]] || gen+=(--no-tcm)
    if [[ -n $DRY_RUN ]]; then
      wl=$(mktemp "${TMPDIR:-/tmp}/work-$tag-XXXXXX.txt")
    else
      mkdir -p "$out"
    fi
    "$PY" "$REPO/tools/al27/make_work_list.py" ${gen[@]+"${gen[@]}"} > "$wl"
  fi
  nvar=NODES_$i
  if [[ -n ${!nvar:-} ]]; then
    plan=$("$PY" "$REPO/tools/al27/farm_plan.py" --json --backend "$BACKEND" \
             --work-list "$wl" --nodes "${!nvar}")
  else
    plan=$("$PY" "$REPO/tools/al27/farm_plan.py" --json --backend "$BACKEND" \
             --work-list "$wl" --target-hours "$TARGET_HOURS" \
             --max-nodes "$MAX_NODES")
  fi
  read -r items n r mk wall_s wall < <(
    "$PY" -c 'import json,sys
j=json.loads(sys.argv[1]); s=j["steps"][0]
print(s["items"], s["nodes"], s["ranks"], "%.3f" % s["makespan_h"],
      s["walltime_s"], s["walltime"])
for w in j["warnings"]: print("WARNING: "+w, file=sys.stderr)' "$plan")
  FLOWS[i]=$flow; MANIFESTS[i]=$manifest; TAGS[i]=$tag; OUTS[i]=$out
  WLISTS[i]=$wl; NODES[i]=$n; RANKS[i]=$r
  total_nodes=$((total_nodes + n))
  (( wall_s > max_wall_s )) && max_wall_s=$wall_s
  summary+="  spec $i $tag: items $items nodes $n ranks $r makespan ${mk} h walltime $wall"$'\n'
done
NSPEC=$i
printf -v PLANNED_WALL '%02d:%02d:00' $((max_wall_s / 3600)) $((max_wall_s % 3600 / 60))
WALLTIME=${WALLTIME:-$PLANNED_WALL}
IFS=: read -r wh wm _ <<< "$WALLTIME"
node_hours=$("$PY" -c 'import sys; print("%.1f" % (int(sys.argv[1]) * (int(sys.argv[2]) + int(sys.argv[3]) / 60.0)))' \
             "$total_nodes" "$((10#$wh))" "$((10#$wm))")

LOGDIR=$(dirname "${FLOWS[1]}")/farm-logs
JOB_NAME=${JOB_NAME:-al27-farm}

if [[ $BACKEND == hip ]]; then
  GPU_OPTS="--gpus-per-task=1 --gpu-bind=closest"
  GPU_SBATCH="#SBATCH --gpus-per-node=8"
else
  GPU_OPTS=""
  GPU_SBATCH=""
fi

JOB="#!/bin/bash
#SBATCH -A ${ACCOUNT:-nph123}
#SBATCH -J $JOB_NAME
#SBATCH -N $total_nodes
#SBATCH -t $WALLTIME
#SBATCH -p ${PARTITION:-batch}
#SBATCH -o $LOGDIR/slurm-%j.out
${GPU_SBATCH}
set -uo pipefail
source \"$REPO/tools/lib/frontier_env.sh\"
imsrg_frontier_modules $BACKEND
export OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=spread OMP_PLACES=threads
# $RPN ranks per node, $THREADS cores each (the 8th core of each L3 region is
# reserved by the low-noise mode). One concurrent step per spec.
echo \"job \$SLURM_JOB_ID start \$(date -Is)\"
pids=()
"
for ((k = 1; k <= NSPEC; k++)); do
  out=${OUTS[k]}
  per_node=$(( (RANKS[k] + NODES[k] - 1) / NODES[k] ))
  JOB+="echo \"step $k (${TAGS[k]}) start \$(date -Is)\"
srun -N ${NODES[k]} -n ${RANKS[k]} --ntasks-per-node=$per_node -c $THREADS \\
     $GPU_OPTS --exact --mem=0 --export=ALL -o \"$out/step-%j-s$k.log\" \\
  \"$out/imsrg.$JOBTAG\" --config \"${FLOWS[k]}/resolved_config.toml\" \\
    --set run.task=transform \\
    --set \"operators.list=[]\" \\
    --set \"output.directory=$out\" \\
    --set \"transform.manifest=${FLOWS[k]}/${MANIFESTS[k]}\" \\
    --set \"transform.work_list=${WLISTS[k]}\" \\
    --set \"transform.failure_manifest=$out/failed-$JOBTAG-s$k.txt\" \\
    --set transform.skip_existing=true &
pids+=(\$!)
"
done
JOB+="status=0
for k in \"\${!pids[@]}\"; do
  st=0
  wait \"\${pids[\$k]}\" || st=\$?
  echo \"step \$((k + 1)) exit status \$st at \$(date -Is)\"
  (( st == 0 )) || status=\$st
done
echo \"job \$SLURM_JOB_ID end \$(date -Is) status \$status\"
exit \$status
"

echo "farm job $JOBTAG: backend $BACKEND, $NSPEC step(s), $total_nodes nodes, walltime $WALLTIME"
printf '%s' "$summary"
echo "  node-hours: $node_hours (nodes x walltime)"

if [[ -n $DRY_RUN ]]; then
  echo "DRY_RUN: would copy $IMSRG_BIN to <out>/imsrg.$JOBTAG"
  echo "-----"
  printf '%s' "$JOB"
  exit 0
fi

mkdir -p "$LOGDIR"
for ((k = 1; k <= NSPEC; k++)); do
  mkdir -p "${OUTS[k]}"
  cp "$IMSRG_BIN" "${OUTS[k]}/imsrg.$JOBTAG"
done
jobid=$(printf '%s' "$JOB" | sbatch --parsable)
echo "submitted job $jobid"
reg=${IMSRG_WORK:-$REPO/..}/runs/al27/farm-submissions.log
mkdir -p "$(dirname "$reg")"
echo "$(date -Is) job $jobid backend $BACKEND nodes $total_nodes walltime $WALLTIME specs $*" >> "$reg"
