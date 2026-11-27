#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
#
# Slurm template of the 27Al response farm (design/50 C10, C13 and section 6):
# run.task = "transform" over the stored Omegas of ONE flow (one Hamiltonian,
# one variant, one Omega format). One MPI rank per GCD (8 per node); every rank
# takes a static cost-weighted share of the work list; the Omegas are read once
# per node into MPI shared memory. Not run at scale yet (2026-11-26).
#
# Usage (from a login node; submits one job, or prints it with DRY_RUN=1):
#   tools/al27/farm_sweep.sh FLOW_DIR [FORMAT_TAG]
#     FLOW_DIR    output directory of the flow (resolved_config.toml and the
#                 manifest <stem>_omegas.toml of output.write_omega)
#     FORMAT_TAG  label of the output directory (default: the manifest's format)
#   Environment: MANIFEST (default: native before me2jp, *_omegas_no.toml
#     before *_omegas.toml), NODES (default: work items / 8 rounded up, at most 604),
#     WALLTIME (default 02:00:00), ACCOUNT (nph123), WORK_LIST (default:
#     tools/al27/make_work_list.py), IMSRG_BIN (default build/frontier-hip/...
#     falls back to frontier-cpu-release), PARTITION (batch), DRY_RUN.
#
# A sweep = 2 Hamiltonians x 3 variants (x 2 Omega formats for the format
# comparison of variant 1): call this script once per flow directory. Resubmit
# a partial farm with WORK_LIST=<out>/failed.txt (the failure manifest lists
# every item that did not finish, also when the job was killed) or with the
# same list and --set transform.skip_existing=true (set below).
set -euo pipefail

FLOW_DIR=${1:?usage: farm_sweep.sh FLOW_DIR [FORMAT_TAG]}
FLOW_DIR=$(cd "$FLOW_DIR" && pwd)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG="$FLOW_DIR/resolved_config.toml"
[[ -f $CONFIG ]] || { echo "no $CONFIG" >&2; exit 2; }
# The manifest: $MANIFEST, else the native one before the ME2JP one, and a
# manifest with normal_ordering_occupations (*_omegas_no.toml, needed with
# [imsrg.space]) before one without.
if [[ -z ${MANIFEST:-} ]]; then
  for m in $(ls "$FLOW_DIR"/*_omegas_no.toml "$FLOW_DIR"/*_omegas.toml \
               2>/dev/null | grep -v _me2jp_) \
           $(ls "$FLOW_DIR"/*_me2jp_omegas_no.toml \
               "$FLOW_DIR"/*_me2jp_omegas.toml 2>/dev/null); do
    MANIFEST=$m
    break
  done
fi
MANIFEST=$(realpath "${MANIFEST:-/nonexistent}")
[[ -f $MANIFEST ]] || { echo "no *_omegas.toml in $FLOW_DIR" >&2; exit 2; }
FORMAT=$(sed -n 's/^format = "\(.*\)"/\1/p' "$MANIFEST")
TAG=${2:-$FORMAT}
OUT="$FLOW_DIR/farm_$TAG"
mkdir -p "$OUT"

WORK_LIST=${WORK_LIST:-$OUT/work.txt}
if [[ ! -f $WORK_LIST ]]; then
  python3 "$REPO/tools/al27/make_work_list.py" > "$WORK_LIST"
fi
ITEMS=$(grep -cv '^\s*\(#\|$\)' "$WORK_LIST")
RANKS_PER_NODE=8
NODES=${NODES:-$(( (ITEMS + RANKS_PER_NODE - 1) / RANKS_PER_NODE ))}
(( NODES > 604 )) && NODES=604
RANKS=$(( NODES * RANKS_PER_NODE ))
(( RANKS > ITEMS )) && RANKS=$ITEMS

IMSRG_BIN=${IMSRG_BIN:-$REPO/build/frontier-hip/apps/imsrg/imsrg}
[[ -x $IMSRG_BIN ]] || IMSRG_BIN=$REPO/build/frontier-cpu-release/apps/imsrg/imsrg

JOB=$(cat <<EOF
#!/bin/bash
#SBATCH -A ${ACCOUNT:-nph123}
#SBATCH -J al27-farm-$TAG
#SBATCH -N $NODES
#SBATCH -t ${WALLTIME:-02:00:00}
#SBATCH -p ${PARTITION:-batch}
#SBATCH -o $OUT/slurm-%j.out
#SBATCH --gpus-per-node=8
set -euo pipefail
source "$REPO/tools/lib/frontier_env.sh"
imsrg_frontier_modules hip
export OMP_NUM_THREADS=7 OMP_PROC_BIND=spread OMP_PLACES=threads
# One rank per GCD: 7 cores each (the 8th core of each L3 region is reserved
# by the low-noise mode); closest GPU binding for the device version.
# (No backslash continuations: inside \$(cat <<EOF) they would join the lines.)
args=(
  --config "$CONFIG"
  --set run.task=transform
  --set "operators.list=[]"
  --set "output.directory=$OUT"
  --set "transform.manifest=$MANIFEST"
  --set "transform.work_list=$WORK_LIST"
  --set transform.skip_existing=true
)
srun -N $NODES -n $RANKS --ntasks-per-node=$RANKS_PER_NODE -c 7 --gpus-per-task=1 --gpu-bind=closest "$IMSRG_BIN" "\${args[@]}"
EOF
)
echo "farm: $ITEMS items, $NODES nodes, $RANKS ranks, Omegas $FORMAT -> $OUT"
if [[ -n ${DRY_RUN:-} ]]; then
  echo "$JOB"
else
  echo "$JOB" | sbatch
fi
