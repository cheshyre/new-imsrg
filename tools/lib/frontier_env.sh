# tools/lib/frontier_env.sh -- OLCF Frontier toolchain and compute-node helpers for tools/*.sh.
# Source it (after imsrg_env.sh); do not execute it. design/20 section 9.
#
#   imsrg_on_frontier           true on a Frontier login or compute node (LMOD_SYSTEM_NAME)
#   imsrg_frontier_modules cpu|hip
#                               loads the pinned modules: PrgEnv-gnu (gcc-native 14.2, the Cray CC
#                               wrapper links cray-mpich and cray-libsci), cmake, ninja, ccache, a
#                               Python >= 3.12 for the venv; `hip` adds rocm. Sets CXX=CC (cpu) and
#                               the MPICH paths hipcc needs (hip).
#   imsrg_frontier_jobid        prints the Slurm job to run builds and tests in, or fails:
#                               $IMSRG_SLURM_JOBID, else $SLURM_JOB_ID, else the newest running job
#                               of this user named $IMSRG_SLURM_JOBNAME (default `imsrg`)
#   imsrg_compute <cmd...>      runs <cmd> on that job's first node: srun --overlap, one task with
#                               $IMSRG_COMPUTE_CPUS cores (default 56), no CPU binding, full env
#
# Get an allocation once (login node):  salloc -A nph123 -N 1 -t 02:00:00 -J imsrg --no-shell
# Several agents may share it: every step uses --overlap.

IMSRG_FRONTIER_CPU_MODULES="PrgEnv-gnu/8.7.0 gcc-native/14.2 cray-mpich cray-libsci \
cmake/4.1.5 ninja/1.13.2 ccache/4.12.1 python/3.14.3"
IMSRG_FRONTIER_ROCM_MODULE="rocm/7.0.2"
: "${IMSRG_COMPUTE_CPUS:=56}"
: "${IMSRG_SLURM_JOBNAME:=imsrg}"

imsrg_on_frontier() {
  [ "${LMOD_SYSTEM_NAME:-}" = frontier ]
}

imsrg_frontier_modules() {  # cpu|hip
  if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    . "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash" || return 1
  fi
  # shellcheck disable=SC2086  # the module list is intentionally word-split
  module load $IMSRG_FRONTIER_CPU_MODULES >/dev/null 2>&1 || {
    echo "frontier_env: module load $IMSRG_FRONTIER_CPU_MODULES failed" >&2; return 1; }
  case ${1:-cpu} in
    cpu)
      export CXX=CC CC=cc ;;
    hip)
      module load "$IMSRG_FRONTIER_ROCM_MODULE" >/dev/null 2>&1 || {
        echo "frontier_env: module load $IMSRG_FRONTIER_ROCM_MODULE failed" >&2; return 1; }
      export CXX=hipcc
      # hipcc is not a Cray wrapper: FindMPI needs to be pointed at cray-mpich.
      export MPI_HOME="${MPICH_DIR:-}" ;;
    *) echo "frontier_env: unknown toolchain '$1' (cpu, hip)" >&2; return 1 ;;
  esac
}

imsrg_frontier_jobid() {
  local j
  j=${IMSRG_SLURM_JOBID:-${SLURM_JOB_ID:-}}
  if [ -z "$j" ]; then
    j=$(squeue -h -u "$USER" -t R -n "$IMSRG_SLURM_JOBNAME" -o '%i %S' 2>/dev/null \
      | sort -k2 -r | head -n 1 | awk '{print $1}')
  fi
  if [ -z "$j" ]; then
    echo "frontier_env: no running Slurm job named '$IMSRG_SLURM_JOBNAME' and no" \
         "IMSRG_SLURM_JOBID/SLURM_JOB_ID; get one with" \
         "'salloc -A nph123 -N 1 -t 02:00:00 -J $IMSRG_SLURM_JOBNAME --no-shell'" >&2
    return 1
  fi
  echo "$j"
}

imsrg_compute() {
  local j
  j=$(imsrg_frontier_jobid) || return 1
  srun --jobid="$j" --overlap -N 1 -n 1 -c "$IMSRG_COMPUTE_CPUS" --cpu-bind=none \
    --export=ALL "$@"
}
