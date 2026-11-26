#!/usr/bin/env bash
# tools/agent_retire.sh -- delete agent clones whose work is safe in the hub (design/41 section 3).
#
# Usage: tools/agent_retire.sh [--merged] [--force-young] [--dry-run] <clone>...
#        tools/agent_retire.sh [--merged] [--force-young] [--dry-run] --all
#
# A clone (a directory directly under $IMSRG_AGENTS_DIR, default $IMSRG_WORK/agents) is deleted
# only if nothing would be lost:
#   * no uncommitted changes to tracked files and no untracked files (ignored files such as build/
#     do not count), in the clone and its submodules;
#   * every local branch's tip is contained in a branch of the hub (fetched first), i.e. pushed;
#   * no stash entries;
#   * with --merged, additionally every local branch is merged into the hub's develop, exists on the
#     hub under its own name, and has moved off the commit the clone was created at
#     (git config imsrg.baseCommit, set by agent_clone.sh; clones without it are kept), and the
#     clone is older than 60 minutes (imsrg.createdAt) unless --force-young is given. This keeps a
#     brand-new clone, whose branch sits at develop's tip, from counting as "merged".
# Otherwise the clone is kept and the reason is printed; the exit status is then 1. --all checks
# every clone (the integrator's routine clean-up after merges uses --all --merged).
set -uo pipefail

merged=0
dry=0
young=0
all=0
targets=()
while [ $# -gt 0 ]; do
  case $1 in
    --merged) merged=1 ;;
    --force-young) young=1 ;;
    --dry-run) dry=1 ;;
    --all) all=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "agent_retire: unknown option $1" >&2; exit 2 ;;
    *) targets+=("$1") ;;
  esac
  shift
done
: "${IMSRG_WORK:?agent_retire: IMSRG_WORK is not set}"
agents_dir=$(cd "${IMSRG_AGENTS_DIR:-$IMSRG_WORK/agents}" 2>/dev/null && pwd -P) || {
  echo "agent_retire: no agents directory" >&2; exit 1; }
if [ "$all" -eq 1 ]; then
  for d in "$agents_dir"/*/; do
    [ -d "$d" ] && targets+=("${d%/}")
  done
fi
if [ ${#targets[@]} -eq 0 ]; then
  [ "$all" -eq 1 ] && exit 0
  echo "agent_retire: no clone given" >&2; exit 2
fi

# check <clone>: prints why the clone must be kept; empty output = safe to delete.
check() {
  local c=$1 dirty ref tip contained base created now br
  if ! git -C "$c" rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git checkout"; return
  fi
  if ! git -C "$c" fetch --quiet --prune origin 2>/dev/null; then
    echo "cannot fetch from the hub"; return
  fi
  dirty=$(git -C "$c" status --porcelain --ignore-submodules=none 2>/dev/null)
  if [ -n "$dirty" ]; then
    echo "uncommitted or untracked files:"; echo "$dirty" | head -n 10 | sed 's/^/    /'
  fi
  if [ -n "$(git -C "$c" stash list 2>/dev/null)" ]; then echo "has stash entries"; fi
  if [ "$merged" -eq 1 ]; then
    base=$(git -C "$c" config imsrg.baseCommit 2>/dev/null)
    created=$(git -C "$c" config imsrg.createdAt 2>/dev/null)
    if [ -z "$base" ]; then
      echo "no creation record (imsrg.baseCommit); cannot tell a fresh clone from a merged one"
    fi
    if [ "$young" -eq 0 ]; then
      now=$(date +%s)
      case $created in
        ''|*[!0-9]*) echo "no creation time (imsrg.createdAt); use --force-young to override" ;;
        *) if [ $((now - created)) -lt 3600 ]; then
             echo "clone is younger than 60 minutes ($(( (now - created) / 60 )) min); use --force-young to override"
           fi ;;
      esac
    fi
  fi
  while read -r ref tip; do
    contained=$(git -C "$c" branch -r --contains "$tip" 2>/dev/null | grep -v -- '->' | head -n 1)
    if [ -z "$contained" ]; then
      echo "branch ${ref#refs/heads/} ($(git -C "$c" rev-parse --short "$tip")) is not pushed"
    elif [ "$merged" -eq 1 ]; then
      br=${ref#refs/heads/}
      if ! git -C "$c" merge-base --is-ancestor "$tip" origin/develop; then
        echo "branch $br is pushed but not merged into develop"
      elif ! git -C "$c" rev-parse -q --verify "refs/remotes/origin/$br" >/dev/null; then
        echo "branch $br does not exist on the hub under that name"
      elif [ -n "$base" ] && [ "$tip" = "$base" ]; then
        echo "branch $br is still at the commit the clone was created at (no work of its own)"
      fi
    fi
  done < <(git -C "$c" for-each-ref --format='%(refname) %(objectname)' refs/heads)
  # A detached HEAD with commits of its own.
  if ! git -C "$c" symbolic-ref -q HEAD >/dev/null; then
    tip=$(git -C "$c" rev-parse HEAD)
    if [ -z "$(git -C "$c" branch -r --contains "$tip" 2>/dev/null | head -n 1)" ]; then
      echo "detached HEAD $(git -C "$c" rev-parse --short HEAD) is not pushed"
    fi
  fi
}

status=0
for t in "${targets[@]}"; do
  c=$(cd "$t" 2>/dev/null && pwd -P) || { echo "agent_retire: $t: no such directory"; status=1; continue; }
  if [ "$(dirname "$c")" != "$agents_dir" ]; then
    echo "agent_retire: $c is not directly under $agents_dir; refusing"; status=1; continue
  fi
  reasons=$(check "$c")
  if [ -n "$reasons" ]; then
    echo "agent_retire: keeping $c:"; echo "$reasons" | sed 's/^/  /'
    status=1
    continue
  fi
  if [ "$dry" -eq 1 ]; then
    echo "agent_retire: would delete $c"
  else
    rm -rf "$c" && echo "agent_retire: deleted $c"
  fi
done
exit $status
