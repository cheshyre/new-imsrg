#!/usr/bin/env bash
# tools/hooks/worktree_create.sh -- Claude Code WorktreeCreate hook (design/41 section 3).
#
# Replaces `git worktree add` for subagents with `isolation: worktree` (imsrg-lead,
# imsrg-implementer, imsrg-mechanic): instead of a worktree of the integrator's clone, the agent
# gets its own clone of the hub, $IMSRG_WORK/agents/<name>, on feature/<name> (tools/agent_clone.sh).
# Base branch: develop (or the branch named in $IMSRG_WORK/agents/.default-base, which the
# integrator sets while an effort builds on an unmerged branch), unless the spawning agent itself runs in an agent clone (a lead spawning
# its workers): then the base is that clone's current branch, which must be pushed to the hub and
# up to date there (the hook fails otherwise, telling the lead to push first).
# Stdin: the hook JSON (fields used: name, cwd). Stdout: the clone path as the last line.
# Every call is logged to $IMSRG_WORK/agents/.hooks.log.
set -uo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${IMSRG_WORK:?IMSRG_WORK is not set}"
agents_dir=${IMSRG_AGENTS_DIR:-$IMSRG_WORK/agents}
mkdir -p "$agents_dir"
logf=$agents_dir/.hooks.log
input=$(cat)
printf '%s worktree_create pid=%s input=%s\n' "$(date '+%F %T')" "$$" "$input" >>"$logf"

name=$(printf '%s' "$input" | jq -r '.name // empty')
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
name=$(printf '%s' "$name" | tr -c 'A-Za-z0-9._-' '-' | sed 's/^[-.]*//; s/-*$//')
[ -n "$name" ] || name=agent-$(date +%Y%m%d-%H%M%S)
# Unique directory name.
n=$name; i=2
while [ -e "$agents_dir/$n" ]; do n=$name-$i; i=$((i + 1)); done
name=$n

base=develop
if [ -s "$agents_dir/.default-base" ]; then
  base=$(head -n 1 "$agents_dir/.default-base" | tr -d '[:space:]')
fi
agents_real=$(cd "$agents_dir" && pwd -P)
cwd_real=$(cd "$cwd" 2>/dev/null && pwd -P || echo "$cwd")
case $cwd_real/ in
  "$agents_real"/*/*)
    rel=${cwd_real#"$agents_real"/}
    parent=$agents_real/${rel%%/*}
    if base=$(git -C "$parent" symbolic-ref -q --short HEAD); then
      local_tip=$(git -C "$parent" rev-parse HEAD)
      hub_tip=$(git ls-remote "$IMSRG_HUB" "refs/heads/$base" | awk '{print $1}')
      if [ "$hub_tip" != "$local_tip" ]; then
        echo "worktree_create: the spawning agent's branch $base in $parent is not pushed or" \
             "not up to date on the hub; run 'git push -u origin HEAD' there first" >&2
        printf '%s worktree_create FAIL parent=%s base=%s not pushed\n' "$(date '+%F %T')" \
          "$parent" "$base" >>"$logf"
        exit 1
      fi
    else
      echo "worktree_create: the spawning agent's clone $parent has a detached HEAD" >&2
      exit 1
    fi ;;
esac

if out=$("$script_dir/../agent_clone.sh" "$name" "$base" 2>>"$logf"); then
  path=$(printf '%s\n' "$out" | tail -n 1)
  printf '%s worktree_create OK %s base=%s\n' "$(date '+%F %T')" "$path" "$base" >>"$logf"
  echo "$path"
else
  echo "worktree_create: tools/agent_clone.sh $name $base failed (see $logf)" >&2
  exit 1
fi
