#!/usr/bin/env bash
# tools/hooks/worktree_remove.sh -- Claude Code WorktreeRemove hook (design/41 section 3).
#
# Deletes an agent clone made by worktree_create.sh, but only through tools/agent_retire.sh, which
# keeps the clone (exit 1, so Claude reports the removal as failed) if anything is uncommitted or
# not pushed to the hub. Paths outside $IMSRG_WORK/agents are never touched.
# Stdin: the hook JSON (field used: worktree_path).
set -uo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${IMSRG_WORK:?IMSRG_WORK is not set}"
agents_dir=${IMSRG_AGENTS_DIR:-$IMSRG_WORK/agents}
logf=$agents_dir/.hooks.log
input=$(cat)
printf '%s worktree_remove input=%s\n' "$(date '+%F %T')" "$input" >>"$logf"
path=$(printf '%s' "$input" | jq -r '.worktree_path // empty')
if [ -z "$path" ]; then echo "worktree_remove: no worktree_path" >&2; exit 1; fi
"$script_dir/../agent_retire.sh" "$path" >>"$logf" 2>&1
rc=$?
printf '%s worktree_remove rc=%s %s\n' "$(date '+%F %T')" "$rc" "$path" >>"$logf"
if [ $rc -ne 0 ]; then
  echo "worktree_remove: kept $path (unpushed or uncommitted work; see $logf)" >&2
fi
exit $rc
