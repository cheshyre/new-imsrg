#!/usr/bin/env bash
# tools/agent_clone.sh -- make a private checkout for one code-writing agent (design/41 section 3).
#
# Usage: tools/agent_clone.sh <name> [<base>]
#
# Clones the hub ($IMSRG_HUB) into $IMSRG_AGENTS_DIR/<name> (default $IMSRG_WORK/agents/<name>),
# creates the branch feature/<name> from origin/<base> (default develop) with origin/<base> as its
# upstream, and borrows the submodules from the integrator's clone ($IMSRG_WORK/main, or
# $IMSRG_SUBMODULE_SOURCE): each submodule is cloned locally from that clone's module repository at
# the pinned commit, so no network is needed. The clone has a single remote, origin = the hub (no
# GitHub remote). Prints the clone path as the last line of stdout (WorktreeCreate hook contract);
# progress goes to stderr. Refuses an existing directory and a base the hub does not have.
#
# The agent may rename its branch (`git branch -m feature/<milestone>-<topic>`) before its first
# push; it pushes with `git push -u origin HEAD`. Remove the clone with tools/agent_retire.sh.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  sed -n '2,16p' "$0" >&2
  exit 2
fi
name=$1
base=${2:-develop}
case $name in
  ''|*/*|.*|*' '*) echo "agent_clone: bad name '$name' (one path component, no spaces)" >&2; exit 2 ;;
esac
: "${IMSRG_HUB:?agent_clone: IMSRG_HUB is not set}"
: "${IMSRG_WORK:?agent_clone: IMSRG_WORK is not set}"
agents_dir=${IMSRG_AGENTS_DIR:-$IMSRG_WORK/agents}
source_clone=${IMSRG_SUBMODULE_SOURCE:-$IMSRG_WORK/main}
dest=$agents_dir/$name
branch=feature/$name

log() { echo "agent_clone: $*" >&2; }

if [ -e "$dest" ]; then
  log "$dest already exists"; exit 1
fi
if ! git ls-remote --exit-code --heads "$IMSRG_HUB" "$base" >/dev/null; then
  log "the hub has no branch '$base' (push it first)"; exit 1
fi
mkdir -p "$agents_dir"

log "cloning $IMSRG_HUB into $dest (branch $branch from origin/$base)"
git clone --quiet --no-checkout "$IMSRG_HUB" "$dest" >&2
git -C "$dest" checkout --quiet -b "$branch" --track "origin/$base" >&2
# A new branch tracks the base for `git pull`; pushes go to a same-named branch on the hub.
git -C "$dest" config push.default current
git -C "$dest" config imsrg.agentBase "$base"
# Creation record for tools/agent_retire.sh --merged: the commit the clone started from and when.
git -C "$dest" config imsrg.baseCommit "$(git -C "$dest" rev-parse HEAD)"
git -C "$dest" config imsrg.createdAt "$(date +%s)"
# Commit identity: that of the integrator's clone, if it has one.
for key in user.name user.email; do
  if value=$(git -C "$source_clone" config "$key"); then git -C "$dest" config "$key" "$value"; fi
done

# borrow <source worktree> <destination worktree>: init each submodule of the destination from the
# corresponding submodule repository of the source, recursively (nanobind has its own submodule).
borrow() {
  local src=$1 dst=$2 key sub_name path src_git
  [ -f "$dst/.gitmodules" ] || return 0
  { git -C "$dst" config -f .gitmodules --get-regexp '^submodule\..*\.path$' || true; } |
  while read -r key path; do
    sub_name=${key#submodule.}
    sub_name=${sub_name%.path}
    # Only an initialized submodule of the source can lend its repository (rev-parse inside an
    # uninitialized one would find the superproject).
    case $(git -C "$src" submodule status -- "$path" 2>/dev/null) in
      [\ +U]*) src_git=$(git -C "$src/$path" rev-parse --absolute-git-dir) ;;
      *) src_git="" ;;
    esac
    if [ -n "$src_git" ]; then
      git -C "$dst" config "submodule.$sub_name.url" "$src_git"
    else
      log "warning: $src/$path is not initialized; fetching $path from its upstream URL"
    fi
    # .gitmodules asks for shallow clones, which local clones ignore with a warning.
    git -C "$dst" -c protocol.file.allow=always submodule update --quiet --init -- "$path" 2>&1 \
      | grep -v -- '--depth is ignored' >&2 || true
    case $(git -C "$dst" submodule status -- "$path") in
      ' '*) ;;
      *) log "submodule $path did not check out at its pinned commit"; return 1 ;;
    esac
    borrow "$src/$path" "$dst/$path"
  done
}
borrow "$source_clone" "$dest"

# The shared venv link (tools/lib/imsrg_env.sh) and the generated-golden location are per machine.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/imsrg_env.sh
. "$script_dir/lib/imsrg_env.sh"
imsrg_link_venv "$dest"

log "ready: $dest on $branch ($(git -C "$dest" rev-parse --short HEAD))"
echo "$dest"
