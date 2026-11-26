#!/usr/bin/env bash
# tools/install_git_hooks.sh -- install the git hooks of the Frontier layout (design/41 section 3).
#
# Usage: tools/install_git_hooks.sh
#   $IMSRG_HUB/hooks/update                 <- tools/git-hooks/hub-update
#   <this clone>/.git/hooks/pre-commit      <- tools/git-hooks/integrator-pre-commit
#   <this clone>/.git/hooks/pre-merge-commit  (same)
# Run it in the integrator's clone only (agent clones get no hooks). Copies, not symlinks, so a
# checkout of another branch cannot change the hooks. Idempotent.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
hooks=$(git -C "$repo" rev-parse --absolute-git-dir)/hooks
: "${IMSRG_HUB:?IMSRG_HUB is not set}"
install -m 755 "$script_dir/git-hooks/hub-update" "$IMSRG_HUB/hooks/update"
install -m 755 "$script_dir/git-hooks/integrator-pre-commit" "$hooks/pre-commit"
install -m 755 "$script_dir/git-hooks/integrator-pre-commit" "$hooks/pre-merge-commit"
git --git-dir="$IMSRG_HUB" config receive.denyNonFastForwards true
git --git-dir="$IMSRG_HUB" config receive.denyDeletes false
echo "installed: $IMSRG_HUB/hooks/update, $hooks/pre-commit, $hooks/pre-merge-commit"
