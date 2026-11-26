#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 The imsrg authors
"""Claude Code PreToolUse hook: keep every subagent inside its own checkout.

design/41 section 3. Runs with the system python3 (3.6 on Frontier), standard library only.

The main session (no `agent_id` in the hook input) is never restricted. Each subagent is bound,
by its `agent_id`, to one place, recorded in $IMSRG_WORK/agents/.bind/<agent_id>:
  * a clone `$IMSRG_WORK/agents/<x>` if its first tool call runs there (worktree isolation via
    tools/hooks/worktree_create.sh), or later if a code-writing agent (imsrg-lead,
    imsrg-implementer, imsrg-mechanic) runs a Bash command starting with `cd <clone> &&`
    (fallback scheme with tools/agent_clone.sh);
  * otherwise `readonly` (reviewers, explorers, planners working in the integrator's clone).
Rules:
  * clone-bound: Edit/Write/NotebookEdit only inside the clone or a temp/scratch directory;
    Bash commands may not name the integrator's clone or another agent's clone, may not run git
    outside the clone, may not push anywhere but origin (the hub), and may not touch remotes;
  * readonly: no file edits outside temp/scratch directories and no git command that changes a
    repository (commit, merge, checkout, push, ...).
Denials print the reason (exit 2), which Claude shows to the agent. Decisions are logged to
$IMSRG_WORK/agents/.hooks.log.
"""

import json
import os
import re
import sys
import time

WRITER_TYPES = ("imsrg-lead", "imsrg-implementer", "imsrg-mechanic")
GIT_READ_SUBCOMMANDS = {
    "log", "diff", "show", "status", "ls-files", "ls-tree", "grep", "rev-parse", "rev-list",
    "merge-base", "blame", "cat-file", "describe", "shortlog", "for-each-ref", "name-rev",
    "show-ref", "fetch", "ls-remote", "whatchanged", "range-diff", "diff-tree", "count-objects",
    "version", "help",
}


def real(path):
  return os.path.realpath(os.path.expanduser(path))


def inside(path, root):
  path, root = real(path), real(root)
  return path == root or path.startswith(root.rstrip("/") + "/")


def temp_dirs():
  dirs = ["/tmp", "/var/tmp", "/dev/null"]
  for var in ("TMPDIR", "CLAUDE_SCRATCHPAD"):
    if os.environ.get(var):
      dirs.append(os.environ[var])
  return dirs


def log(agents_dir, msg):
  try:
    with open(os.path.join(agents_dir, ".hooks.log"), "a") as f:
      f.write("%s agent_guard %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
  except OSError:
    pass


def deny(agents_dir, who, reason):
  log(agents_dir, "DENY %s: %s" % (who, reason))
  sys.stderr.write("agent_guard: " + reason + "\n")
  sys.exit(2)


def leading_cd(command):
  """The directory of a leading `cd <dir> &&` (or `;`), else None."""
  m = re.match(r"\s*cd\s+(\"[^\"]+\"|'[^']+'|\S+)\s*(&&|;)", command)
  if not m:
    return None
  return os.path.expandvars(m.group(1).strip("\"'"))


def git_invocations(command):
  """Yields (options-before-subcommand, subcommand, rest) for each `git ...` in the command."""
  for m in re.finditer(r"(?:^|[;&|(`]|\$\()\s*(?:\w+=\S*\s+)*git((?:\s+-{1,2}[\w-]+(?:[ =](?!-)\S+)?)*)\s+([\w-]+)([^;&|]*)",
                       command):
    yield m.group(1), m.group(2), m.group(3)


def main():
  try:
    data = json.load(sys.stdin)
  except ValueError:
    return 0
  agent_id = data.get("agent_id")
  if not agent_id:
    return 0  # the main session (integrator)
  work = os.environ.get("IMSRG_WORK")
  if not work:
    return 0
  agents_dir = os.environ.get("IMSRG_AGENTS_DIR", os.path.join(work, "agents"))
  main_clone = os.path.join(work, "main")
  agent_type = data.get("agent_type", "")
  tool = data.get("tool_name", "")
  tin = data.get("tool_input") or {}
  cwd = data.get("cwd") or os.getcwd()
  who = "%s(%s)" % (agent_type, agent_id[:12])

  bind_dir = os.path.join(agents_dir, ".bind")
  os.makedirs(bind_dir, exist_ok=True)
  bind_file = os.path.join(bind_dir, re.sub(r"[^\w.-]", "_", agent_id))
  binding = None
  if os.path.exists(bind_file):
    with open(bind_file) as f:
      binding = f.read().strip()

  def clone_of(path):
    """The agent clone directory containing path, or None."""
    rp, ra = real(path), real(agents_dir)
    if not rp.startswith(ra + "/"):
      return None
    first = rp[len(ra) + 1:].split("/")[0]
    if not first or first.startswith("."):
      return None
    return os.path.join(ra, first)

  command = tin.get("command", "") if tool == "Bash" else ""
  if binding is None or binding == "readonly":
    new = clone_of(cwd)
    if new is None and binding is None and tool == "Bash" and agent_type in WRITER_TYPES:
      d = leading_cd(command)
      if d:
        new = clone_of(d)
    elif new is None and binding == "readonly" and tool == "Bash" and agent_type in WRITER_TYPES:
      d = leading_cd(command)
      if d:
        new = clone_of(d)
    if new is not None:
      binding = new
      with open(bind_file, "w") as f:
        f.write(binding + "\n")
      log(agents_dir, "BIND %s -> %s" % (who, binding))
    elif binding is None:
      binding = "readonly"
      with open(bind_file, "w") as f:
        f.write(binding + "\n")
      log(agents_dir, "BIND %s -> readonly (cwd %s)" % (who, cwd))

  edit_path = None
  if tool in ("Edit", "Write", "MultiEdit"):
    edit_path = tin.get("file_path")
  elif tool == "NotebookEdit":
    edit_path = tin.get("notebook_path")

  if binding == "readonly":
    if edit_path is not None:
      p = edit_path if os.path.isabs(edit_path) else os.path.join(cwd, edit_path)
      if not any(inside(p, t) for t in temp_dirs()):
        deny(agents_dir, who, "this agent has no checkout of its own (read-only); it may not "
             "edit %s. Code-writing agents work in their clone under %s." % (edit_path, agents_dir))
    if command:
      for _, sub, _rest in git_invocations(command):
        if sub == "branch" and not re.search(r"\s-[dDmMcC]\b|--delete|--move|--copy|--force",
                                             _rest):
          continue
        if sub not in GIT_READ_SUBCOMMANDS and sub != "branch":
          deny(agents_dir, who, "read-only agent: 'git %s' would change a repository" % sub)
        if sub == "branch":
          deny(agents_dir, who, "read-only agent: 'git branch%s' would change a repository"
               % _rest)
    return 0

  clone = binding
  if edit_path is not None:
    p = edit_path if os.path.isabs(edit_path) else os.path.join(cwd, edit_path)
    if not inside(p, clone) and not any(inside(p, t) for t in temp_dirs()):
      deny(agents_dir, who, "edits are confined to your clone %s (got %s)" % (clone, edit_path))
    return 0

  if not command:
    return 0
  # Paths of other checkouts may not appear in a command at all.
  for root in [main_clone] + [os.path.join(agents_dir, d) for d in os.listdir(agents_dir)
                              if not d.startswith(".")]:
    if inside(root, clone):
      continue
    for form in {root, real(root)}:
      if re.search(re.escape(form) + r"(/|\b|$)", command):
        deny(agents_dir, who, "your checkout is %s; commands may not refer to %s" % (clone, form))
  if re.search(r"(^|[^\w])(\$IMSRG_WORK|\$\{IMSRG_WORK\})/main\b", command):
    deny(agents_dir, who, "your checkout is %s; commands may not refer to the integrator's "
         "clone" % clone)
  eff = leading_cd(command) or cwd
  for opts, sub, rest in git_invocations(command):
    m = re.search(r"(?:-C|--git-dir|--work-tree)[ =](\S+)", opts)
    target = m.group(1).strip("\"'") if m else eff
    target = os.path.expandvars(target)
    if not os.path.isabs(target):
      target = os.path.join(eff, target)
    if not inside(target, clone):
      deny(agents_dir, who, "git may only run inside your clone %s (this command runs it in %s); "
           "start commands with 'cd %s &&'" % (clone, target, clone))
    if sub == "remote" and re.search(r"\b(add|remove|rm|rename|set-url|set-head|prune)\b", rest):
      deny(agents_dir, who, "agents never change git remotes")
    if sub == "push":
      words = rest.split()
      positional = [w for w in words if not w.startswith("-")]
      if positional and positional[0] != "origin":
        deny(agents_dir, who, "agents push only to origin (the hub), never to '%s'"
             % positional[0])
      if re.search(r"github|https?://|git@", rest):
        deny(agents_dir, who, "agents never push to GitHub")
      if re.search(r"(^|\s)(--force|-f|--force-with-lease|--delete|-d|--mirror|--all)\b", rest):
        deny(agents_dir, who, "agents never force-push, delete or mirror branches")
      for w in positional[1:]:
        dst = w.split(":")[-1].lstrip("+")
        if dst in ("develop", "main", "refs/heads/develop", "refs/heads/main"):
          deny(agents_dir, who, "agents push only their feature branch, never %s" % dst)
  return 0


if __name__ == "__main__":
  sys.exit(main())
