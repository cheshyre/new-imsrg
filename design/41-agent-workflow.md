# 41 — Agent workflow (how the unsupervised session works)

*Frontier layout since 2026-11-26: sections 2 and 3 describe the hub and the per-agent clones;
the owner interview after the Frontier port decides proposals P-F1 and P-F2.*

## 1. Roles and models *(user: "Opus agents orchestrate bigger efforts, which then orchestrate Haiku or Sonnet agents for small tasks")*

| Role | Model | Agent definition | Does | Does not |
|------|-------|------------------|------|----------|
| Orchestrator (main session) | Opus | — | Owns `develop`, picks the next milestone from `40-roadmap.md`, spawns milestone leads, merges finished branches, keeps `STATUS.md` and `QUESTIONS.md` current | Write large amounts of code itself |
| Milestone lead | Opus | `.claude/agents/imsrg-lead.md` | Plans the milestone into small tasks (≤ ~300 lines of code each), spawns workers, reviews their diffs, runs `tools/ci.sh`, makes the final commits on the milestone branch, reports back | Merge into `develop` |
| Implementer | Sonnet | `.claude/agents/imsrg-implementer.md` | One well-specified task: code + tests + docstrings, run the module's tests, commit on the branch it is given | Change the design or the module DAG without asking its lead |
| Mechanic | Haiku | `.claude/agents/imsrg-mechanic.md` | Mechanical tasks: CMake wiring, boilerplate, renames, docstring passes, formatting, simple scripts, golden-data plumbing | Numerics or kernel logic |
| Reviewer | Sonnet (Opus for kernels and IMSRG(3)) | `.claude/agents/imsrg-reviewer.md` | Read-only review of a branch diff against the design docs, device-code rules (`12` §5), and tests; reports findings | Edit files |

Nesting is supported (up to 3 levels below the main session by default), so the
orchestrator → lead → worker structure works as described.

## 2. Git workflow (hub on Frontier, since 2026-11-26; proposal P-F1, owner interview decides)

* Repositories: the bare **hub** `$IMSRG_HUB` holds every branch; `$IMSRG_WORK/main` is the
  **integrator's clone** (remotes `origin` = hub, `github` = `$IMSRG_GITHUB`), used by the main
  session and the owner; each code-writing agent gets its own clone `$IMSRG_WORK/agents/<name>`
  (only remote: `origin` = hub).
* `main` is never touched. The integration branch is **`develop`**. One branch per task:
  `feature/<milestone>-<topic>`, cut from `develop` (an agent clone starts on
  `feature/<name>`; the agent renames it with `git branch -m` before its first push). A
  milestone lead's workers branch from the lead's branch as pushed to the hub.
* Commits are small and modular. The message gives a subject line (what), a body (why, and any
  design decision taken), and the trailer `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
  naming the model that wrote it.
* **Pushes.** Agents push only their own feature branch, only to the hub
  (`git push -u origin HEAD`), and push before they report: unpushed work is lost when a clone is
  retired. Agents never touch GitHub or git remotes and never force-push.
* **Integration (integrator only, `IMSRG_INTEGRATOR=1`).** In `$IMSRG_WORK/main`:
  `git fetch origin`, merge the feature branch into `develop` with `--no-ff`, run
  `tools/ci.sh frontier-cpu` on the merged result (plus `ctest -L slow` for numerics merges),
  then `git push origin develop` and `git push github develop`. If a merge breaks `develop`,
  revert the merge commit and fix it on the branch.
* **GitHub policy (proposal P-F2).** Only the integrator fetches, pulls and pushes `github`:
  GitHub changes come in with `git merge --ff-only`; `develop` goes to GitHub only after CI passed
  and after it was pushed to the hub; never force-push, never delete or rename a remote branch,
  never rewrite published history, `main` stays untouched. Feature branches are not pushed to
  GitHub.
* **Enforcement** (`tools/install_git_hooks.sh`): the hub's `update` hook rejects every update of
  `main`, updates of `develop` without `IMSRG_INTEGRATOR=1`, tags, and (with
  `receive.denyNonFastForwards=true`) non-fast-forward updates; the integrator's clone has
  `pre-commit` and `pre-merge-commit` hooks that reject commits without `IMSRG_INTEGRATOR=1`; the
  Claude Code hook `tools/hooks/agent_guard.py` (section 3) stops agents earlier, with a message.
* Never use bare `git stash`; use WIP commits.
* *Superseded (Mac, D31/D44):* runs started in app worktrees, `develop` advanced with
  `git update-ref`, nothing was pushed, and the owner re-synced the main checkout after each run.
* **Pacing (D31):** at most 3 leads, Sonnet-heavy delegation; when the 5-hour usage limit stops
  agents, resume each with `SendMessage` after the reset. An owner interview follows every run
  (D42).

## 3. Parallel work: one clone per code-writing agent (proposal P-F1, scheme (a))

* **Who gets a clone.** `imsrg-lead`, `imsrg-implementer` and `imsrg-mechanic` declare
  `isolation: worktree` in their frontmatter. Read-only agents (`imsrg-reviewer`, Explore, Plan,
  web fetchers) run in `$IMSRG_WORK/main` without isolation and change nothing there.
* **How (Claude Code hooks, `.claude/settings.json`).** The `WorktreeCreate` hook
  (`tools/hooks/worktree_create.sh`) replaces `git worktree add`: it runs
  `tools/agent_clone.sh <name> <base>`, which clones the hub into `$IMSRG_WORK/agents/<name>`
  (`<name>` = `agent-<agent id>`), checks out `feature/<name>` from `origin/<base>`, borrows the
  submodules from the integrator's clone (local clones, no network), copies the git identity and
  links the venv, and prints the path (8 s). `<base>` is `develop`, or the branch in
  `$IMSRG_WORK/agents/.default-base` (set by the integrator while an effort builds on an
  unmerged branch), or, when the spawning agent itself runs in an agent clone (the hook's `cwd`
  is that clone), that clone's current branch, which must be pushed and current on the hub (the
  spawn fails otherwise, telling the lead to push). The `PreToolUse` hook
  (`tools/hooks/agent_guard.py`) binds each subagent by `agent_id` to the clone its first tool call
  runs in (otherwise: read-only) and denies edits outside that clone (and temp directories),
  commands that name the integrator's clone or another agent's clone, git commands run outside
  the clone, pushes other than to `origin`, pushes of `develop`/`main`, force pushes and remote
  changes; read-only agents may run only read-only git commands. Claude Code itself also refuses
  an isolated agent's edits outside its worktree. Every hook call is logged to
  `$IMSRG_WORK/agents/.hooks.log`.
* **Tested 2026-11-26** with headless `claude -p` sessions: a mechanic got its clone, committed
  and pushed its branch, and its edit of the integrator's clone, `git -C <main>`, `cd <main>` and
  a push to `develop` were denied; a lead committed and pushed in its clone and its mechanic's
  clone started from the lead's pushed commit; the hub rejected a direct push to `develop` and
  `main`, and the integrator's clone rejected a commit without `IMSRG_INTEGRATOR=1`.
* **Removal.** Claude Code does not call `WorktreeRemove` when a subagent finishes (also not for
  an unchanged clone), so clones stay until retired. `tools/agent_retire.sh` (also the
  `WorktreeRemove` hook) deletes a clone only if it has no uncommitted or untracked files, no
  stash, and every local branch is contained in a hub branch; `--merged` also requires every
  branch to be merged into the hub's `develop`. The integrator runs
  `tools/agent_retire.sh --all --merged` after merges; clones it keeps are reported with the
  reason.
* **Fallback (scheme (b), not needed so far).** Without the WorktreeCreate hook, the integrator or
  a lead runs `tools/agent_clone.sh <task> [<base>]` itself, names the clone in the brief, and
  every command starts with `cd <clone> &&`; `agent_guard.py` binds a code-writing agent to the
  clone of its first `cd <clone> &&` command, so the same guards apply.
* **Builds.** One build tree per clone (`build/<preset>`); Kokkos comes from `$IMSRG_DEPS_PREFIX`
  and ccache (`$CCACHE_DIR`, `CCACHE_BASEDIR=$IMSRG_WORK`) is shared, so a new clone compiles
  mostly from cache. All agents share the compute allocation `imsrg` through `srun --overlap`.
* Merge conflicts are resolved by the integrator (or delegated to the lead that owns the
  branch), never by deleting the other side's work.

## 4. Briefing subagents (they know only what you tell them)

Every task brief contains:
1. the goal, and the acceptance test(s) that define done;
2. the design-doc sections and `design/reference/*.md` sections to read, plus the exact
   reference source files (with line ranges) for numerics;
3. the files to create or touch (and files NOT to touch);
4. the branch name to use (the agent renames `feature/<name>` to it), the base it starts from,
   the commit-message convention, and that it pushes before reporting;
5. what to report back: a ≤ 200-word summary, test results, open questions (which are also
   appended to `design/QUESTIONS.md` with the task id).

## 5. Context management *(user: compact at ~60 % for Haiku/Sonnet, ~80 % for Opus)*

* Claude Code supports a global auto-compact percentage (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`,
  which can only lower the default threshold) and, per agent definition, `autoCompactWindow`
  (tokens). Configuration (to be enabled by the user; see `design/claude-settings.proposed.json`):
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80` for everything, plus a smaller `autoCompactWindow` in the
  Sonnet/Haiku agent definitions so that workers compact at roughly 60 % of their real window.
  **Checked at the start of the run (2026-11-11, Claude Code docs):** `autoCompactWindow` is the
  absolute token count at which that subagent auto-compacts (integer, 100000–1000000; it only
  takes effect below the inherited window and is ignored when `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
  is set). Sonnet 5.5, Haiku 5.5 and Opus 5.x run with a 1M-token window by default, so the old
  value 150000 meant ~15 %, not 60 %. The agent files now use `autoCompactWindow: 600000` for the
  Sonnet/Haiku agents (implementer, mechanic, reviewer) and `800000` for `imsrg-lead` (Opus). How
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` combines with `autoCompactWindow` is not documented; do not
  set both (see Q19). **Decided (owner, 2026-11-15, D31):** per-agent windows only. The docs say
  the percentage override also applies to subagents, on top of their `autoCompactWindow` (80 would
  compact Sonnet/Haiku agents at ~480k tokens), so it is not set; the orchestrator gets
  `"autoCompactWindow": 800000` from the session settings, which agents with their own frontmatter
  value do not inherit (`design/claude-settings.proposed.json`). The frontmatter field needs Claude
  Code ≥ v2.1.296 and a plain integer; `CLAUDE_CODE_AUTO_COMPACT_WINDOW` would override everything
  and must not be set. Agent definitions load at session start, so the new values apply from the
  next session on.
* More important than compaction: **keep tasks small** so workers finish well before they need
  it, and **keep the orchestrator lean**. Leads report summaries, not diffs. Detailed results
  live in files (`STATUS.md`, `design/benchmarks/`, test logs), not in the conversation.
* After a compaction, re-read `design/STATUS.md` and the current milestone section of
  `40-roadmap.md` before continuing.

## 6. Status and questions

* `design/STATUS.md`: the orchestrator appends one entry per merged branch (date/time, branch,
  what landed, test status, follow-ups). Its "Current state" section at the top is always
  up to date. A fresh session or a post-compaction context starts there.
* `design/QUESTIONS.md`: every question that needs the user goes here (id, date, context,
  options, what was assumed meanwhile). **Never block on a question:** choose the most
  reversible option, record the assumption, and continue.

## 7. Quality rules for every task

* Tests first or alongside. No merge without tests.
* Every public function has a docstring (conventions, units, complexity for kernels).
* No new dependency or module edge without updating `10-architecture.md` and the layering check.
* Numerical mismatch vs reference → investigate until it is understood. Never loosen a tolerance
  silently. Document reference bugs in `30-testing-and-validation.md` §6.
* Performance claims need a benchmark entry.
