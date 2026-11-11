# 41 — Agent workflow (how the unsupervised session works)

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

## 2. Git workflow (local only *(user)*)

* `main` is never touched. The integration branch is **`develop`**, created from the
  design-plan commit at the start of the run.
* One branch per task: `feature/<milestone>-<topic>` (e.g. `feature/m3-two-body-channels`),
  created from `develop`. A milestone lead may stack task branches on its milestone branch
  `feature/<milestone>` and merge them there.
* Commits are small and modular. The message gives a subject line (what), a body (why, and any
  design decision taken), and the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  (or the model that actually wrote it).
* Before merging into `develop`: `tools/ci.sh dev-openmp` passes; the branch is rebased on, or
  has merged, the current `develop`; then `git merge --no-ff`. If a merge breaks `develop`,
  revert the merge commit immediately and fix it on the branch.
* Never `git push`. Never use bare `git stash`: the stash is shared across worktrees. Use WIP
  commits instead.

## 3. Parallel work

* Parallel leads use `isolation: "worktree"`, one worktree per milestone, so they never edit
  the same checkout. Workers inside one lead's milestone share the lead's worktree and must
  touch disjoint files, or run one after another. Workers that share a checkout use separate
  build directories (`-B build/<task>`) or build one at a time; two builds must never run in the
  same build directory.
* Kokkos is built once per backend into the deps prefix (`tools/bootstrap_deps.sh`), so a new
  worktree only compiles our code. ccache is shared.
* Merge conflicts are resolved by the orchestrator (or delegated to the lead that owns the
  branch), never by deleting the other side's work.

## 4. Briefing subagents (they know only what you tell them)

Every task brief contains:
1. the goal, and the acceptance test(s) that define done;
2. the design-doc sections and `design/reference/*.md` sections to read, plus the exact
   reference source files (with line ranges) for numerics;
3. the files to create or touch (and files NOT to touch);
4. the branch and worktree to work in, and the commit-message convention;
5. what to report back: a ≤ 200-word summary, test results, open questions (which are also
   appended to `design/QUESTIONS.md` with the task id).

## 5. Context management *(user: compact at ~60 % for Haiku/Sonnet, ~80 % for Opus)*

* Claude Code supports a global auto-compact percentage (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`,
  which can only lower the default threshold) and, per agent definition, `autoCompactWindow`
  (tokens). Configuration (to be enabled by the user; see `design/claude-settings.proposed.json`):
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80` for everything, plus a smaller `autoCompactWindow` in the
  Sonnet/Haiku agent definitions so that workers compact at roughly 60 % of their real window.
  Check the exact semantics of `autoCompactWindow` at the start of the run, and adjust the agent
  files if needed.
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
