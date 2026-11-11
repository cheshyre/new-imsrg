@AGENTS.md

# Claude-specific notes

* **Orchestration** follows `design/41-agent-workflow.md`. The main session (Opus) orchestrates;
  Opus milestone leads (`.claude/agents/imsrg-lead.md`) delegate small tasks to Sonnet
  implementers (`imsrg-implementer`), Haiku mechanics (`imsrg-mechanic`), and read-only reviewers
  (`imsrg-reviewer`). Brief subagents completely: they know only what you tell them (see §4 of the
  workflow doc).
* **Start of every session, and after every compaction:** read `design/STATUS.md` (current state),
  then the active milestone in `design/40-roadmap.md`, then the design docs that milestone names.
* **Keep context small.** Delegate file-heavy reading. Ask subagents for summaries of at most
  200 words. Put details into files (`design/STATUS.md`, `design/benchmarks/`, test logs), not
  into the conversation.
* **Parallel leads** run with `isolation: "worktree"`. Merge their branches into `develop`
  yourself, after `tools/ci.sh` passes on the merged result.
* Run long builds and test suites in the background when possible, and never poll in tight loops.
* When a subagent reports a numerical mismatch, do not accept "close enough". Have it find the
  cause, or record it as a question with evidence.
* Commit trailer: `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that
  wrote the change (e.g. `Claude Opus 5.5`, `Claude Sonnet 5.5`, `Claude Haiku 5.5`).
