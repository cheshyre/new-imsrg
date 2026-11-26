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
* **Frontier (since 2026-11-26).** The main session is the integrator in `$IMSRG_WORK/main`:
  it commits there with `IMSRG_INTEGRATOR=1`, merges feature branches from the hub into `develop`
  (`--no-ff`) after `tools/ci.sh frontier-cpu` passes on the merged result, then pushes `develop`
  to the hub and to GitHub (design/41 §2). Code-writing agents (`imsrg-lead`, `imsrg-implementer`,
  `imsrg-mechanic`) get their own clone of the hub through the WorktreeCreate hook
  (`isolation: worktree`, design/41 §3) and push only their feature branch; tell each one the
  branch name to use. Read-only agents work in this clone. Set `$IMSRG_WORK/agents/.default-base`
  when agents must start from an unmerged branch, and retire merged clones with
  `tools/agent_retire.sh --all --merged`.
* **Compute node.** Builds and tests run in the allocation named `imsrg`
  (`salloc -A nph123 -N 1 -t 02:00:00 -J imsrg --no-shell`; queue the next one with
  `sbatch --dependency=afterany:<job> -J imsrg --wrap "sleep 7190"` before it ends).
* Run long builds and test suites in the background when possible, and never poll in tight loops.
* When a subagent reports a numerical mismatch, do not accept "close enough". Have it find the
  cause, or record it as a question with evidence.
* Commit trailer: `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that
  wrote the change (e.g. `Claude Opus 5.5`, `Claude Sonnet 5.5`, `Claude Haiku 5.5`).
