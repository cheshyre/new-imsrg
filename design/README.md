# Design documents

Reading order for a new agent or developer:

| File | Content |
|------|---------|
| `../AGENTS.md` | Golden rules, build commands, conventions (read first) |
| `STATUS.md` | Current state and progress log |
| `00-requirements-and-decisions.md` | Requirements, scope and priorities, decision log (D1–D18) |
| `10-architecture.md` | Repository layout, module DAG, core types, runtime, header policy |
| `12-compute-backend.md` | Kokkos facade, linalg, kernel strategy, device-code rules, MPI plan |
| `13-physics-modules.md` | Per-module specification and what must match imsrg++ |
| `14-imsrg3-plan.md` | IMSRG(3): layout benchmark, brute-force validation, N7 / 3f2 / triples |
| `15-operator-dump-format.md` | Text format for exchanging operators with imsrg-devel (golden data) |
| `16-config-and-cli.md` | TOML schema, legacy `key=value` mapping of every old parameter, operator-name grammar |
| `18-python-api.md` | nanobind Python API plan and the mapping from the old pyIMSRG |
| `20-build-and-tooling.md` | Dependencies, CMake structure, presets, scripts, reference build, Makefile |
| `30-testing-and-validation.md` | Test layers, tolerances, golden data, integration test matrix, deviation log |
| `40-roadmap.md` | Milestones M0–M13 with acceptance criteria and agent roles |
| `41-agent-workflow.md` | How the multi-agent session works: roles, git, parallelism, briefs, context |
| `QUESTIONS.md` | Open questions for the project owner (append, never block) |
| `NEXT_SESSION_PROMPT.md` | The prompt that starts the unsupervised implementation run |
| `claude-settings.proposed.json` | Proposed Claude Code settings for the unattended run (owner installs) |
| `reference/01..05-*.md` | Detailed analyses of imsrg-devel with file:line citations (feature checklists) |

**When documents disagree**, this order wins: `00` (decisions) > `10` > `12` > `14` > `13` > the
rest. The `reference/` files describe the **old** code; their "suggestions for the rewrite" (e.g.
the layering proposal in `reference/01`) do not bind us. Question numbers inside
`16-config-and-cli.md` and `18-python-api.md` are local to those files; the canonical list is
`QUESTIONS.md`. Use the file conventions of `10-architecture.md` (`.hpp`/`.cpp`) even where a doc
writes `.hh`.

Reference analyses:
* `reference/01-core-data-model.md`: ModelSpace, channels, Operator, TwoBodyME, 3B storage, AngMom, bugs
* `reference/02-commutators.md`: every commutator term, dispatch flags, algorithms, hotspots, GPU fit
* `reference/03-solvers-generators-driver.md`: driver pipeline, methods, BCH, generators, outputs
* `reference/04-hf-mbpt-io.md`: HF, natural orbitals, RPA, every file format (me2j loop order!)
* `reference/05-operators-bindings-tests-build.md`: operator catalog, pyIMSRG API, tests, build, GSL/Boost usage
