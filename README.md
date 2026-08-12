# EvoCI

EvoCI is a self-improving multi-agent runtime for CI failure recovery.

Current release: **0.7.0**, focused on execution isolation, durable retry semantics,
self-evolution attribution, and fair benchmark budgets.

It dynamically coordinates specialized coding agents, remembers past incidents across runs, and
promotes successful execution patterns into versioned reusable capability packages.

## Core systems

- **Dynamic Multi-Agent Orchestration** — LangGraph owns planning, bounded fan-out/fan-in,
  diagnosis, approval, repair, verification, review, and terminal state.
- **Tool-using leaf agents** — investigators, fixer, and reviewer use one shared bounded tool-loop
  runtime over the existing permission-aware filesystem, Git, command, and skill tools. Their final
  answers remain validated Pydantic models.
- **Cross-run long-term memory** — SQLite and FTS5 retain success and failure episodes plus
  namespaced semantic facts, with retrieved, selected, and used attribution kept distinct.
- **Self-evolving capability registry** — reusable procedures become immutable, validated package
  versions, gather trial evidence, and are promoted or rejected by configurable policy.

## Multi-Agent Orchestration

LangGraph remains the only multi-agent control plane. The coordinator creates independent
investigation tasks and dispatches them with `Send`; reducer-backed state joins evidence before
diagnosis. Coordinator and diagnoser are structured one-shot calls. Each investigator, the fixer,
and the reviewer owns only a local bounded model → tool → result loop with configurable iteration
and tool-call limits.

Role permissions differ. Investigators can read, search, inspect Git, and run allowlisted commands,
but executable tools run in a disposable copy and cannot write the real workspace. The reviewer has
the same execution isolation while reading the real final patch. The fixer can edit and test a
private staging copy with independent filesystem and Git metadata. Its structured full-file edits
then return to LangGraph, which performs deterministic risk checks, pauses for approval where
required, and is the only component that applies those edits to the real workspace.

All current-task agents share atomic run-level model-call and tool-call budgets. A rejected or failed
repair is rolled back to its attempt baseline before the next fixer receives verification feedback,
review blockers, and an attempt summary. Patch application is desired-state idempotent, so replay
after a crash treats an already-applied create, update, or delete as a successful no-op.

Every attempted model call and every actual tool call/result becomes an append-only event. A
`TrajectoryView` projects agents, evidence, diagnoses, patches, failed attempts, tool results,
created/modified files, verification history, and memory/skill attribution from that event stream.
Tool metrics and learning triggers use this projection rather than proposal fields.

## Long-Term Memory

Retrieval records three separate stages: all retrieved IDs, the budget-selected IDs injected into
worker context, and the IDs that a structured leaf output says materially influenced its answer.
The same distinction applies to capabilities. Claimed IDs are validated against the selected
context before `MemoryUsed` or `SkillUsed` is accepted.

Both successful and failed terminal paths persist an episode. Failure episodes retain the failure
reason, attempted hypotheses, verification failures, evidence IDs, actual tools, and repair-attempt
count. Successful runs may additionally consolidate a durable semantic fact; full trajectories stay
in the event store rather than being copied into long-term memory.

## Self-Evolving Capability Registry

Trial and active packages are both retrievable, with active versions ranked first. Workers receive
`SKILL.md` plus declared resource metadata. `run_skill_script` is a formal registry tool and can
execute only a selected trial/active package's declared script; its result emits `SkillUsed`.

After a run, per-execution traces update success/failure counts, average tool calls, average attempts,
patch association, last-used time, and utility. An explicit skill-script failure remains failure
evidence even when the agent later succeeds without it. Trial promotion defaults to two uses, two
successes, a 75% success rate, and rejection after two failures; all thresholds are configurable
with `EVO_*` variables. A colliding `new_skill` slug is rejected. An `update_skill` decision must
identify an exact skill ID and parent version; only this explicit lineage can create a later version.
The parent remains active until the child earns promotion, then becomes superseded.

Experience mining reads the full trajectory, including helper scripts created through real tools.
The curator first performs deterministic lifecycle maintenance and an indexed duplicate shortlist.
An auxiliary model (falling back to the main model) reviews complete package manifests, files,
tests, utility, and provenance. A merge is always a new validated trial; its sources remain active
until that merge later earns promotion.

## Quick start

```bash
cp .env.example .env
uv sync
uv run evoci doctor
uv run pytest
uv run ruff check .
uv run mypy src
```

For a deterministic demonstration that does not require a model API, run:

```bash
uv run evoci demo
```

The demo deliberately starts with a broken calculator, launches three investigators with dynamic
`Send`, reduces their evidence, repairs one source file, runs the real test suite, asks an independent
reviewer, checkpoints the run, and writes a condensed episode.

## CLI

```bash
evoci doctor
evoci run TASK_ID --task-file task.json
evoci resume RUN_ID
evoci runs list
evoci runs inspect RUN_ID
evoci memory search "pytest import" --namespace repo:owner/repo
evoci skills list
evoci skills show SKILL_ID
evoci skills curate
evoci benchmark --manifest benchmarks/mvp-30.jsonl \
  --dataset /path/to/ci-repair-bench.jsonl --variant multi
evoci benchmark --manifest benchmarks/continual.jsonl \
  --dataset /path/to/ci-repair-bench.jsonl --variant evo --continual
```

Live task files contain only the repository identity, CI failure, and an already prepared workspace:

```json
{
  "repo": {"owner": "owner", "name": "repo", "base_commit": "abc123"},
  "ci_failure": {
    "summary": "pytest failed",
    "log_excerpt": "AssertionError",
    "failed_commands": [["python", "-m", "pytest", "-q"]],
    "task_family": "test"
  },
  "workspace_path": "/absolute/path/to/worktree"
}
```

## Safety and durability

- Investigator and reviewer reads inspect the real workspace, while commands, tests, and non-writing
  skill scripts execute in disposable copies. Worker and manifest permissions must both allow a
  skill write before it can reach a writable worker workspace.
- Fixer tool writes are confined to a staging copy with independent Git metadata; graph-owned apply
  happens only after deterministic checks and any required approval.
- Commands use argv execution without a shell, an executable allowlist, workspace validation,
  restricted environment variables, output truncation, and timeouts.
- Workflow, manifest-plus-lockfile, deletion, and broad changes pause through a durable interrupt
  before any file is written.
- SQLite checkpoints retain successful parallel writes. Desired-state patch application makes
  create/update/delete replay safe after a crash, while rejected attempts are rolled back before
  retry.
- Skill scripts require a selected trial/active package, declared file membership, static safety
  validation, a restricted environment, a workspace boundary, and a timeout.
- Core repair success is frozen before best-effort memory, mining, promotion, or curator work;
  post-run failures emit `LearningError` without overturning the repair outcome.

These are application-level guardrails, not an OS/container security boundary.

Sandbox/security isolation intentionally remains deferred in this iteration.

## Evaluation

The benchmark adapter uses separate `AgentTaskView` and `GroundTruth` models; success commits, gold
diffs, changed files, and error types cannot enter worker context. Ground truth is loaded only after
execution for evaluation. The CLI checks out each failing commit into a dedicated worktree and runs
the selected implementation, rather than a placeholder metrics callback. Benchmark-only approval
accepts every durable HITL interrupt because the task already runs in an isolated benchmark
workspace; interactive runs still require a human decision:

| Variant | Runtime behavior |
| --- | --- |
| `single` | One bounded coding leaf; no LangGraph multi-agent flow, memory, or capabilities |
| `multi` | Full LangGraph orchestration; memory and capabilities disabled |
| `multi-memory` | Full orchestration plus persistent memory; capabilities disabled |
| `evo` | Full orchestration, memory, capabilities, learning, promotion, and curator |

CI-Repair-Bench workflow YAML is never interpreted as a shell command. Structured workflow and log
values are normalized explicitly; only reliable commands extracted from failed-step logs are replayed.
If no command is available, benchmark verification is `not_available`, not resolved.

Harness outcomes (`agent_declared_success`, `targeted_verification_passed`, and `review_passed`) are
reported separately from evaluator-owned `benchmark_verification_status`. Before the agent runs, a
pristine failing-commit workspace must reproduce failure with the candidate command. A command that
already passes is an invalid oracle and yields `not_available`. `benchmark_resolved` is true only
for fail-before/pass-after replay. Final changed files and line counts come only from Git state in
the real benchmark workspace; trajectory staging writes remain auxiliary `attempted_files`.
`gold_file_overlap` is diagnostic and never a correctness oracle.

Aggregate output separates evaluator coverage from repair success: `evaluation_coverage` is the
fraction with `passed`/`failed` verification, while `benchmark_success_rate` is passed divided only
by those evaluable tasks. Repair-phase and post-run model/tool call counts are also reported
separately.

Other metrics—including model calls, tool calls/failures, workers, rounds, attribution, and learning—
come from the real result and trajectory events. `--continual` reuses one memory/capability registry
across sequential tasks and writes `continual_learning.csv` with the learning curve.

`benchmarks/mvp-30.jsonl` has the required balanced 30 slots, currently marked with explicit skip
reasons until a local CI-Repair-Bench export is selected and its repositories are reproducibly
available. Three offline rows exercise adapter and leakage behavior in `tests/fixtures/ci_tasks`.

Live runs require `EVO_MODEL_BASE_URL`, `EVO_MODEL_API_KEY`, and `EVO_MODEL_NAME`. EvoCI does not
read Codex, Claude Code, or personal agent configuration.

## Status and boundaries

The local MVP focuses on orchestration, durable execution, memory, capability evolution, and real
fixture/live-export benchmark execution. Remote repository checkout and model-backed runs require
the corresponding network and endpoint access. Distributed execution, Web UI, MCP server, TUI,
multi-provider routing, and OS-level sandboxing are outside this iteration.

See [architecture](docs/architecture.md), [memory](docs/memory.md),
[capabilities](docs/skills.md), and [evaluation](docs/evaluation.md) for design details.
