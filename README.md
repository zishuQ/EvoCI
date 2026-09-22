# EvoCI

EvoCI is a self-improving multi-agent runtime for CI failure recovery.

Current release: **0.7.0**, focused on execution isolation, durable retry semantics,
self-evolution attribution, and fair benchmark budgets.

It dynamically coordinates specialized coding agents, remembers past incidents across runs, and
turns successful execution patterns into a single current reusable capability package.

## Core systems

- **Supervisor–Worker orchestration** — LangGraph owns planning, bounded worker fan-out/fan-in,
  patch integration, approval, verification, rollback, and terminal state.
- **Tool-using leaf agents** — Supervisor (read-only) and Worker (investigate or repair) share one
  bounded tool-loop runtime over the existing permission-aware filesystem, Git, command, and skill
  tools. Final Worker answers are short reports; the harness collects staged edits.
- **Cross-run long-term memory** — SQLite and FTS5 retain success and failure episodes plus
  repository-scoped facts, with retrieved, selected, and used attribution kept distinct.
- **Self-evolving capability registry** — reusable procedures become one current validated package
  with append-only usage memory; failed updates leave the current package unchanged.

## Multi-Agent Orchestration

LangGraph remains the only multi-agent control plane. The Supervisor plans investigate or repair
tasks and dispatches at most two workers with `Send`. Reducer-backed worker results join before
integration. After a batch is integrated, the harness runs formal verification. A code failure
rolls the batch back and returns evidence to the Supervisor.

Role permissions differ. The Supervisor can read, search, and inspect Git, but cannot write or run
tests. Investigate workers are read-only. Repair workers edit a private staging copy with exact
`write_scope` paths. The harness collects staged edits, performs deterministic risk checks, pauses
for approval where required, and is the only component that applies those edits to the real
workspace. Learning (Memory Consolidator and SkillMiner) reuses the Worker gateway after the run.

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
count. Successful runs may additionally consolidate a durable repository fact; full trajectories stay
in the event store rather than being copied into long-term memory.

## Self-Evolving Capability Registry

Enabled packages are retrievable. Workers receive `SKILL.md`, recent usage memory, and declared
resource metadata. `run_skill_script` is a formal registry tool and can execute only a selected
enabled package's declared script; its result emits `SkillUsed`.

After a run, actually used skills receive success/failure counts and an append-only `memory.md`
entry. Retrieved-but-unused skills do not get memory. An explicit skill-script failure remains
failure evidence even when the agent later succeeds without it. A colliding `new_skill` slug is
rejected. An `update_skill` decision names an exact skill ID, validates a new package, keeps one
previous backup, and leaves `memory.md` in place.

Skill mining reads the full trajectory, including helper scripts created through real tools. It can
create a skill after a success with no prior skill, or update a used skill after success. Repair
failures do not create skills; they may append failure memory for used skills. Infrastructure and
model failures do not learn.

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

The demo deliberately starts with a broken calculator, asks the Supervisor to dispatch a Worker,
repairs one source file, runs the real verification command, checkpoints the run, and writes a
condensed episode.

## CLI

For a local Git repository, provide its normal verification command. EvoCI reproduces the
failure in a disposable copy, captures the logs, prepares the task automatically, and enters
the existing repair and learning workflow. A command that already passes uses no model calls.

```bash
evoci fix --command "python -m pytest -q"
evoci fix --repo /path/to/repository --command "python -m pytest -q"
evoci fix --command "python -m pytest -q" --prepare-only
```

Install with `uv tool install .` to use `evoci` from other repositories. Install the target
project's test dependencies first and activate its environment. Model settings are read from
exported `EVO_*` variables or the target repository's `.env`. If using `.env.example`, remove
`EVO_RUNTIME_DIR` to allow automatic runtime relocation, or set it to an absolute directory
outside the target repository. `--prepare-only` needs no model credentials and saves a task
and preflight report under the configured state directory. Existing local edits are included
in reproduction; successful repair edits are kept in the target repository.

This entrance accepts one argv command, without shell pipes, redirection, or chaining. It
is intended for reproducible command failures; it does not introduce an interactive chat UI.

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
- SQLite checkpoints resume a serial apply. Desired-state patch application makes
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
