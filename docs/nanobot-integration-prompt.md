# Nanobot Integration Task

You are integrating an existing Python project, EvoCI, with Nanobot. Work in the
EvoCI repository first, then inspect the locally available Nanobot source or its
official documentation before choosing APIs. Do not invent Nanobot APIs or assume
a particular plugin layout.

## Product Goal

Use Nanobot as the conversational entry point and interaction shell for EvoCI.
Keep EvoCI as the CI-repair execution engine. A user should be able to start,
observe, inspect, and resume an EvoCI run through Nanobot without losing EvoCI's
durability, safety, or evidence trail.

This is an integration, not a rewrite or a migration of EvoCI's orchestration to
Nanobot.

## Existing EvoCI Architecture

- `src/evoci/graph/builder.py`: LangGraph is the sole multi-agent control plane.
  It plans bounded parallel investigations, diagnoses, repairs, verifies, reviews,
  and checkpoints the run.
- `src/evoci/runtime/`: SQLite checkpoints, append-only events, and run metadata.
- `src/evoci/tools/`: permission-aware filesystem, Git, shell, and skill tools.
  Investigators and reviewers execute commands in disposable copies; the fixer
  edits staging copies; the graph applies approved patches to the real workspace.
- `src/evoci/capability/`: immutable versioned capability packages with a
  `candidate -> trial -> active/rejected` lifecycle. A skill must include a
  validated `SKILL.md`, manifest, optional resources/tests, and safety checks.
- `src/evoci/cli.py`: `evoci run`, `evoci resume`, `evoci runs report`,
  `evoci skills list`, and `evoci skills curate`.

Important current behavior:

- `evoci run` creates a new run ID every time. Only `evoci resume RUN_ID` resumes
  a saved run.
- Events are the source of truth. Do not infer execution state from terminal text.
- The CLI already creates user-facing summaries for plan, failed commands, key
  evidence, diagnosis, patch, verification, review, and completion.
- Detailed evidence is available through `evoci runs report RUN_ID`.
- EvoCI's skill extraction exists but real model output has sometimes failed the
  strict `SKILL.md` schema. Do not describe self-evolution as working end-to-end
  until a real candidate has been validated and registered.

## Integration Constraints

1. Preserve the EvoCI graph, event store, checkpoint store, capability registry,
   safety checks, patch approval, and run IDs. Nanobot must not directly edit the
   target repository or bypass EvoCI's tool policy.
2. Treat Nanobot as a thin adapter. It may invoke EvoCI through an in-process
   adapter if its runtime permits, or through the installed `evoci` CLI otherwise.
   Keep the boundary behind a small interface so the transport can change.
3. Do not expose private chain-of-thought, raw model messages, API keys, or full
   tool payloads. Present only evidence-grounded progress and user-relevant
   artifacts.
4. Do not add a second multi-agent orchestration system. Nanobot should dispatch
   user commands and render EvoCI state; LangGraph remains in control of repair.
5. Do not add a web UI, Docker service, MCP server, or external database unless
   the inspected Nanobot integration mechanism strictly requires one.
6. Maintain backwards compatibility with the existing EvoCI CLI and tests.

## Required User Experience

Nanobot should support a concise command or intent flow equivalent to:

- Start: `repair <task-file>` or an equivalent structured invocation.
- Progress: show phase updates based on persisted EvoCI events.
- Report: show plan, failed commands, evidence summaries, root cause, patch,
  verification, review, and the canonical EvoCI run ID.
- Resume: `resume <run-id>`.
- Inspect: `report <run-id>`.
- Skills: list active and trial EvoCI skills, including status and provenance.

Progress must read like a coding agent's external working summary, for example:

```text
Plan: 4 independent checks
1. Reproduce the reported test failure
2. Compare implementation and test contracts

Reproduced failure: python -m unittest -q (exit 1)
AssertionError: 0.25 != 25.0

Evidence: calculator.py returns value / total while the test and docstring require
a 0-100 percentage value.

Proposed repair: multiply the normalized ratio by 100 in calculator.py.
Verification: passed (python -m unittest -q)
Review: passed
```

Do not emit raw `ModelCall`, `ToolCall`, `ToolResult`, or agent implementation
labels as the main user-facing experience. Deduplicate identical failures emitted
by concurrent investigators.

## Implementation Plan

1. Inspect Nanobot and identify its supported extension/command/tool mechanism.
   Record the chosen integration point and why it is stable enough to depend on.
2. Create a small EvoCI adapter with operations such as `start`, `resume`,
   `get_run`, `list_events`, `render_progress`, and `render_report`. The adapter
   must preserve the run ID returned by EvoCI.
3. Implement the Nanobot-facing command or tool wrapper using that adapter.
4. Build an event-to-message projection. Reuse EvoCI's event semantics and make
   duplicate suppression deterministic. Progress polling/subscription must stop
   at terminal events and tolerate client disconnects; `resume` must remain safe.
5. Add tests using fake or temporary EvoCI stores, without requiring a model API
   or a network connection. Cover a new run, a resumed run, duplicate failed
   command suppression, terminal success/failure, and report rendering.
6. Run EvoCI's formatting, typing, and test suite. Run Nanobot-specific tests in
   the smallest available local setup. State clearly if an integration test is
   unavailable because Nanobot is not present locally.

## Skill Evolution Follow-up

As a separate, scoped improvement, make EvoCI's skill extraction reliable:

- Give the experience-mining model an explicit valid `SKILL.md` template with
  frontmatter and exactly these required headings: `# Purpose`, `# When to Use`,
  `# Procedure`, `# Pitfalls`, `# Verification`, `# Bundled Resources`.
- Use structured output validation and, when a candidate is invalid, perform one
  bounded repair attempt using the schema errors.
- If skill extraction still fails, store a memory decision or record a visible
  non-fatal learning warning. Never make the CI repair itself fail.
- Verify one real or deterministic fixture can create a validated trial skill and
  that later retrieval exposes it to a relevant run.

## Acceptance Criteria

- Existing `evoci run`, `evoci resume`, and `evoci runs report` behavior remains
  intact.
- Nanobot can start and resume an EvoCI run and reports the canonical run ID.
- User-visible progress contains plan items, meaningful failed command summaries,
  evidence summaries, diagnosis, patch, verification, and review.
- Nanobot does not obtain direct write access to an EvoCI target workspace.
- The integration adds focused tests and passes `ruff`, `mypy`, and `pytest` in
  EvoCI.
- Documentation describes setup, the chosen Nanobot extension point, command
  examples, limitations, and the ownership boundary between Nanobot and EvoCI.

First response: inspect both codebases, state the exact Nanobot integration API
you found, and propose the smallest implementation slice. Then implement it; do
not stop at a plan.
