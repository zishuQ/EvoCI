# Evaluation

The adapter constructs two unrelated Pydantic models from every benchmark row. It explicitly
normalizes workflow text or structured workflow data into `workflow_yaml`, and string or structured
step logs into `log_text` plus `FailedStep` records. Only explicit, shell-free `Run ...` lines or
structured command fields become `candidate_failed_commands`; an unparseable command produces an
empty list rather than a guess. Workers receive this `NormalizedCIFailure` through `AgentTaskView`.
The evaluator separately retains success commit, gold diff, changed files, and normalized error type.
Both models forbid unknown fields, and tests inspect serialized worker data for leakage.

The CLI materializes the failing commit in a dedicated worktree, invokes the selected real runtime,
then loads `GroundTruth` for evaluation. The ablation variants keep the model and overall budgets
constant while switching actual runtime components:

| Variant | Orchestration | Long-term memory | Capabilities + curator |
| --- | --- | --- | --- |
| `single` | Single agent | No | No |
| `multi` | Multi-agent | No | No |
| `multi-memory` | Multi-agent | Yes | No |
| `evo` | Multi-agent | Yes | Yes |

Harness verification and benchmark evaluation are separate. `targeted_verification_passed` and
`review_passed` describe the agent-controlled repair loop. Before agent execution,
`FailedCommandReplayVerifier` replays dataset-derived commands in a pristine failing-commit
workspace. At least one candidate must fail there to become a valid oracle. A command that already
passes, cannot execute, or cannot be parsed yields `not_available`. After execution, only the
preflight-confirmed command is replayed in the final workspace. `benchmark_resolved` therefore means
fail-before/pass-after, never merely agent success or command-after success.

Benchmark workspaces automatically approve every durable HITL interrupt because they are disposable;
interactive repair keeps manual approval. All variants share one atomic run-level repair model/tool
budget. The serial Worker consumes that budget together with any task-level call limits. Evolution
work is outside the repair cap and is reported as `post_run_model_calls` and `post_run_tool_calls`.

Final patch scope comes only from `git status --porcelain`, `git diff --name-only`, and
`git diff --numstat` in the real benchmark workspace, including untracked files. Trajectory file
records are reported separately as `attempted_files`; they never affect `benchmark_resolved`, final
file counts, or `gold_file_overlap`. Gold-file overlap is auxiliary patch analysis, not a correctness
oracle.

The runner writes `runs.jsonl`, `aggregate.json`, `by_error_type.csv`, and
`continual_learning.csv` with the explicit benchmark metric names above. `evaluation_coverage` is
the fraction of completed tasks with `passed` or `failed` verification;
`benchmark_success_rate` is passed divided by that evaluable subset. `not_available` is reported
separately and is never counted as an agent failure. In `--continual` mode,
sequential tasks share memory and the capability registry; the learning CSV records task family,
benchmark verification, tool calls, attempts, memory hits, skill use, and registry growth. The
bundled 30-slot manifest remains explicitly skipped until reproducible local dataset rows are chosen.
