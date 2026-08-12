# Long-term memory

EvoCI keeps three layers distinct:

| Layer | Meaning | Storage |
| --- | --- | --- |
| Episode | What happened in one successful or failed run | Condensed SQLite row + FTS5 |
| Semantic memory | A durable fact useful across runs | Namespaced SQLite row + FTS5 |
| Capability | A reusable procedure and optional executable resources | Immutable package + registry |

Semantic namespaces are exact: `repo:owner/name`, `family:error-type`, or `global:ci`. Retrieval asks
for at most five semantic facts and three same-repository episodes, then enforces a 6,000-character
context budget. Retrieved, selected, and actually used identifiers remain distinct telemetry.

Only the post-run consolidator may commit semantic memory. Model output is a typed candidate; the
harness validates its namespace and performs an idempotent commit derived from run and content.
Complete logs and tool output remain in the trajectory/artifact layer rather than memory or graph
state.

Failure episodes include a terminal reason, attempted hypotheses, verification failures, actual
tools, evidence IDs, and attempt count. Usage claims from structured leaf outputs are intersected
with selected memory IDs before `MemoryUsed` is emitted.
