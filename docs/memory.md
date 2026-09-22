# Long-term memory

EvoCI keeps three layers distinct:

| Layer | Meaning | Storage |
| --- | --- | --- |
| Episode | What happened in one successful or failed run | Condensed SQLite row + FTS5 |
| Long-term memory | A durable fact about the current repository | Repository-scoped SQLite row + FTS5 |
| Capability | A reusable procedure plus optional executable resources | Current package + `memory.md` |

Long-term facts are strictly scoped to the current repository. The model cannot choose a namespace;
the harness binds each fact to `RepoSpec.full_name`. Workflows belong in Skills, not long-term
memory. Retrieval asks for same-repository episodes and repository facts, then enforces a
6,000-character context budget. Retrieved, selected, and actually used identifiers remain distinct
telemetry.

Only the post-run consolidator may commit a long-term fact. Model output is a typed
`LongTermFactCandidate`; the harness validates content and performs an idempotent commit derived
from run, repository, and content. Complete logs and tool output remain in the trajectory/artifact
layer rather than memory or graph state.

Failure episodes include a terminal reason, attempted hypotheses, verification failures, actual
tools, evidence IDs, and attempt count. Usage claims from structured leaf outputs are intersected
with selected memory IDs before `MemoryUsed` is emitted.

Old Semantic Memory schemas are not migrated. Opening a legacy `memory.sqlite` fails with a request
to start from a fresh state or campaign directory.
