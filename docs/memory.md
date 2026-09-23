# Long-term memory

EvoCI keeps three layers distinct:

| Layer | Meaning | Storage |
| --- | --- | --- |
| Episode | What happened in one successful or failed run | Condensed SQLite row + FTS5 |
| Long-term memory | A durable fact about the current repository | Repository-scoped SQLite row + FTS5 |
| Capability | A reusable procedure plus optional executable resources | Current package + `memory.md` |

Long-term facts are strictly scoped to the current repository. The model cannot choose a namespace;
the harness binds each fact to `RepoSpec.full_name`. Workflows belong in Skills, not long-term
memory. Retrieval selects current-repository facts first, then same-repository episodes with
a matching failure fingerprint, followed by broader Episode matches. Broader matches can include
other repositories; their namespace stays visible. Supervisor and Worker receive brief entries within
a 10,000-character catalog budget. They can call `read_memory` by an ID visible to that invocation
to inspect complete content in bounded pages; Worker only sees IDs forwarded in `fact_refs`.
Reading an entry is a tool call, not a claim of use. Retrieved, selected, and actually used identifiers
remain distinct telemetry.

Only the post-run consolidator may commit a long-term fact. Model output is a typed
`LongTermFactCandidate`; the harness validates content and performs an idempotent commit derived
from run, repository, and content. Complete logs and tool output remain in the trajectory/artifact
layer rather than memory or graph state.

Failure episodes include a terminal reason, attempted hypotheses, verification failures, actual
tools, evidence IDs, and attempt count. Usage claims from structured leaf outputs are intersected
with selected memory IDs before `MemoryUsed` is emitted.

Old Semantic Memory schemas are not migrated. Opening a legacy `memory.sqlite` fails with a request
to start from a fresh state or campaign directory.
