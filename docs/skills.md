# Capability registry

A capability is a single current package, not a versioned prompt fragment. `SKILL.md` and
`manifest.json` are required; scripts, references, templates, and tests are optional. Package files
are hashed and made read-only. A revised capability replaces `package/` after validation and keeps
one `previous/` backup. `memory.md` stays beside the package and is never hashed into the manifest.

```text
skills/
└── debug-pytest-collection/
    ├── package/
    │   ├── SKILL.md
    │   ├── manifest.json
    │   └── scripts/
    ├── previous/
    └── memory.md
```

`CandidateValidator` checks only the Skill package itself: required directory and manifest files,
file hashes, path containment, secret scans, dangerous text, Python AST safety, and, if `tests/`
exists, `pytest tests` in a disposable copy. It does not run repository-level test commands during
install. Whether the current repair is valid is already decided by the repair graph and official
evaluator. How a future agent should verify a repair after using the Skill is written in
`SkillSpec.verification` and rendered as the `# Verification` section of `SKILL.md`. Failed
validation deletes the candidate and leaves the current package unchanged.

`run_skill_script` is a formal leaf tool and is limited to enabled capabilities selected for the
current run. Its trace records skill ID, agent, stable invocation, resource, and execution result.
Explicit script failure is always failure evidence. Disabled skills are not retrieved or executed.

Statistics distinguish retrieval, selection, and execution. The registry rejects a colliding
`new_skill` slug. `update_skill` names an exact skill ID, validates a new package, moves the current
package to `previous/`, and installs the candidate as `package/`. Skill usage memory is append-only,
idempotent on `run_id + skill_id`, and injected into agent context separately from the procedure.
FAILURE memory is counterevidence, not a recommended procedure.
