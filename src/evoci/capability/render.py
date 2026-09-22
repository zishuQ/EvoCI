"""Deterministic SKILL.md rendering from a model-visible SkillSpec."""

from __future__ import annotations

from evoci.capability.models import SkillSpec

REQUIRED_SKILL_SECTIONS = (
    "# Purpose",
    "# When to Use",
    "# Procedure",
    "# Pitfalls",
    "# Verification",
    "# Bundled Resources",
)


def render_skill_markdown(spec: SkillSpec) -> str:
    """Render a complete, valid SKILL.md. Models never have to emit this format."""

    return (
        f"---\nname: {spec.name}\ndescription: {spec.description}\n---\n\n"
        f"# Purpose\n{spec.purpose.strip()}\n\n"
        f"# When to Use\n{spec.when_to_use.strip()}\n\n"
        f"# Procedure\n{spec.procedure.strip()}\n\n"
        f"# Pitfalls\n{spec.pitfalls.strip()}\n\n"
        f"# Verification\n{spec.verification.strip()}\n\n"
        f"# Bundled Resources\n{spec.bundled_resources.strip()}\n"
    )
