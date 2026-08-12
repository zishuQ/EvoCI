"""Read-only materialization of selected packages into a run workspace."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from evoci.capability.models import SkillFile
from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import SkillHit
from evoci.tools.policy import PolicyViolation


class CapabilityMaterializer:
    def __init__(self, registry: CapabilityRegistry, runtime_root: Path) -> None:
        self.registry = registry
        self.runtime_root = runtime_root.resolve()

    def materialize(
        self,
        hits: list[SkillHit],
        *,
        run_id: str,
        workspace: Path,
    ) -> list[Path]:
        """Materialize immutable skills outside the target repository tree."""

        target = workspace.resolve()
        runtime_root = self.runtime_root
        if runtime_root == target or target in runtime_root.parents:
            raise PolicyViolation("capability runtime root must be outside target workspace")
        run_key = hashlib.sha256(run_id.encode()).hexdigest()[:20]
        runtime_root = runtime_root / f"run-{run_key}" / "skills"
        runtime_root.mkdir(parents=True, exist_ok=True)
        destinations: list[Path] = []
        for hit in hits:
            record = self.registry.get(hit.skill_id, hit.version)
            if record is None or record.manifest.status not in {"trial", "active"}:
                raise PolicyViolation(f"skill is not materializable: {hit.skill_id} v{hit.version}")
            source = Path(record.package_path).resolve()
            destination = (
                runtime_root / record.manifest.skill_id / f"v{record.manifest.version}"
            ).resolve()
            if runtime_root not in destination.parents:
                raise PolicyViolation("materialized skill path escaped runtime root")
            if destination.exists():
                self._verify(destination, record.manifest.files)
                destinations.append(destination)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            self._verify(destination, record.manifest.files)
            for item in destination.rglob("*"):
                if item.is_file():
                    item.chmod(item.stat().st_mode & ~0o222)
            destinations.append(destination)
        return destinations

    @staticmethod
    def _verify(destination: Path, files: list[SkillFile]) -> None:
        for raw_file in files:
            path = raw_file.path
            expected = raw_file.sha256
            target = (destination / path).resolve()
            if destination not in target.parents or not target.is_file():
                raise PolicyViolation(f"materialized skill file is missing or unsafe: {path}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise PolicyViolation(f"materialized skill hash mismatch: {path}")
