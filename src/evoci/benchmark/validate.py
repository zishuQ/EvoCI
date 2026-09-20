"""Gold/reference validation for official-image SWE-bench tasks. Never calls a model."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from evoci.benchmark.adapters import CIRepairBenchAdapter
from evoci.benchmark.docker import DockerImageManager, DockerReplayVerifier
from evoci.benchmark.models import BenchmarkManifestEntry, DockerTaskSpec
from evoci.benchmark.runner import BenchmarkRunner


@dataclass
class ReferenceTaskResult:
    task_id: str
    status: str
    details: str
    ftp_baseline: dict[str, str] = field(default_factory=dict)
    ftp_reference: dict[str, str] = field(default_factory=dict)
    ptp_reference: dict[str, str] = field(default_factory=dict)


@dataclass
class ReferenceValidationReport:
    results: list[ReferenceTaskResult]

    @property
    def valid(self) -> int:
        return sum(item.status == "valid" for item in self.results)

    @property
    def invalid(self) -> int:
        return sum(item.status == "invalid" for item in self.results)

    @property
    def infra_error(self) -> int:
        return sum(item.status == "infra_error" for item in self.results)

    def as_dict(self) -> dict[str, object]:
        return {
            "tasks": len(self.results),
            "valid": self.valid,
            "invalid": self.invalid,
            "infra_error": self.infra_error,
            "results": [
                {
                    "task_id": item.task_id,
                    "status": item.status,
                    "details": item.details,
                    "ftp_baseline": item.ftp_baseline,
                    "ftp_reference": item.ftp_reference,
                    "ptp_reference": item.ptp_reference,
                }
                for item in self.results
            ],
        }


def _local_pytest(
    workspace: Path, nodes: list[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "-m", "pytest", "-rA", "--tb=short", "--no-header", *nodes],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _validate_local(
    adapter: CIRepairBenchAdapter,
    task_id: str,
    timeout: float,
) -> ReferenceTaskResult:
    prepared = adapter.prepare_task(task_id)
    if prepared.workspace_path is None:
        return ReferenceTaskResult(task_id, "infra_error", "no local workspace for non-Docker task")
    source = Path(prepared.workspace_path)
    if not source.is_dir():
        return ReferenceTaskResult(task_id, "infra_error", f"workspace missing: {source}")
    nodes = adapter.fail_to_pass_nodes(task_id)
    if not nodes:
        return ReferenceTaskResult(task_id, "infra_error", "no local FAIL_TO_PASS nodes")
    reference = adapter.ground_truth(task_id).diff
    with tempfile.TemporaryDirectory() as tmp:
        baseline = Path(tmp) / "baseline"
        shutil.copytree(source, baseline, symlinks=True)
        try:
            before = _local_pytest(baseline, nodes, timeout)
        except subprocess.TimeoutExpired:
            return ReferenceTaskResult(task_id, "infra_error", "local baseline timed out")
        if before.returncode == 0:
            return ReferenceTaskResult(
                task_id,
                "invalid",
                "local FAIL_TO_PASS already passed before reference patch",
            )
        if before.returncode not in {1}:
            return ReferenceTaskResult(
                task_id,
                "infra_error",
                (
                    f"local baseline infra exit {before.returncode}: "
                    f"{(before.stderr or before.stdout)[-2000:]}"
                ),
            )
        if not reference.strip():
            return ReferenceTaskResult(task_id, "infra_error", "no reference patch for local task")
        gold = Path(tmp) / "gold"
        shutil.copytree(source, gold, symlinks=True)
        apply = subprocess.run(
            ["git", "-C", str(gold), "apply", "--whitespace=nowarn", "-"],
            input=reference,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if apply.returncode != 0:
            return ReferenceTaskResult(
                task_id, "infra_error", f"local reference patch apply failed: {apply.stderr}"
            )
        try:
            after = _local_pytest(gold, nodes, timeout)
        except subprocess.TimeoutExpired:
            return ReferenceTaskResult(task_id, "infra_error", "local reference tests timed out")
        if after.returncode != 0:
            return ReferenceTaskResult(
                task_id,
                "invalid",
                f"local reference tests failed: {(after.stdout or after.stderr)[-2000:]}",
            )
        return ReferenceTaskResult(task_id, "valid", "local baseline failed and reference passed")


def _validate_docker(
    spec: DockerTaskSpec,
    pinned_image: str,
    *,
    timeout: float,
    cli: object | None = None,
) -> ReferenceTaskResult:
    if not spec.reference_patch.strip():
        return ReferenceTaskResult(spec.task_id, "infra_error", "no reference patch in dataset")
    verifier = DockerReplayVerifier(
        spec,
        pinned_image,
        cli=cli,  # type: ignore[arg-type]
        timeout=timeout,
    )
    try:
        baseline = verifier.score_patches(
            phase="reference-baseline", candidate_patch="", run_pass_to_pass=False
        )
        if baseline.infra_reason:
            return ReferenceTaskResult(
                spec.task_id,
                "infra_error",
                f"baseline infra_error: {baseline.infra_reason}",
                ftp_baseline=baseline.ftp_results,
            )
        if not baseline.ftp_executed or not baseline.any_ftp_failed:
            return ReferenceTaskResult(
                spec.task_id,
                "invalid",
                f"FAIL_TO_PASS must run and fail on baseline+test_patch: {baseline.ftp_results}",
                ftp_baseline=baseline.ftp_results,
            )
        gold = verifier.score_patches(
            phase="reference-gold",
            candidate_patch=spec.reference_patch,
            run_pass_to_pass=True,
        )
        if gold.infra_reason:
            return ReferenceTaskResult(
                spec.task_id,
                "infra_error",
                f"reference infra_error: {gold.infra_reason}",
                ftp_baseline=baseline.ftp_results,
                ftp_reference=gold.ftp_results,
                ptp_reference=gold.ptp_results,
            )
        if not gold.all_ftp_passed:
            return ReferenceTaskResult(
                spec.task_id,
                "invalid",
                f"FAIL_TO_PASS must pass with reference patch: {gold.ftp_results}",
                ftp_baseline=baseline.ftp_results,
                ftp_reference=gold.ftp_results,
                ptp_reference=gold.ptp_results,
            )
        if not gold.all_ptp_passed:
            return ReferenceTaskResult(
                spec.task_id,
                "invalid",
                f"PASS_TO_PASS must pass with reference patch: {gold.ptp_results}",
                ftp_baseline=baseline.ftp_results,
                ftp_reference=gold.ftp_results,
                ptp_reference=gold.ptp_results,
            )
        return ReferenceTaskResult(
            spec.task_id,
            "valid",
            "baseline FAIL_TO_PASS failed; reference FAIL_TO_PASS and PASS_TO_PASS passed",
            ftp_baseline=baseline.ftp_results,
            ftp_reference=gold.ftp_results,
            ptp_reference=gold.ptp_results,
        )
    finally:
        verifier.close()


def validate_reference(
    *,
    dataset: Path,
    manifest: Path,
    docker_official_images: bool,
    apply_reference_patch: bool,
    task_ids: list[str] | None = None,
    timeout: float = 1800.0,
    audit_path: Path | None = None,
    cli: object | None = None,
) -> ReferenceValidationReport:
    """Validate gold patches against FAIL_TO_PASS / PASS_TO_PASS. No LLM, no learning."""

    if not docker_official_images:
        raise ValueError("reference validation requires --docker-official-images")
    if not apply_reference_patch:
        raise ValueError("reference validation requires --apply-reference-patch")
    entries = BenchmarkRunner.load_manifest(manifest)
    if task_ids:
        wanted = set(task_ids)
        entries = [entry for entry in entries if entry.task_id in wanted]
        missing = wanted.difference(entry.task_id for entry in entries)
        if missing:
            raise ValueError(f"task ids not in manifest: {sorted(missing)}")
    adapter = CIRepairBenchAdapter(dataset)
    specs: dict[str, DockerTaskSpec] = {}
    results: list[ReferenceTaskResult] = []
    pending: list[BenchmarkManifestEntry] = []
    for entry in entries:
        if entry.skipped:
            results.append(
                ReferenceTaskResult(entry.task_id, "invalid", "task is skipped in manifest")
            )
            continue
        spec = adapter.docker_task_spec(entry.task_id)
        if spec is None:
            pending.append(entry)
            continue
        specs[entry.task_id] = spec
        pending.append(entry)
    pinned: dict[str, str] = {}
    if specs:
        pinned = DockerImageManager(cli=cli).prepare(  # type: ignore[arg-type]
            list(specs.values()), audit_path=audit_path
        )
    for entry in pending:
        spec = specs.get(entry.task_id)
        if spec is None:
            results.append(_validate_local(adapter, entry.task_id, timeout))
            continue
        results.append(_validate_docker(spec, pinned[entry.task_id], timeout=timeout, cli=cli))
    return ReferenceValidationReport(results)
