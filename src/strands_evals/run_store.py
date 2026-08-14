"""Versioned local storage for repetition-aware experiment runs."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .components import ComponentSnapshot
from .packaged_tasks import MetricDefinition, PackagedTaskRef
from .trials import ExecutionRequest, TrialExecutor, TrialLimits, TrialResult


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_case_id(case_id: str) -> str:
    encoded = base64.urlsafe_b64encode(case_id.encode()).decode().rstrip("=")
    return encoded or "_"


def _framework_snapshot() -> ComponentSnapshot:
    """Identify the installed Evals build and exact local package contents."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        version = importlib.metadata.version("strands-agents-evals")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return ComponentSnapshot(
        kind="strands-evals-framework",
        config={
            "distribution_version": version,
            "source_digest": digest.hexdigest(),
        },
    )


class RunManifest(BaseModel):
    """Immutable logical definition of one local experiment run."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    fingerprint: str = Field(min_length=64, max_length=64)
    framework: ComponentSnapshot
    target: ComponentSnapshot
    executor: ComponentSnapshot
    task_environment: ComponentSnapshot
    limits: TrialLimits
    max_concurrency: int = Field(ge=1)
    repetitions: int = Field(ge=1)
    ordered_case_ids: tuple[str, ...]
    encoded_case_ids: dict[str, str]
    packaged_tasks: dict[str, PackagedTaskRef]
    metrics: tuple[MetricDefinition, ...]

    @classmethod
    def build(
        cls,
        *,
        target: ComponentSnapshot,
        executor: TrialExecutor,
        execution: ExecutionRequest,
        repetitions: int,
        ordered_case_ids: list[str],
        packaged_tasks: dict[str, PackagedTaskRef],
    ) -> "RunManifest":
        """Freeze the semantic run definition and calculate its fingerprint."""
        metrics_by_name: dict[str, MetricDefinition] = {}
        for task in packaged_tasks.values():
            for metric in task.metrics:
                existing = metrics_by_name.get(metric.name)
                if existing is not None and existing != metric:
                    raise ValueError(f"Metric {metric.name!r} has conflicting definitions")
                metrics_by_name[metric.name] = metric

        encoded_case_ids = {case_id: _encode_case_id(case_id) for case_id in ordered_case_ids}
        framework = _framework_snapshot()
        definition = {
            "schema_version": 1,
            "framework": framework.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "executor": executor.snapshot().model_dump(mode="json"),
            "task_environment": execution.task_environment.snapshot().model_dump(mode="json"),
            "limits": execution.limits.model_dump(mode="json"),
            "max_concurrency": execution.max_concurrency,
            "repetitions": repetitions,
            "ordered_case_ids": ordered_case_ids,
            "encoded_case_ids": encoded_case_ids,
            "packaged_tasks": {
                case_id: packaged_tasks[case_id].model_dump(mode="json") for case_id in ordered_case_ids
            },
            "metrics": [
                metric.model_dump(mode="json")
                for metric in sorted(
                    metrics_by_name.values(),
                    key=lambda value: value.name,
                )
            ],
        }
        return cls(
            run_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            fingerprint=_fingerprint(definition),
            framework=framework,
            target=target,
            executor=executor.snapshot(),
            task_environment=execution.task_environment.snapshot(),
            limits=execution.limits,
            max_concurrency=execution.max_concurrency,
            repetitions=repetitions,
            ordered_case_ids=tuple(ordered_case_ids),
            encoded_case_ids=encoded_case_ids,
            packaged_tasks=packaged_tasks,
            metrics=tuple(metrics_by_name[name] for name in sorted(metrics_by_name)),
        )


class LocalRunResultStore:
    """Atomic local manifest and result storage for self-contained runs."""

    def __init__(self, root: Path, manifest: RunManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.run_dir = root / "runs" / manifest.run_id
        self.results_dir = self.run_dir / "results"

    @classmethod
    def open(
        cls,
        root: Path,
        requested: RunManifest,
        *,
        resume_run_id: str | None,
    ) -> "LocalRunResultStore":
        """Create a run or resume one with the exact same fingerprint."""
        if resume_run_id is not None:
            manifest_path = root / "runs" / resume_run_id / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Run {resume_run_id!r} has no manifest at {manifest_path}")
            stored = RunManifest.model_validate_json(manifest_path.read_text())
            if stored.fingerprint != requested.fingerprint:
                raise ValueError(
                    f"Run {resume_run_id!r} fingerprint mismatch: "
                    f"stored={stored.fingerprint}, requested={requested.fingerprint}"
                )
            return cls(root, stored)

        store = cls(root, requested)
        store.results_dir.mkdir(parents=True, exist_ok=False)
        store._atomic_write(
            store.run_dir / "manifest.json",
            requested.model_dump_json(indent=2),
        )
        return store

    def result_path(self, case_id: str, repetition: int) -> Path:
        """Return the stable path for one logical result coordinate."""
        encoded = self.manifest.encoded_case_ids[case_id]
        return self.results_dir / encoded / f"{repetition}.json"

    def load(self, case_id: str, repetition: int) -> TrialResult | None:
        """Load a committed terminal result, if present."""
        path = self.result_path(case_id, repetition)
        if not path.is_file():
            return None
        return TrialResult.model_validate_json(path.read_text())

    def save(self, result: TrialResult) -> None:
        """Atomically commit one terminal result."""
        path = self.result_path(result.case_id, result.repetition)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, result.model_dump_json(indent=2))

    def load_all(self) -> list[TrialResult]:
        """Load results in case-major, repetition-minor order."""
        results: list[TrialResult] = []
        for case_id in self.manifest.ordered_case_ids:
            for repetition in range(1, self.manifest.repetitions + 1):
                result = self.load(case_id, repetition)
                if result is not None:
                    results.append(result)
        return results

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(content + "\n")
        os.replace(temporary, path)
