"""Load packaged benchmarks into self-contained Strands Evals experiments."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case import Case
from .experiment import Experiment
from .packaged_tasks import (
    ComponentRef,
    MetricDefinition,
    PackagedTaskRef,
    TaskRequirements,
)
from .targets import EvaluationTarget


def _prepared_task_paths(source: Path) -> list[Path]:
    from harbor.models.task.task import Task  # optional extra: harbor

    if Task.is_valid_dir(source):
        return [source.resolve()]
    if source.is_dir():
        tasks = [child.resolve() for child in sorted(source.iterdir()) if child.is_dir() and Task.is_valid_dir(child)]
        if tasks:
            return tasks
    return []


def _registry_task_paths(
    name: str,
    *,
    version: str | None,
    registry_path: Path | None,
    output_dir: Path,
    task_ids: list[str] | None,
    limit: int | None,
    overwrite: bool,
) -> list[Path]:
    """Resolve an existing Harbor registry dataset."""
    from harbor.models.job.config import DatasetConfig  # optional extra: harbor
    from harbor.tasks.client import TaskClient

    async def resolve() -> list[Path]:
        dataset = DatasetConfig(
            name=name,
            version=version,
            registry_path=registry_path,
            task_names=task_ids,
            n_tasks=limit,
            overwrite=overwrite,
            download_dir=output_dir,
        )
        task_configs = await dataset.get_task_configs()
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = await TaskClient().download_tasks(
            [config.get_task_id() for config in task_configs],
            overwrite=overwrite,
            output_dir=output_dir,
            export=True,
        )
        return [result.path.resolve() for result in downloaded.results]

    return asyncio.run(resolve())


def _directory_revision(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metric_definitions(task: Any) -> tuple[MetricDefinition, ...]:
    configured = task.config.metadata.get("benchmark_metrics", ["reward"])
    if not isinstance(configured, list) or not all(isinstance(value, str) and value for value in configured):
        raise ValueError(f"{task.task_dir}/task.toml metadata.benchmark_metrics must be a list of non-empty strings")
    primary = task.config.metadata.get("benchmark_primary_metric")
    if primary is not None and primary not in configured:
        raise ValueError(
            f"{task.task_dir}/task.toml benchmark_primary_metric {primary!r} is not present in benchmark_metrics"
        )
    return tuple(MetricDefinition(name=name, primary=name == primary) for name in configured)


def _requirements(task: Any) -> TaskRequirements:
    environment = task.config.environment
    task_os = getattr(environment.os, "value", environment.os)
    return TaskRequirements(
        os=str(task_os) if task_os is not None else None,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpus=environment.gpus,
        workdir=environment.workdir,
    )


def _load_cases(
    benchmark_name: str,
    task_paths: list[Path],
    requested_revision: str | None,
) -> tuple[list[Case[str, str]], dict[str, PackagedTaskRef]]:
    from harbor.models.task.task import Task  # optional extra: harbor

    cases: list[Case[str, str]] = []
    packaged_tasks: dict[str, PackagedTaskRef] = {}
    for task_path in task_paths:
        task = Task(task_path)
        case_id = str(task.config.metadata.get("instance_id") or task.name)
        if case_id in packaged_tasks:
            raise ValueError(f"Benchmark contains duplicate case ID {case_id!r}")
        workload_revision = str(
            requested_revision or task.config.metadata.get("benchmark_revision") or task.config.task.version
        )
        metrics = _metric_definitions(task)
        package_revision = _directory_revision(task_path)
        packaged_tasks[case_id] = PackagedTaskRef(
            case_id=case_id,
            workload_name=benchmark_name,
            workload_revision=workload_revision,
            scorer_revision=str(task.config.metadata.get("scorer_revision") or package_revision),
            package=ComponentRef(
                kind="harbor-task-directory",
                value=str(task_path.resolve()),
                revision=package_revision,
            ),
            requirements=_requirements(task),
            metrics=metrics,
        )
        cases.append(
            Case(
                name=case_id,
                input=task.instruction,
                metadata={
                    "benchmark": benchmark_name,
                    "workload_revision": workload_revision,
                },
            )
        )
    return cases, packaged_tasks


@dataclass(frozen=True)
class Benchmark:
    """Prepared cases and executor-neutral packaged task references."""

    name: str
    cases: list[Case[str, str]]
    packaged_tasks: dict[str, PackagedTaskRef]
    work_dir: Path

    def experiment(
        self,
        *,
        target: EvaluationTarget,
        repetitions: int = 1,
    ) -> Experiment[str, str]:
        """Build a self-contained experiment from this benchmark."""
        return Experiment(
            cases=self.cases,
            target=target,
            repetitions=repetitions,
            packaged_tasks=self.packaged_tasks,
        )


def load_benchmark(
    name: str,
    *,
    source: Path | str | None = None,
    task_paths: Iterable[Path | str] | None = None,
    revision: str | None = None,
    registry_path: Path | str | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    work_dir: Path | str | None = None,
    overwrite: bool = False,
    adapter_options: dict[str, Any] | None = None,
) -> Benchmark:
    """Prepare benchmark-native data as packaged cases.

    The returned references are executor-neutral. Their opaque package values
    are interpreted only by the installed Harbor integration.
    """
    root = Path(work_dir or f".strands-evals/{name}")
    resolved_paths = [Path(path).resolve() for path in (task_paths or [])]
    source_path = Path(source).expanduser() if source is not None else None
    resolved_registry_path = Path(registry_path).expanduser() if registry_path is not None else None
    if not resolved_paths and source_path is not None:
        resolved_paths = _prepared_task_paths(source_path)

    if not resolved_paths:
        from harbor.benchmark_adapters import (  # optional extra: harbor
            has_benchmark_adapter,
            prepare_benchmark,
        )

        if source_path is not None or has_benchmark_adapter(name):
            resolved_paths = prepare_benchmark(
                name,
                source=source_path,
                output_dir=root / "tasks",
                revision=revision,
                task_ids=task_ids,
                limit=limit,
                overwrite=overwrite,
                options=adapter_options,
            )
        else:
            resolved_paths = _registry_task_paths(
                name,
                version=revision,
                registry_path=resolved_registry_path,
                output_dir=root / "tasks",
                task_ids=task_ids,
                limit=limit,
                overwrite=overwrite,
            )

    if limit is not None:
        resolved_paths = resolved_paths[:limit]
    if not resolved_paths:
        raise ValueError(f"Benchmark {name!r} did not resolve any tasks")

    cases, packaged_tasks = _load_cases(name, resolved_paths, revision)
    return Benchmark(
        name=name,
        cases=cases,
        packaged_tasks=packaged_tasks,
        work_dir=root,
    )


__all__ = ["Benchmark", "load_benchmark"]
