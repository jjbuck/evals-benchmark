from pathlib import Path
from types import SimpleNamespace

from strands_evals.benchmarks import load_benchmark
from strands_evals.targets import StrandsAgentTarget


def _write_task(root: Path, name: str) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Inspect the repository.\n")
    (task / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "task.toml").write_text(
        f"""
schema_version = "1.4"
source = "example"

[task]
name = "example/{name}"
version = "1"
authors = []

[metadata]
instance_id = "{name}"
benchmark_revision = "dataset-revision"
scorer_revision = "scorer-revision"
benchmark_primary_metric = "accuracy"
benchmark_metrics = ["accuracy", "precision"]

[environment]
workdir = "/workspace"
""".strip()
        + "\n"
    )
    return task


def test_loads_prepared_harbor_tasks_as_packaged_references(tmp_path: Path):
    task = _write_task(tmp_path, "one")

    benchmark = load_benchmark(
        "example",
        source=tmp_path,
        work_dir=tmp_path / "work",
    )

    assert [case.name for case in benchmark.cases] == ["one"]
    reference = benchmark.packaged_tasks["one"]
    assert reference.workload_revision == "dataset-revision"
    assert reference.scorer_revision == "scorer-revision"
    assert reference.package.kind == "harbor-task-directory"
    assert reference.package.value == str(task.resolve())
    assert [metric.name for metric in reference.metrics] == [
        "accuracy",
        "precision",
    ]
    assert reference.metrics[0].primary is True


def test_builds_self_contained_experiment_with_configurable_repetitions(
    tmp_path: Path,
):
    _write_task(tmp_path, "one")
    benchmark = load_benchmark("example", source=tmp_path)
    target = StrandsAgentTarget(
        factory="example:create_agent",
        model={"model_id": "example-model"},
    )

    experiment = benchmark.experiment(target=target, repetitions=3)

    assert experiment.target is target
    assert experiment.repetitions == 3
    assert experiment.packaged_tasks == benchmark.packaged_tasks
    assert [case.name for case in experiment.cases] == ["one"]


def test_uses_task_package_digest_when_scorer_revision_is_not_declared(
    tmp_path: Path,
):
    task = _write_task(tmp_path, "one")
    config = task / "task.toml"
    config.write_text(config.read_text().replace('scorer_revision = "scorer-revision"\n', ""))

    benchmark = load_benchmark("example", source=tmp_path)
    reference = benchmark.packaged_tasks["one"]

    assert reference.scorer_revision == reference.package.revision


def test_loads_existing_harbor_registry_dataset(monkeypatch, tmp_path: Path):
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import TaskConfig
    from harbor.tasks.client import TaskClient

    task = _write_task(tmp_path / "published", "django__django-13741")
    registry = tmp_path / "registry.json"
    registry.write_text("[]")
    captured: dict = {}

    async def get_task_configs(self):
        captured["dataset"] = self
        return [TaskConfig(path=task)]

    async def download_tasks(self, task_ids, **kwargs):
        captured["task_ids"] = task_ids
        captured["download"] = kwargs
        return SimpleNamespace(results=[SimpleNamespace(path=task)])

    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    monkeypatch.setattr(TaskClient, "download_tasks", download_tasks)

    benchmark = load_benchmark(
        "swebench-verified",
        registry_path=registry,
        task_ids=["django__django-13741"],
        limit=1,
        work_dir=tmp_path / "work",
    )

    dataset = captured["dataset"]
    assert dataset.name == "swebench-verified"
    assert dataset.registry_path == registry
    assert dataset.task_names == ["django__django-13741"]
    assert dataset.n_tasks == 1
    assert captured["download"]["export"] is True
    assert benchmark.packaged_tasks["django__django-13741"].package.value == str(task.resolve())
