from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.components import ComponentSnapshot
from strands_evals.packaged_tasks import (
    ComponentRef,
    MetricDefinition,
    PackagedTaskRef,
    TaskRequirements,
)
from strands_evals.runner import run_experiment
from strands_evals.targets import StrandsAgentTarget
from strands_evals.task_environments import LocalContainers
from strands_evals.trials import (
    ExecutionRequest,
    TrialError,
    TrialErrorPhase,
    TrialLimits,
    TrialRequest,
    TrialResult,
    TrialStatus,
)


def _packaged_task(case_id: str) -> PackagedTaskRef:
    return PackagedTaskRef(
        case_id=case_id,
        workload_name="contextbench",
        workload_revision="dataset-revision",
        scorer_revision="scorer-revision",
        package=ComponentRef(
            kind="fixture-task",
            value=f"/tasks/{case_id}",
            revision="task-revision",
        ),
        requirements=TaskRequirements(
            os="linux",
            workdir="/workspace",
        ),
        metrics=(
            MetricDefinition(
                name="contextbench_file_coverage",
                primary=True,
            ),
            MetricDefinition(name="contextbench_file_precision"),
        ),
    )


def _experiment(*, repetitions: int = 2) -> Experiment:
    case = Case(name="case/one", input="Inspect the repository")
    return Experiment(
        cases=[case],
        target=StrandsAgentTarget(
            factory="example:create_agent",
            source_revision="abc123",
            source_dirty=False,
            model={"model_id": "example/model"},
        ),
        repetitions=repetitions,
        packaged_tasks={"case/one": _packaged_task("case/one")},
    )


def _execution(
    tmp_path: Path,
    *,
    resume_run_id: str | None = None,
) -> ExecutionRequest:
    return ExecutionRequest(
        task_environment=LocalContainers(),
        max_concurrency=2,
        limits=TrialLimits(),
        output_dir=tmp_path,
        resume_run_id=resume_run_id,
    )


class _Executor:
    def __init__(self) -> None:
        self.requests: list[TrialRequest] = []

    def snapshot(self) -> ComponentSnapshot:
        return ComponentSnapshot(
            kind="fixture-executor",
            config={"revision": "executor-revision"},
        )

    async def run(self, request: TrialRequest) -> TrialResult:
        self.requests.append(request)
        if request.repetition == 1:
            return TrialResult(
                case_id=request.case.name,
                repetition=request.repetition,
                status=TrialStatus.COMPLETED,
                metrics={
                    "contextbench_file_coverage": 0.0,
                    "contextbench_file_precision": 1.0,
                },
            )
        return TrialResult(
            case_id=request.case.name,
            repetition=request.repetition,
            status=TrialStatus.AGENT_FAILURE,
            error=TrialError(
                phase=TrialErrorPhase.AGENT,
                error_type="RuntimeError",
                message="agent failed",
            ),
        )


async def test_runner_persists_repetitions_and_preserves_zero(tmp_path: Path):
    executor = _Executor()

    report = await run_experiment(
        _experiment(),
        executor=executor,
        execution=_execution(tmp_path),
    )

    run_dir = tmp_path / "runs" / report.manifest.run_id
    assert len(executor.requests) == 2
    assert len(list((run_dir / "results").rglob("*.json"))) == 2
    assert report.results[0].metrics["contextbench_file_coverage"] == 0.0
    coverage = report.metric_summaries["contextbench_file_coverage"]
    assert coverage.valid == 1
    assert coverage.failed == 1
    assert coverage.invalid == 0
    assert coverage.mean == 0.0
    assert report.compatibility_overall_score == 0.0


async def test_runner_resumes_without_rerunning_terminal_results(tmp_path: Path):
    first_executor = _Executor()
    first = await run_experiment(
        _experiment(),
        executor=first_executor,
        execution=_execution(tmp_path),
    )
    resumed_executor = _Executor()

    resumed = await run_experiment(
        _experiment(),
        executor=resumed_executor,
        execution=_execution(
            tmp_path,
            resume_run_id=first.manifest.run_id,
        ),
    )

    assert resumed.manifest.run_id == first.manifest.run_id
    assert resumed_executor.requests == []
    assert resumed.results == first.results


async def test_runner_rejects_resume_with_changed_definition(tmp_path: Path):
    first = await run_experiment(
        _experiment(),
        executor=_Executor(),
        execution=_execution(tmp_path),
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        await run_experiment(
            _experiment(repetitions=3),
            executor=_Executor(),
            execution=_execution(
                tmp_path,
                resume_run_id=first.manifest.run_id,
            ),
        )


def test_experiment_serializes_self_contained_run_definition():
    experiment = _experiment()

    restored = Experiment.from_dict(experiment.to_dict())

    assert restored.repetitions == 2
    assert restored.target.snapshot() == experiment.target.snapshot()
    assert restored.packaged_tasks == experiment.packaged_tasks


@pytest.mark.parametrize(
    ("cases", "packaged_tasks", "message"),
    [
        ([], {}, "at least one case"),
        (
            [Case(name="case/one", input="Inspect")],
            {
                "case/one": _packaged_task("different-case"),
            },
            "case IDs do not match",
        ),
        (
            [Case(name="case/one", input="Inspect")],
            {
                "case/one": _packaged_task("case/one"),
                "extra": _packaged_task("extra"),
            },
            "no matching case",
        ),
    ],
)
async def test_runner_rejects_inconsistent_packaged_experiments(
    cases,
    packaged_tasks,
    message,
    tmp_path: Path,
):
    experiment = Experiment(
        cases=cases,
        target=StrandsAgentTarget(factory="example:create_agent"),
        packaged_tasks=packaged_tasks,
    )

    with pytest.raises(ValueError, match=message):
        await run_experiment(
            experiment,
            executor=_Executor(),
            execution=_execution(tmp_path),
        )
