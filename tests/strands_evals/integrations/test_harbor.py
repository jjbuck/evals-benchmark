import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext

from strands_evals import Case
from strands_evals.integrations import harbor as harbor_integration
from strands_evals.integrations.harbor import HarborTrialExecutor
from strands_evals.packaged_tasks import (
    ComponentRef,
    MetricDefinition,
    PackagedTaskRef,
    TaskRequirements,
)
from strands_evals.targets import StrandsAgentTarget
from strands_evals.task_environments import LocalContainers
from strands_evals.trials import (
    ExecutionRequest,
    TrialErrorPhase,
    TrialLimits,
    TrialRequest,
    TrialStatus,
)


class _Config:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


def _request(tmp_path: Path) -> TrialRequest:
    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "task.toml").write_text("task")
    package = PackagedTaskRef(
        case_id="case-one",
        workload_name="contextbench",
        workload_revision="dataset-revision",
        scorer_revision="scorer-revision",
        package=ComponentRef(
            kind="harbor-task-directory",
            value=str(task_path),
            revision=harbor_integration._directory_revision(task_path),
        ),
        requirements=TaskRequirements(workdir="/workspace"),
        metrics=(
            MetricDefinition(name="coverage", primary=True),
            MetricDefinition(name="precision"),
        ),
    )
    return TrialRequest(
        run_id="run-one",
        case=Case(name="case-one", input="Inspect the repository"),
        packaged_task=package,
        target=StrandsAgentTarget(
            factory="example:create_agent",
            kwargs={"window": 40},
            model={"model_id": "example/model"},
        ),
        repetition=1,
        execution=ExecutionRequest(
            task_environment=LocalContainers(),
            max_concurrency=1,
            limits=TrialLimits(),
            output_dir=tmp_path,
        ),
    )


def _timing():
    return SimpleNamespace(started_at=None, finished_at=None)


class _Trial:
    result = None

    def __init__(self, result, root: Path):
        self.result = result
        self.paths = SimpleNamespace(
            result_path=root / "result.json",
            agent_dir=root / "agent",
            verifier_dir=root / "verifier",
        )
        self.paths.agent_dir.mkdir(parents=True)
        self.paths.verifier_dir.mkdir()

    async def run(self):
        return self.result


def _harbor_types(trial: _Trial, captured: dict):
    class Trial:
        @staticmethod
        async def create(config):
            captured["config"] = config
            return trial

    return {
        "AgentConfig": _Config,
        "EnvironmentConfig": _Config,
        "EnvironmentType": SimpleNamespace(DOCKER="docker"),
        "TaskConfig": _Config,
        "Trial": Trial,
        "TrialConfig": _Config,
    }


async def test_harbor_executor_preserves_native_zero_and_unavailable_metric(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    harbor_result = SimpleNamespace(
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"coverage": 0.0}),
        verifier=_timing(),
        agent_setup=_timing(),
        agent_execution=_timing(),
        environment_setup=_timing(),
        started_at=None,
        finished_at=None,
        agent_result=AgentContext(
            n_input_tokens=10,
            n_output_tokens=3,
            metadata={"output": "done"},
        ),
        trial_uri="file:///trial",
    )
    trial = _Trial(harbor_result, tmp_path / "trial")
    monkeypatch.setattr(
        harbor_integration,
        "_harbor",
        lambda: _harbor_types(trial, captured),
    )

    result = await HarborTrialExecutor(tmp_path / "trials").run(_request(tmp_path))

    assert result.status == TrialStatus.COMPLETED
    assert result.metrics == {"coverage": 0.0}
    assert result.unavailable_metrics == {"precision"}
    assert result.output == "done"
    assert result.usage.input_tokens == 10
    config = captured["config"]
    assert config.agent.name == "strands"
    assert config.agent.kwargs["factory"] == "example:create_agent"
    assert config.environment.type == "docker"


async def test_harbor_executor_classifies_agent_failure(monkeypatch, tmp_path: Path):
    captured = {}
    harbor_result = SimpleNamespace(
        exception_info=SimpleNamespace(
            exception_type="RuntimeError",
            exception_message="agent failed",
            exception_traceback="traceback",
        ),
        verifier_result=None,
        verifier=None,
        agent_setup=_timing(),
        agent_execution=_timing(),
        environment_setup=_timing(),
        started_at=None,
        finished_at=None,
        agent_result=AgentContext(metadata={"output": ""}),
        trial_uri="file:///trial",
    )
    trial = _Trial(harbor_result, tmp_path / "trial")
    monkeypatch.setattr(
        harbor_integration,
        "_harbor",
        lambda: _harbor_types(trial, captured),
    )

    result = await HarborTrialExecutor(tmp_path / "trials").run(_request(tmp_path))

    assert result.status == TrialStatus.AGENT_FAILURE
    assert result.metrics == {}
    assert result.error.phase == TrialErrorPhase.AGENT
    assert result.error.message == "agent failed"


async def test_harbor_executor_classifies_verified_agent_timeout(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    harbor_result = SimpleNamespace(
        exception_info=SimpleNamespace(
            exception_type="AgentTimeoutError",
            exception_message="Agent execution timed out after 120 seconds",
            exception_traceback="traceback",
        ),
        verifier_result=SimpleNamespace(rewards={"coverage": 0.5}),
        verifier=_timing(),
        agent_setup=_timing(),
        agent_execution=_timing(),
        environment_setup=_timing(),
        started_at=None,
        finished_at=None,
        agent_result=AgentContext(metadata={"output": "partial"}),
        trial_uri="file:///trial",
    )
    trial = _Trial(harbor_result, tmp_path / "trial")
    monkeypatch.setattr(
        harbor_integration,
        "_harbor",
        lambda: _harbor_types(trial, captured),
    )

    result = await HarborTrialExecutor(tmp_path / "trials").run(_request(tmp_path))

    assert result.status == TrialStatus.AGENT_FAILURE
    assert result.metrics == {"coverage": 0.5}
    assert result.error.phase == TrialErrorPhase.AGENT
    assert result.error.error_type == "AgentTimeoutError"


async def test_harbor_executor_preserves_partial_evidence_on_cancellation(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    harbor_result = SimpleNamespace(
        exception_info=None,
        verifier_result=None,
        verifier=None,
        agent_setup=_timing(),
        agent_execution=_timing(),
        environment_setup=_timing(),
        started_at=None,
        finished_at=None,
        agent_result=AgentContext(
            n_input_tokens=12,
            n_output_tokens=4,
            metadata={"output": "partial"},
        ),
        trial_uri="file:///trial",
    )

    class CancelledTrial(_Trial):
        async def run(self):
            raise asyncio.CancelledError

    trial = CancelledTrial(harbor_result, tmp_path / "trial")
    (trial.paths.agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "session_id": "partial-session",
                "steps": [
                    {
                        "source": "user",
                        "message": "Inspect",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        harbor_integration,
        "_harbor",
        lambda: _harbor_types(trial, captured),
    )

    result = await HarborTrialExecutor(tmp_path / "trials").run(_request(tmp_path))

    assert result.status == TrialStatus.CANCELLED
    assert result.output == "partial"
    assert result.usage.input_tokens == 12
    assert result.session.session_id == "partial-session"
    assert {artifact.kind for artifact in result.artifacts} == {
        "harbor-trial",
        "atif-trajectory",
    }
    assert result.error.phase == TrialErrorPhase.CANCELLATION


async def test_harbor_executor_preserves_partial_evidence_on_executor_failure(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    harbor_result = SimpleNamespace(
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"coverage": 0.75}),
        verifier=_timing(),
        agent_setup=_timing(),
        agent_execution=_timing(),
        environment_setup=_timing(),
        started_at=None,
        finished_at=None,
        agent_result=AgentContext(
            n_input_tokens=20,
            n_output_tokens=5,
            metadata={"output": "done"},
        ),
        trial_uri="file:///trial",
    )

    class FailingFinalizeTrial(_Trial):
        async def run(self):
            raise RuntimeError("artifact export failed")

    trial = FailingFinalizeTrial(harbor_result, tmp_path / "trial")
    (trial.paths.result_path).write_text("{}")
    monkeypatch.setattr(
        harbor_integration,
        "_harbor",
        lambda: _harbor_types(trial, captured),
    )

    result = await HarborTrialExecutor(tmp_path / "trials").run(_request(tmp_path))

    assert result.status == TrialStatus.INFRASTRUCTURE_INVALID
    assert result.metrics == {"coverage": 0.75}
    assert result.output == "done"
    assert result.usage.input_tokens == 20
    assert {artifact.kind for artifact in result.artifacts} == {
        "harbor-trial",
        "harbor-result",
    }
    assert result.error.phase == TrialErrorPhase.EXECUTOR
    assert result.error.message == "artifact export failed"
