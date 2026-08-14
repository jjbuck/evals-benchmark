"""Translate provisional Strands Evals trial contracts into Harbor trials."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import traceback
from pathlib import Path
from typing import Any

from ..components import ComponentSnapshot
from ..mappers.atif_session_mapper import ATIFSessionMapper
from ..targets import StrandsAgentTarget
from ..task_environments import LocalContainers
from ..trials import (
    ArtifactRef,
    PhaseTiming,
    TrialError,
    TrialErrorPhase,
    TrialRequest,
    TrialResult,
    TrialStatus,
    Usage,
)


def _harbor() -> dict[str, Any]:
    try:
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
        )
        from harbor.trial.trial import Trial
    except ImportError as error:
        raise ImportError(
            "Harbor integration requires the optional dependency. Install strands-agents-evals[harbor]."
        ) from error
    return {
        "AgentConfig": AgentConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "EnvironmentType": EnvironmentType,
        "TaskConfig": TaskConfig,
        "Trial": Trial,
        "TrialConfig": TrialConfig,
    }


def _elapsed_seconds(timing: Any) -> float | None:
    if timing is None or timing.started_at is None or timing.finished_at is None:
        return None
    return max((timing.finished_at - timing.started_at).total_seconds(), 0.0)


def _directory_revision(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _python_source_revision(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.py")):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_path(request: TrialRequest) -> Path:
    package = request.packaged_task.package
    if package.kind != "harbor-task-directory":
        raise ValueError(f"HarborTrialExecutor cannot resolve package kind {package.kind!r}")
    path = Path(package.value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Packaged Harbor task not found: {path}")
    observed_revision = _directory_revision(path)
    if observed_revision != package.revision:
        raise ValueError(
            f"Packaged Harbor task changed after the run was defined: {path}; "
            f"expected {package.revision}, observed {observed_revision}"
        )
    return path


def _phase_status(result: Any) -> tuple[TrialStatus, TrialErrorPhase | None]:
    if result.exception_info is None:
        if result.verifier_result is None:
            return TrialStatus.VERIFIER_INVALID, TrialErrorPhase.VERIFIER
        return TrialStatus.COMPLETED, None
    # Harbor still runs the verifier after these recoverable agent failures, so
    # verifier timing alone does not identify the phase that raised the error.
    if result.exception_info.exception_type in {
        "AgentTimeoutError",
        "NonZeroAgentExitCodeError",
    }:
        return TrialStatus.AGENT_FAILURE, TrialErrorPhase.AGENT
    if result.verifier is not None:
        return TrialStatus.VERIFIER_INVALID, TrialErrorPhase.VERIFIER
    if result.agent_execution is not None or result.agent_setup is not None:
        return TrialStatus.AGENT_FAILURE, TrialErrorPhase.AGENT
    return TrialStatus.INFRASTRUCTURE_INVALID, TrialErrorPhase.ENVIRONMENT


def _error_from_harbor(result: Any, phase: TrialErrorPhase | None) -> TrialError | None:
    error = result.exception_info
    if error is None:
        if phase == TrialErrorPhase.VERIFIER:
            return TrialError(
                phase=phase,
                error_type="MissingVerifierResult",
                message="Harbor trial completed without a verifier result",
            )
        return None
    return TrialError(
        phase=phase or TrialErrorPhase.EXECUTOR,
        error_type=error.exception_type,
        message=error.exception_message,
        traceback=error.exception_traceback,
    )


def _artifacts(trial: Any, result: Any) -> list[ArtifactRef]:
    artifacts = []
    trial_uri = getattr(result, "trial_uri", None)
    if trial_uri:
        artifacts.append(ArtifactRef(kind="harbor-trial", uri=trial_uri))
    candidates = {
        "harbor-result": trial.paths.result_path,
        "atif-trajectory": trial.paths.agent_dir / "trajectory.json",
        "strands-result": trial.paths.agent_dir / "strands-result.json",
        "contextbench-diagnostics": (trial.paths.verifier_dir / "contextbench_metrics.json"),
    }
    for kind, path in candidates.items():
        if path.is_file():
            artifacts.append(ArtifactRef(kind=kind, uri=path.resolve().as_uri()))
    return artifacts


class HarborTrialExecutor:
    """Execute packaged tasks through Harbor's local trial lifecycle."""

    def __init__(self, trials_dir: Path | str) -> None:
        self.trials_dir = Path(trials_dir)
        self._mapper = ATIFSessionMapper()

    def snapshot(self) -> ComponentSnapshot:
        """Record the installed integration and dependency versions."""

        def version(distribution: str) -> str | None:
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                return None

        import harbor

        harbor_root = Path(harbor.__file__).resolve().parent
        return ComponentSnapshot(
            kind="harbor-trial-executor",
            config={
                "harbor_version": version("harbor"),
                "harbor_source_digest": _python_source_revision(harbor_root),
                "strands_version": version("strands-agents"),
                "adapter": "harbor.agents.installed.strands:StrandsAgent",
            },
        )

    async def run(self, request: TrialRequest) -> TrialResult:
        """Run one Harbor trial and preserve phase-specific terminal state."""
        if not isinstance(request.target, StrandsAgentTarget):
            raise ValueError("HarborTrialExecutor currently supports only StrandsAgentTarget")
        if not isinstance(request.execution.task_environment, LocalContainers):
            raise ValueError("HarborTrialExecutor currently supports only LocalContainers")

        try:
            task_path = _package_path(request)
            harbor = _harbor()
            environment = harbor["EnvironmentConfig"](
                type=harbor["EnvironmentType"].DOCKER,
                kwargs=request.execution.task_environment.options,
            )
            model_name = request.target.model.get("model_id")
            config = harbor["TrialConfig"](
                task=harbor["TaskConfig"](path=task_path),
                trials_dir=self.trials_dir / request.run_id,
                timeout_multiplier=request.execution.limits.timeout_multiplier,
                agent_timeout_multiplier=(request.execution.limits.agent_timeout_multiplier),
                verifier_timeout_multiplier=(request.execution.limits.verifier_timeout_multiplier),
                agent=harbor["AgentConfig"](
                    name="strands",
                    model_name=str(model_name) if model_name is not None else None,
                    kwargs={
                        "factory": request.target.factory,
                        "factory_kwargs": request.target.kwargs,
                    },
                ),
                environment=environment,
            )
            trial = await harbor["Trial"].create(config)
        except Exception as error:
            return TrialResult(
                case_id=request.packaged_task.case_id,
                repetition=request.repetition,
                status=TrialStatus.INFRASTRUCTURE_INVALID,
                error=TrialError(
                    phase=TrialErrorPhase.ENVIRONMENT,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback="".join(
                        traceback.format_exception(
                            type(error),
                            error,
                            error.__traceback__,
                        )
                    ),
                ),
            )

        try:
            harbor_result = await trial.run()
        except asyncio.CancelledError as error:
            return self._map_result(
                request,
                trial,
                trial.result,
                status=TrialStatus.CANCELLED,
                error=TrialError(
                    phase=TrialErrorPhase.CANCELLATION,
                    error_type=type(error).__name__,
                    message=str(error),
                ),
            )
        except Exception as error:
            partial_result = getattr(trial, "result", None)
            if partial_result is None:
                return TrialResult(
                    case_id=request.packaged_task.case_id,
                    repetition=request.repetition,
                    status=TrialStatus.INFRASTRUCTURE_INVALID,
                    error=TrialError(
                        phase=TrialErrorPhase.EXECUTOR,
                        error_type=type(error).__name__,
                        message=str(error),
                        traceback="".join(
                            traceback.format_exception(
                                type(error),
                                error,
                                error.__traceback__,
                            )
                        ),
                    ),
                )
            return self._map_result(
                request,
                trial,
                partial_result,
                status=TrialStatus.INFRASTRUCTURE_INVALID,
                error=TrialError(
                    phase=TrialErrorPhase.EXECUTOR,
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback="".join(
                        traceback.format_exception(
                            type(error),
                            error,
                            error.__traceback__,
                        )
                    ),
                ),
            )

        return self._map_result(request, trial, harbor_result)

    def _map_result(
        self,
        request: TrialRequest,
        trial: Any,
        harbor_result: Any,
        *,
        status: TrialStatus | None = None,
        error: TrialError | None = None,
    ) -> TrialResult:
        """Map Harbor's result while retaining evidence from partial trials."""
        inferred_status, error_phase = _phase_status(harbor_result)
        rewards = (harbor_result.verifier_result.rewards if harbor_result.verifier_result is not None else None) or {}
        declared_metrics = {metric.name for metric in request.packaged_task.metrics}
        metrics = {name: float(value) for name, value in rewards.items() if name in declared_metrics}
        unavailable = declared_metrics - metrics.keys()
        agent_context = harbor_result.agent_result
        output = str((agent_context.metadata or {}).get("output") or "") if agent_context is not None else None
        trajectory_path = trial.paths.agent_dir / "trajectory.json"
        session = None
        if trajectory_path.is_file():
            try:
                session = self._mapper.map_to_session(
                    json.loads(trajectory_path.read_text()),
                    session_id=request.case.session_id,
                )
            except Exception:
                session = None

        return TrialResult(
            case_id=request.packaged_task.case_id,
            repetition=request.repetition,
            status=status or inferred_status,
            output=output,
            metrics=metrics,
            unavailable_metrics=unavailable,
            session=session,
            usage=Usage(
                input_tokens=(agent_context.n_input_tokens if agent_context is not None else None),
                cache_read_tokens=(agent_context.n_cache_tokens if agent_context is not None else None),
                output_tokens=(agent_context.n_output_tokens if agent_context is not None else None),
                cost_usd=(agent_context.cost_usd if agent_context is not None else None),
            ),
            timing=PhaseTiming(
                environment_setup_seconds=_elapsed_seconds(harbor_result.environment_setup),
                agent_setup_seconds=_elapsed_seconds(harbor_result.agent_setup),
                agent_execution_seconds=_elapsed_seconds(harbor_result.agent_execution),
                verification_seconds=_elapsed_seconds(harbor_result.verifier),
                total_seconds=(
                    max(
                        (harbor_result.finished_at - harbor_result.started_at).total_seconds(),
                        0.0,
                    )
                    if harbor_result.started_at is not None and harbor_result.finished_at is not None
                    else None
                ),
            ),
            artifacts=_artifacts(trial, harbor_result),
            error=error or _error_from_harbor(harbor_result, error_phase),
        )


__all__ = ["HarborTrialExecutor"]
