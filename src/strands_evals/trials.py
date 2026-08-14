"""Provisional execution contracts for packaged evaluation trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .case import Case
from .components import ComponentSnapshot
from .packaged_tasks import PackagedTaskRef
from .targets import EvaluationTarget
from .task_environments import TaskEnvironmentSelection
from .types.trace import Session


class TrialStatus(str, Enum):
    """Terminal status of one logical case repetition."""

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    VERIFIER_INVALID = "verifier_invalid"
    CANCELLED = "cancelled"


class TrialErrorPhase(str, Enum):
    """Execution phase responsible for an unsuccessful trial."""

    ENVIRONMENT = "environment"
    AGENT = "agent"
    VERIFIER = "verifier"
    EXECUTOR = "executor"
    CANCELLATION = "cancellation"


class TrialLimits(BaseModel):
    """Run-level limit overrides that affect trial semantics."""

    model_config = ConfigDict(frozen=True)

    timeout_multiplier: float = Field(default=1.0, gt=0)
    agent_timeout_multiplier: float | None = Field(default=None, gt=0)
    verifier_timeout_multiplier: float | None = Field(default=None, gt=0)


class Usage(BaseModel):
    """Token and cost measurements retained from one trial."""

    input_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class PhaseTiming(BaseModel):
    """Elapsed seconds for measured execution phases."""

    environment_setup_seconds: float | None = Field(default=None, ge=0)
    agent_setup_seconds: float | None = Field(default=None, ge=0)
    agent_execution_seconds: float | None = Field(default=None, ge=0)
    verification_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float | None = Field(default=None, ge=0)


class ArtifactRef(BaseModel):
    """Reference to an executor-owned diagnostic or result artifact."""

    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    media_type: str | None = None


class TrialError(BaseModel):
    """Structured terminal error without collapsing it into a score."""

    phase: TrialErrorPhase
    error_type: str = Field(min_length=1)
    message: str
    traceback: str | None = None


class TrialResult(BaseModel):
    """Executor-neutral terminal result for one logical coordinate."""

    schema_version: int = 1
    case_id: str = Field(min_length=1)
    repetition: int = Field(ge=1)
    status: TrialStatus
    output: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    unavailable_metrics: set[str] = Field(default_factory=set)
    session: Session | None = None
    usage: Usage = Field(default_factory=Usage)
    timing: PhaseTiming = Field(default_factory=PhaseTiming)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    error: TrialError | None = None


@dataclass(frozen=True)
class ExecutionRequest:
    """Execution conditions supplied when an experiment is run."""

    task_environment: TaskEnvironmentSelection
    max_concurrency: int
    limits: TrialLimits
    output_dir: Path
    resume_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


@dataclass(frozen=True)
class TrialRequest:
    """One case/repetition request passed to a trial executor."""

    run_id: str
    case: Case
    packaged_task: PackagedTaskRef
    target: EvaluationTarget
    repetition: int
    execution: ExecutionRequest


class TrialExecutor(Protocol):
    """Backend that executes one packaged trial and returns a terminal result."""

    def snapshot(self) -> ComponentSnapshot:
        """Return the executor declaration used in the run manifest."""

    async def run(self, request: TrialRequest) -> TrialResult:
        """Execute one logical trial without raising ordinary phase failures."""
