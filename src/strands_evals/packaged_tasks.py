"""Executor-neutral references to packaged benchmark tasks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComponentRef(BaseModel):
    """Opaque versioned reference interpreted by a benchmark integration."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    value: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class TaskRequirements(BaseModel):
    """Recorded execution requirements derived from the packaged task."""

    model_config = ConfigDict(frozen=True)

    os: str | None = None
    architecture: str | None = None
    cpus: int | None = Field(default=None, ge=1)
    memory_mb: int | None = Field(default=None, ge=1)
    storage_mb: int | None = Field(default=None, ge=1)
    gpus: int | None = Field(default=None, ge=0)
    workdir: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MetricDefinition(BaseModel):
    """One native metric emitted by a packaged task verifier."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str | None = None
    primary: bool = False


class PackagedTaskRef(BaseModel):
    """Stable task identity plus an opaque resolver reference."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    workload_name: str = Field(min_length=1)
    workload_revision: str = Field(min_length=1)
    scorer_revision: str = Field(min_length=1)
    package: ComponentRef
    requirements: TaskRequirements
    metrics: tuple[MetricDefinition, ...] = ()
