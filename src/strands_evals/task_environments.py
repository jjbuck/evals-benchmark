"""Provisional task-environment selections for experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .components import ComponentSnapshot, canonical_json_object


@runtime_checkable
class TaskEnvironmentSelection(Protocol):
    """A declarative task placement interpreted by a trial executor."""

    def snapshot(self) -> ComponentSnapshot:
        """Return the placement declaration for a run manifest."""


@dataclass(frozen=True)
class LocalContainers:
    """Run packaged task environments in Harbor's local Docker provider."""

    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "options",
            canonical_json_object(
                self.options,
                field_name="local container options",
            ),
        )

    def snapshot(self) -> ComponentSnapshot:
        """Return the executor-neutral local-container registry selection."""
        return ComponentSnapshot(
            kind="local-containers",
            config={"options": self.options},
        )


def task_environment_from_snapshot(
    snapshot: ComponentSnapshot,
) -> TaskEnvironmentSelection:
    """Reconstruct a supported provisional task-environment selection."""
    if snapshot.kind != "local-containers" or snapshot.version != 1:
        raise ValueError(f"Unsupported task environment {snapshot.kind!r} version {snapshot.version}")
    return LocalContainers(options=dict(snapshot.config.get("options") or {}))


__all__ = [
    "LocalContainers",
    "TaskEnvironmentSelection",
    "task_environment_from_snapshot",
]
