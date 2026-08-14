"""Provisional declarative targets for self-contained experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .components import ComponentSnapshot, canonical_json_object


@runtime_checkable
class EvaluationTarget(Protocol):
    """A configured system under test that can be snapshotted for execution."""

    def snapshot(self) -> ComponentSnapshot:
        """Return the portable declaration used by executors and run manifests."""


@dataclass(frozen=True)
class StrandsAgentTarget:
    """An importable Strands agent factory and its declared study configuration."""

    factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    source_revision: str | None = None
    source_dirty: bool | None = None
    model: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module, separator, symbol = self.factory.partition(":")
        if not separator or not module.strip() or not symbol.strip():
            raise ValueError("StrandsAgentTarget.factory must use an importable 'module:symbol' reference")
        object.__setattr__(
            self,
            "kwargs",
            canonical_json_object(self.kwargs, field_name="target kwargs"),
        )
        object.__setattr__(
            self,
            "model",
            canonical_json_object(self.model, field_name="target model"),
        )

    def snapshot(self) -> ComponentSnapshot:
        """Return a deterministic declaration without introspecting factory code."""
        return ComponentSnapshot(
            kind="strands-agent",
            config={
                "factory": self.factory,
                "kwargs": self.kwargs,
                "source_revision": self.source_revision,
                "source_dirty": self.source_dirty,
                "model": self.model,
            },
        )


def target_from_snapshot(snapshot: ComponentSnapshot) -> EvaluationTarget:
    """Reconstruct a supported provisional target from its snapshot."""
    if snapshot.kind != "strands-agent" or snapshot.version != 1:
        raise ValueError(f"Unsupported evaluation target {snapshot.kind!r} version {snapshot.version}")
    config = snapshot.config
    return StrandsAgentTarget(
        factory=str(config.get("factory") or ""),
        kwargs=dict(config.get("kwargs") or {}),
        source_revision=config.get("source_revision"),
        source_dirty=config.get("source_dirty"),
        model=dict(config.get("model") or {}),
    )


__all__ = [
    "EvaluationTarget",
    "StrandsAgentTarget",
    "target_from_snapshot",
]
