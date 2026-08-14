"""Serializable component declarations used by provisional execution contracts."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_json_object(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    """Validate and copy a JSON-serializable object."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain only JSON values: {error}") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return decoded


class ComponentSnapshot(BaseModel):
    """Versioned, JSON-serializable declaration of one configured component."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)
