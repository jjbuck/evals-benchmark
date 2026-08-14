"""Named-metric report for repetition-aware packaged experiment runs."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from .run_store import RunManifest
from .trials import TrialResult, TrialStatus


class MetricSummary(BaseModel):
    """Availability and aggregate values for one native metric."""

    name: str
    valid: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    failed: int = Field(ge=0)
    invalid: int = Field(ge=0)
    mean: float | None = None


class RunReport(BaseModel):
    """Complete versioned report built from committed terminal records."""

    schema_version: int = 1
    manifest: RunManifest
    results: list[TrialResult]
    metric_summaries: dict[str, MetricSummary]
    compatibility_overall_score: float | None = None

    @classmethod
    def build(
        cls,
        manifest: RunManifest,
        results: list[TrialResult],
    ) -> "RunReport":
        """Build metric summaries without averaging unrelated metrics."""
        declared_names = {metric.name for metric in manifest.metrics}
        returned_names = {name for result in results for name in result.metrics}
        metric_names = sorted(declared_names | returned_names)
        summaries: dict[str, MetricSummary] = {}
        for name in metric_names:
            values: list[float] = []
            unavailable = 0
            failed = 0
            invalid = 0
            for result in results:
                if result.status == TrialStatus.COMPLETED:
                    if name in result.metrics:
                        values.append(result.metrics[name])
                    else:
                        unavailable += 1
                elif result.status == TrialStatus.AGENT_FAILURE:
                    failed += 1
                else:
                    invalid += 1
            summaries[name] = MetricSummary(
                name=name,
                valid=len(values),
                unavailable=unavailable,
                failed=failed,
                invalid=invalid,
                mean=sum(values) / len(values) if values else None,
            )

        primary = [metric.name for metric in manifest.metrics if metric.primary]
        compatibility_score = summaries[primary[0]].mean if len(primary) == 1 else None
        return cls(
            manifest=manifest,
            results=results,
            metric_summaries=summaries,
            compatibility_overall_score=compatibility_score,
        )

    def write(self, path: Path) -> None:
        """Atomically write this report."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n")
        os.replace(temporary, path)
