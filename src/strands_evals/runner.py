"""Local runner for self-contained packaged experiments."""

from __future__ import annotations

import asyncio
import traceback

from .experiment import Experiment
from .run_report import RunReport
from .run_store import LocalRunResultStore, RunManifest
from .trials import (
    ExecutionRequest,
    TrialError,
    TrialErrorPhase,
    TrialExecutor,
    TrialRequest,
    TrialResult,
    TrialStatus,
)


async def run_experiment(
    experiment: Experiment,
    *,
    executor: TrialExecutor,
    execution: ExecutionRequest,
) -> RunReport:
    """Execute or resume a self-contained packaged experiment locally."""
    target = experiment.target
    if target is None:
        raise ValueError("A packaged experiment requires an evaluation target")

    cases = experiment.cases
    case_ids = [case.name for case in cases]
    if any(case_id is None or not case_id for case_id in case_ids):
        raise ValueError("Every packaged experiment case must have a stable name")
    stable_case_ids = [case_id for case_id in case_ids if case_id is not None]
    if not stable_case_ids:
        raise ValueError("A packaged experiment requires at least one case")
    if len(stable_case_ids) != len(set(stable_case_ids)):
        raise ValueError("Packaged experiment case names must be unique")

    packaged_tasks = experiment.packaged_tasks
    missing = [case_id for case_id in stable_case_ids if case_id not in packaged_tasks]
    if missing:
        raise ValueError(f"Cases have no packaged task reference: {', '.join(missing)}")
    extra = sorted(set(packaged_tasks) - set(stable_case_ids))
    if extra:
        raise ValueError("Packaged task references have no matching case: " + ", ".join(extra))
    mismatched = [
        f"{case_id!r} -> {packaged_tasks[case_id].case_id!r}"
        for case_id in stable_case_ids
        if packaged_tasks[case_id].case_id != case_id
    ]
    if mismatched:
        raise ValueError("Packaged task reference case IDs do not match their keys: " + ", ".join(mismatched))

    requested_manifest = RunManifest.build(
        target=target.snapshot(),
        executor=executor,
        execution=execution,
        repetitions=experiment.repetitions,
        ordered_case_ids=stable_case_ids,
        packaged_tasks=packaged_tasks,
    )
    store = LocalRunResultStore.open(
        execution.output_dir,
        requested_manifest,
        resume_run_id=execution.resume_run_id,
    )

    semaphore = asyncio.Semaphore(execution.max_concurrency)
    case_by_id = {case.name: case for case in cases if case.name is not None}

    async def execute_one(case_id: str, repetition: int) -> None:
        if store.load(case_id, repetition) is not None:
            return
        request = TrialRequest(
            run_id=store.manifest.run_id,
            case=case_by_id[case_id],
            packaged_task=packaged_tasks[case_id],
            target=target,
            repetition=repetition,
            execution=execution,
        )
        async with semaphore:
            try:
                result = await executor.run(request)
                if result.case_id != case_id or result.repetition != repetition:
                    raise ValueError(
                        "Trial executor returned the wrong result coordinate: "
                        f"expected {case_id}/{repetition}, got "
                        f"{result.case_id}/{result.repetition}"
                    )
            except asyncio.CancelledError as error:
                store.save(
                    TrialResult(
                        case_id=case_id,
                        repetition=repetition,
                        status=TrialStatus.CANCELLED,
                        error=TrialError(
                            phase=TrialErrorPhase.CANCELLATION,
                            error_type=type(error).__name__,
                            message=str(error),
                        ),
                    )
                )
                raise
            except Exception as error:
                result = TrialResult(
                    case_id=case_id,
                    repetition=repetition,
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
            store.save(result)

    tasks = [
        asyncio.create_task(execute_one(case_id, repetition))
        for case_id in stable_case_ids
        for repetition in range(1, experiment.repetitions + 1)
        if store.load(case_id, repetition) is None
    ]
    if tasks:
        await asyncio.gather(*tasks)

    results = store.load_all()
    expected_count = len(stable_case_ids) * experiment.repetitions
    if len(results) != expected_count:
        raise RuntimeError(
            f"Run {store.manifest.run_id} has {len(results)} terminal results; expected {expected_count}"
        )
    report = RunReport.build(store.manifest, results)
    report.write(store.run_dir / "report.json")
    return report


__all__ = ["run_experiment"]
