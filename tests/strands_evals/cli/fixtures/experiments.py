"""Self-contained experiment factories used by CLI tests."""

from strands_evals import Case, Experiment
from strands_evals.packaged_tasks import (
    ComponentRef,
    MetricDefinition,
    PackagedTaskRef,
    TaskRequirements,
)
from strands_evals.targets import StrandsAgentTarget


def packaged_experiment() -> Experiment:
    case_id = "fixture/case"
    return Experiment(
        cases=[Case(name=case_id, input="Inspect the repository")],
        target=StrandsAgentTarget(
            factory="example:create_agent",
            source_revision="factory-revision",
            source_dirty=False,
            model={"model_id": "fixture-model"},
        ),
        repetitions=1,
        packaged_tasks={
            case_id: PackagedTaskRef(
                case_id=case_id,
                workload_name="fixture-benchmark",
                workload_revision="dataset-revision",
                scorer_revision="scorer-revision",
                package=ComponentRef(
                    kind="fixture-task",
                    value="/fixture/task",
                    revision="task-revision",
                ),
                requirements=TaskRequirements(
                    os="linux",
                    workdir="/workspace",
                ),
                metrics=(
                    MetricDefinition(
                        name="fixture_coverage",
                        primary=True,
                    ),
                ),
            )
        },
    )
