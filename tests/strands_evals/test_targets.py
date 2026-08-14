import pytest

from strands_evals.targets import StrandsAgentTarget, target_from_snapshot


def test_strands_target_snapshot_round_trips_deterministically():
    target = StrandsAgentTarget(
        factory="example.agents:create_agent",
        kwargs={"window": 40, "nested": {"enabled": True}},
        source_revision="abc123",
        source_dirty=False,
        model={"model_id": "example/model"},
    )

    snapshot = target.snapshot()
    restored = target_from_snapshot(snapshot)

    assert restored == target
    assert snapshot.kind == "strands-agent"
    assert snapshot.model_dump_json() == target.snapshot().model_dump_json()


def test_strands_target_rejects_non_importable_factory():
    with pytest.raises(ValueError, match="module:symbol"):
        StrandsAgentTarget(factory="create_agent")


def test_strands_target_rejects_non_json_configuration():
    with pytest.raises(ValueError, match="target kwargs"):
        StrandsAgentTarget(
            factory="example:create_agent",
            kwargs={"invalid": object()},
        )
