import pytest

from strands_evals.mappers.atif_session_mapper import ATIFSessionMapper
from strands_evals.types.trace import AgentInvocationSpan, ToolExecutionSpan


def test_maps_agent_and_tool_evidence_from_atif():
    session = ATIFSessionMapper().map_to_session(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "atif-session",
            "trajectory_id": "trajectory-one",
            "agent": {
                "tool_definitions": [
                    {
                        "name": "sandbox_file_editor",
                        "description": "Read files",
                    }
                ]
            },
            "steps": [
                {
                    "source": "user",
                    "message": "Inspect the repository.",
                    "timestamp": "2026-08-13T12:00:00Z",
                },
                {
                    "source": "agent",
                    "message": "Called sandbox_file_editor",
                    "timestamp": "2026-08-13T12:00:01Z",
                    "tool_calls": [
                        {
                            "tool_call_id": "read-one",
                            "function_name": "sandbox_file_editor",
                            "arguments": {
                                "command": "view",
                                "path": "/workspace/example.py",
                            },
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "read-one",
                                "content": "     1  value = True",
                                "extra": {"error": "partial read"},
                            }
                        ]
                    },
                },
                {
                    "source": "agent",
                    "message": "Found the relevant file.",
                    "timestamp": "2026-08-13T12:00:02Z",
                },
            ],
        },
        session_id="fallback-session",
    )

    assert session.session_id == "atif-session"
    [trace] = session.traces
    assert trace.trace_id == "trajectory-one"
    invocation = trace.spans[0]
    assert isinstance(invocation, AgentInvocationSpan)
    assert invocation.user_prompt == "Inspect the repository."
    assert invocation.agent_response == "Found the relevant file."
    assert [tool.name for tool in invocation.available_tools] == ["sandbox_file_editor"]
    tool = trace.spans[1]
    assert isinstance(tool, ToolExecutionSpan)
    assert tool.tool_call.tool_call_id == "read-one"
    assert tool.tool_result.content == "     1  value = True"
    assert tool.tool_result.error == "partial read"


def test_rejects_non_object_atif():
    with pytest.raises(TypeError, match="ATIF trajectory must be a dictionary"):
        ATIFSessionMapper().map_to_session([], session_id="fallback-session")
