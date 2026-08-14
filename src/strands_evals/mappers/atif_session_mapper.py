"""Map Agent Trajectory Interchange Format (ATIF) data to Evals sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    SpanUnion,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)
from .session_mapper import SessionMapper


def _timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class ATIFSessionMapper(SessionMapper):
    """Convert one ATIF trajectory into a Strands Evals ``Session``."""

    def map_to_session(self, data: Any, session_id: str) -> Session:
        if not isinstance(data, dict):
            raise TypeError("ATIF trajectory must be a dictionary")

        resolved_session_id = str(data.get("session_id") or session_id)
        trace_id = str(data.get("trajectory_id") or uuid4())
        steps = data.get("steps") or []
        first_time = _timestamp(steps[0].get("timestamp")) if steps else datetime.now(timezone.utc)
        last_time = _timestamp(steps[-1].get("timestamp")) if steps else first_time
        instruction = next(
            (str(step.get("message") or "") for step in steps if step.get("source") == "user"),
            "",
        )
        response = next(
            (
                str(step.get("message") or "")
                for step in reversed(steps)
                if step.get("source") == "agent" and not step.get("tool_calls")
            ),
            "",
        )

        spans: list[SpanUnion] = [
            AgentInvocationSpan(
                span_info=SpanInfo(
                    trace_id=trace_id,
                    span_id=str(uuid4()),
                    session_id=resolved_session_id,
                    start_time=first_time,
                    end_time=last_time,
                ),
                user_prompt=instruction,
                agent_response=response,
                available_tools=self._tool_configs(data),
                metadata={"source_format": data.get("schema_version", "ATIF")},
            )
        ]

        for step in steps:
            observations = {
                result.get("source_call_id"): result
                for result in (step.get("observation") or {}).get("results", [])
                if isinstance(result, dict)
            }
            step_time = _timestamp(step.get("timestamp"))
            for call in step.get("tool_calls") or []:
                call_id = str(call.get("tool_call_id") or uuid4())
                observation = observations.get(call_id, {})
                extra = observation.get("extra") or {}
                content = observation.get("content")
                spans.append(
                    ToolExecutionSpan(
                        span_info=SpanInfo(
                            trace_id=trace_id,
                            span_id=str(uuid4()),
                            session_id=resolved_session_id,
                            start_time=step_time,
                            end_time=step_time,
                        ),
                        tool_call=ToolCall(
                            name=str(call.get("function_name") or ""),
                            arguments=call.get("arguments") or {},
                            tool_call_id=call_id,
                        ),
                        tool_result=ToolResult(
                            content=content if isinstance(content, str) else str(content or ""),
                            error=extra.get("error"),
                            tool_call_id=call_id,
                        ),
                        metadata={"source_format": data.get("schema_version", "ATIF")},
                    )
                )

        return Session(
            session_id=resolved_session_id,
            traces=[
                Trace(
                    trace_id=trace_id,
                    session_id=resolved_session_id,
                    spans=spans,
                )
            ],
        )

    @staticmethod
    def _tool_configs(data: dict[str, Any]) -> list[ToolConfig]:
        definitions = (data.get("agent") or {}).get("tool_definitions") or []
        configs: list[ToolConfig] = []
        for definition in definitions:
            function = definition.get("function", definition)
            name = function.get("name")
            if not name:
                continue
            configs.append(
                ToolConfig(
                    name=str(name),
                    description=function.get("description"),
                    parameters=function.get("parameters"),
                )
            )
        return configs
