"""Unit tests for the Groq+Instructor ReAct loop module."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.react_instructor_agent import (
    GroqReActAgent,
    ReActConfig,
    ReActRunResult,
    ReActStep,
    ReActTool,
    ReActToolCall,
)
from src.config.settings import Settings


class _FinalModel(BaseModel):
    answer: str


class _LookupArgs(BaseModel):
    query: str


@pytest.mark.asyncio
async def test_react_loop_executes_tools_and_finishes() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    seen_messages: list[list[dict[str, str]]] = []

    async def fake_text_generator(messages: list[dict[str, str]]) -> str:
        seen_messages.append([m.copy() for m in messages])
        if len(seen_messages) == 1:
            return "Need evidence. I should call lookup with query aspirin and cancer."
        return "I now have enough evidence to conclude aspirin may help cardiovascular disease."

    async def fake_step_parser(text: str, allowed_tools: list[str]) -> ReActStep:
        assert "lookup" in allowed_tools
        if "Need evidence" in text:
            return ReActStep(
                thought="Need source evidence first",
                sufficient_information=False,
                tool_calls=[
                    ReActToolCall(name="lookup", arguments={"query": "aspirin mechanism"}),
                ],
                synthesis="",
            )
        return ReActStep(
            thought="Enough evidence collected",
            sufficient_information=True,
            tool_calls=[],
            synthesis="Aspirin has plausible cardiovascular relevance.",
        )

    async def fake_final_parser(text: str, output_model: type[_FinalModel]) -> _FinalModel:
        return output_model(answer=text)

    async def lookup(query: str) -> dict[str, Any]:
        return {"query": query, "hits": [{"pmid": "12345678"}]}

    agent = GroqReActAgent(
        settings=settings,
        model_id="groq:openai/gpt-oss-120b",
        config=ReActConfig(max_iterations=4),
        text_generator=fake_text_generator,
        step_parser=fake_step_parser,
        final_parser=fake_final_parser,
    )

    result: ReActRunResult[_FinalModel] = await agent.run(
        query="Assess aspirin disease associations",
        tools=[
            ReActTool(
                name="lookup",
                description="Lookup biomedical evidence",
                handler=lookup,
                args_model=_LookupArgs,
            )
        ],
        output_model=_FinalModel,
    )

    assert result.output.answer == "Aspirin has plausible cardiovascular relevance."
    assert len(result.iterations) == 2
    assert result.iterations[0].executed_tools
    assert result.iterations[0].executed_tools[0]["ok"] is True
    assert "TOOL_RESULTS" in seen_messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_react_loop_handles_unknown_tool_and_continues() -> None:
    settings = Settings(_env_file=None, groq_api_key="test-key")

    async def fake_text_generator(messages: list[dict[str, str]]) -> str:
        if len(messages) < 4:
            return "Need more data; call missing_tool with x=1"
        return "Enough now."

    async def fake_step_parser(text: str, allowed_tools: list[str]) -> ReActStep:
        if "Need more data" in text:
            return ReActStep(
                thought="Try a tool",
                sufficient_information=False,
                tool_calls=[ReActToolCall(name="missing_tool", arguments={"x": 1})],
                synthesis="",
            )
        return ReActStep(
            thought="Stop",
            sufficient_information=True,
            tool_calls=[],
            synthesis="Final synthesis",
        )

    async def fake_final_parser(text: str, output_model: type[_FinalModel]) -> _FinalModel:
        return output_model(answer=text)

    agent = GroqReActAgent(
        settings=settings,
        model_id="groq:openai/gpt-oss-120b",
        text_generator=fake_text_generator,
        step_parser=fake_step_parser,
        final_parser=fake_final_parser,
    )

    result = await agent.run(
        query="Assess aspirin",
        tools=[],
        output_model=_FinalModel,
    )

    assert result.output.answer == "Final synthesis"
    assert result.iterations[0].executed_tools[0]["ok"] is False
    assert "Unknown tool name" in result.iterations[0].executed_tools[0]["error"]
