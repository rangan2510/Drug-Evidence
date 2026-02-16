"""Groq ReAct loop with Instructor-based structured parsing.

This module implements a text-only ReAct orchestration pattern that avoids
provider-native function calling. The control loop is:

1. Send query + available tools to Groq (free-form text response).
2. Parse that free-form text with Instructor into ``ReActStep``:
   - thought
   - sufficient_information (yes/no)
   - tool_calls
3. Execute requested tools locally (with Pydantic argument validation).
4. Send tool results back as memory context.
5. Repeat until sufficient information is reached.
6. Parse final free-form synthesis into ``DrugDiseasePrediction`` via a second
   Instructor call.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

import httpx
import instructor
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError

from src.config.settings import Settings
from src.schemas.prediction import DrugDiseasePrediction


TOutput = TypeVar("TOutput", bound=BaseModel)


class ReActToolCall(BaseModel):
    """A single requested tool call parsed from free-form model text."""

    name: str = Field(..., description="Tool name to execute")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool",
    )


class ReActStep(BaseModel):
    """Structured interpretation of one free-form ReAct step."""

    thought: str = Field(..., description="Model's reasoning for this step")
    sufficient_information: bool = Field(
        ...,
        description="Whether enough evidence has been gathered to finalize",
    )
    tool_calls: list[ReActToolCall] = Field(default_factory=list)
    synthesis: str = Field(
        default="",
        description="Current free-form synthesis from the model",
    )


class ReActConfig(BaseModel):
    """Runtime controls for ReAct execution."""

    max_iterations: int = 8
    max_tool_calls_per_step: int = 4
    max_parse_failures: int = 2
    max_tool_result_chars: int = 3000
    generation_temperature: float = 0.0
    generation_max_tokens: int = 4096
    step_parse_max_tokens: int = 1200
    final_parse_max_tokens: int = 16384


@dataclass(slots=True)
class ReActTool:
    """Executable tool metadata for the ReAct loop."""

    name: str
    description: str
    handler: Callable[..., Awaitable[Any] | Any]
    args_model: type[BaseModel] | None = None

    def manifest(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema() if self.args_model else {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        return {
            "name": self.name,
            "description": self.description,
            "arguments_schema": schema,
        }


@dataclass(slots=True)
class ReActIterationTrace:
    """Trace for one loop iteration."""

    iteration: int
    raw_assistant_text: str
    parsed_step: ReActStep | None = None
    executed_tools: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None


@dataclass(slots=True)
class ReActRunResult(Generic[TOutput]):
    """Final run output and debug traces."""

    output: TOutput
    final_freeform: str
    iterations: list[ReActIterationTrace]


@dataclass(slots=True)
class ReActBaselineResult:
    """Baseline-like container for easy replacement experiments."""

    prediction: DrugDiseasePrediction
    raw_text: str
    raw_model_id: str
    iteration_count: int
    extra: dict[str, Any] = field(default_factory=dict)


class TavilySearchArgs(BaseModel):
    """Arguments for Tavily web search tool."""

    query: str


class GroqReActAgent:
    """Text-only ReAct controller backed by Groq + Instructor parsers."""

    def __init__(
        self,
        *,
        settings: Settings,
        model_id: str,
        step_parser_model_id: str | None = None,
        final_parser_model_id: str | None = None,
        config: ReActConfig | None = None,
        text_generator: Callable[[list[dict[str, str]]], Awaitable[str]] | None = None,
        step_parser: Callable[[str, list[str]], Awaitable[ReActStep]] | None = None,
        final_parser: Callable[[str, type[TOutput]], Awaitable[TOutput]] | None = None,
    ) -> None:
        self.settings = settings
        self.config = config or ReActConfig()
        self.model_name = model_id.split(":", 1)[1] if ":" in model_id else model_id
        self.step_parser_model_name = (
            step_parser_model_id.split(":", 1)[1]
            if step_parser_model_id and ":" in step_parser_model_id
            else (step_parser_model_id or self.model_name)
        )
        self.final_parser_model_name = (
            final_parser_model_id.split(":", 1)[1]
            if final_parser_model_id and ":" in final_parser_model_id
            else (final_parser_model_id or self.model_name)
        )

        self._openai: AsyncOpenAI | None = None
        self._step_instructor: Any | None = None
        self._final_instructor: Any | None = None

        if text_generator is None or step_parser is None or final_parser is None:
            self._openai = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

            self._step_instructor = instructor.from_openai(
                self._openai,
                mode=instructor.Mode.JSON,
            )
            self._final_instructor = instructor.from_openai(
                self._openai,
                mode=instructor.Mode.JSON,
            )

        self._text_generator = text_generator or self._generate_text
        self._step_parser = step_parser or self._parse_step
        self._final_parser = final_parser or self._parse_final

    async def run(
        self,
        *,
        query: str,
        tools: list[ReActTool],
        output_model: type[TOutput] = DrugDiseasePrediction,
    ) -> ReActRunResult[TOutput]:
        """Run the full ReAct loop and return structured output."""
        tools_by_name = {tool.name: tool for tool in tools}
        tool_manifest = [tool.manifest() for tool in tools]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_initial_user_prompt(query, tool_manifest),
            },
        ]

        traces: list[ReActIterationTrace] = []
        parse_failures = 0
        final_freeform = ""

        for iteration in range(1, self.config.max_iterations + 1):
            raw_text = await self._text_generator(messages)
            messages.append({"role": "assistant", "content": raw_text})

            trace = ReActIterationTrace(
                iteration=iteration,
                raw_assistant_text=raw_text,
            )

            try:
                step = await self._step_parser(raw_text, list(tools_by_name))
                trace.parsed_step = step
                parse_failures = 0
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                trace.parse_error = f"{type(exc).__name__}: {exc}"
                traces.append(trace)

                if parse_failures > self.config.max_parse_failures:
                    final_freeform = raw_text
                    break

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer could not be parsed into the required "
                            "ReAct step format. Respond again with clear thought, "
                            "sufficient information decision, and explicit tool calls."
                        ),
                    }
                )
                continue

            traces.append(trace)
            final_freeform = step.synthesis or raw_text

            if step.sufficient_information:
                break

            if not step.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Insufficient information was indicated, but no tool calls were "
                            "provided. Call one or more tools or mark sufficient_information=true."
                        ),
                    }
                )
                continue

            tool_results = await self._execute_tool_calls(step.tool_calls, tools_by_name)
            trace.executed_tools.extend(tool_results)
            messages.append(
                {
                    "role": "user",
                    "content": self._render_tool_results(tool_results),
                }
            )

        final_output = await self._final_parser(final_freeform, output_model)
        return ReActRunResult(
            output=final_output,
            final_freeform=final_freeform,
            iterations=traces,
        )

    async def _generate_text(self, messages: list[dict[str, str]]) -> str:
        if self._openai is None:
            msg = "OpenAI client is not configured"
            raise RuntimeError(msg)
        try:
            response = await self._openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.generation_temperature,
                max_tokens=self.config.generation_max_tokens,
            )
        except BadRequestError as exc:
            body: Any = getattr(exc, "body", None)
            error_obj: dict[str, Any] = {}
            if isinstance(body, dict):
                error_obj = body.get("error", {}) if isinstance(body.get("error"), dict) else body

            code = error_obj.get("code")
            failed_generation = error_obj.get("failed_generation")

            if not failed_generation:
                message = str(exc)
                match = re.search(r'"failed_generation"\s*:\s*"(.+?)"\s*}', message)
                if match:
                    failed_generation = bytes(match.group(1), "utf-8").decode("unicode_escape")
                if "tool_use_failed" in message and code is None:
                    code = "tool_use_failed"

            if code == "tool_use_failed" and failed_generation:
                try:
                    payload = json.loads(failed_generation)
                    name = payload.get("name", "")
                    arguments = payload.get("arguments", {})
                    return (
                        "Attempting tool call in text-only mode. "
                        f"Tool requested: {name} with arguments: {arguments}. "
                        "More information is still required."
                    )
                except (TypeError, ValueError):
                    return str(failed_generation)
            raise
        content = response.choices[0].message.content
        if not content:
            return ""
        return content

    async def _parse_step(self, freeform_text: str, allowed_tools: list[str]) -> ReActStep:
        if self._step_instructor is None:
            msg = "Step parser client is not configured"
            raise RuntimeError(msg)
        return await self._step_instructor.create(
            model=self.step_parser_model_name,
            response_model=ReActStep,
            max_tokens=self.config.step_parse_max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _STEP_PARSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Allowed tools: {allowed_tools}\n\n"
                        f"Assistant free-form text:\n{freeform_text}"
                    ),
                },
            ],
        )

    async def _parse_final(self, freeform_text: str, output_model: type[TOutput]) -> TOutput:
        if self._final_instructor is None:
            msg = "Final parser client is not configured"
            raise RuntimeError(msg)
        return await self._final_instructor.create(
            model=self.final_parser_model_name,
            response_model=output_model,
            max_tokens=self.config.final_parse_max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _FINAL_PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": freeform_text},
            ],
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[ReActToolCall],
        tools_by_name: dict[str, ReActTool],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for call in tool_calls[: self.config.max_tool_calls_per_step]:
            tool = tools_by_name.get(call.name)
            if tool is None:
                results.append(
                    {
                        "tool": call.name,
                        "arguments": call.arguments,
                        "ok": False,
                        "error": "Unknown tool name",
                    }
                )
                continue

            try:
                kwargs = self._validate_tool_arguments(tool, call.arguments)
            except ValidationError as exc:
                results.append(
                    {
                        "tool": call.name,
                        "arguments": call.arguments,
                        "ok": False,
                        "error": f"Argument validation failed: {exc}",
                    }
                )
                continue

            try:
                value = tool.handler(**kwargs)
                if inspect.isawaitable(value):
                    value = await value

                results.append(
                    {
                        "tool": call.name,
                        "arguments": kwargs,
                        "ok": True,
                        "result": value,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "tool": call.name,
                        "arguments": kwargs,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return results

    def _validate_tool_arguments(
        self,
        tool: ReActTool,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if tool.args_model is None:
            return arguments
        validated = tool.args_model.model_validate(arguments)
        return validated.model_dump()

    def _render_tool_results(self, tool_results: list[dict[str, Any]]) -> str:
        rendered = json.dumps(tool_results, ensure_ascii=False, default=str)
        limit = self.config.max_tool_result_chars
        if len(rendered) <= limit:
            clipped = rendered
        else:
            clipped = rendered[:limit] + "..."

        return (
            "TOOL_RESULTS\n"
            "------------\n"
            f"{clipped}\n\n"
            "Use these results for the next reasoning step."
        )

    def _build_initial_user_prompt(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> str:
        tools_json = json.dumps(tools, ensure_ascii=False, default=str)
        return (
            f"Query:\n{query}\n\n"
            "Available tools (name, description, argument schema):\n"
            f"{tools_json}\n\n"
            "Respond with your best biomedical reasoning. You may request tool calls "
            "in free-form text when needed."
        )


def build_tavily_tool(
    *,
    settings: Settings,
    allowed_domains: list[str] | None = None,
) -> ReActTool:
    """Build a ReAct-compatible Tavily search tool."""

    async def _tavily_search(query: str) -> list[dict[str, str]]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": "advanced",
            "include_raw_content": False,
            "include_domains": allowed_domains or [],
            "exclude_domains": [],
        }
        async with httpx.AsyncClient() as http:
            response = await http.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.tavily_api_key}",
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:2000],
            }
            for item in results
        ]

    return ReActTool(
        name="web_search",
        description="Search biomedical web sources for evidence",
        handler=_tavily_search,
        args_model=TavilySearchArgs,
    )


async def run_react_baseline(
    prompt: str,
    settings: Settings,
    *,
    model_id: str,
    step_parser_model_id: str | None = None,
    final_parser_model_id: str | None = None,
    allowed_domains: list[str] | None = None,
    config: ReActConfig | None = None,
) -> ReActBaselineResult:
    """Run the ReAct loop as a baseline-style agent over Groq."""
    agent = GroqReActAgent(
        settings=settings,
        model_id=model_id,
        step_parser_model_id=step_parser_model_id,
        final_parser_model_id=final_parser_model_id,
        config=config,
    )
    tools = [build_tavily_tool(settings=settings, allowed_domains=allowed_domains)]

    result = await agent.run(
        query=prompt,
        tools=tools,
        output_model=DrugDiseasePrediction,
    )
    return ReActBaselineResult(
        prediction=result.output,
        raw_text=result.final_freeform,
        raw_model_id=model_id,
        iteration_count=len(result.iterations),
        extra={
            "iterations": [
                {
                    "iteration": t.iteration,
                    "parsed_step": t.parsed_step.model_dump() if t.parsed_step else None,
                    "executed_tools": t.executed_tools,
                    "parse_error": t.parse_error,
                }
                for t in result.iterations
            ]
        },
    )


_GENERATION_SYSTEM_PROMPT = """\
You are operating in a ReAct loop for biomedical analysis.

At each step, reason about what information is still missing.
If more evidence is needed, request explicit tool calls by naming tools and
arguments clearly in plain text.
If you have enough evidence, provide a concise synthesis and indicate that
sufficient information has been reached.

Do not fabricate tool outputs.
"""

_STEP_PARSE_SYSTEM_PROMPT = """\
You are a parser that converts free-form assistant text into structured data.

Extract exactly:
- thought: concise reasoning summary
- sufficient_information: true/false
- tool_calls: list of {name, arguments}
- synthesis: the assistant's latest free-form synthesis text

If no tool calls are requested, return tool_calls as an empty list.
Never invent tool names that are not in the allowed tool list.
"""

_FINAL_PARSE_SYSTEM_PROMPT = """\
You are a strict biomedical data extraction assistant.
Convert the provided free-form analysis into the required response schema.
Preserve details from the source text and do not fabricate unsupported claims.
"""
