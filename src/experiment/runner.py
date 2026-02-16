"""Experiment runner -- execute pipeline and websearch arms against a single drug.

``run_pipeline_arm`` builds a fresh evidence agent for the arm's model,
runs it with the supplied ``EvidenceDeps``, and returns an ``ArmResult``
containing the structured prediction, usage stats, and timing.

``run_websearch_arm`` calls the unified ``run_baseline()`` which uses a
PydanticAI agent with Tavily web search.  All frontier providers (OpenAI,
Anthropic) go through the same code path.

All arms return the same ``ArmResult`` schema for apples-to-apples comparison.

Usage
-----
::

    from src.experiment.runner import run_pipeline_arm, run_websearch_arm
    from src.experiment.arms import PIPELINE_ARMS, WEBSEARCH_ARMS

    result = await run_pipeline_arm(
        arm=PIPELINE_ARMS["pipeline-gpt41"],
        deps=deps,
    )
    ws = await run_websearch_arm(
        arm=WEBSEARCH_ARMS["websearch-gpt52"],
        drug_name="aspirin",
        chembl_id="CHEMBL25",
        settings=settings,
    )
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from src.agents.baseline_agent import (
    build_baseline_prompt,
    run_baseline,
)


def _strip_pending_tool_calls(
    messages: list,
) -> list:
    """Remove trailing messages with unprocessed tool calls.

    When ``UsageLimitExceeded`` fires mid-turn the last ``ModelResponse``
    may contain ``ToolCallPart``s that were never answered.  PydanticAI
    refuses a new user prompt in that state, so we trim backward until
    the conversation ends on a clean boundary.
    """
    while messages:
        last = messages[-1]
        if isinstance(last, ModelResponse) and any(
            isinstance(p, ToolCallPart) for p in last.parts
        ):
            messages.pop()
            continue
        break
    return messages
from src.agents.deps import EvidenceDeps
from src.agents.evidence_agent import build_evidence_agent_with_context
from src.config.models import MODEL_REGISTRY, ModelSpec
from src.config.settings import Settings
from src.experiment.arms import ArmConfig, ArmType
from src.schemas.prediction import DrugDiseasePrediction

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


class ArmResult(BaseModel):
    """Outcome of running a single arm on a single drug."""

    arm_id: str
    drug_name: str
    model_id: str = Field(..., description="Resolved pydantic_ai model string")
    prediction: DrugDiseasePrediction | None = Field(
        default=None,
        description="Structured prediction; None if the run errored",
    )
    usage: dict = Field(
        default_factory=dict,
        description="Token / request usage from RunUsage (serialised as dict)",
    )
    wall_clock_seconds: float = Field(
        default=0.0, description="Wall-clock duration of the agent run"
    )
    error: str | None = Field(
        default=None, description="Error message if the run failed"
    )


# ------------------------------------------------------------------
# Default user prompt template
# ------------------------------------------------------------------

_DEFAULT_PROMPT_TEMPLATE = (
    "Analyse the drug '{drug_name}'"
    "{chembl_clause}"
    " and predict disease associations with mechanistic evidence chains."
)


def _build_prompt(deps: EvidenceDeps, override: str | None = None) -> str:
    """Construct the user prompt sent to the evidence agent."""
    if override:
        return override
    chembl_clause = f" (ChEMBL ID: {deps.chembl_id})" if deps.chembl_id else ""
    return _DEFAULT_PROMPT_TEMPLATE.format(
        drug_name=deps.drug_name,
        chembl_clause=chembl_clause,
    )


# ------------------------------------------------------------------
# Single-arm runner
# ------------------------------------------------------------------


async def run_pipeline_arm(
    arm: ArmConfig,
    deps: EvidenceDeps,
    registry: dict[str, ModelSpec] | None = None,
    *,
    prompt: str | None = None,
    retries: int = 3,
) -> ArmResult:
    """Run one pipeline arm for the drug configured in *deps*.

    A **fresh** evidence agent is built for each call so there is no
    shared mutable state between concurrent arm executions.

    Parameters
    ----------
    arm:
        Arm configuration (must be ``ArmType.PIPELINE``).
    deps:
        Pre-populated dependency container with drug info + services.
    registry:
        Model registry (defaults to module-level ``MODEL_REGISTRY``).
    prompt:
        Custom user prompt override.  When ``None`` a sensible default
        is constructed from ``deps.drug_name`` and ``deps.chembl_id``.
    retries:
        PydanticAI retry count for validation / model errors.

    Returns
    -------
    ArmResult
        Always returned -- errors are captured in ``ArmResult.error``
        rather than raised.
    """
    reg = registry or MODEL_REGISTRY
    user_prompt = _build_prompt(deps, prompt)
    model_id = f"<unresolved:{arm.model_key}>"

    t0 = time.perf_counter()
    try:
        model_id = arm.resolve_model_id(reg)

        logger.info(
            "Starting arm '%s' (model=%s) for drug '%s'",
            arm.arm_id,
            model_id,
            deps.drug_name,
        )
        # Use staged (MongoDB) tools when an evidence store is available
        use_staged = deps.evidence_store is not None
        agent = build_evidence_agent_with_context(
            model_id=model_id,
            retries=retries,
            use_staged=use_staged,
        )

        # Raise the default request limit (50) so the agent has room for
        # multiple tool-call rounds before producing structured output.
        request_limit = max(20, int(deps.settings.pipeline_request_limit))

        # Cap total tool calls to prevent excessive sequential invocations
        # (e.g., Anthropic models validating PMIDs one-by-one).
        tool_calls_limit = 25

        # Set generous max_tokens so large structured outputs (many diseases
        # with evidence chains) are not truncated -- the default 4096/8192
        # is often too small for Anthropic models producing 15+ associations.
        settings: ModelSettings = {"max_tokens": 16384}

        with capture_run_messages() as messages:
            try:
                result = await agent.run(
                    user_prompt,
                    deps=deps,
                    usage_limits=UsageLimits(
                        request_limit=request_limit,
                        tool_calls_limit=tool_calls_limit,
                    ),
                    model_settings=settings,
                )
            except UsageLimitExceeded:
                # Tool-call limit hit -- force a final answer from evidence
                # gathered so far by re-running with no tool calls allowed.
                logger.warning(
                    "Arm '%s' hit tool_calls_limit (%d) for '%s' -- "
                    "forcing final structured output from gathered evidence",
                    arm.arm_id,
                    tool_calls_limit,
                    deps.drug_name,
                )
                # Strip trailing ModelResponse with unprocessed ToolCallParts
                # so PydanticAI accepts the new user prompt.
                clean = _strip_pending_tool_calls(list(messages))
                result = await agent.run(
                    "You have reached the tool call limit. "
                    "Using ONLY the evidence you have already gathered above, "
                    "produce your final DrugDiseasePrediction now.",
                    deps=deps,
                    message_history=clean,
                    usage_limits=UsageLimits(
                        request_limit=5,
                        tool_calls_limit=0,
                    ),
                    model_settings=settings,
                )

        elapsed = time.perf_counter() - t0

        run_usage = result.usage()
        usage_dict = dataclasses.asdict(run_usage)

        logger.info(
            "Arm '%s' completed for '%s' in %.1f s  (tokens: in=%d out=%d, requests=%d)",
            arm.arm_id,
            deps.drug_name,
            elapsed,
            run_usage.input_tokens,
            run_usage.output_tokens,
            run_usage.requests,
        )

        return ArmResult(
            arm_id=arm.arm_id,
            drug_name=deps.drug_name,
            model_id=model_id,
            prediction=result.output,
            usage=usage_dict,
            wall_clock_seconds=elapsed,
        )

    except (UnexpectedModelBehavior, Exception) as exc:
        elapsed = time.perf_counter() - t0
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Arm '%s' failed for '%s' after %.1f s: %s",
            arm.arm_id,
            deps.drug_name,
            elapsed,
            error_msg,
        )
        # Walk the exception cause chain to surface the root validation error
        cause = exc.__cause__
        depth = 0
        while cause and depth < 5:
            logger.error(
                "  Caused by [%d]: %s: %s",
                depth,
                type(cause).__name__,
                cause,
            )
            cause = cause.__cause__
            depth += 1

        return ArmResult(
            arm_id=arm.arm_id,
            drug_name=deps.drug_name,
            model_id=model_id,
            wall_clock_seconds=elapsed,
            error=error_msg,
        )


# ------------------------------------------------------------------
# Multi-arm convenience runner
# ------------------------------------------------------------------


async def run_all_pipeline_arms(
    deps: EvidenceDeps,
    arms: Sequence[ArmConfig] | None = None,
    registry: dict[str, ModelSpec] | None = None,
    *,
    prompt: str | None = None,
    retries: int = 3,
) -> list[ArmResult]:
    """Run multiple pipeline arms **sequentially** for one drug.

    Arms are executed one at a time to respect provider rate limits
    (especially Groq).  Phase 8's ``ExperimentRunner`` will add
    configurable concurrency with semaphores.

    Parameters
    ----------
    deps:
        Pre-populated dependency container.
    arms:
        Arms to run.  Defaults to all 4 ``PIPELINE_ARMS``.
    registry:
        Model registry.
    prompt:
        Optional prompt override (same for all arms).
    retries:
        PydanticAI retry count.

    Returns
    -------
    list[ArmResult]
        One result per arm, in execution order.
    """
    from src.experiment.arms import PIPELINE_ARMS

    arm_list = list(arms) if arms is not None else list(PIPELINE_ARMS.values())
    results: list[ArmResult] = []

    logger.info(
        "Running %d pipeline arms for drug '%s'",
        len(arm_list),
        deps.drug_name,
    )

    for arm in arm_list:
        if arm.arm_type != ArmType.PIPELINE:
            logger.warning(
                "Skipping non-pipeline arm '%s' in run_all_pipeline_arms",
                arm.arm_id,
            )
            continue
        result = await run_pipeline_arm(
            arm=arm,
            deps=deps,
            registry=registry,
            prompt=prompt,
            retries=retries,
        )
        results.append(result)

    logger.info(
        "Completed %d / %d arms for '%s'  (errors: %d)",
        len(results),
        len(arm_list),
        deps.drug_name,
        sum(1 for r in results if r.error),
    )
    return results


# ------------------------------------------------------------------
# Web-search arm runner -- Tavily only (baseline comparison)
# ------------------------------------------------------------------


async def run_websearch_arm(
    arm: ArmConfig,
    drug_name: str,
    chembl_id: str | None = None,
    settings: Settings | None = None,
    registry: dict[str, ModelSpec] | None = None,
    *,
    prompt: str | None = None,
    retries: int = 3,
) -> ArmResult:
    """Run one websearch arm for a drug using a PydanticAI agent + Tavily.

    All frontier providers (OpenAI, Anthropic) go through the unified
    ``run_baseline()`` which creates a PydanticAI agent with Tavily web
    search restricted to biomedical domains.

    No ``EvidenceDeps`` are required -- websearch arms rely only on LLM
    knowledge and Tavily web search results.

    Parameters
    ----------
    arm:
        Arm configuration (must be ``ArmType.WEBSEARCH``).
    drug_name:
        Name of the drug to analyse.
    chembl_id:
        Optional ChEMBL identifier (included in the prompt).
    settings:
        Application settings (API keys, etc.).  Defaults to ``Settings()``.
    registry:
        Model registry (defaults to module-level ``MODEL_REGISTRY``).
    prompt:
        Custom user prompt override.
    retries:
        PydanticAI retry count.

    Returns
    -------
    ArmResult
        Always returned -- errors are captured in ``ArmResult.error``
        rather than raised.
    """
    reg = registry or MODEL_REGISTRY
    cfg = settings or Settings()
    user_prompt = build_baseline_prompt(drug_name, chembl_id, prompt)
    model_id = f"<unresolved:{arm.model_key}>"

    t0 = time.perf_counter()
    try:
        model_id = arm.resolve_model_id(reg)

        logger.info(
            "Starting websearch arm '%s' (model=%s, web_search=%s) for drug '%s'",
            arm.arm_id,
            model_id,
            arm.uses_web_search,
            drug_name,
        )

        domains = arm.allowed_search_domains or None

        br = await run_baseline(
            user_prompt,
            cfg,
            pydantic_ai_model_id=model_id,
            uses_web_search=arm.uses_web_search,
            allowed_domains=domains,
            retries=retries,
        )

        elapsed = time.perf_counter() - t0

        usage_dict = {
            "input_tokens": br.input_tokens,
            "output_tokens": br.output_tokens,
            "total_tokens": br.total_tokens,
            "web_search_requests": br.web_search_requests,
        }

        logger.info(
            "Websearch arm '%s' completed for '%s' in %.1f s  (tokens: in=%d out=%d)",
            arm.arm_id,
            drug_name,
            elapsed,
            br.input_tokens,
            br.output_tokens,
        )

        return ArmResult(
            arm_id=arm.arm_id,
            drug_name=drug_name,
            model_id=model_id,
            prediction=br.prediction,
            usage=usage_dict,
            wall_clock_seconds=elapsed,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Websearch arm '%s' failed for '%s' after %.1f s: %s",
            arm.arm_id,
            drug_name,
            elapsed,
            error_msg,
        )
        return ArmResult(
            arm_id=arm.arm_id,
            drug_name=drug_name,
            model_id=model_id,
            wall_clock_seconds=elapsed,
            error=error_msg,
        )
