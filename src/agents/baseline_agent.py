"""Frontier baseline runners -- PydanticAI agents with optional Tavily search.

Baseline arms give frontier LLMs their **best-case** scenario: the same
``DrugDiseasePrediction`` output schema but with web search restricted to
biomedical domains instead of our retrieval pipeline.

All providers (OpenAI, Anthropic) are accessed through a **single unified
code path** using PydanticAI ``Agent(output_type=str)``.  When web search
is enabled, a Tavily ``@agent.tool`` is registered on the agent.

The agent replies in **free-form text** -- no forced structured output.
A second pass with ``instructor`` then extracts ``DrugDiseasePrediction``
from that text using the **same LLM** that produced it.

4 baseline arms:

* **OpenAI GPT-5.2 no search** -- PydanticAI ``"openai:gpt-5.2-..."``
* **OpenAI GPT-5.2 + Tavily** -- PydanticAI ``"openai:gpt-5.2-..."`` + Tavily tool
* **Anthropic Claude Opus 4.6 no search** -- PydanticAI ``"anthropic:claude-opus-4-6"``
* **Anthropic Claude Opus 4.6 + Tavily** -- PydanticAI ``"anthropic:claude-opus-4-6"`` + Tavily tool

All runners return a ``BaselineResult`` dataclass with the parsed
``DrugDiseasePrediction``, raw free-form text, token usage, and
web-search request counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import instructor
from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from src.config.settings import Settings
from src.schemas.prediction import DrugDiseasePrediction

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------
# Shared system prompt (used by all three providers)
# ------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """\
You are a biomedical expert.  Given a drug name, identify diseases it
may be therapeutically relevant for and construct mechanistic evidence
chains supporting each prediction.

EVIDENCE CHAINS
---------------
For each disease association, build one or more chains of the form:

    drug -> [binds/inhibits/activates] -> target_gene
          -> [participates_in/modulates] -> pathway
          -> [associated_with] -> disease

Each edge must include:
  - source_entity and target_entity (use specific gene symbols, e.g. PTGS2)
  - relationship (one of: binds, inhibits, activates, upregulates,
    downregulates, modulates, transports, metabolizes, participates_in,
    associated_with)
  - evidence_snippet (quote or close paraphrase from a source)
  - pmid (PubMed ID if available; null otherwise)

INSTRUCTIONS
------------
- Use web search (if available) to find supporting evidence from biomedical
  literature.  Prefer results from PubMed, ClinicalTrials.gov, OpenTargets,
  DGIdb, PharmGKB, EBI, and UniProt.
- For each prediction, cite specific papers (PMID, DOI, or URL).
- Include ALL diseases with non-trivial evidence, even if confidence is low.
- Do NOT fabricate PMIDs or evidence snippets.
- The reasoning field should summarise your analysis strategy (2-4 sentences).
- Do not use emojis anywhere in the output.
"""


# ------------------------------------------------------------------
# Result container (provider-agnostic)
# ------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Outcome of a single native-SDK baseline call."""

    prediction: DrugDiseasePrediction
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    web_search_requests: int = 0
    raw_model_id: str = ""
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# User prompt builder
# ------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Analyse the drug '{drug_name}'"
    "{chembl_clause}"
    " and predict disease associations with mechanistic evidence chains."
    " Cite specific papers (PMID or DOI) where possible."
)


def build_baseline_prompt(
    drug_name: str,
    chembl_id: str | None = None,
    override: str | None = None,
) -> str:
    """Construct the user prompt for a baseline run."""
    if override:
        return override
    chembl_clause = f" (ChEMBL ID: {chembl_id})" if chembl_id else ""
    return _PROMPT_TEMPLATE.format(drug_name=drug_name, chembl_clause=chembl_clause)


# ------------------------------------------------------------------
# Instructor-based extraction (second pass)
# ------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are a data-extraction assistant.  The user will provide a free-form "
    "biomedical analysis of a drug.  Extract the structured information exactly "
    "as described by the response schema.  Preserve all diseases, confidence "
    "scores, evidence chains, PMIDs, and reasoning from the source text.  Do "
    "not fabricate data that is not present in the source text."
)


async def extract_prediction(
    raw_text: str,
    *,
    provider_model: str,
    api_key: str | None = None,
    max_tokens: int = 16384,
) -> DrugDiseasePrediction:
    """Extract ``DrugDiseasePrediction`` from free-form LLM text using Instructor.

    Uses ``instructor.from_provider()`` with the **same model** that produced
    the free-form text so the extraction quality matches the original LLM.

    Args:
        raw_text: Free-form analysis text produced by a baseline LLM.
        provider_model: Instructor provider string, e.g.
            ``"openai/gpt-5.2-2025-12-11"``, ``"anthropic/claude-opus-4-6"``,
            ``"groq/qwen/qwen3-32b"``.
        api_key: Provider API key.  Falls back to env vars when ``None``.
        max_tokens: Maximum tokens for the extraction call.

    Returns:
        Validated ``DrugDiseasePrediction`` instance.
    """
    client = instructor.from_provider(
        provider_model,
        async_client=True,
        api_key=api_key,
    )

    # OpenAI newer models reject ``max_tokens`` and require
    # ``max_completion_tokens`` instead.  Anthropic and Groq still
    # use ``max_tokens``.
    provider_prefix = provider_model.split("/", 1)[0].lower()
    if provider_prefix == "openai":
        token_kwargs: dict = {"max_completion_tokens": max_tokens}
    else:
        token_kwargs = {"max_tokens": max_tokens}

    prediction = await client.create(
        response_model=DrugDiseasePrediction,
        **token_kwargs,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": raw_text},
        ],
    )

    return prediction


# ======================================================================
# PydanticAI baseline -- unified for all providers
# ======================================================================

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass
class _TavilyDeps:
    """Lightweight deps for baseline agents with Tavily web search."""

    tavily_api_key: str
    allowed_domains: list[str] | None = None


def build_baseline_agent(
    model_id: str,
    *,
    uses_web_search: bool = True,
    retries: int = 2,
) -> Agent[_TavilyDeps, str]:
    """Build a PydanticAI agent for any provider with optional Tavily search.

    The agent uses ``output_type=str`` so the LLM replies in free-form
    text.  Structured extraction happens in a second pass via Instructor.

    Exposed as a public function so tests can inspect the agent without
    running it against a live API.

    Args:
        model_id: PydanticAI model string, e.g. ``"openai:gpt-5.2-2025-12-11"``
            or ``"anthropic:claude-opus-4-6"``.
        uses_web_search: Whether to register the Tavily web search tool.
        retries: Maximum result-parse retries.
    """
    agent: Agent[_TavilyDeps, str] = Agent(
        model=model_id,
        output_type=str,
        deps_type=_TavilyDeps,
        retries=retries,
        instructions=BASELINE_SYSTEM_PROMPT,
    )

    if uses_web_search:

        @agent.tool
        async def web_search(
            ctx: RunContext[_TavilyDeps],
            query: str,
        ) -> list[dict]:
            """Search the web for biomedical evidence using Tavily.

            Args:
                query: The search query (e.g. "aspirin COX-2 inhibition cancer").

            Returns:
                A list of search result dicts with title, url, and content.
            """
            import httpx

            logger.info("[TOOL] web_search (Tavily) | query=%r", query)
            payload: dict = {
                "query": query,
                "search_depth": "advanced",
                "max_results": 10,
                "include_answer": "advanced",
                "include_raw_content": False,
                "include_domains": ctx.deps.allowed_domains or [],
                "exclude_domains": [],
            }
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    _TAVILY_SEARCH_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {ctx.deps.tavily_api_key}",
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            logger.info("[TOOL] web_search (Tavily) -> %d results", len(results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:2000],
                }
                for r in results
            ]

    return agent


def _resolve_instructor_provider(
    pydantic_ai_id: str,
    settings: Settings,
) -> tuple[str, str]:
    """Derive Instructor provider string and API key from a PydanticAI model id.

    Args:
        pydantic_ai_id: e.g. ``"openai:gpt-5.2-2025-12-11"`` or
            ``"anthropic:claude-opus-4-6"``.
        settings: Application settings containing API keys.

    Returns:
        Tuple of (instructor_provider_model, api_key).
        e.g. ``("openai/gpt-5.2-2025-12-11", "sk-...")``.
    """
    prefix, model_name = pydantic_ai_id.split(":", 1)
    provider = prefix.lower()

    if provider == "openai":
        return f"openai/{model_name}", settings.openai_api_key
    elif provider == "anthropic":
        return f"anthropic/{model_name}", settings.anthropic_api_key
    elif provider == "groq":
        return f"groq/{model_name}", settings.groq_api_key
    else:
        msg = f"Unsupported provider prefix: {provider!r}"
        raise ValueError(msg)


async def run_baseline(
    prompt: str,
    settings: Settings,
    *,
    pydantic_ai_model_id: str,
    uses_web_search: bool = True,
    allowed_domains: list[str] | None = None,
    retries: int = 2,
) -> BaselineResult:
    """Run a baseline via PydanticAI with any supported provider.

    The agent replies in **free-form text** (``output_type=str``).
    A second pass with Instructor extracts ``DrugDiseasePrediction`` from
    that text using the same model.

    When ``uses_web_search`` is True, a Tavily ``@agent.tool`` is
    registered so the LLM can issue web searches.

    Args:
        prompt: User prompt (from ``build_baseline_prompt``).
        settings: Application settings (API keys, etc.).
        pydantic_ai_model_id: PydanticAI model string, e.g.
            ``"openai:gpt-5.2-2025-12-11"`` or ``"anthropic:claude-opus-4-6"``.
        uses_web_search: Whether to enable Tavily web search.
        allowed_domains: Domain allow-list for Tavily search.
        retries: PydanticAI retry count.

    Returns:
        ``BaselineResult`` with prediction, raw text, and usage stats.
    """
    agent = build_baseline_agent(
        model_id=pydantic_ai_model_id,
        uses_web_search=uses_web_search,
        retries=retries,
    )

    deps = _TavilyDeps(
        tavily_api_key=settings.tavily_api_key,
        allowed_domains=allowed_domains,
    )

    logger.info(
        "Baseline: model=%s, web_search=%s, domains=%s",
        pydantic_ai_model_id,
        uses_web_search,
        allowed_domains,
    )

    # --- Pass 1: free-form text response ---
    # Cap tool calls to prevent excessive web searches
    with capture_run_messages() as messages:
        try:
            result = await agent.run(
                prompt,
                deps=deps,
                usage_limits=UsageLimits(request_limit=50, tool_calls_limit=5),
            )
        except UsageLimitExceeded:
            # Tool-call limit hit -- force a final text answer from
            # whatever web search results were gathered so far.
            logger.warning(
                "Baseline '%s' hit tool_calls_limit -- "
                "forcing final answer from gathered evidence",
                pydantic_ai_model_id,
            )
            # Strip trailing ModelResponse with unprocessed ToolCallParts
            # so PydanticAI accepts the new user prompt.
            clean = _strip_pending_tool_calls(list(messages))
            result = await agent.run(
                "You have reached the tool call limit. "
                "Using ONLY the information you have already gathered above, "
                "produce your final comprehensive analysis now.",
                deps=deps,
                message_history=clean,
                usage_limits=UsageLimits(request_limit=5, tool_calls_limit=0),
            )
    run_usage = result.usage()

    raw_text = result.output  # str -- free-form LLM text

    # --- Pass 2: Instructor extraction ---
    instructor_model, api_key = _resolve_instructor_provider(
        pydantic_ai_model_id, settings,
    )
    prediction = await extract_prediction(
        raw_text,
        provider_model=instructor_model,
        api_key=api_key,
    )

    return BaselineResult(
        prediction=prediction,
        raw_text=raw_text,
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
        total_tokens=run_usage.input_tokens + run_usage.output_tokens,
        raw_model_id=pydantic_ai_model_id,
    )
