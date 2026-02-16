"""Smoke tests for all three agent types -- no real LLM or API calls.

Covers:
1. Evidence agent (staged + live) -- construction, tool wiring, TestModel run
2. Baseline agent -- construction, Tavily tool registration, prompt builder
3. Groq ReAct instructor agent -- loop execution, tool dispatch, parse failures

All tests use PydanticAI TestModel or injected fakes so they run offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from src.agents.deps import EvidenceDeps
from src.agents.evidence_agent import (
    build_evidence_agent,
    build_evidence_agent_with_context,
)
from src.agents.baseline_agent import (
    BASELINE_SYSTEM_PROMPT,
    BaselineResult,
    _TavilyDeps,
    build_baseline_agent,
    build_baseline_prompt,
)
from src.agents.react_instructor_agent import (
    GroqReActAgent,
    ReActConfig,
    ReActRunResult,
    ReActStep,
    ReActTool,
    ReActToolCall,
    ReActBaselineResult,
)
from src.agents.tools.mongo_tools import (
    get_evidence_overview,
    lookup_staged_chembl,
    lookup_staged_dgidb,
    lookup_staged_opentargets,
    lookup_staged_pharmgkb,
    lookup_staged_pubchem,
    lookup_staged_pubmed,
    search_staged_evidence,
    filter_by_target,
    filter_by_disease,
)
from src.config.settings import Settings
from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)
from src.schemas.prediction import (
    DrugDiseasePrediction,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ======================================================================
# Shared helpers
# ======================================================================


def _settings() -> Settings:
    return Settings(_env_file=None, entrez_email="test@example.com")


def _prediction_args() -> dict[str, Any]:
    """Dict for TestModel.custom_output_args producing a DrugDiseasePrediction."""
    return {
        "drug_name": "metformin",
        "drug_chembl_id": "CHEMBL1431",
        "associations": [
            {
                "disease_name": "type 2 diabetes",
                "disease_id": "MESH:D003924",
                "predicted": True,
                "confidence": 0.92,
                "evidence_chains": [
                    {
                        "edges": [
                            {
                                "source_entity": "metformin",
                                "target_entity": "AMPK",
                                "relationship": "activates",
                                "evidence_snippet": "Metformin activates AMPK via LKB1 signalling",
                                "pmid": "11960013",
                            },
                            {
                                "source_entity": "AMPK",
                                "target_entity": "type 2 diabetes",
                                "relationship": "associated_with",
                                "evidence_snippet": "AMPK activation improves insulin sensitivity",
                                "pmid": "16356169",
                            },
                        ],
                        "summary": "Metformin activates AMPK, improving insulin sensitivity in T2D",
                        "confidence": 0.92,
                    }
                ],
            }
        ],
        "reasoning": "AMPK activation is the primary mechanism of metformin in diabetes.",
    }


def _mock_deps(
    drug_name: str = "metformin",
    chembl_id: str | None = "CHEMBL1431",
    evidence_store: Any = None,
) -> EvidenceDeps:
    settings = _settings()
    return EvidenceDeps(
        settings=settings,
        vector_store=MagicMock(),
        aggregator=MagicMock(),
        evidence_store=evidence_store,
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=4091,
    )


def _mock_ctx(deps: EvidenceDeps | None = None) -> RunContext[EvidenceDeps]:
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps or _mock_deps()
    return ctx


# ======================================================================
# 1. Evidence agent -- construction
# ======================================================================


class TestEvidenceAgentConstruction:
    """Verify staged and live agents register the correct tool sets."""

    def test_staged_agent_tools(self) -> None:
        model = TestModel(custom_output_args=_prediction_args())
        agent = build_evidence_agent(model, use_staged=True)
        tool_names = set(agent._function_toolset.tools.keys())
        # Staged tools
        assert "get_evidence_overview" in tool_names
        assert "search_staged_evidence" in tool_names
        assert "lookup_staged_dgidb" in tool_names
        assert "lookup_staged_opentargets" in tool_names
        assert "lookup_staged_pubchem" in tool_names
        assert "lookup_staged_chembl" in tool_names
        assert "lookup_staged_pharmgkb" in tool_names
        assert "lookup_staged_pubmed" in tool_names
        assert "filter_by_target" in tool_names
        assert "filter_by_disease" in tool_names
        # Shared validation tools
        assert "validate_pmids" in tool_names
        assert "validate_doi" in tool_names
        # Live-only tools should NOT be present
        assert "lookup_dgidb" not in tool_names
        assert "search_pubmed" not in tool_names

    def test_live_agent_tools(self) -> None:
        model = TestModel(custom_output_args=_prediction_args())
        agent = build_evidence_agent(model, use_staged=False)
        tool_names = set(agent._function_toolset.tools.keys())
        assert "search_evidence" in tool_names
        assert "search_pubmed" in tool_names
        assert "lookup_dgidb" in tool_names
        assert "lookup_opentargets" in tool_names
        assert "lookup_pubchem" in tool_names
        assert "lookup_chembl" in tool_names
        assert "lookup_pharmgkb" in tool_names
        assert "validate_pmids" in tool_names
        assert "validate_doi" in tool_names
        # Staged tools should NOT be present
        assert "get_evidence_overview" not in tool_names
        assert "lookup_staged_dgidb" not in tool_names

    def test_context_agent_has_system_prompt(self) -> None:
        model = TestModel(custom_output_args=_prediction_args())
        agent = build_evidence_agent_with_context(model, use_staged=True)
        assert len(agent._system_prompt_functions) >= 1

    def test_output_type_is_prediction(self) -> None:
        model = TestModel(custom_output_args=_prediction_args())
        agent = build_evidence_agent(model)
        # The agent should produce DrugDiseasePrediction
        assert agent._output_type is DrugDiseasePrediction


# ======================================================================
# 2. Evidence agent -- run with TestModel (no real LLM)
# ======================================================================


class TestEvidenceAgentRun:
    """Run evidence agent with TestModel, skip tool calls, verify output."""

    @pytest.mark.asyncio
    async def test_staged_agent_produces_prediction(self) -> None:
        model = TestModel(custom_output_args=_prediction_args(), call_tools=[])
        agent = build_evidence_agent(model, use_staged=True)
        deps = _mock_deps()

        result = await agent.run(
            "Analyse metformin (CHEMBL1431) and predict disease associations.",
            deps=deps,
        )

        pred = result.output
        assert isinstance(pred, DrugDiseasePrediction)
        assert pred.drug_name == "metformin"
        assert pred.drug_chembl_id == "CHEMBL1431"
        assert len(pred.associations) == 1
        assert pred.associations[0].disease_name == "type 2 diabetes"
        assert pred.associations[0].predicted is True
        assert pred.associations[0].confidence == 0.92

    @pytest.mark.asyncio
    async def test_prediction_has_mechanistic_chains(self) -> None:
        model = TestModel(custom_output_args=_prediction_args(), call_tools=[])
        agent = build_evidence_agent(model, use_staged=True)
        deps = _mock_deps()

        result = await agent.run("Analyse metformin.", deps=deps)
        chain = result.output.associations[0].evidence_chains[0]
        assert len(chain.edges) == 2
        assert chain.edges[0].relationship == EdgeType.ACTIVATES
        assert chain.edges[0].target_entity == "AMPK"
        assert chain.edges[1].relationship == EdgeType.ASSOCIATED_WITH

    @pytest.mark.asyncio
    async def test_prediction_schema_roundtrip(self) -> None:
        model = TestModel(custom_output_args=_prediction_args(), call_tools=[])
        agent = build_evidence_agent(model, use_staged=True)
        deps = _mock_deps()

        result = await agent.run("Analyse metformin.", deps=deps)
        raw = result.output.model_dump()
        validated = DrugDiseasePrediction.model_validate(raw)
        assert validated.drug_name == "metformin"
        assert validated.associations[0].confidence == 0.92

    @pytest.mark.asyncio
    async def test_live_agent_also_produces_prediction(self) -> None:
        model = TestModel(custom_output_args=_prediction_args(), call_tools=[])
        agent = build_evidence_agent(model, use_staged=False)
        deps = _mock_deps()

        result = await agent.run("Analyse metformin.", deps=deps)
        assert isinstance(result.output, DrugDiseasePrediction)


# ======================================================================
# 3. Staged MongoDB tools -- unit tests
# ======================================================================


class TestMongoToolsSmoke:
    """Verify mongo tools return expected shapes with mock EvidenceStore."""

    @pytest.mark.asyncio
    async def test_get_evidence_overview(self) -> None:
        store = AsyncMock()
        store.get_evidence_summary = AsyncMock(return_value={
            "total_docs": 42,
            "sources": {"opentargets": 10, "dgidb": 8, "pubchem": 5,
                        "pharmgkb": 4, "chembl": 6, "pubmed": 7, "reactome": 2},
            "distinct_targets": ["AMPK", "SLC22A1"],
            "distinct_diseases": ["type 2 diabetes"],
        })
        ctx = _mock_ctx(_mock_deps(evidence_store=store))

        result = await get_evidence_overview(ctx)
        assert result["total_docs"] == 42
        assert "reactome" in result["sources"]

    @pytest.mark.asyncio
    async def test_search_staged_evidence(self) -> None:
        store = AsyncMock()
        store.search_text = AsyncMock(return_value=[
            {
                "text": "Metformin activates AMPK",
                "source": "chembl",
                "drug_name": "metformin",
                "target_symbol": "AMPK",
                "disease_name": "",
                "evidence_type": "mechanism_of_action",
                "score": None,
                "citation": {"pmid": "11960013"},
            }
        ])
        ctx = _mock_ctx(_mock_deps(evidence_store=store))

        results = await search_staged_evidence(ctx, "AMPK activation", limit=5)
        assert len(results) == 1
        assert results[0]["source"] == "chembl"
        assert results[0]["target_symbol"] == "AMPK"

    @pytest.mark.asyncio
    async def test_lookup_staged_source_tools(self) -> None:
        """Each source lookup returns formatted docs from the store."""
        doc_template = {
            "text": "some evidence",
            "source": "dgidb",
            "drug_name": "metformin",
            "target_symbol": "SLC22A1",
            "disease_name": "",
            "evidence_type": "drug_gene_interaction",
            "score": 0.9,
            "citation": {"pmid": "99999"},
        }

        tools_and_sources = [
            (lookup_staged_dgidb, "dgidb"),
            (lookup_staged_opentargets, "opentargets"),
            (lookup_staged_pubchem, "pubchem"),
            (lookup_staged_chembl, "chembl"),
            (lookup_staged_pharmgkb, "pharmgkb"),
        ]

        for tool_fn, source_name in tools_and_sources:
            store = AsyncMock()
            store.find_by_source = AsyncMock(return_value=[
                {**doc_template, "source": source_name}
            ])
            ctx = _mock_ctx(_mock_deps(evidence_store=store))

            results = await tool_fn(ctx)
            assert len(results) == 1, f"{source_name} should return 1 doc"
            assert results[0]["source"] == source_name

    @pytest.mark.asyncio
    async def test_lookup_staged_pubmed(self) -> None:
        store = AsyncMock()
        store.find_by_source = AsyncMock(return_value=[
            {
                "text": "Metformin abstract text",
                "source": "pubmed",
                "drug_name": "metformin",
                "citation": {"pmid": "11960013", "title": "Metformin study", "year": 2001},
            }
        ])
        ctx = _mock_ctx(_mock_deps(evidence_store=store))

        results = await lookup_staged_pubmed(ctx)
        assert len(results) == 1
        assert results[0]["pmid"] == "11960013"
        assert results[0]["source"] == "pubmed"
        assert "abstract" in results[0]

    @pytest.mark.asyncio
    async def test_filter_by_target(self) -> None:
        store = AsyncMock()
        store.find_by_target = AsyncMock(return_value=[
            {"text": "AMPK evidence", "source": "chembl", "drug_name": "metformin",
             "target_symbol": "AMPK", "disease_name": "", "evidence_type": "moa",
             "score": None, "citation": {}}
        ])
        ctx = _mock_ctx(_mock_deps(evidence_store=store))

        results = await filter_by_target(ctx, "AMPK", limit=10)
        assert len(results) == 1
        store.find_by_target.assert_called_once_with("metformin", "AMPK", limit=10)

    @pytest.mark.asyncio
    async def test_filter_by_disease(self) -> None:
        store = AsyncMock()
        store.find_by_disease = AsyncMock(return_value=[
            {"text": "T2D evidence", "source": "opentargets", "drug_name": "metformin",
             "target_symbol": "", "disease_name": "type 2 diabetes", "evidence_type": "pathway",
             "score": 0.8, "citation": {"pmid": "99999"}}
        ])
        ctx = _mock_ctx(_mock_deps(evidence_store=store))

        results = await filter_by_disease(ctx, "type 2 diabetes", limit=10)
        assert len(results) == 1
        store.find_by_disease.assert_called_once_with("metformin", "type 2 diabetes", limit=10)

    @pytest.mark.asyncio
    async def test_tools_handle_no_evidence_store(self) -> None:
        """All mongo tools return error dict when evidence_store is None."""
        ctx = _mock_ctx(_mock_deps(evidence_store=None))

        for tool_fn in [
            get_evidence_overview,
            lambda c: search_staged_evidence(c, "test"),
            lookup_staged_dgidb,
            lookup_staged_opentargets,
            lookup_staged_pubchem,
            lookup_staged_chembl,
            lookup_staged_pharmgkb,
            lookup_staged_pubmed,
            lambda c: filter_by_target(c, "AMPK"),
            lambda c: filter_by_disease(c, "diabetes"),
        ]:
            result = await tool_fn(ctx)
            if isinstance(result, dict):
                assert "error" in result
            elif isinstance(result, list) and result:
                assert "error" in result[0]


# ======================================================================
# 4. Baseline agent -- construction and prompt
# ======================================================================


class TestBaselineAgentSmoke:
    """Baseline agent construction and prompt builder."""

    def test_build_with_web_search(self) -> None:
        agent = build_baseline_agent("test", uses_web_search=True)
        tool_names = set(agent._function_toolset.tools.keys())
        assert "web_search" in tool_names

    def test_build_without_web_search(self) -> None:
        agent = build_baseline_agent("test", uses_web_search=False)
        tool_names = set(agent._function_toolset.tools.keys())
        assert "web_search" not in tool_names
        assert len(tool_names) == 0

    def test_output_type_is_str(self) -> None:
        agent = build_baseline_agent("test")
        assert agent._output_type is str

    def test_system_prompt_present(self) -> None:
        assert "biomedical expert" in BASELINE_SYSTEM_PROMPT.lower()
        assert "evidence chains" in BASELINE_SYSTEM_PROMPT.lower()

    def test_prompt_builder_basic(self) -> None:
        prompt = build_baseline_prompt("metformin")
        assert "metformin" in prompt
        assert "ChEMBL" not in prompt

    def test_prompt_builder_with_chembl(self) -> None:
        prompt = build_baseline_prompt("metformin", chembl_id="CHEMBL1431")
        assert "metformin" in prompt
        assert "CHEMBL1431" in prompt

    def test_prompt_builder_override(self) -> None:
        prompt = build_baseline_prompt("metformin", override="Custom prompt text")
        assert prompt == "Custom prompt text"

    def test_baseline_result_dataclass(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="metformin",
            associations=[],
            reasoning="Test.",
        )
        result = BaselineResult(prediction=pred)
        assert result.raw_text == ""
        assert result.input_tokens == 0
        assert result.total_tokens == 0
        assert result.web_search_requests == 0

    @pytest.mark.asyncio
    async def test_baseline_agent_run_with_test_model(self) -> None:
        """Run baseline agent with TestModel (output_type=str)."""
        model = TestModel(custom_output_text="Metformin analysis: T2D via AMPK activation.")
        agent = build_baseline_agent(model, uses_web_search=False)
        deps = _TavilyDeps(tavily_api_key="tvly-test")

        result = await agent.run(
            "Analyse metformin and predict disease associations.",
            deps=deps,
        )
        assert isinstance(result.output, str)
        assert len(result.output) > 0


# ======================================================================
# 5. Groq ReAct instructor agent -- loop tests
# ======================================================================


class _TestOutput(BaseModel):
    answer: str
    confidence: float = 0.5


class _SearchArgs(BaseModel):
    query: str
    max_results: int = 5


class TestReActAgentSmoke:
    """ReAct loop execution with injected fakes -- no real Groq calls."""

    @pytest.mark.asyncio
    async def test_basic_loop_with_tool_call(self) -> None:
        """Agent calls a tool, gets result, then declares sufficient info."""
        settings = Settings(_env_file=None, groq_api_key="test-key")
        call_index = {"n": 0}

        async def text_gen(messages: list[dict[str, str]]) -> str:
            call_index["n"] += 1
            if call_index["n"] == 1:
                return "I should search for metformin mechanism."
            return "Metformin activates AMPK, treating T2D. I have enough."

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            if "should search" in text:
                return ReActStep(
                    thought="Search for evidence",
                    sufficient_information=False,
                    tool_calls=[
                        ReActToolCall(name="search", arguments={"query": "metformin AMPK"})
                    ],
                )
            return ReActStep(
                thought="Evidence collected",
                sufficient_information=True,
                tool_calls=[],
                synthesis="Metformin activates AMPK, treating T2D.",
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer=text, confidence=0.9)

        async def search_handler(query: str, max_results: int = 5) -> dict:
            return {"hits": [{"pmid": "11960013", "text": "AMPK activation"}]}

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            config=ReActConfig(max_iterations=5),
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result: ReActRunResult[_TestOutput] = await agent.run(
            query="Assess metformin disease associations",
            tools=[
                ReActTool(
                    name="search",
                    description="Search biomedical evidence",
                    handler=search_handler,
                    args_model=_SearchArgs,
                ),
            ],
            output_model=_TestOutput,
        )

        assert result.output.answer == "Metformin activates AMPK, treating T2D."
        assert result.output.confidence == 0.9
        assert len(result.iterations) == 2
        # First iteration should have executed the search tool
        assert len(result.iterations[0].executed_tools) == 1
        assert result.iterations[0].executed_tools[0]["ok"] is True

    @pytest.mark.asyncio
    async def test_loop_stops_at_max_iterations(self) -> None:
        """Agent that never declares sufficient info hits max_iterations."""
        settings = Settings(_env_file=None, groq_api_key="test-key")

        async def text_gen(messages: list[dict[str, str]]) -> str:
            return "Still searching..."

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            return ReActStep(
                thought="Need more",
                sufficient_information=False,
                tool_calls=[
                    ReActToolCall(name="search", arguments={"query": "metformin"})
                ],
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer="Ran out of iterations", confidence=0.1)

        async def search_handler(query: str, max_results: int = 5) -> dict:
            return {"hits": []}

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            config=ReActConfig(max_iterations=3),
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result = await agent.run(
            query="metformin",
            tools=[
                ReActTool(name="search", description="Search", handler=search_handler,
                          args_model=_SearchArgs),
            ],
            output_model=_TestOutput,
        )

        assert len(result.iterations) == 3
        assert result.output.answer == "Ran out of iterations"

    @pytest.mark.asyncio
    async def test_loop_handles_unknown_tool_gracefully(self) -> None:
        """Unknown tool calls produce error entries but don't crash."""
        settings = Settings(_env_file=None, groq_api_key="test-key")
        call_count = {"n": 0}

        async def text_gen(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Call nonexistent_tool"
            return "Done."

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            if "nonexistent_tool" in text:
                return ReActStep(
                    thought="Try bad tool",
                    sufficient_information=False,
                    tool_calls=[ReActToolCall(name="nonexistent_tool", arguments={})],
                )
            return ReActStep(
                thought="Done",
                sufficient_information=True,
                synthesis="No good evidence found.",
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer=text, confidence=0.2)

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result = await agent.run(
            query="metformin",
            tools=[],
            output_model=_TestOutput,
        )

        assert result.output.answer == "No good evidence found."
        assert result.iterations[0].executed_tools[0]["ok"] is False
        assert "Unknown tool name" in result.iterations[0].executed_tools[0]["error"]

    @pytest.mark.asyncio
    async def test_loop_recovers_from_parse_failure(self) -> None:
        """A parse failure should be retried, not crash the loop."""
        settings = Settings(_env_file=None, groq_api_key="test-key")
        call_count = {"n": 0}

        async def text_gen(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Garbled output that won't parse"
            return "Proper response with enough info."

        parse_count = {"n": 0}

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            parse_count["n"] += 1
            if parse_count["n"] == 1:
                raise ValueError("Simulated parse failure")
            return ReActStep(
                thought="Recovered",
                sufficient_information=True,
                synthesis="Metformin treats T2D.",
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer=text, confidence=0.7)

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            config=ReActConfig(max_iterations=5, max_parse_failures=3),
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result = await agent.run(
            query="metformin",
            tools=[],
            output_model=_TestOutput,
        )

        assert result.output.answer == "Metformin treats T2D."
        # First iteration had parse error, second succeeded
        assert result.iterations[0].parse_error is not None
        assert result.iterations[1].parsed_step is not None

    @pytest.mark.asyncio
    async def test_no_tools_immediate_synthesis(self) -> None:
        """Agent with no tool calls and immediate sufficient_information."""
        settings = Settings(_env_file=None, groq_api_key="test-key")

        async def text_gen(messages: list[dict[str, str]]) -> str:
            return "I already know metformin treats diabetes via AMPK."

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            return ReActStep(
                thought="Already know this",
                sufficient_information=True,
                synthesis="Metformin: T2D via AMPK.",
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer=text, confidence=0.95)

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result = await agent.run(
            query="metformin",
            tools=[],
            output_model=_TestOutput,
        )

        assert len(result.iterations) == 1
        assert result.output.confidence == 0.95

    @pytest.mark.asyncio
    async def test_multiple_tools_per_step(self) -> None:
        """Agent requesting multiple tool calls in a single step."""
        settings = Settings(_env_file=None, groq_api_key="test-key")
        call_count = {"n": 0}

        async def text_gen(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Search for targets and diseases simultaneously."
            return "Found AMPK and T2D evidence."

        async def step_parse(text: str, tools: list[str]) -> ReActStep:
            if "simultaneously" in text:
                return ReActStep(
                    thought="Parallel lookups",
                    sufficient_information=False,
                    tool_calls=[
                        ReActToolCall(name="search", arguments={"query": "metformin targets"}),
                        ReActToolCall(name="search", arguments={"query": "metformin diseases"}),
                    ],
                )
            return ReActStep(
                thought="Got it",
                sufficient_information=True,
                synthesis="AMPK + T2D confirmed.",
            )

        async def final_parse(text: str, model: type[_TestOutput]) -> _TestOutput:
            return model(answer=text, confidence=0.88)

        calls_made: list[str] = []

        async def search_handler(query: str, max_results: int = 5) -> dict:
            calls_made.append(query)
            return {"hits": [{"text": f"Result for: {query}"}]}

        agent = GroqReActAgent(
            settings=settings,
            model_id="groq:test-model",
            config=ReActConfig(max_iterations=4, max_tool_calls_per_step=4),
            text_generator=text_gen,
            step_parser=step_parse,
            final_parser=final_parse,
        )

        result = await agent.run(
            query="metformin",
            tools=[ReActTool(name="search", description="Search", handler=search_handler,
                             args_model=_SearchArgs)],
            output_model=_TestOutput,
        )

        assert len(calls_made) == 2
        assert "metformin targets" in calls_made
        assert "metformin diseases" in calls_made
        assert result.iterations[0].executed_tools[0]["ok"] is True


# ======================================================================
# 6. ReAct data structures
# ======================================================================


class TestReActDataStructures:
    """Verify ReAct schema models validate correctly."""

    def test_react_step_minimal(self) -> None:
        step = ReActStep(thought="test", sufficient_information=True)
        assert step.tool_calls == []
        assert step.synthesis == ""

    def test_react_tool_call(self) -> None:
        tc = ReActToolCall(name="search", arguments={"query": "aspirin"})
        assert tc.name == "search"
        assert tc.arguments == {"query": "aspirin"}

    def test_react_config_defaults(self) -> None:
        cfg = ReActConfig()
        assert cfg.max_iterations == 8
        assert cfg.max_tool_calls_per_step == 4
        assert cfg.max_parse_failures == 2

    def test_react_tool_manifest(self) -> None:
        tool = ReActTool(
            name="search",
            description="Search evidence",
            handler=lambda: None,
            args_model=_SearchArgs,
        )
        manifest = tool.manifest()
        assert manifest["name"] == "search"
        assert "arguments_schema" in manifest
        assert "query" in manifest["arguments_schema"].get("properties", {})

    def test_react_baseline_result(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="metformin", associations=[], reasoning="Test."
        )
        r = ReActBaselineResult(
            prediction=pred,
            raw_text="some text",
            raw_model_id="groq:test",
            iteration_count=3,
        )
        assert r.iteration_count == 3
        assert r.prediction.drug_name == "metformin"
