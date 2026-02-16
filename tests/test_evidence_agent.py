"""Tests for the evidence extraction agent and its tools.

Uses PydanticAI's ``TestModel`` to exercise the agent without real LLM calls.
Tool functions are tested independently with mock dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from src.agents.deps import EvidenceDeps
from src.agents.evidence_agent import (
    build_evidence_agent,
    build_evidence_agent_with_context,
)
from src.agents.tools.db_tools import (
    lookup_chembl,
    lookup_dgidb,
    lookup_opentargets,
    lookup_pharmgkb,
    lookup_pubchem,
)
from src.agents.tools.search_tools import search_evidence, search_pubmed
from src.agents.tools.validation_tools import validate_doi, validate_pmids
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
from src.vector.store import SearchResult


# ======================================================================
# Fixtures
# ======================================================================

def _make_settings() -> Settings:
    """Create a minimal Settings for tests (no API calls)."""
    return Settings(
        _env_file=None,
        entrez_email="test@example.com",
    )


def _make_evidence_doc(
    text: str = "aspirin inhibits COX-2",
    source: EvidenceSource = EvidenceSource.CHEMBL,
    drug_name: str = "aspirin",
    **kwargs: Any,
) -> EvidenceDocument:
    return EvidenceDocument(
        text=text,
        source=source,
        evidence_type=kwargs.pop("evidence_type", EvidenceType.MECHANISM_OF_ACTION),
        drug_name=drug_name,
        citation=Citation(pmid=kwargs.pop("pmid", "12345678")),
        **kwargs,
    )


def _make_search_result(
    text: str = "aspirin inhibits COX-2",
    score: float = 0.85,
    **payload_extra: Any,
) -> SearchResult:
    payload = {
        "text": text,
        "source": "chembl",
        "drug_name": "aspirin",
        "target_symbol": "PTGS2",
        "disease_name": "",
        "pmid": "12345678",
        "doi": "",
        **payload_extra,
    }
    return SearchResult(chunk_id="abc123", score=score, text=text, payload=payload)


def _make_prediction_args() -> dict:
    """Return dict that TestModel can use as custom_output_args for DrugDiseasePrediction."""
    return {
        "drug_name": "aspirin",
        "drug_chembl_id": "CHEMBL25",
        "associations": [
            {
                "disease_name": "colorectal cancer",
                "disease_id": "EFO_0000365",
                "predicted": True,
                "confidence": 0.82,
                "evidence_chains": [
                    {
                        "edges": [
                            {
                                "source_entity": "aspirin",
                                "target_entity": "PTGS2",
                                "relationship": "inhibits",
                                "evidence_snippet": "Aspirin irreversibly inhibits COX-2",
                                "pmid": "12345678",
                            },
                            {
                                "source_entity": "PTGS2",
                                "target_entity": "colorectal cancer",
                                "relationship": "associated_with",
                                "evidence_snippet": "COX-2 overexpression promotes colorectal tumorigenesis",
                                "pmid": "23456789",
                            },
                        ],
                        "summary": "Aspirin inhibits COX-2, reducing colorectal cancer risk",
                        "confidence": 0.82,
                    }
                ],
            }
        ],
        "reasoning": "Queried vector store and databases. Found strong COX-2 inhibition evidence.",
    }


def _mock_deps(
    drug_name: str = "aspirin",
    chembl_id: str | None = "CHEMBL25",
    pubchem_cid: int | None = 2244,
) -> EvidenceDeps:
    """Create EvidenceDeps with mock objects for tool tests."""
    settings = _make_settings()
    vector_store = MagicMock()
    aggregator = MagicMock()

    # Mock all data clients on the aggregator
    aggregator._dgidb = AsyncMock()
    aggregator._opentargets = AsyncMock()
    aggregator._pubchem = AsyncMock()
    aggregator._chembl = AsyncMock()
    aggregator._pharmgkb = AsyncMock()
    aggregator._pubmed = AsyncMock()

    return EvidenceDeps(
        settings=settings,
        vector_store=vector_store,
        aggregator=aggregator,
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=pubchem_cid,
    )


def _mock_run_context(
    deps: EvidenceDeps | None = None,
) -> RunContext[EvidenceDeps]:
    """Create a minimal RunContext for tool tests."""
    if deps is None:
        deps = _mock_deps()
    # RunContext is a dataclass -- construct it directly
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


# ======================================================================
# Agent construction tests
# ======================================================================

class TestAgentConstruction:
    """Verify agent builds correctly with all tools registered."""

    def test_build_with_test_model(self) -> None:
        model = TestModel(custom_output_args=_make_prediction_args())
        agent = build_evidence_agent(model)
        assert agent is not None
        tool_names = set(agent._function_toolset.tools.keys())
        expected = {
            "search_evidence",
            "search_pubmed",
            "lookup_dgidb",
            "lookup_opentargets",
            "lookup_pubchem",
            "lookup_chembl",
            "lookup_pharmgkb",
            "validate_pmids",
            "validate_doi",
        }
        assert tool_names == expected

    def test_build_with_context_has_system_prompt(self) -> None:
        model = TestModel(custom_output_args=_make_prediction_args())
        agent = build_evidence_agent_with_context(model)
        # Should have at least one system prompt function (the dynamic one)
        assert len(agent._system_prompt_functions) >= 1

    def test_tool_count(self) -> None:
        model = TestModel(custom_output_args=_make_prediction_args())
        agent = build_evidence_agent(model)
        assert len(agent._function_toolset.tools) == 9


# ======================================================================
# Agent run tests (with TestModel, no real LLM)
# ======================================================================

class TestAgentRun:
    """Run the agent with TestModel and verify structured output."""

    @pytest.mark.asyncio
    async def test_run_produces_valid_prediction(self) -> None:
        """Agent run with TestModel should produce a valid DrugDiseasePrediction."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],  # skip tool calls, just produce output
        )
        agent = build_evidence_agent(model)
        deps = _mock_deps()

        result = await agent.run(
            "Analyse aspirin (CHEMBL25) and predict disease associations.",
            deps=deps,
        )

        prediction = result.output
        assert isinstance(prediction, DrugDiseasePrediction)
        assert prediction.drug_name == "aspirin"
        assert prediction.drug_chembl_id == "CHEMBL25"
        assert len(prediction.associations) == 1
        assert prediction.associations[0].disease_name == "colorectal cancer"
        assert prediction.associations[0].predicted is True

    @pytest.mark.asyncio
    async def test_prediction_has_evidence_chains(self) -> None:
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        agent = build_evidence_agent(model)
        deps = _mock_deps()

        result = await agent.run("Analyse aspirin.", deps=deps)
        assoc = result.output.associations[0]
        assert len(assoc.evidence_chains) == 1
        chain = assoc.evidence_chains[0]
        assert len(chain.edges) == 2
        assert chain.edges[0].relationship.value == "inhibits"
        assert chain.edges[1].relationship.value == "associated_with"

    @pytest.mark.asyncio
    async def test_prediction_schema_validation(self) -> None:
        """Verify the output can round-trip through Pydantic validation."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        agent = build_evidence_agent(model)
        deps = _mock_deps()

        result = await agent.run("Analyse aspirin.", deps=deps)
        raw = result.output.model_dump()
        validated = DrugDiseasePrediction.model_validate(raw)
        assert validated.drug_name == "aspirin"
        assert len(validated.associations) == 1

    @pytest.mark.asyncio
    async def test_run_with_context_agent(self) -> None:
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        agent = build_evidence_agent_with_context(model)
        deps = _mock_deps()

        result = await agent.run("Analyse aspirin.", deps=deps)
        assert isinstance(result.output, DrugDiseasePrediction)


# ======================================================================
# Search tool tests
# ======================================================================

class TestSearchTools:
    """Test search tools with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_search_evidence_returns_formatted_results(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.vector_store.hybrid_search = MagicMock(
            return_value=[
                _make_search_result("aspirin inhibits COX-2", 0.9),
                _make_search_result("COX-2 in colorectal cancer", 0.7),
            ]
        )

        with patch("src.agents.tools.search_tools.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = ctx.deps.vector_store.hybrid_search(
                "aspirin mechanism", limit=10, drug_filter="aspirin"
            )
            results = await search_evidence(ctx, "aspirin mechanism", limit=10)

        assert len(results) == 2
        assert results[0]["source"] == "chembl"
        assert results[0]["target_symbol"] == "PTGS2"
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_search_pubmed_returns_formatted_results(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._pubmed.search_and_fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    text="Abstract about aspirin and cancer",
                    source=EvidenceSource.PUBMED,
                    evidence_type=EvidenceType.LITERATURE,
                ),
            ]
        )

        results = await search_pubmed(ctx, "cancer mechanism", max_results=5)

        assert len(results) == 1
        assert results[0]["source"] == "pubmed"
        assert "abstract" in results[0]
        assert results[0]["pmid"] == "12345678"


# ======================================================================
# DB tool tests
# ======================================================================

class TestDBTools:
    """Test DB lookup tools with mocked clients."""

    @pytest.mark.asyncio
    async def test_lookup_dgidb(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._dgidb.fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    source=EvidenceSource.DGIDB,
                    evidence_type=EvidenceType.DRUG_GENE_INTERACTION,
                    target_symbol="PTGS2",
                    metadata={"interaction_type": "inhibitor"},
                ),
            ]
        )

        results = await lookup_dgidb(ctx)
        assert len(results) == 1
        assert results[0]["target_symbol"] == "PTGS2"
        assert results[0]["interaction_type"] == "inhibitor"

    @pytest.mark.asyncio
    async def test_lookup_opentargets(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._opentargets.fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    source=EvidenceSource.OPENTARGETS,
                    evidence_type=EvidenceType.PATHWAY,
                    target_symbol="PTGS2",
                    disease_name="colorectal cancer",
                    disease_id="EFO_0000365",
                    score=0.75,
                ),
            ]
        )

        results = await lookup_opentargets(ctx)
        assert len(results) == 1
        assert results[0]["disease_name"] == "colorectal cancer"
        assert results[0]["score"] == 0.75

    @pytest.mark.asyncio
    async def test_lookup_opentargets_no_chembl_id(self) -> None:
        deps = _mock_deps(chembl_id=None)
        ctx = _mock_run_context(deps)

        results = await lookup_opentargets(ctx)
        assert len(results) == 1
        assert "error" in results[0]

    @pytest.mark.asyncio
    async def test_lookup_pubchem(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._pubchem.fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    source=EvidenceSource.PUBCHEM,
                    evidence_type=EvidenceType.PHARMACOLOGICAL_ACTION,
                ),
            ]
        )

        results = await lookup_pubchem(ctx)
        assert len(results) == 1
        assert results[0]["evidence_type"] == "pharmacological_action"

    @pytest.mark.asyncio
    async def test_lookup_chembl(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._chembl.fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    source=EvidenceSource.CHEMBL,
                    evidence_type=EvidenceType.MECHANISM_OF_ACTION,
                    target_symbol="PTGS2",
                ),
            ]
        )

        results = await lookup_chembl(ctx)
        assert len(results) == 1
        assert results[0]["target_symbol"] == "PTGS2"

    @pytest.mark.asyncio
    async def test_lookup_chembl_no_chembl_id(self) -> None:
        deps = _mock_deps(chembl_id=None)
        ctx = _mock_run_context(deps)

        results = await lookup_chembl(ctx)
        assert len(results) == 1
        assert "error" in results[0]

    @pytest.mark.asyncio
    async def test_lookup_pharmgkb(self) -> None:
        ctx = _mock_run_context()
        ctx.deps.aggregator._pharmgkb.fetch = AsyncMock(
            return_value=[
                _make_evidence_doc(
                    source=EvidenceSource.PHARMGKB,
                    evidence_type=EvidenceType.CLINICAL_ANNOTATION,
                    disease_name="pain",
                ),
            ]
        )

        results = await lookup_pharmgkb(ctx)
        assert len(results) == 1
        assert results[0]["disease_name"] == "pain"
        assert results[0]["evidence_type"] == "clinical_annotation"


# ======================================================================
# Validation tool tests
# ======================================================================

class TestValidationTools:
    """Test citation validation tools."""

    @pytest.mark.asyncio
    async def test_validate_pmids_valid(self) -> None:
        ctx = _mock_run_context()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "title": "Test Article",
                    "pubdate": "2023 Jan",
                },
            }
        }

        with patch("src.agents.tools.validation_tools.cached_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            results = await validate_pmids(ctx, ["12345678"])

        assert len(results) == 1
        assert results[0]["valid"] is True
        assert results[0]["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_validate_pmids_invalid(self) -> None:
        ctx = _mock_run_context()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "uids": ["99999999"],
                "99999999": {"error": "cannot get document summary"},
            }
        }

        with patch("src.agents.tools.validation_tools.cached_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            results = await validate_pmids(ctx, ["99999999"])

        assert len(results) == 1
        assert results[0]["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_pmids_empty_list(self) -> None:
        ctx = _mock_run_context()
        results = await validate_pmids(ctx, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_validate_doi_valid(self) -> None:
        ctx = _mock_run_context()

        with patch("src.agents.tools.validation_tools.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.url = "https://example.com/article"
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            result = await validate_doi(ctx, "10.1038/s41586-020-2649-2")

        assert result["valid"] is True
        assert result["doi"] == "10.1038/s41586-020-2649-2"

    @pytest.mark.asyncio
    async def test_validate_doi_not_found(self) -> None:
        ctx = _mock_run_context()

        with patch("src.agents.tools.validation_tools.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            result = await validate_doi(ctx, "10.9999/fake")

        assert result["valid"] is False


# ======================================================================
# Schema round-trip tests
# ======================================================================

class TestSchemaRoundTrip:
    """Verify DrugDiseasePrediction serialization fidelity."""

    def test_prediction_json_round_trip(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="aspirin",
            drug_chembl_id="CHEMBL25",
            associations=[
                ScoredAssociation(
                    disease_name="colorectal cancer",
                    disease_id="EFO_0000365",
                    predicted=True,
                    confidence=0.85,
                    evidence_chains=[
                        EvidenceChain(
                            edges=[
                                MechanisticEdge(
                                    source_entity="aspirin",
                                    target_entity="PTGS2",
                                    relationship=EdgeType.INHIBITS,
                                    evidence_snippet="Aspirin inhibits COX-2",
                                    pmid="12345678",
                                ),
                            ],
                            summary="Aspirin inhibits COX-2 linked to colorectal cancer",
                            confidence=0.85,
                        )
                    ],
                )
            ],
            reasoning="Evidence from ChEMBL and PubMed.",
        )
        raw = pred.model_dump_json()
        restored = DrugDiseasePrediction.model_validate_json(raw)
        assert restored.drug_name == "aspirin"
        assert restored.associations[0].evidence_chains[0].edges[0].relationship == EdgeType.INHIBITS

    def test_empty_associations_valid(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="unknown_drug",
            associations=[],
            reasoning="No evidence found.",
        )
        assert len(pred.associations) == 0

    def test_multiple_chains_per_association(self) -> None:
        chain1 = EvidenceChain(
            edges=[
                MechanisticEdge(
                    source_entity="aspirin",
                    target_entity="PTGS2",
                    relationship=EdgeType.INHIBITS,
                    evidence_snippet="COX-2 inhibition",
                ),
            ],
            summary="Direct COX-2 path",
            confidence=0.9,
        )
        chain2 = EvidenceChain(
            edges=[
                MechanisticEdge(
                    source_entity="aspirin",
                    target_entity="NF-kB",
                    relationship=EdgeType.INHIBITS,
                    evidence_snippet="NF-kB pathway inhibition",
                ),
                MechanisticEdge(
                    source_entity="NF-kB",
                    target_entity="colorectal cancer",
                    relationship=EdgeType.ASSOCIATED_WITH,
                    evidence_snippet="NF-kB promotes tumor growth",
                ),
            ],
            summary="NF-kB pathway",
            confidence=0.65,
        )
        assoc = ScoredAssociation(
            disease_name="colorectal cancer",
            predicted=True,
            confidence=0.8,
            evidence_chains=[chain1, chain2],
        )
        assert len(assoc.evidence_chains) == 2
        assert assoc.evidence_chains[1].edges[0].target_entity == "NF-kB"
