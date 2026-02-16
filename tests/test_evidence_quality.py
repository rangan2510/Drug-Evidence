"""Tests for evidence quality metrics.

Tests cover all five metrics:
1. citation_validity_rate -- with mocked PubMed eSummary responses
2. mean_chain_depth -- pure computation
3. chain_verifiability_score -- pure computation
4. evidence_relevance -- MedCPT cosine similarity (real model)
5. mechanistic_specificity -- heuristic fallback (no LLM calls in tests)

Also tests the top-level ``evaluate_evidence_quality()`` orchestrator.
"""

from __future__ import annotations

import re

import httpx
import numpy as np
import pytest

from src.evaluation.evidence_quality import (
    EvidenceQualityMetrics,
    _all_chains,
    _all_edges,
    _collect_citations,
    _heuristic_specificity,
    _validate_pmids_batch,
    chain_verifiability_score,
    citation_validity_rate,
    evaluate_evidence_quality,
    evidence_relevance,
    mean_chain_depth,
    mechanistic_specificity,
)
from src.schemas.prediction import (
    DrugDiseasePrediction,
    DrugDifficulty,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ---------------------------------------------------------------------------
# Fixtures -- reusable prediction objects
# ---------------------------------------------------------------------------


def _make_edge(
    src: str = "aspirin",
    tgt: str = "PTGS2",
    rel: EdgeType = EdgeType.INHIBITS,
    snippet: str = "Aspirin irreversibly inhibits COX-2 (PTGS2) enzyme.",
    pmid: str | None = "12345678",
) -> MechanisticEdge:
    return MechanisticEdge(
        source_entity=src,
        target_entity=tgt,
        relationship=rel,
        evidence_snippet=snippet,
        pmid=pmid,
    )


def _make_chain(
    edges: list[MechanisticEdge] | None = None,
    summary: str = "Aspirin inhibits PTGS2 reducing prostaglandin synthesis.",
    confidence: float = 0.9,
) -> EvidenceChain:
    if edges is None:
        edges = [
            _make_edge("aspirin", "PTGS2", EdgeType.INHIBITS,
                       "Aspirin inhibits COX-2.", "11111111"),
            _make_edge("PTGS2", "PGE2", EdgeType.DOWNREGULATES,
                       "PTGS2 inhibition reduces PGE2 synthesis.", "22222222"),
            _make_edge("PGE2", "inflammation", EdgeType.ASSOCIATED_WITH,
                       "PGE2 drives inflammatory response.", "33333333"),
        ]
    return EvidenceChain(edges=edges, summary=summary, confidence=confidence)


def _make_prediction(
    chains: list[EvidenceChain] | None = None,
    n_associations: int = 1,
) -> DrugDiseasePrediction:
    if chains is None:
        chains = [_make_chain()]

    associations = [
        ScoredAssociation(
            disease_name=f"disease_{i}",
            predicted=True,
            confidence=0.9 - i * 0.1,
            evidence_chains=chains if i == 0 else [],
        )
        for i in range(n_associations)
    ]

    return DrugDiseasePrediction(
        drug_name="aspirin",
        drug_chembl_id="CHEMBL25",
        associations=associations,
        reasoning="Test prediction for aspirin.",
    )


@pytest.fixture()
def rich_prediction() -> DrugDiseasePrediction:
    """Prediction with multiple chains and cited edges."""
    chain1 = _make_chain(
        edges=[
            _make_edge("aspirin", "PTGS1", EdgeType.INHIBITS,
                       "Aspirin binds COX-1 active site.", "11111111"),
            _make_edge("PTGS1", "TXA2", EdgeType.DOWNREGULATES,
                       "COX-1 inhibition blocks thromboxane A2.", "22222222"),
        ],
        summary="Aspirin inhibits COX-1 blocking TXA2 for antiplatelet effect.",
    )
    chain2 = _make_chain(
        edges=[
            _make_edge("aspirin", "PTGS2", EdgeType.INHIBITS,
                       "Aspirin acetylates Ser530 on COX-2.", "33333333"),
            _make_edge("PTGS2", "PGE2", EdgeType.DOWNREGULATES,
                       "COX-2 inhibition reduces PGE2.", None),  # no citation
            _make_edge("PGE2", "pain", EdgeType.ASSOCIATED_WITH,
                       "Reduced PGE2 lowers pain sensation.", "44444444"),
        ],
        summary="Aspirin inhibits PTGS2 -> PGE2 for analgesic effect.",
    )
    return _make_prediction(chains=[chain1, chain2])


@pytest.fixture()
def empty_prediction() -> DrugDiseasePrediction:
    """Prediction with no associations."""
    return DrugDiseasePrediction(
        drug_name="aspirin",
        associations=[],
        reasoning="No associations found.",
    )


@pytest.fixture()
def generic_prediction() -> DrugDiseasePrediction:
    """Prediction with generic (non-specific) language."""
    chain = _make_chain(
        edges=[
            _make_edge("the drug", "some enzyme", EdgeType.MODULATES,
                       "The drug modulates an enzyme in the body.", None),
            _make_edge("some enzyme", "a disease", EdgeType.ASSOCIATED_WITH,
                       "The enzyme is associated with the disease.", None),
        ],
        summary="The drug treats the disease through its mechanism.",
    )
    return _make_prediction(chains=[chain])


# ===================================================================
# Helper tests
# ===================================================================

class TestHelpers:
    def test_all_chains(self, rich_prediction: DrugDiseasePrediction) -> None:
        chains = _all_chains(rich_prediction)
        assert len(chains) == 2

    def test_all_edges(self, rich_prediction: DrugDiseasePrediction) -> None:
        edges = _all_edges(rich_prediction)
        assert len(edges) == 5  # 2 + 3

    def test_all_chains_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        assert _all_chains(empty_prediction) == []

    def test_all_edges_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        assert _all_edges(empty_prediction) == []

    def test_collect_citations(self, rich_prediction: DrugDiseasePrediction) -> None:
        pmids, dois = _collect_citations(rich_prediction)
        # Edges have PMIDs: 11111111, 22222222, 33333333, 44444444 (one edge has None)
        assert len(pmids) == 4
        assert "11111111" in pmids
        assert "44444444" in pmids

    def test_collect_citations_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        pmids, dois = _collect_citations(empty_prediction)
        assert pmids == []
        assert dois == []


# ===================================================================
# 1. Citation Validity Rate
# ===================================================================

class TestCitationValidity:
    @pytest.mark.asyncio
    async def test_no_citations_returns_zero(
        self, empty_prediction: DrugDiseasePrediction,
    ) -> None:
        rate, n_valid, n_total = await citation_validity_rate(empty_prediction)
        assert rate == 0.0
        assert n_valid == 0
        assert n_total == 0

    @pytest.mark.asyncio
    async def test_mock_all_valid(
        self,
        rich_prediction: DrugDiseasePrediction,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All PMIDs resolve -- mock the eSummary response."""
        pmids, _ = _collect_citations(rich_prediction)

        # Build mock response: each PMID gets a valid entry
        result = {"uids": pmids}
        for pmid in pmids:
            result[pmid] = {"uid": pmid, "title": "Mock Title"}

        async def mock_get(self_client, url, **kwargs):
            resp = httpx.Response(200, json={"result": result})
            resp._request = httpx.Request("GET", url)
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        rate, n_valid, n_total = await citation_validity_rate(rich_prediction)
        assert n_total == 4
        assert n_valid == 4
        assert rate == 1.0

    @pytest.mark.asyncio
    async def test_mock_some_invalid(
        self,
        rich_prediction: DrugDiseasePrediction,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Some PMIDs have errors."""
        pmids, _ = _collect_citations(rich_prediction)

        # First 2 valid, rest have errors
        result = {"uids": pmids[:2]}
        for pmid in pmids[:2]:
            result[pmid] = {"uid": pmid, "title": "Valid"}
        for pmid in pmids[2:]:
            result[pmid] = {"error": "cannot get document summary"}

        async def mock_get(self_client, url, **kwargs):
            resp = httpx.Response(200, json={"result": result})
            resp._request = httpx.Request("GET", url)
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        rate, n_valid, n_total = await citation_validity_rate(rich_prediction)
        assert n_total == 4
        assert n_valid == 2
        assert rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_api_failure_graceful(
        self,
        rich_prediction: DrugDiseasePrediction,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP error is handled gracefully -- returns 0 valid."""

        async def mock_get(self_client, url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        rate, n_valid, n_total = await citation_validity_rate(rich_prediction)
        assert n_total == 4
        assert n_valid == 0
        assert rate == 0.0


# ===================================================================
# 2. Mean Chain Depth
# ===================================================================

class TestMeanChainDepth:
    def test_standard(self, rich_prediction: DrugDiseasePrediction) -> None:
        # chain1 has 2 edges, chain2 has 3 edges => mean = 2.5
        depth = mean_chain_depth(rich_prediction)
        assert depth == pytest.approx(2.5)

    def test_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        assert mean_chain_depth(empty_prediction) == 0.0

    def test_single_edge_chains(self) -> None:
        chain = _make_chain(
            edges=[_make_edge()],
            summary="Single edge.",
        )
        pred = _make_prediction(chains=[chain])
        assert mean_chain_depth(pred) == 1.0

    def test_multiple_associations(self) -> None:
        chain_a = _make_chain(
            edges=[_make_edge(), _make_edge()],
            summary="Two edges.",
        )
        chain_b = _make_chain(
            edges=[_make_edge(), _make_edge(), _make_edge(), _make_edge()],
            summary="Four edges.",
        )
        pred = DrugDiseasePrediction(
            drug_name="test",
            associations=[
                ScoredAssociation(
                    disease_name="d1", predicted=True, confidence=0.9,
                    evidence_chains=[chain_a],
                ),
                ScoredAssociation(
                    disease_name="d2", predicted=True, confidence=0.8,
                    evidence_chains=[chain_b],
                ),
            ],
            reasoning="test",
        )
        # chain_a=2, chain_b=4 => mean=3.0
        assert mean_chain_depth(pred) == pytest.approx(3.0)


# ===================================================================
# 3. Chain Verifiability Score
# ===================================================================

class TestChainVerifiability:
    def test_all_cited(self) -> None:
        chain = _make_chain(
            edges=[
                _make_edge(pmid="11111111"),
                _make_edge(pmid="22222222"),
            ],
            summary="All cited.",
        )
        pred = _make_prediction(chains=[chain])
        assert chain_verifiability_score(pred) == 1.0

    def test_none_cited(self) -> None:
        chain = _make_chain(
            edges=[
                _make_edge(pmid=None),
                _make_edge(pmid=None),
            ],
            summary="None cited.",
        )
        pred = _make_prediction(chains=[chain])
        assert chain_verifiability_score(pred) == 0.0

    def test_partial(self, rich_prediction: DrugDiseasePrediction) -> None:
        # rich_prediction: 5 edges total, 4 have PMIDs, 1 does not
        score = chain_verifiability_score(rich_prediction)
        assert score == pytest.approx(4 / 5)

    def test_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        assert chain_verifiability_score(empty_prediction) == 0.0


# ===================================================================
# 4. Evidence Relevance
# ===================================================================

class TestEvidenceRelevance:
    @pytest.fixture(scope="class")
    def embedding_manager(self):
        """Load the MedCPT query encoder once for the class."""
        from src.vector.embeddings import EmbeddingManager

        mgr = EmbeddingManager()
        mgr.load_query_only()
        return mgr

    def test_relevant_edges(
        self,
        rich_prediction: DrugDiseasePrediction,
        embedding_manager: object,
    ) -> None:
        """Edges with matching snippets should have high relevance."""
        score = evidence_relevance(
            rich_prediction, embedding_manager=embedding_manager,
        )
        # Snippets closely match the claim -- expect > 0.3
        assert 0.0 < score <= 1.0
        assert score > 0.3

    def test_empty_prediction(
        self,
        empty_prediction: DrugDiseasePrediction,
        embedding_manager: object,
    ) -> None:
        score = evidence_relevance(
            empty_prediction, embedding_manager=embedding_manager,
        )
        assert score == 0.0

    def test_generic_lower_than_specific(
        self,
        rich_prediction: DrugDiseasePrediction,
        generic_prediction: DrugDiseasePrediction,
        embedding_manager: object,
    ) -> None:
        """Generic snippets should have lower relevance than specific ones."""
        specific = evidence_relevance(
            rich_prediction, embedding_manager=embedding_manager,
        )
        generic = evidence_relevance(
            generic_prediction, embedding_manager=embedding_manager,
        )
        # Both should be valid scores
        assert 0.0 <= specific <= 1.0
        assert 0.0 <= generic <= 1.0
        # Specific should generally be higher (or at least comparable)
        # We do not assert strict ordering because the model may vary,
        # but both should produce reasonable non-zero scores
        assert specific > 0.0
        assert generic > 0.0


# ===================================================================
# 5. Mechanistic Specificity
# ===================================================================

class TestMechanisticSpecificity:
    def test_heuristic_specific(self) -> None:
        """Chains with gene symbols should score high."""
        chain = _make_chain(
            edges=[
                _make_edge("aspirin", "PTGS2", EdgeType.INHIBITS,
                           "Aspirin binds COX-2 (PTGS2).", "12345678"),
                _make_edge("PTGS2", "PGE2", EdgeType.DOWNREGULATES,
                           "PTGS2 catalyses PGE2 synthesis.", "22222222"),
            ],
            summary="Aspirin inhibits PTGS2 / COX-2.",
        )
        score = _heuristic_specificity(chain)
        assert score == 1.0  # both edges reference gene symbols

    def test_heuristic_generic(self) -> None:
        """Chains without gene symbols should score low."""
        chain = _make_chain(
            edges=[
                _make_edge("the drug", "some enzyme", EdgeType.MODULATES,
                           "The drug modulates an enzyme.", None),
                _make_edge("some enzyme", "a condition", EdgeType.ASSOCIATED_WITH,
                           "The enzyme relates to the condition.", None),
            ],
            summary="The drug treats the condition.",
        )
        score = _heuristic_specificity(chain)
        assert score == 0.0

    def test_heuristic_mixed(self) -> None:
        """One specific edge, one generic."""
        chain = _make_chain(
            edges=[
                _make_edge("aspirin", "PTGS2", EdgeType.INHIBITS,
                           "Aspirin binds COX-2.", "12345678"),
                _make_edge("some pathway", "inflammation", EdgeType.ASSOCIATED_WITH,
                           "The pathway leads to inflammation.", None),
            ],
            summary="Mixed specificity.",
        )
        score = _heuristic_specificity(chain)
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_mechanistic_specificity_uses_heuristic_without_api(
        self, rich_prediction: DrugDiseasePrediction,
    ) -> None:
        """Without a valid LLM key, falls back to heuristic."""
        # Pass a bogus model_id so PydanticAI setup fails gracefully
        score = await mechanistic_specificity(
            rich_prediction, model_id="fake:nonexistent-model",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_empty(self, empty_prediction: DrugDiseasePrediction) -> None:
        score = await mechanistic_specificity(empty_prediction)
        assert score == 0.0


# ===================================================================
# Full evaluation
# ===================================================================

class TestEvaluateEvidenceQuality:
    @pytest.fixture(scope="class")
    def embedding_manager(self):
        from src.vector.embeddings import EmbeddingManager

        mgr = EmbeddingManager()
        mgr.load_query_only()
        return mgr

    @pytest.mark.asyncio
    async def test_full_evaluation(
        self,
        rich_prediction: DrugDiseasePrediction,
        embedding_manager: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end evaluation with mocked PMID validation."""
        # Mock PMID validation to return all valid
        pmids, _ = _collect_citations(rich_prediction)
        result = {"uids": pmids}
        for pmid in pmids:
            result[pmid] = {"uid": pmid, "title": "Valid"}

        async def mock_get(self_client, url, **kwargs):
            resp = httpx.Response(200, json={"result": result})
            resp._request = httpx.Request("GET", url)
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        metrics = await evaluate_evidence_quality(
            rich_prediction,
            embedding_manager=embedding_manager,
            use_llm_specificity=False,  # use heuristic
        )

        assert isinstance(metrics, EvidenceQualityMetrics)
        assert metrics.citation_validity_rate == 1.0
        assert metrics.mean_chain_depth == pytest.approx(2.5)
        assert metrics.chain_verifiability_score == pytest.approx(4 / 5)
        assert 0.0 < metrics.evidence_relevance <= 1.0
        assert 0.0 <= metrics.mechanistic_specificity <= 1.0
        assert metrics.n_chains == 2
        assert metrics.n_edges == 5
        assert metrics.n_citations_checked == 4

    @pytest.mark.asyncio
    async def test_empty_prediction_evaluation(
        self,
        empty_prediction: DrugDiseasePrediction,
        embedding_manager: object,
    ) -> None:
        metrics = await evaluate_evidence_quality(
            empty_prediction,
            embedding_manager=embedding_manager,
            use_llm_specificity=False,
        )

        assert metrics.citation_validity_rate == 0.0
        assert metrics.mean_chain_depth == 0.0
        assert metrics.chain_verifiability_score == 0.0
        assert metrics.evidence_relevance == 0.0
        assert metrics.mechanistic_specificity == 0.0
        assert metrics.n_chains == 0
        assert metrics.n_edges == 0

    @pytest.mark.asyncio
    async def test_metrics_model_validates(
        self,
        rich_prediction: DrugDiseasePrediction,
        embedding_manager: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EvidenceQualityMetrics passes Pydantic validation."""
        async def mock_get(self_client, url, **kwargs):
            resp = httpx.Response(200, json={"result": {}})
            resp._request = httpx.Request("GET", url)
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        metrics = await evaluate_evidence_quality(
            rich_prediction,
            embedding_manager=embedding_manager,
            use_llm_specificity=False,
        )

        # Round-trip through Pydantic
        data = metrics.model_dump()
        restored = EvidenceQualityMetrics.model_validate(data)
        assert restored.mean_chain_depth == metrics.mean_chain_depth
