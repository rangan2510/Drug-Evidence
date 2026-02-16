"""Tests for Pydantic schemas -- evidence and prediction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)
from src.schemas.prediction import (
    DrugDiseasePrediction,
    DrugDifficulty,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ------------------------------------------------------------------
# evidence.py
# ------------------------------------------------------------------


class TestCitation:
    def test_empty_citation(self) -> None:
        c = Citation()
        assert c.pmid is None and c.doi is None

    def test_full_citation(self) -> None:
        c = Citation(pmid="12345678", doi="10.1234/test", title="A paper", year=2025)
        assert c.pmid == "12345678"
        assert c.year == 2025


class TestEvidenceDocument:
    def test_minimal_document(self) -> None:
        doc = EvidenceDocument(
            text="Aspirin inhibits COX-2.",
            source=EvidenceSource.PUBMED,
            evidence_type=EvidenceType.LITERATURE,
            drug_name="aspirin",
        )
        assert doc.source == "pubmed"
        assert doc.target_symbol is None

    def test_full_document(self) -> None:
        doc = EvidenceDocument(
            text="Imatinib binds BCR-ABL.",
            source=EvidenceSource.CHEMBL,
            evidence_type=EvidenceType.BINDING_ASSAY,
            drug_name="imatinib",
            drug_chembl_id="CHEMBL941",
            target_symbol="ABL1",
            disease_name="Chronic myeloid leukemia",
            score=0.95,
        )
        assert doc.drug_chembl_id == "CHEMBL941"
        assert doc.score == 0.95

    def test_source_enum_validation(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceDocument(
                text="bad",
                source="not_a_source",  # type: ignore[arg-type]
                evidence_type=EvidenceType.LITERATURE,
                drug_name="x",
            )


# ------------------------------------------------------------------
# prediction.py
# ------------------------------------------------------------------


class TestMechanisticEdge:
    def test_basic_edge(self) -> None:
        edge = MechanisticEdge(
            source_entity="imatinib",
            target_entity="ABL1",
            relationship=EdgeType.INHIBITS,
            evidence_snippet="Imatinib inhibits ABL1 kinase activity.",
            pmid="11423618",
        )
        assert edge.relationship == "inhibits"


class TestEvidenceChain:
    def test_single_edge_chain(self) -> None:
        chain = EvidenceChain(
            edges=[
                MechanisticEdge(
                    source_entity="aspirin",
                    target_entity="COX-2",
                    relationship=EdgeType.INHIBITS,
                    evidence_snippet="Aspirin irreversibly acetylates COX-2.",
                )
            ],
            summary="Aspirin inhibits COX-2.",
            confidence=0.9,
        )
        assert len(chain.edges) == 1

    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceChain(edges=[], summary="empty", confidence=0.5)

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceChain(
                edges=[
                    MechanisticEdge(
                        source_entity="a",
                        target_entity="b",
                        relationship=EdgeType.BINDS,
                        evidence_snippet="x",
                    )
                ],
                summary="s",
                confidence=1.5,
            )


class TestDrugDiseasePrediction:
    def test_full_prediction(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="metformin",
            drug_chembl_id="CHEMBL1431",
            associations=[
                ScoredAssociation(
                    disease_name="Type 2 diabetes",
                    predicted=True,
                    confidence=0.95,
                    evidence_chains=[
                        EvidenceChain(
                            edges=[
                                MechanisticEdge(
                                    source_entity="metformin",
                                    target_entity="AMPK",
                                    relationship=EdgeType.ACTIVATES,
                                    evidence_snippet="Metformin activates AMPK.",
                                ),
                                MechanisticEdge(
                                    source_entity="AMPK",
                                    target_entity="hepatic gluconeogenesis",
                                    relationship=EdgeType.INHIBITS,
                                    evidence_snippet="AMPK inhibits hepatic gluconeogenesis.",
                                ),
                            ],
                            summary="Metformin activates AMPK, reducing hepatic glucose output.",
                            confidence=0.92,
                        )
                    ],
                )
            ],
            reasoning="Strong mechanistic evidence via AMPK pathway.",
        )
        assert pred.associations[0].predicted is True
        assert len(pred.associations[0].evidence_chains[0].edges) == 2

    def test_roundtrip_json(self) -> None:
        pred = DrugDiseasePrediction(
            drug_name="aspirin",
            associations=[
                ScoredAssociation(
                    disease_name="Pain", predicted=True, confidence=0.8
                )
            ],
            reasoning="Well-known analgesic.",
        )
        json_str = pred.model_dump_json()
        restored = DrugDiseasePrediction.model_validate_json(json_str)
        assert restored.drug_name == "aspirin"
        assert restored.associations[0].confidence == 0.8
