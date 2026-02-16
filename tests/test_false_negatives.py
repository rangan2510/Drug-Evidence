"""Tests for src.evaluation.false_negatives -- FN taxonomy and analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.false_negatives import (
    AggregateFNSummary,
    FalseNegativeCategory,
    FalseNegativeRecord,
    FalseNegativeSummary,
    NEAR_MISS_THRESHOLD,
    _cosine_similarity_matrix,
    _categorise_fn,
    aggregate_fn_summaries,
    analyse_false_negatives,
    find_nearest_prediction,
    source_coverage_analysis,
)
from src.schemas.prediction import (
    DrugDiseasePrediction,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_prediction(
    drug: str = "Aspirin",
    associations: list[tuple[str, float, bool]] | None = None,
) -> DrugDiseasePrediction:
    """Build a minimal prediction for testing."""
    if associations is None:
        associations = [
            ("Heart Disease", 0.9, True),
            ("Stroke", 0.7, True),
            ("Diabetes", 0.4, False),
            ("Cancer", 0.2, False),
        ]
    assocs = []
    for name, conf, pred in associations:
        assocs.append(
            ScoredAssociation(
                disease_name=name,
                predicted=pred,
                confidence=conf,
                evidence_chains=[
                    EvidenceChain(
                        edges=[
                            MechanisticEdge(
                                source_entity=drug,
                                target_entity=name,
                                relationship=EdgeType.ASSOCIATED_WITH,
                                evidence_snippet="test snippet",
                            )
                        ],
                        summary="test",
                        confidence=conf,
                    )
                ],
            )
        )
    return DrugDiseasePrediction(
        drug_name=drug,
        associations=assocs,
        reasoning="test reasoning",
    )


@pytest.fixture
def sample_prediction() -> DrugDiseasePrediction:
    return _make_prediction()


# ------------------------------------------------------------------
# FalseNegativeCategory enum tests
# ------------------------------------------------------------------

class TestFalseNegativeCategory:
    def test_all_six_categories(self):
        assert len(FalseNegativeCategory) == 6

    def test_values(self):
        expected = {
            "no_evidence_found",
            "low_retrieval_score",
            "low_llm_confidence",
            "below_threshold",
            "not_in_candidates",
            "name_mismatch",
        }
        assert {cat.value for cat in FalseNegativeCategory} == expected


# ------------------------------------------------------------------
# FalseNegativeRecord tests
# ------------------------------------------------------------------

class TestFalseNegativeRecord:
    def test_basic_creation(self):
        rec = FalseNegativeRecord(
            drug="Aspirin",
            disease_gt="Asthma",
            category=FalseNegativeCategory.NOT_IN_CANDIDATES,
        )
        assert rec.drug == "Aspirin"
        assert rec.disease_gt == "Asthma"
        assert rec.closest_prediction is None
        assert rec.evidence_sources_checked == []

    def test_with_all_fields(self):
        rec = FalseNegativeRecord(
            drug="Aspirin",
            disease_gt="Asthma",
            category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
            closest_prediction="Asthmatic Bronchitis",
            similarity_to_closest=0.75,
            retrieval_score=0.2,
            llm_confidence=0.6,
            evidence_sources_checked=["opentargets", "dgidb"],
        )
        assert rec.similarity_to_closest == 0.75
        assert len(rec.evidence_sources_checked) == 2


# ------------------------------------------------------------------
# FalseNegativeSummary tests
# ------------------------------------------------------------------

class TestFalseNegativeSummary:
    def test_fn_rate(self):
        summary = FalseNegativeSummary(
            drug_name="X", arm_id="test",
            n_ground_truth=10, n_true_positives=6, n_false_negatives=4,
        )
        assert summary.fn_rate == pytest.approx(0.4)

    def test_fn_rate_zero_gt(self):
        summary = FalseNegativeSummary(
            drug_name="X", arm_id="test",
            n_ground_truth=0, n_true_positives=0, n_false_negatives=0,
        )
        assert summary.fn_rate == 0.0

    def test_category_counts(self):
        records = [
            FalseNegativeRecord(
                drug="X", disease_gt="A",
                category=FalseNegativeCategory.NOT_IN_CANDIDATES,
            ),
            FalseNegativeRecord(
                drug="X", disease_gt="B",
                category=FalseNegativeCategory.NOT_IN_CANDIDATES,
            ),
            FalseNegativeRecord(
                drug="X", disease_gt="C",
                category=FalseNegativeCategory.LOW_LLM_CONFIDENCE,
            ),
        ]
        summary = FalseNegativeSummary(
            drug_name="X", arm_id="test",
            n_ground_truth=5, n_true_positives=2, n_false_negatives=3,
            records=records,
        )
        counts = summary.category_counts()
        assert counts["not_in_candidates"] == 2
        assert counts["low_llm_confidence"] == 1
        assert counts["no_evidence_found"] == 0


# ------------------------------------------------------------------
# Cosine similarity tests
# ------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(a, a)
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(a, b)
        assert sim[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        sim = _cosine_similarity_matrix(a, b)
        assert sim[0, 0] == pytest.approx(-1.0, abs=1e-5)

    def test_matrix_shape(self):
        a = np.random.randn(3, 10).astype(np.float32)
        b = np.random.randn(5, 10).astype(np.float32)
        sim = _cosine_similarity_matrix(a, b)
        assert sim.shape == (3, 5)

    def test_self_similarity_diagonal(self):
        a = np.random.randn(4, 8).astype(np.float32)
        sim = _cosine_similarity_matrix(a, a)
        for i in range(4):
            assert sim[i, i] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------
# find_nearest_prediction tests
# ------------------------------------------------------------------

class TestFindNearestPrediction:
    def test_empty_predictions(self):
        name, sim = find_nearest_prediction("Asthma", [])
        assert name is None
        assert sim is None

    def test_exact_match_fallback(self):
        name, sim = find_nearest_prediction(
            "Heart Disease",
            ["Cancer", "Heart Disease", "Stroke"],
        )
        assert name == "Heart Disease"
        assert sim == 1.0

    def test_no_match_fallback(self):
        name, sim = find_nearest_prediction(
            "Asthma",
            ["Cancer", "Diabetes"],
        )
        # Returns first prediction with sim 0.0 when no exact match
        assert name == "Cancer"
        assert sim == 0.0

    def test_with_embeddings(self):
        # Create embeddings where gt[0] is closest to pred[1]
        gt_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pred_emb = np.array(
            [[0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        name, sim = find_nearest_prediction(
            "Asthma",
            ["Cancer", "Asthmatic Bronchitis", "Flu"],
            gt_embeddings=gt_emb,
            pred_embeddings=pred_emb,
            gt_index=0,
        )
        assert name == "Asthmatic Bronchitis"
        assert sim is not None
        assert sim > 0.9  # high similarity


# ------------------------------------------------------------------
# Categorisation tests
# ------------------------------------------------------------------

class TestCategoriseFN:
    def test_not_in_candidates(self, sample_prediction):
        """Disease not in prediction associations at all."""
        rec = _categorise_fn(
            "Asthma", sample_prediction,
        )
        assert rec.category == FalseNegativeCategory.NOT_IN_CANDIDATES
        assert rec.drug == "Aspirin"
        assert rec.disease_gt == "Asthma"

    def test_below_threshold(self):
        """Disease present with predicted=False and low confidence."""
        pred = _make_prediction(associations=[
            ("Asthma", 0.3, False),
            ("Heart Disease", 0.9, True),
        ])
        rec = _categorise_fn(
            "asthma", pred,
            retrieval_scores={"asthma": 0.6},
            evidence_sources={"asthma": ["opentargets"]},
            threshold=0.5,
            retrieval_gate=0.3,
        )
        assert rec.category == FalseNegativeCategory.BELOW_THRESHOLD

    def test_low_retrieval_score(self):
        """Disease present but retrieval score below gate."""
        pred = _make_prediction(associations=[
            ("Asthma", 0.8, False),
        ])
        rec = _categorise_fn(
            "asthma", pred,
            retrieval_scores={"asthma": 0.1},
            evidence_sources={"asthma": ["opentargets"]},
            retrieval_gate=0.3,
        )
        assert rec.category == FalseNegativeCategory.LOW_RETRIEVAL_SCORE
        assert rec.retrieval_score == 0.1

    def test_low_llm_confidence(self):
        """Disease present, good retrieval, but LLM not confident."""
        pred = _make_prediction(associations=[
            ("Asthma", 0.1, False),
        ])
        rec = _categorise_fn(
            "asthma", pred,
            retrieval_scores={"asthma": 0.8},
            evidence_sources={"asthma": ["opentargets"]},
            retrieval_gate=0.3,
        )
        assert rec.category == FalseNegativeCategory.LOW_LLM_CONFIDENCE
        assert rec.llm_confidence == 0.1

    def test_name_mismatch_via_embedding(self):
        """Disease not in candidates but high embedding similarity."""
        gt_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        pred_emb = np.array([[0.95, 0.05]], dtype=np.float32)

        pred = _make_prediction(associations=[
            ("Cardiac Disease", 0.9, True),
        ])
        rec = _categorise_fn(
            "Heart Disease", pred,
            predicted_names=["Cardiac Disease"],
            gt_embeddings=gt_emb,
            pred_embeddings=pred_emb,
            gt_index=0,
            near_miss_threshold=0.7,
        )
        assert rec.category == FalseNegativeCategory.NAME_MISMATCH
        assert rec.closest_prediction == "Cardiac Disease"
        assert rec.similarity_to_closest is not None
        assert rec.similarity_to_closest > 0.7

    def test_no_evidence_found(self):
        """Disease in candidates but no evidence sources and no retrieval score."""
        pred = _make_prediction(associations=[
            ("Asthma", 0.5, False),
        ])
        rec = _categorise_fn(
            "asthma", pred,
            retrieval_scores=None,
            evidence_sources={},  # no sources
        )
        assert rec.category == FalseNegativeCategory.NO_EVIDENCE_FOUND


# ------------------------------------------------------------------
# analyse_false_negatives tests
# ------------------------------------------------------------------

class TestAnalyseFalseNegatives:
    def test_basic_analysis(self, sample_prediction):
        gt = {"heart disease", "stroke", "asthma"}
        summary = analyse_false_negatives(
            sample_prediction, gt, arm_id="test",
        )
        assert isinstance(summary, FalseNegativeSummary)
        assert summary.drug_name == "Aspirin"
        assert summary.n_ground_truth == 3
        assert summary.n_true_positives == 2  # Heart Disease, Stroke
        assert summary.n_false_negatives == 1  # Asthma
        assert len(summary.records) == 1
        assert summary.records[0].disease_gt == "asthma"

    def test_all_true_positives(self):
        pred = _make_prediction(associations=[
            ("Heart Disease", 0.9, True),
            ("Stroke", 0.7, True),
        ])
        gt = {"heart disease", "stroke"}
        summary = analyse_false_negatives(pred, gt, arm_id="test")
        assert summary.n_false_negatives == 0
        assert len(summary.records) == 0

    def test_all_false_negatives(self):
        pred = _make_prediction(associations=[
            ("Cancer", 0.9, True),
        ])
        gt = {"heart disease", "stroke"}
        summary = analyse_false_negatives(pred, gt, arm_id="test")
        assert summary.n_false_negatives == 2
        assert len(summary.records) == 2

    def test_empty_ground_truth(self, sample_prediction):
        summary = analyse_false_negatives(
            sample_prediction, set(), arm_id="test",
        )
        assert summary.n_ground_truth == 0
        assert summary.n_false_negatives == 0

    def test_with_retrieval_scores(self):
        pred = _make_prediction(associations=[
            ("Heart Disease", 0.9, True),
            ("Asthma", 0.1, False),
        ])
        gt = {"heart disease", "asthma"}
        summary = analyse_false_negatives(
            pred, gt, arm_id="test",
            retrieval_scores={"asthma": 0.05},
            evidence_sources={"asthma": ["pubmed"]},
            retrieval_gate=0.3,
        )
        assert summary.n_false_negatives == 1
        rec = summary.records[0]
        assert rec.category == FalseNegativeCategory.LOW_RETRIEVAL_SCORE

    def test_fn_rate_matches(self, sample_prediction):
        gt = {"heart disease", "stroke", "asthma", "flu", "gout"}
        summary = analyse_false_negatives(
            sample_prediction, gt, arm_id="test",
        )
        expected_rate = summary.n_false_negatives / summary.n_ground_truth
        assert summary.fn_rate == pytest.approx(expected_rate)


# ------------------------------------------------------------------
# Aggregate FN summary tests
# ------------------------------------------------------------------

class TestAggregateFNSummary:
    def _make_summaries(self) -> list[FalseNegativeSummary]:
        return [
            FalseNegativeSummary(
                drug_name="Drug1", arm_id="test",
                n_ground_truth=5, n_true_positives=3, n_false_negatives=2,
                records=[
                    FalseNegativeRecord(
                        drug="Drug1", disease_gt="A",
                        category=FalseNegativeCategory.NOT_IN_CANDIDATES,
                    ),
                    FalseNegativeRecord(
                        drug="Drug1", disease_gt="B",
                        category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
                        similarity_to_closest=0.8,
                    ),
                ],
            ),
            FalseNegativeSummary(
                drug_name="Drug2", arm_id="test",
                n_ground_truth=3, n_true_positives=1, n_false_negatives=2,
                records=[
                    FalseNegativeRecord(
                        drug="Drug2", disease_gt="C",
                        category=FalseNegativeCategory.NAME_MISMATCH,
                        similarity_to_closest=0.9,
                    ),
                    FalseNegativeRecord(
                        drug="Drug2", disease_gt="D",
                        category=FalseNegativeCategory.BELOW_THRESHOLD,
                    ),
                ],
            ),
        ]

    def test_totals(self):
        summaries = self._make_summaries()
        agg = aggregate_fn_summaries(summaries, arm_id="test")
        assert agg.n_drugs == 2
        assert agg.total_ground_truth == 8
        assert agg.total_true_positives == 4
        assert agg.total_false_negatives == 4

    def test_category_counts(self):
        summaries = self._make_summaries()
        agg = aggregate_fn_summaries(summaries, arm_id="test")
        assert agg.category_counts["not_in_candidates"] == 1
        assert agg.category_counts["low_retrieval_score"] == 1
        assert agg.category_counts["name_mismatch"] == 1
        assert agg.category_counts["below_threshold"] == 1

    def test_near_miss_detection(self):
        summaries = self._make_summaries()
        agg = aggregate_fn_summaries(summaries, arm_id="test")
        # Drug1/B has sim=0.8 (above 0.7) -> near miss
        # Drug2/C is NAME_MISMATCH -> always near miss
        assert agg.near_miss_count >= 2

    def test_overall_fn_rate(self):
        summaries = self._make_summaries()
        agg = aggregate_fn_summaries(summaries, arm_id="test")
        assert agg.overall_fn_rate == pytest.approx(4 / 8)

    def test_category_fractions(self):
        summaries = self._make_summaries()
        agg = aggregate_fn_summaries(summaries, arm_id="test")
        fracs = agg.category_fractions()
        assert sum(fracs.values()) == pytest.approx(1.0)

    def test_empty_summaries(self):
        agg = aggregate_fn_summaries([], arm_id="test")
        assert agg.n_drugs == 0
        assert agg.overall_fn_rate == 0.0


# ------------------------------------------------------------------
# Source coverage analysis tests
# ------------------------------------------------------------------

class TestSourceCoverage:
    def test_basic_coverage(self):
        summaries = [
            FalseNegativeSummary(
                drug_name="X", arm_id="test",
                n_ground_truth=3, n_true_positives=1, n_false_negatives=2,
                records=[
                    FalseNegativeRecord(
                        drug="X", disease_gt="A",
                        category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
                        evidence_sources_checked=["opentargets", "dgidb"],
                    ),
                    FalseNegativeRecord(
                        drug="X", disease_gt="B",
                        category=FalseNegativeCategory.NOT_IN_CANDIDATES,
                        evidence_sources_checked=["opentargets"],
                    ),
                ],
            ),
        ]
        coverage = source_coverage_analysis(summaries)
        assert coverage["opentargets"] == 2
        assert coverage["dgidb"] == 1

    def test_empty_sources(self):
        summaries = [
            FalseNegativeSummary(
                drug_name="X", arm_id="test",
                n_ground_truth=1, n_true_positives=0, n_false_negatives=1,
                records=[
                    FalseNegativeRecord(
                        drug="X", disease_gt="A",
                        category=FalseNegativeCategory.NO_EVIDENCE_FOUND,
                    ),
                ],
            ),
        ]
        coverage = source_coverage_analysis(summaries)
        assert coverage == {}

    def test_sorted_by_count(self):
        summaries = [
            FalseNegativeSummary(
                drug_name="X", arm_id="test",
                n_ground_truth=3, n_true_positives=0, n_false_negatives=3,
                records=[
                    FalseNegativeRecord(
                        drug="X", disease_gt="A",
                        category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
                        evidence_sources_checked=["pubmed", "opentargets"],
                    ),
                    FalseNegativeRecord(
                        drug="X", disease_gt="B",
                        category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
                        evidence_sources_checked=["pubmed"],
                    ),
                    FalseNegativeRecord(
                        drug="X", disease_gt="C",
                        category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
                        evidence_sources_checked=["pubmed", "chembl", "opentargets"],
                    ),
                ],
            ),
        ]
        coverage = source_coverage_analysis(summaries)
        keys = list(coverage.keys())
        # pubmed appears most (3), then opentargets (2), then chembl (1)
        assert keys[0] == "pubmed"
        assert coverage["pubmed"] == 3
