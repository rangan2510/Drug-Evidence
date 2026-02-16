"""Tests for src.evaluation.sensitivity -- weight sweep, threshold sweep, ablations."""

from __future__ import annotations

import pytest

from src.evaluation.sensitivity import (
    ABLATION_CONFIGS,
    AblationConfig,
    AblationResult,
    AblationType,
    AggregateSweepPoint,
    CachedScore,
    HeatmapCell,
    SweepPoint,
    ThresholdSweepResult,
    WeightSweepResult,
    aggregate_ablation_results,
    aggregate_threshold_sweeps,
    aggregate_weight_sweeps,
    build_heatmap_data,
    extract_cached_scores,
    threshold_sweep,
    weight_sweep,
)
from src.schemas.prediction import (
    DrugDiseasePrediction,
    EvidenceChain,
    MechanisticEdge,
    EdgeType,
    ScoredAssociation,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_prediction(
    drug: str = "Aspirin",
    associations: list[tuple[str, float, bool]] | None = None,
) -> DrugDiseasePrediction:
    """Helper: build a minimal DrugDiseasePrediction.

    associations is a list of (disease_name, confidence, predicted).
    """
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
                                evidence_snippet="test",
                            )
                        ],
                        summary="test chain",
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


@pytest.fixture
def sample_ground_truth() -> set[str]:
    return {"heart disease", "stroke", "asthma"}


@pytest.fixture
def sample_cached_scores() -> list[CachedScore]:
    return [
        CachedScore(disease_name="Heart Disease", retrieval_score=0.8, llm_confidence=0.9),
        CachedScore(disease_name="Stroke", retrieval_score=0.6, llm_confidence=0.7),
        CachedScore(disease_name="Diabetes", retrieval_score=0.3, llm_confidence=0.4),
        CachedScore(disease_name="Cancer", retrieval_score=0.1, llm_confidence=0.2),
    ]


# ------------------------------------------------------------------
# CachedScore tests
# ------------------------------------------------------------------

class TestCachedScore:
    def test_combined_score_equal_weight(self):
        cs = CachedScore(disease_name="X", retrieval_score=0.8, llm_confidence=0.6)
        assert cs.combined_score(0.5) == pytest.approx(0.7)

    def test_combined_score_retrieval_only(self):
        cs = CachedScore(disease_name="X", retrieval_score=0.8, llm_confidence=0.6)
        assert cs.combined_score(1.0) == pytest.approx(0.8)

    def test_combined_score_llm_only(self):
        cs = CachedScore(disease_name="X", retrieval_score=0.8, llm_confidence=0.6)
        assert cs.combined_score(0.0) == pytest.approx(0.6)

    def test_combined_score_asymmetric(self):
        cs = CachedScore(disease_name="X", retrieval_score=1.0, llm_confidence=0.0)
        assert cs.combined_score(0.75) == pytest.approx(0.75)


# ------------------------------------------------------------------
# extract_cached_scores tests
# ------------------------------------------------------------------

class TestExtractCachedScores:
    def test_with_retrieval_scores(self, sample_prediction):
        ret_scores = {
            "heart disease": 0.85,
            "stroke": 0.65,
        }
        cached = extract_cached_scores(sample_prediction, ret_scores)
        assert len(cached) == 4
        # Heart Disease matched
        assert cached[0].disease_name == "Heart Disease"
        assert cached[0].retrieval_score == 0.85
        assert cached[0].llm_confidence == 0.9

    def test_without_retrieval_scores(self, sample_prediction):
        cached = extract_cached_scores(sample_prediction)
        assert len(cached) == 4
        # All retrieval scores default to 0.0
        for cs in cached:
            assert cs.retrieval_score == 0.0

    def test_llm_confidence_preserved(self, sample_prediction):
        cached = extract_cached_scores(sample_prediction)
        confidences = {cs.disease_name: cs.llm_confidence for cs in cached}
        assert confidences["Heart Disease"] == 0.9
        assert confidences["Cancer"] == 0.2


# ------------------------------------------------------------------
# SweepPoint / recompute tests
# ------------------------------------------------------------------

class TestRecomputeMetrics:
    def test_all_above_threshold(self, sample_cached_scores, sample_ground_truth):
        from src.evaluation.sensitivity import _recompute_metrics
        pt = _recompute_metrics(sample_cached_scores, sample_ground_truth, 0.5, 0.0)
        # All 4 diseases above threshold 0.0
        assert pt.n_predicted == 4
        assert pt.n_ground_truth == 3

    def test_high_threshold_filters(self, sample_cached_scores, sample_ground_truth):
        from src.evaluation.sensitivity import _recompute_metrics
        pt = _recompute_metrics(sample_cached_scores, sample_ground_truth, 0.5, 0.9)
        # Only Heart Disease (0.85) might pass at threshold 0.9
        # With w=0.5: Heart = 0.5*0.8 + 0.5*0.9 = 0.85 -- below 0.9
        assert pt.n_predicted == 0

    def test_llm_only_weights(self, sample_cached_scores, sample_ground_truth):
        from src.evaluation.sensitivity import _recompute_metrics
        # w_retrieval=0 -> pure LLM confidence
        pt = _recompute_metrics(sample_cached_scores, sample_ground_truth, 0.0, 0.5)
        # Heart=0.9, Stroke=0.7 pass; Diabetes=0.4, Cancer=0.2 fail
        assert pt.n_predicted == 2
        assert pt.precision_at_1 == 1.0  # Heart Disease is in GT

    def test_retrieval_only_weights(self, sample_cached_scores, sample_ground_truth):
        from src.evaluation.sensitivity import _recompute_metrics
        pt = _recompute_metrics(sample_cached_scores, sample_ground_truth, 1.0, 0.5)
        # Heart=0.8, Stroke=0.6 pass; Diabetes=0.3, Cancer=0.1 fail
        assert pt.n_predicted == 2

    def test_empty_ground_truth(self, sample_cached_scores):
        from src.evaluation.sensitivity import _recompute_metrics
        pt = _recompute_metrics(sample_cached_scores, set(), 0.5, 0.5)
        assert pt.recall_at_1 == 0.0
        assert pt.recall_at_10 == 0.0

    def test_empty_cached_scores(self, sample_ground_truth):
        from src.evaluation.sensitivity import _recompute_metrics
        pt = _recompute_metrics([], sample_ground_truth, 0.5, 0.5)
        assert pt.n_predicted == 0
        assert pt.precision_at_1 == 0.0


# ------------------------------------------------------------------
# Weight sweep tests
# ------------------------------------------------------------------

class TestWeightSweep:
    def test_default_11_points(self, sample_cached_scores, sample_ground_truth):
        result = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test-arm", drug_name="Aspirin",
        )
        assert isinstance(result, WeightSweepResult)
        assert result.drug_name == "Aspirin"
        assert result.arm_id == "test-arm"
        assert len(result.points) == 11  # 0.0 to 1.0 in 0.1 steps

    def test_weight_range(self, sample_cached_scores, sample_ground_truth):
        result = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
        )
        weights = [p.w_retrieval for p in result.points]
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_w_llm_complement(self, sample_cached_scores, sample_ground_truth):
        result = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
        )
        for pt in result.points:
            assert pt.w_retrieval + pt.w_llm == pytest.approx(1.0)

    def test_custom_step(self, sample_cached_scores, sample_ground_truth):
        result = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
            start=0.0, end=1.0, step=0.5,
        )
        assert len(result.points) == 3  # 0.0, 0.5, 1.0

    def test_threshold_fixed(self, sample_cached_scores, sample_ground_truth):
        result = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
            threshold=0.7,
        )
        for pt in result.points:
            assert pt.threshold == 0.7


# ------------------------------------------------------------------
# Threshold sweep tests
# ------------------------------------------------------------------

class TestThresholdSweep:
    def test_default_13_points(self, sample_cached_scores, sample_ground_truth):
        result = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test-arm", drug_name="Aspirin",
        )
        assert isinstance(result, ThresholdSweepResult)
        assert len(result.points) == 13  # 0.3 to 0.9 in 0.05 steps

    def test_threshold_range(self, sample_cached_scores, sample_ground_truth):
        result = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
        )
        thresholds = [p.threshold for p in result.points]
        assert thresholds[0] == pytest.approx(0.3)
        assert thresholds[-1] == pytest.approx(0.9)

    def test_higher_threshold_fewer_predicted(self, sample_cached_scores, sample_ground_truth):
        result = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
        )
        # n_predicted should be non-increasing as threshold rises
        n_pred = [p.n_predicted for p in result.points]
        for i in range(1, len(n_pred)):
            assert n_pred[i] <= n_pred[i - 1]

    def test_weight_stored(self, sample_cached_scores, sample_ground_truth):
        result = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="test", drug_name="X",
            w_retrieval=0.3,
        )
        assert result.w_retrieval == 0.3
        assert result.w_llm == pytest.approx(0.7)


# ------------------------------------------------------------------
# Aggregation tests
# ------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_weight_sweeps(self, sample_cached_scores, sample_ground_truth):
        sweep1 = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="arm1", drug_name="Drug1",
        )
        sweep2 = weight_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="arm1", drug_name="Drug2",
        )
        agg = aggregate_weight_sweeps([sweep1, sweep2])
        assert len(agg) == 11
        assert all(isinstance(p, AggregateSweepPoint) for p in agg)
        assert agg[0].n_drugs == 2

    def test_aggregate_empty(self):
        assert aggregate_weight_sweeps([]) == []

    def test_aggregate_threshold_sweeps(self, sample_cached_scores, sample_ground_truth):
        sweep1 = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="arm1", drug_name="Drug1",
        )
        sweep2 = threshold_sweep(
            sample_cached_scores, sample_ground_truth,
            arm_id="arm1", drug_name="Drug2",
        )
        agg = aggregate_threshold_sweeps([sweep1, sweep2])
        assert len(agg) == 13
        assert agg[0].n_drugs == 2


# ------------------------------------------------------------------
# Ablation tests
# ------------------------------------------------------------------

class TestAblation:
    def test_all_ablation_types_defined(self):
        for at in AblationType:
            assert at in ABLATION_CONFIGS

    def test_ablation_config_labels(self):
        for at, cfg in ABLATION_CONFIGS.items():
            assert cfg.label == at.value
            assert cfg.description  # non-empty

    def test_no_dgidb_config(self):
        cfg = ABLATION_CONFIGS[AblationType.NO_DGIDB]
        assert "dgidb" in cfg.disabled_sources
        assert cfg.tools_enabled is True

    def test_dense_only_config(self):
        cfg = ABLATION_CONFIGS[AblationType.DENSE_ONLY]
        assert cfg.search_mode == "dense"
        assert cfg.disabled_sources == []

    def test_no_tools_config(self):
        cfg = ABLATION_CONFIGS[AblationType.NO_TOOLS]
        assert cfg.tools_enabled is False

    def test_ablation_result_delta(self):
        result = AblationResult(
            ablation_type=AblationType.NO_DGIDB,
            arm_id="test",
            drug_name="X",
            baseline_p_at_10=0.8,
            ablated_p_at_10=0.6,
            baseline_r_at_10=0.7,
            ablated_r_at_10=0.5,
        )
        assert result.delta_p_at_10 == pytest.approx(-0.2)
        assert result.delta_r_at_10 == pytest.approx(-0.2)

    def test_aggregate_ablation_results(self):
        results = [
            AblationResult(
                ablation_type=AblationType.NO_DGIDB,
                arm_id="test", drug_name="A",
                baseline_p_at_10=0.8, ablated_p_at_10=0.6,
                baseline_r_at_10=0.7, ablated_r_at_10=0.5,
            ),
            AblationResult(
                ablation_type=AblationType.NO_DGIDB,
                arm_id="test", drug_name="B",
                baseline_p_at_10=0.6, ablated_p_at_10=0.4,
                baseline_r_at_10=0.5, ablated_r_at_10=0.3,
            ),
        ]
        agg = aggregate_ablation_results(results)
        assert agg.n_drugs == 2
        assert agg.mean_delta_p_at_10 == pytest.approx(-0.2)
        assert agg.mean_delta_r_at_10 == pytest.approx(-0.2)

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_ablation_results([])


# ------------------------------------------------------------------
# Heatmap data tests
# ------------------------------------------------------------------

class TestHeatmapData:
    def test_grid_dimensions(self, sample_cached_scores, sample_ground_truth):
        cells = build_heatmap_data(
            sample_cached_scores, sample_ground_truth,
            metric="precision_at_10",
        )
        # 11 weight values x 13 threshold values = 143 cells
        assert len(cells) == 11 * 13
        assert all(isinstance(c, HeatmapCell) for c in cells)

    def test_custom_grid(self, sample_cached_scores, sample_ground_truth):
        cells = build_heatmap_data(
            sample_cached_scores, sample_ground_truth,
            metric="recall_at_10",
            w_start=0.0, w_end=0.5, w_step=0.25,
            t_start=0.3, t_end=0.5, t_step=0.1,
        )
        # 3 weights x 3 thresholds = 9 cells
        assert len(cells) == 9

    def test_invalid_metric_raises(self, sample_cached_scores, sample_ground_truth):
        with pytest.raises(ValueError, match="metric must be one of"):
            build_heatmap_data(
                sample_cached_scores, sample_ground_truth,
                metric="invalid_metric",
            )

    def test_values_bounded(self, sample_cached_scores, sample_ground_truth):
        cells = build_heatmap_data(
            sample_cached_scores, sample_ground_truth,
            metric="precision_at_1",
        )
        for c in cells:
            assert 0.0 <= c.value <= 1.0
