"""Tests for Phase 8 modules: cache, accuracy, difficulty, orchestrator.

All tests use mocked data -- no live API calls, no Qdrant, no LLMs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.accuracy import (
    AggregateMetrics,
    DrugMetrics,
    aggregate_metrics,
    evaluate_prediction,
)
from src.experiment.cache import ResultCache
from src.experiment.difficulty import (
    ClassifiedDrug,
    classify_batch,
    classify_difficulty,
)
from src.experiment.runner import ArmResult
from src.schemas.prediction import (
    DrugDifficulty,
    DrugDiseasePrediction,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ======================================================================
# Fixtures
# ======================================================================

def _make_prediction(
    drug_name: str = "aspirin",
    diseases: list[tuple[str, float, bool]] | None = None,
) -> DrugDiseasePrediction:
    """Build a prediction with given (disease_name, confidence, predicted) tuples."""
    if diseases is None:
        diseases = [
            ("colorectal cancer", 0.9, True),
            ("heart disease", 0.7, True),
            ("diabetes", 0.3, False),
        ]
    associations = []
    for name, conf, pred in diseases:
        associations.append(
            ScoredAssociation(
                disease_name=name,
                predicted=pred,
                confidence=conf,
                evidence_chains=[
                    EvidenceChain(
                        edges=[
                            MechanisticEdge(
                                source_entity=drug_name,
                                target_entity="TARGET",
                                relationship="associated_with",
                                evidence_snippet="test evidence",
                            )
                        ],
                        summary=f"{drug_name} -> {name}",
                        confidence=conf,
                    )
                ],
            )
        )
    return DrugDiseasePrediction(
        drug_name=drug_name,
        drug_chembl_id="CHEMBL25",
        associations=associations,
        reasoning="Test reasoning",
    )


def _make_arm_result(
    drug_name: str = "aspirin",
    arm_id: str = "pipeline-gpt-oss",
    model_id: str = "groq:openai/gpt-oss-120b",
    prediction: DrugDiseasePrediction | None = None,
    error: str | None = None,
) -> ArmResult:
    """Build a minimal ArmResult."""
    return ArmResult(
        arm_id=arm_id,
        drug_name=drug_name,
        model_id=model_id,
        prediction=prediction,
        wall_clock_seconds=1.5,
        error=error,
    )


# ======================================================================
# ResultCache tests
# ======================================================================

class TestResultCache:
    """Tests for the hash-based result cache."""

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        pred = _make_prediction()
        result = _make_arm_result(prediction=pred)

        key = cache.put(result)
        assert key  # non-empty hex string
        assert cache.has("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")

        loaded = cache.get("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")
        assert loaded is not None
        assert loaded.drug_name == "aspirin"
        assert loaded.arm_id == "pipeline-gpt-oss"
        assert loaded.prediction is not None
        assert loaded.prediction.drug_name == "aspirin"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        assert cache.get("unknown", "arm", "model") is None

    def test_has_returns_false_for_missing(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        assert not cache.has("aspirin", "pipeline-gpt-oss", "model")

    def test_put_does_not_overwrite(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        pred = _make_prediction()
        r1 = _make_arm_result(prediction=pred)
        key1 = cache.put(r1)

        # Second put with same key should not overwrite
        r2 = _make_arm_result(prediction=pred, error="should not be saved")
        key2 = cache.put(r2)
        assert key1 == key2

        loaded = cache.get("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")
        assert loaded is not None
        assert loaded.error is None  # original value preserved

    def test_load_all(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        pred = _make_prediction()

        cache.put(_make_arm_result(prediction=pred, arm_id="arm-a"))
        cache.put(_make_arm_result(prediction=pred, arm_id="arm-b"))

        all_results = cache.load_all()
        assert len(all_results) == 2

    def test_count(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        assert cache.count() == 0

        pred = _make_prediction()
        cache.put(_make_arm_result(prediction=pred))
        assert cache.count() == 1

    def test_clear(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        pred = _make_prediction()
        cache.put(_make_arm_result(prediction=pred))
        assert cache.count() == 1

        removed = cache.clear()
        assert removed == 1
        assert cache.count() == 0

    def test_corrupt_cache_entry_returns_none(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        # Write a corrupt JSON file
        key = ResultCache.key("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")
        corrupt_path = (tmp_path / "results" / f"{key}.json")
        corrupt_path.write_text("not valid json {{{", encoding="utf-8")

        result = cache.get("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")
        assert result is None

    def test_result_with_none_prediction(self, tmp_path: Path) -> None:
        cache = ResultCache(tmp_path / "results")
        result = _make_arm_result(prediction=None, error="Model failed")
        cache.put(result)

        loaded = cache.get("aspirin", "pipeline-gpt-oss", "groq:openai/gpt-oss-120b")
        assert loaded is not None
        assert loaded.prediction is None
        assert loaded.error == "Model failed"


# ======================================================================
# Accuracy metrics tests
# ======================================================================

class TestAccuracyMetrics:
    """Tests for precision, recall, and AUC computation."""

    def test_perfect_p_at_1(self) -> None:
        pred = _make_prediction(
            diseases=[("colorectal cancer", 0.9, True)],
        )
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.precision_at_1 == 1.0

    def test_zero_p_at_1(self) -> None:
        pred = _make_prediction(
            diseases=[("unknown disease", 0.9, True)],
        )
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.precision_at_1 == 0.0

    def test_precision_at_10_partial(self) -> None:
        # 3 predictions, 1 correct out of 3
        pred = _make_prediction(
            diseases=[
                ("colorectal cancer", 0.9, True),
                ("wrong1", 0.7, True),
                ("wrong2", 0.5, True),
            ],
        )
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        # P@10 = 1/3 (only 3 predictions, all within top 10)
        assert abs(m.precision_at_10 - 1 / 3) < 0.01

    def test_recall_at_1(self) -> None:
        pred = _make_prediction(
            diseases=[
                ("colorectal cancer", 0.9, True),
                ("heart disease", 0.7, True),
            ],
        )
        # 2 ground truth diseases, top-1 only captures 1
        gt = {"colorectal cancer", "heart disease"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.recall_at_1 == 0.5

    def test_recall_at_10_perfect(self) -> None:
        pred = _make_prediction(
            diseases=[
                ("colorectal cancer", 0.9, True),
                ("heart disease", 0.7, True),
            ],
        )
        gt = {"colorectal cancer", "heart disease"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.recall_at_10 == 1.0

    def test_empty_prediction_returns_zero(self) -> None:
        pred = _make_prediction(diseases=[])
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.precision_at_1 == 0.0
        assert m.recall_at_10 == 0.0

    def test_empty_ground_truth_returns_zero_recall(self) -> None:
        pred = _make_prediction(
            diseases=[("colorectal cancer", 0.9, True)],
        )
        m = evaluate_prediction(pred, set(), arm_id="test")
        assert m.recall_at_1 == 0.0
        assert m.recall_at_10 == 0.0

    def test_roc_auc_perfect(self) -> None:
        # All positives ranked above negatives -> AUC = 1.0
        pred = _make_prediction(
            diseases=[
                ("disease_a", 0.9, True),  # positive
                ("disease_b", 0.8, True),  # positive
                ("disease_c", 0.2, True),  # negative
                ("disease_d", 0.1, True),  # negative
            ],
        )
        gt = {"disease_a", "disease_b"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.roc_auc is not None
        assert abs(m.roc_auc - 1.0) < 0.01

    def test_roc_auc_none_for_single_class(self) -> None:
        # All predictions are positive (no negatives) -> AUC undefined
        pred = _make_prediction(
            diseases=[("disease_a", 0.9, True)],
        )
        gt = {"disease_a"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.roc_auc is None

    def test_case_insensitive_matching(self) -> None:
        pred = _make_prediction(
            diseases=[("Colorectal Cancer", 0.9, True)],
        )
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert m.precision_at_1 == 1.0

    def test_difficulty_stored_in_metrics(self) -> None:
        pred = _make_prediction(
            diseases=[("colorectal cancer", 0.9, True)],
        )
        gt = {"colorectal cancer"}
        m = evaluate_prediction(pred, gt, arm_id="test", difficulty=DrugDifficulty.HARD)
        assert m.difficulty == DrugDifficulty.HARD

    def test_true_positives_at_10(self) -> None:
        pred = _make_prediction(
            diseases=[
                ("colorectal cancer", 0.9, True),
                ("wrong", 0.5, True),
                ("heart disease", 0.3, True),
            ],
        )
        gt = {"colorectal cancer", "heart disease"}
        m = evaluate_prediction(pred, gt, arm_id="test")
        assert "colorectal cancer" in [d.lower() for d in m.true_positives_at_10] or \
               "Colorectal Cancer" in m.true_positives_at_10


class TestAggregateMetrics:
    """Tests for aggregate_metrics over multiple drugs."""

    def test_aggregate_two_drugs(self) -> None:
        m1 = DrugMetrics(
            drug_name="aspirin",
            arm_id="test",
            precision_at_1=1.0,
            precision_at_10=0.5,
            recall_at_1=1.0,
            recall_at_10=0.5,
            roc_auc=0.8,
        )
        m2 = DrugMetrics(
            drug_name="ibuprofen",
            arm_id="test",
            precision_at_1=0.0,
            precision_at_10=0.3,
            recall_at_1=0.0,
            recall_at_10=0.7,
            roc_auc=0.6,
        )
        agg = aggregate_metrics([m1, m2], "test")
        assert agg.n_drugs == 2
        assert abs(agg.mean_precision_at_1 - 0.5) < 0.01
        assert abs(agg.mean_precision_at_10 - 0.4) < 0.01
        assert agg.mean_roc_auc is not None
        assert abs(agg.mean_roc_auc - 0.7) < 0.01

    def test_aggregate_empty_list(self) -> None:
        agg = aggregate_metrics([], "test")
        assert agg.n_drugs == 0
        assert agg.mean_precision_at_1 == 0.0

    def test_aggregate_by_difficulty(self) -> None:
        m1 = DrugMetrics(
            drug_name="aspirin",
            arm_id="test",
            difficulty=DrugDifficulty.EASY,
            precision_at_10=0.8,
        )
        m2 = DrugMetrics(
            drug_name="ibuprofen",
            arm_id="test",
            difficulty=DrugDifficulty.HARD,
            precision_at_10=0.1,
        )
        agg = aggregate_metrics([m1, m2], "test")
        assert "easy" in agg.by_difficulty
        assert "hard" in agg.by_difficulty
        assert agg.by_difficulty["easy"].n_drugs == 1
        assert abs(agg.by_difficulty["easy"].mean_precision_at_10 - 0.8) < 0.01
        assert abs(agg.by_difficulty["hard"].mean_precision_at_10 - 0.1) < 0.01


# ======================================================================
# Difficulty classifier tests
# ======================================================================

class TestDifficultyClassifier:
    """Tests for difficulty classification based on baseline-gpt5-nosearch P@10."""

    def test_easy_classification(self) -> None:
        # P@10 > 0.4 -> EASY
        pred = _make_prediction(
            diseases=[
                ("d1", 0.9, True),
                ("d2", 0.8, True),
                ("d3", 0.7, True),
                ("d4", 0.6, True),
                ("d5", 0.5, True),
            ],
        )
        result = _make_arm_result(
            arm_id="baseline-gpt5-nosearch",
            prediction=pred,
        )
        gt = {"d1", "d2", "d3", "d4", "d5"}
        diff = classify_difficulty(result, gt)
        assert diff == DrugDifficulty.EASY

    def test_hard_classification(self) -> None:
        # All predictions wrong -> P@10 = 0 -> HARD
        pred = _make_prediction(
            diseases=[("wrong1", 0.9, True), ("wrong2", 0.5, True)],
        )
        result = _make_arm_result(
            arm_id="baseline-gpt5-nosearch",
            prediction=pred,
        )
        gt = {"colorectal cancer", "heart disease"}
        diff = classify_difficulty(result, gt)
        assert diff == DrugDifficulty.HARD

    def test_medium_classification(self) -> None:
        # 2/10 correct -> P@10 = 0.2 -> MEDIUM
        diseases = [
            ("d1", 0.9, True),  # correct
            ("d2", 0.8, True),  # correct
            *[(f"wrong{i}", 0.5 - i * 0.01, True) for i in range(8)],
        ]
        pred = _make_prediction(diseases=diseases)
        result = _make_arm_result(
            arm_id="baseline-gpt5-nosearch",
            prediction=pred,
        )
        gt = {"d1", "d2"}
        diff = classify_difficulty(result, gt)
        assert diff == DrugDifficulty.MEDIUM

    def test_none_prediction_returns_hard(self) -> None:
        result = _make_arm_result(
            arm_id="baseline-gpt5-nosearch",
            prediction=None,
            error="Model errored",
        )
        diff = classify_difficulty(result, {"disease_a"})
        assert diff == DrugDifficulty.HARD

    def test_classify_batch(self) -> None:
        pred_easy = _make_prediction(
            drug_name="drug_easy",
            diseases=[("d1", 0.9, True), ("d2", 0.8, True), ("d3", 0.7, True)],
        )
        pred_hard = _make_prediction(
            drug_name="drug_hard",
            diseases=[("wrong", 0.9, True)],
        )
        nosearch = {
            "drug_easy": _make_arm_result(
                drug_name="drug_easy",
                arm_id="baseline-gpt5-nosearch",
                prediction=pred_easy,
            ),
            "drug_hard": _make_arm_result(
                drug_name="drug_hard",
                arm_id="baseline-gpt5-nosearch",
                prediction=pred_hard,
            ),
        }
        gts = {
            "drug_easy": {"d1", "d2", "d3"},
            "drug_hard": {"real_disease"},
        }
        classified = classify_batch(nosearch, gts)
        assert classified["drug_easy"].difficulty == DrugDifficulty.EASY
        assert classified["drug_hard"].difficulty == DrugDifficulty.HARD


# ======================================================================
# Orchestrator tests (mocked)
# ======================================================================

class TestExperimentRunner:
    """Test ExperimentRunner with fully mocked services."""

    @pytest.fixture
    def mock_settings(self) -> Settings:
        from src.config.settings import Settings
        return Settings(_env_file=None, entrez_email="test@example.com")

    @pytest.mark.asyncio(loop_scope="session")
    async def test_runner_creates_cache_dir(self, tmp_path: Path) -> None:
        from src.experiment.orchestrator import ExperimentRunner

        cache_dir = tmp_path / "test_cache"
        runner = ExperimentRunner(
            cache_dir=str(cache_dir),
            resume=False,
        )
        assert cache_dir.exists()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_experiment_report_summary_table(self) -> None:
        from src.experiment.orchestrator import ExperimentReport

        report = ExperimentReport(
            n_drugs=10,
            n_arms=2,
            wall_clock_seconds=100.0,
            arm_aggregates={
                "arm-a": AggregateMetrics(
                    arm_id="arm-a",
                    n_drugs=10,
                    mean_precision_at_1=0.5,
                    mean_precision_at_10=0.4,
                    mean_recall_at_1=0.3,
                    mean_recall_at_10=0.6,
                    mean_roc_auc=0.75,
                ),
                "arm-b": AggregateMetrics(
                    arm_id="arm-b",
                    n_drugs=10,
                    mean_precision_at_1=0.7,
                    mean_precision_at_10=0.6,
                    mean_recall_at_1=0.5,
                    mean_recall_at_10=0.8,
                    mean_roc_auc=None,
                ),
            },
        )
        table = report.summary_table()
        assert "arm-a" in table
        assert "arm-b" in table
        assert "n/a" in table  # arm-b has no AUC
        assert "0.500" in table  # arm-a P@1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_difficulty_summary(self) -> None:
        from src.experiment.orchestrator import ExperimentReport

        report = ExperimentReport(
            n_drugs=3,
            n_arms=1,
            wall_clock_seconds=10.0,
            difficulty_map={
                "drug_a": ClassifiedDrug("drug_a", DrugDifficulty.EASY, 0.5),
                "drug_b": ClassifiedDrug("drug_b", DrugDifficulty.HARD, 0.0),
                "drug_c": ClassifiedDrug("drug_c", DrugDifficulty.MEDIUM, 0.2),
            },
        )
        summary = report.difficulty_summary()
        assert "easy=1" in summary
        assert "medium=1" in summary
        assert "hard=1" in summary

    @pytest.mark.asyncio(loop_scope="session")
    async def test_drug_result_dataclass(self) -> None:
        from src.experiment.orchestrator import DrugResult

        dr = DrugResult(drug_name="aspirin")
        assert dr.drug_name == "aspirin"
        assert dr.ground_truth_diseases == set()
        assert dr.arm_results == {}
        assert dr.difficulty is None


# ======================================================================
# CLI tests (argument parsing only)
# ======================================================================

class TestCLI:
    """Tests for the main.py argument parser."""

    def test_default_args(self) -> None:
        from src.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.drugs is None
        assert args.arms is None
        assert args.cache_dir == ".cache/results"
        assert args.no_resume is False
        assert args.log_level == "INFO"

    def test_custom_args(self) -> None:
        from src.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--drugs", "10",
            "--arms", "pipeline-gpt-oss", "baseline-gpt5-search",
            "--cache-dir", "/tmp/test",
            "--no-resume",
            "--log-level", "DEBUG",
        ])
        assert args.drugs == 10
        assert args.arms == ["pipeline-gpt-oss", "baseline-gpt5-search"]
        assert args.cache_dir == "/tmp/test"
        assert args.no_resume is True
        assert args.log_level == "DEBUG"

    def test_validate_arms_valid(self) -> None:
        from src.main import _validate_arms
        result = _validate_arms(["pipeline-gpt-oss", "baseline-gpt5-search"])
        assert result == ["pipeline-gpt-oss", "baseline-gpt5-search"]

    def test_validate_arms_none(self) -> None:
        from src.main import _validate_arms
        assert _validate_arms(None) is None

    def test_validate_arms_invalid_exits(self) -> None:
        from src.main import _validate_arms
        with pytest.raises(SystemExit):
            _validate_arms(["nonexistent-arm"])

    def test_eval_flags_default(self) -> None:
        from src.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.skip_fn_analysis is False
        assert args.run_evidence_quality is False
        assert args.skip_sensitivity is False

    def test_eval_flags_set(self) -> None:
        from src.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--skip-fn-analysis",
            "--run-evidence-quality",
            "--skip-sensitivity",
        ])
        assert args.skip_fn_analysis is True
        assert args.run_evidence_quality is True
        assert args.skip_sensitivity is True


# ======================================================================
# Eval wiring tests
# ======================================================================

class TestEvalWiring:
    """Test that evaluation modules are wired into the orchestrator correctly."""

    def _build_drug_result(
        self,
        drug_name: str = "aspirin",
        arm_ids: list[str] | None = None,
        gt_diseases: set[str] | None = None,
    ) -> "DrugResult":
        """Build a DrugResult with predictions for the given arms."""
        from src.experiment.orchestrator import DrugResult
        arm_ids = arm_ids or ["pipeline-gpt-oss"]
        gt_diseases = gt_diseases or {"colorectal cancer", "heart disease"}

        dr = DrugResult(drug_name=drug_name)
        dr.ground_truth_diseases = gt_diseases

        for arm_id in arm_ids:
            pred = _make_prediction(
                drug_name=drug_name,
                diseases=[
                    ("colorectal cancer", 0.9, True),
                    ("heart disease", 0.7, True),
                    ("diabetes", 0.3, False),
                ],
            )
            dr.arm_results[arm_id] = _make_arm_result(
                drug_name=drug_name,
                arm_id=arm_id,
                prediction=pred,
            )
        return dr

    def test_compute_fn_analysis(self, tmp_path: Path) -> None:
        from src.experiment.arms import PIPELINE_ARMS
        from src.experiment.orchestrator import ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
        )
        dr = self._build_drug_result(arm_ids=["pipeline-gpt-oss"])
        drug_results = {"aspirin": dr}
        arms = {"pipeline-gpt-oss": PIPELINE_ARMS["pipeline-gpt-oss"]}

        fn_summaries = runner._compute_fn_analysis(drug_results, arms)

        assert "pipeline-gpt-oss" in fn_summaries
        summary = fn_summaries["pipeline-gpt-oss"]
        assert summary.arm_id == "pipeline-gpt-oss"
        assert summary.n_drugs == 1
        assert summary.total_ground_truth >= 1
        # diabetes was predicted=False -> at least 0 FN depending on GT
        # GT is {colorectal cancer, heart disease}, both predicted=True
        assert summary.total_true_positives == 2
        assert summary.total_false_negatives == 0

    def test_compute_fn_analysis_with_false_negatives(self, tmp_path: Path) -> None:
        from src.experiment.arms import PIPELINE_ARMS
        from src.experiment.orchestrator import DrugResult, ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
        )
        # Ground truth includes "lung cancer" which is NOT predicted
        dr = DrugResult(drug_name="aspirin")
        dr.ground_truth_diseases = {"colorectal cancer", "lung cancer"}
        pred = _make_prediction(
            diseases=[
                ("colorectal cancer", 0.9, True),
                ("heart disease", 0.7, True),
            ],
        )
        dr.arm_results["pipeline-gpt-oss"] = _make_arm_result(
            arm_id="pipeline-gpt-oss", prediction=pred,
        )
        arms = {"pipeline-gpt-oss": PIPELINE_ARMS["pipeline-gpt-oss"]}
        fn_summaries = runner._compute_fn_analysis({"aspirin": dr}, arms)

        summary = fn_summaries["pipeline-gpt-oss"]
        assert summary.total_false_negatives == 1
        assert summary.total_true_positives == 1
        assert len(summary.category_counts) > 0

    def test_compute_sensitivity(self, tmp_path: Path) -> None:
        from src.experiment.arms import PIPELINE_ARMS
        from src.experiment.orchestrator import ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
        )
        dr = self._build_drug_result(arm_ids=["pipeline-gpt-oss"])
        drug_results = {"aspirin": dr}
        arms = {"pipeline-gpt-oss": PIPELINE_ARMS["pipeline-gpt-oss"]}

        heatmap, sweeps = runner._compute_sensitivity(drug_results, arms)

        # Heatmap should have cells (11 weight x 13 threshold = 143 cells)
        assert len(heatmap) > 0
        assert all(isinstance(cell.value, float) for cell in heatmap)

        # Weight sweeps should exist for the arm
        assert "pipeline-gpt-oss" in sweeps
        assert len(sweeps["pipeline-gpt-oss"]) == 1  # 1 drug
        sweep = sweeps["pipeline-gpt-oss"][0]
        assert sweep.drug_name == "aspirin"
        assert len(sweep.points) == 11  # 0.0 to 1.0 step 0.1

    def test_compute_sensitivity_empty_predictions(self, tmp_path: Path) -> None:
        from src.experiment.arms import PIPELINE_ARMS
        from src.experiment.orchestrator import DrugResult, ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
        )
        dr = DrugResult(drug_name="aspirin")
        dr.arm_results["pipeline-gpt-oss"] = _make_arm_result(
            arm_id="pipeline-gpt-oss", prediction=None, error="failed",
        )
        arms = {"pipeline-gpt-oss": PIPELINE_ARMS["pipeline-gpt-oss"]}
        heatmap, sweeps = runner._compute_sensitivity({"aspirin": dr}, arms)

        assert heatmap == []
        assert sweeps == {}

    def test_experiment_report_has_eval_fields(self) -> None:
        from src.experiment.orchestrator import ExperimentReport

        report = ExperimentReport(
            n_drugs=5,
            n_arms=2,
            wall_clock_seconds=10.0,
        )
        # New fields should have sensible defaults
        assert report.fn_summaries == {}
        assert report.evidence_quality == {}
        assert report.heatmap_data == []
        assert report.weight_sweeps == {}

    def test_experiment_report_with_eval_data(self) -> None:
        from src.evaluation.evidence_quality import EvidenceQualityMetrics
        from src.evaluation.false_negatives import AggregateFNSummary
        from src.evaluation.sensitivity import HeatmapCell, WeightSweepResult
        from src.experiment.orchestrator import ExperimentReport

        fn = AggregateFNSummary(arm_id="arm-a", n_drugs=5)
        eq = EvidenceQualityMetrics(
            citation_validity_rate=0.8,
            mean_chain_depth=3.0,
            chain_verifiability_score=0.7,
            evidence_relevance=0.6,
            mechanistic_specificity=0.5,
        )
        heatmap = [HeatmapCell(w_retrieval=0.5, threshold=0.5, value=0.4)]

        report = ExperimentReport(
            n_drugs=5,
            n_arms=1,
            wall_clock_seconds=10.0,
            fn_summaries={"arm-a": fn},
            evidence_quality={"arm-a": eq},
            heatmap_data=heatmap,
        )
        assert report.fn_summaries["arm-a"].n_drugs == 5
        assert report.evidence_quality["arm-a"].citation_validity_rate == 0.8
        assert len(report.heatmap_data) == 1

    def test_runner_eval_flags_disabled(self, tmp_path: Path) -> None:
        from src.experiment.orchestrator import ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
            run_fn_analysis=False,
            run_evidence_quality=False,
            run_sensitivity=False,
        )
        assert runner._run_fn_analysis is False
        assert runner._run_evidence_quality is False
        assert runner._run_sensitivity is False

    def test_runner_eval_flags_enabled(self, tmp_path: Path) -> None:
        from src.experiment.orchestrator import ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
            run_fn_analysis=True,
            run_evidence_quality=True,
            run_sensitivity=True,
        )
        assert runner._run_fn_analysis is True
        assert runner._run_evidence_quality is True
        assert runner._run_sensitivity is True

    def test_compute_fn_multiple_arms(self, tmp_path: Path) -> None:
        from src.experiment.arms import BASELINE_ARMS, PIPELINE_ARMS
        from src.experiment.orchestrator import ExperimentRunner

        runner = ExperimentRunner(
            cache_dir=str(tmp_path / "cache"),
            resume=False,
        )
        dr = self._build_drug_result(
            arm_ids=["pipeline-gpt-oss", "baseline-gpt5-search"],
        )
        drug_results = {"aspirin": dr}
        arms = {
            "pipeline-gpt-oss": PIPELINE_ARMS["pipeline-gpt-oss"],
            "baseline-gpt5-search": BASELINE_ARMS["baseline-gpt5-search"],
        }
        fn_summaries = runner._compute_fn_analysis(drug_results, arms)
        assert len(fn_summaries) == 2
        assert "pipeline-gpt-oss" in fn_summaries
        assert "baseline-gpt5-search" in fn_summaries
