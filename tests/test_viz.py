"""Tests for src.viz.plots and src.viz.report."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.accuracy import AggregateMetrics, DrugMetrics, aggregate_metrics
from src.evaluation.evidence_quality import EvidenceQualityMetrics
from src.evaluation.false_negatives import AggregateFNSummary, FalseNegativeCategory
from src.evaluation.sensitivity import (
    AblationType,
    AggregateAblationResult,
    HeatmapCell,
)
from src.experiment.difficulty import ClassifiedDrug
from src.experiment.orchestrator import DrugResult, ExperimentReport
from src.experiment.runner import ArmResult
from src.schemas.prediction import (
    DrugDifficulty,
    DrugDiseasePrediction,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)
from src.viz.plots import (
    generate_all_plots,
    plot_ablation_delta,
    plot_cost_bars,
    plot_evidence_radar,
    plot_fn_distribution,
    plot_precision_recall_bars,
    plot_pr_scatter,
    plot_roc_curves,
    plot_score_distribution,
    plot_sensitivity_heatmap,
)
from src.viz.report import (
    generate_report,
    write_report,
    _paired_wilcoxon,
    _bonferroni_correction,
)


# ------------------------------------------------------------------
# Fixtures (shared with test_export.py -- minimal version)
# ------------------------------------------------------------------

def _make_prediction(drug_name: str, diseases: list[str]) -> DrugDiseasePrediction:
    associations = []
    for d in diseases:
        chain = EvidenceChain(
            edges=[MechanisticEdge(
                source_entity=drug_name, target_entity=d,
                relationship=EdgeType.ASSOCIATED_WITH,
                evidence_snippet="Test evidence.",
            )],
            summary=f"{drug_name} treats {d}",
            confidence=0.85,
        )
        associations.append(ScoredAssociation(
            disease_name=d, predicted=True, confidence=0.85,
            evidence_chains=[chain],
        ))
    return DrugDiseasePrediction(
        drug_name=drug_name, associations=associations, reasoning="Test.",
    )


def _make_arm_result(drug_name: str, arm_id: str, model_id: str) -> ArmResult:
    return ArmResult(
        arm_id=arm_id, drug_name=drug_name, model_id=model_id,
        prediction=_make_prediction(drug_name, ["disease A", "disease B"]),
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        wall_clock_seconds=1.5,
    )


def _make_drug_metrics(drug_name: str, arm_id: str) -> DrugMetrics:
    return DrugMetrics(
        drug_name=drug_name, arm_id=arm_id,
        difficulty=DrugDifficulty.MEDIUM,
        n_ground_truth=5, n_predicted=3,
        precision_at_1=1.0, precision_at_10=0.6,
        recall_at_1=0.2, recall_at_10=0.6,
        roc_auc=0.75,
        true_positives_at_10=["disease A", "disease B"],
    )


@pytest.fixture()
def mini_report() -> ExperimentReport:
    drugs = ["aspirin", "ibuprofen"]
    arm_ids = ["pipeline-gpt-oss", "baseline-gpt5-nosearch"]
    model_ids = ["groq:openai/gpt-oss-120b", "openai:gpt-5.2-2025-12-11"]

    drug_results: dict[str, DrugResult] = {}
    all_metrics: list[DrugMetrics] = []

    for drug in drugs:
        arm_results = {}
        metrics = {}
        for arm_id, model_id in zip(arm_ids, model_ids):
            arm_results[arm_id] = _make_arm_result(drug, arm_id, model_id)
            dm = _make_drug_metrics(drug, arm_id)
            metrics[arm_id] = dm
            all_metrics.append(dm)
        drug_results[drug] = DrugResult(
            drug_name=drug, chembl_id=f"CHEMBL_{drug.upper()}",
            pubchem_cid=12345,
            ground_truth_diseases={"disease A", "disease B", "disease X"},
            difficulty=DrugDifficulty.MEDIUM,
            arm_results=arm_results, metrics=metrics,
            n_evidence_docs=10, n_chunks_indexed=30,
        )

    arm_aggregates = {}
    for arm_id in arm_ids:
        arm_metrics = [m for m in all_metrics if m.arm_id == arm_id]
        arm_aggregates[arm_id] = aggregate_metrics(arm_metrics, arm_id)

    difficulty_map = {
        drug: ClassifiedDrug(drug_name=drug, difficulty=DrugDifficulty.MEDIUM, reference_p_at_10=0.4)
        for drug in drugs
    }

    return ExperimentReport(
        n_drugs=2, n_arms=2, wall_clock_seconds=5.678,
        drug_results=drug_results, difficulty_map=difficulty_map,
        arm_aggregates=arm_aggregates, cached_results=1,
    )


# ------------------------------------------------------------------
# Plot tests
# ------------------------------------------------------------------

class TestPrecisionRecallBars:
    def test_returns_figure(self, mini_report):
        fig = plot_precision_recall_bars(mini_report.arm_aggregates)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self, mini_report, tmp_path):
        fig = plot_precision_recall_bars(mini_report.arm_aggregates)
        path = tmp_path / "pr_bars.png"
        fig.savefig(path)
        import matplotlib.pyplot as plt
        plt.close(fig)
        assert path.exists()
        assert path.stat().st_size > 0


class TestRocCurves:
    def test_returns_figure(self, mini_report):
        arm_dm = {}
        for dr in mini_report.drug_results.values():
            for arm_id, dm in dr.metrics.items():
                arm_dm.setdefault(arm_id, []).append(dm)
        fig = plot_roc_curves(arm_dm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPrScatter:
    def test_returns_figure(self, mini_report):
        arm_dm = {}
        for dr in mini_report.drug_results.values():
            for arm_id, dm in dr.metrics.items():
                arm_dm.setdefault(arm_id, []).append(dm)
        fig = plot_pr_scatter(arm_dm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestEvidenceRadar:
    def test_returns_figure(self):
        eq = EvidenceQualityMetrics(
            citation_validity_rate=0.8, mean_chain_depth=2.5,
            chain_verifiability_score=0.6, evidence_relevance=0.7,
            mechanistic_specificity=0.5,
        )
        fig = plot_evidence_radar({"arm-a": eq, "arm-b": eq})
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestCostBars:
    def test_returns_figure(self):
        usage = {
            "arm-a": {"input_tokens": 1000, "output_tokens": 500},
            "arm-b": {"input_tokens": 2000, "output_tokens": 800},
        }
        fig = plot_cost_bars(usage)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSensitivityHeatmap:
    def test_returns_figure_with_data(self):
        cells = [
            HeatmapCell(w_retrieval=0.0, threshold=0.3, value=0.5),
            HeatmapCell(w_retrieval=0.0, threshold=0.5, value=0.6),
            HeatmapCell(w_retrieval=0.5, threshold=0.3, value=0.4),
            HeatmapCell(w_retrieval=0.5, threshold=0.5, value=0.7),
        ]
        fig = plot_sensitivity_heatmap(cells)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_figure_empty(self):
        fig = plot_sensitivity_heatmap([])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestFnDistribution:
    def test_returns_figure(self):
        s = AggregateFNSummary(
            arm_id="arm-a", n_drugs=5,
            total_ground_truth=100, total_true_positives=60,
            total_false_negatives=40,
            category_counts={
                "no_evidence_found": 10, "low_retrieval_score": 8,
                "low_llm_confidence": 7, "below_threshold": 5,
                "not_in_candidates": 6, "name_mismatch": 4,
            },
        )
        fig = plot_fn_distribution({"arm-a": s})
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestScoreDistribution:
    def test_returns_figure(self, mini_report):
        arm_dm = {}
        for dr in mini_report.drug_results.values():
            for arm_id, dm in dr.metrics.items():
                arm_dm.setdefault(arm_id, []).append(dm)
        fig = plot_score_distribution(arm_dm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestAblationDelta:
    def test_returns_figure(self):
        results = [
            AggregateAblationResult(
                ablation_type=AblationType.NO_DGIDB, n_drugs=10,
                mean_delta_p_at_10=-0.05,
            ),
            AggregateAblationResult(
                ablation_type=AblationType.DENSE_ONLY, n_drugs=10,
                mean_delta_p_at_10=-0.12,
            ),
            AggregateAblationResult(
                ablation_type=AblationType.NO_TOOLS, n_drugs=10,
                mean_delta_p_at_10=0.01,
            ),
        ]
        fig = plot_ablation_delta(results)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestGenerateAllPlots:
    def test_produces_core_plots(self, mini_report, tmp_path):
        paths = generate_all_plots(mini_report, output_dir=tmp_path)
        # At minimum: precision_recall_bars, roc_curves, pr_curves, cost_bars, score_distribution
        assert len(paths) >= 5
        for p in paths.values():
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0

    def test_creates_output_dir(self, mini_report, tmp_path):
        new_dir = tmp_path / "sub" / "plots"
        generate_all_plots(mini_report, output_dir=new_dir)
        assert new_dir.exists()


# ------------------------------------------------------------------
# Report tests
# ------------------------------------------------------------------

class TestPairedWilcoxon:
    def test_identical_values(self):
        stat, p = _paired_wilcoxon([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert p == 1.0

    def test_different_values(self):
        stat, p = _paired_wilcoxon(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7],
            [0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3],
        )
        assert p < 0.05

    def test_too_few_samples(self):
        import math
        stat, p = _paired_wilcoxon([0.5], [0.3])
        assert math.isnan(p)


class TestBonferroniCorrection:
    def test_basic(self):
        corrected = _bonferroni_correction([0.01, 0.04, 0.05])
        assert corrected[0] == pytest.approx(0.03)
        assert corrected[1] == pytest.approx(0.12)
        assert corrected[2] == pytest.approx(0.15)

    def test_caps_at_one(self):
        corrected = _bonferroni_correction([0.5, 0.5])
        assert corrected[0] == 1.0
        assert corrected[1] == 1.0


class TestGenerateReport:
    def test_returns_markdown_string(self, mini_report):
        md = generate_report(mini_report)
        assert isinstance(md, str)
        assert "# Drug-Disease Prediction Experiment Report" in md

    def test_contains_accuracy_table(self, mini_report):
        md = generate_report(mini_report)
        assert "## Accuracy Metrics (All Drugs)" in md
        assert "pipeline-gpt-oss" in md
        assert "baseline-gpt5-nosearch" in md

    def test_contains_difficulty_breakdown(self, mini_report):
        md = generate_report(mini_report)
        assert "## Accuracy by Difficulty" in md
        assert "### Medium" in md

    def test_contains_statistical_tests(self, mini_report):
        md = generate_report(mini_report)
        assert "## Statistical Significance" in md

    def test_contains_evidence_quality_when_provided(self, mini_report):
        eq = EvidenceQualityMetrics(
            citation_validity_rate=0.8, mean_chain_depth=2.5,
            chain_verifiability_score=0.6, evidence_relevance=0.7,
            mechanistic_specificity=0.5,
        )
        md = generate_report(mini_report, evidence_quality={"pipeline-gpt-oss": eq})
        assert "## Evidence Quality" in md

    def test_contains_fn_summary_when_provided(self, mini_report):
        s = AggregateFNSummary(
            arm_id="pipeline-gpt-oss", n_drugs=2,
            total_ground_truth=10, total_true_positives=6,
            total_false_negatives=4,
            category_counts={"name_mismatch": 2, "no_evidence_found": 2},
        )
        md = generate_report(mini_report, fn_summaries={"pipeline-gpt-oss": s})
        assert "## False Negative Analysis" in md

    def test_contains_plot_references(self, mini_report, tmp_path):
        plot_paths = {
            "precision_recall_bars": tmp_path / "precision_recall_bars.png",
            "roc_curves": tmp_path / "roc_curves.png",
        }
        md = generate_report(mini_report, plot_paths=plot_paths)
        assert "## Plots" in md
        assert "precision_recall_bars.png" in md


class TestWriteReport:
    def test_writes_file(self, mini_report, tmp_path):
        path = write_report(mini_report, output_dir=tmp_path)
        assert path.exists()
        assert path.name == "report.md"
        content = path.read_text(encoding="utf-8")
        assert "# Drug-Disease Prediction Experiment Report" in content
