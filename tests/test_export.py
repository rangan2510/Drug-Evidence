"""Tests for src.viz.export -- CSV and JSON artefact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.accuracy import AggregateMetrics, DrugMetrics
from src.experiment.difficulty import ClassifiedDrug
from src.experiment.orchestrator import DrugResult, ExperimentReport
from src.experiment.runner import ArmResult
from src.schemas.prediction import (
    DrugDifficulty,
    DrugDiseasePrediction,
    EvidenceChain,
    MechanisticEdge,
    EdgeType,
    ScoredAssociation,
)
from src.viz.export import export_results


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_prediction(drug_name: str, diseases: list[str]) -> DrugDiseasePrediction:
    """Build a minimal DrugDiseasePrediction for testing."""
    associations = []
    for d in diseases:
        chain = EvidenceChain(
            edges=[
                MechanisticEdge(
                    source_entity=drug_name,
                    target_entity=d,
                    relationship=EdgeType.ASSOCIATED_WITH,
                    evidence_snippet="Test evidence snippet.",
                ),
            ],
            summary=f"{drug_name} treats {d}",
            confidence=0.85,
        )
        associations.append(
            ScoredAssociation(
                disease_name=d,
                predicted=True,
                confidence=0.85,
                evidence_chains=[chain],
            )
        )
    return DrugDiseasePrediction(
        drug_name=drug_name,
        associations=associations,
        reasoning="Test reasoning.",
    )


def _make_arm_result(
    drug_name: str,
    arm_id: str,
    model_id: str,
    diseases: list[str] | None = None,
    error: str | None = None,
) -> ArmResult:
    """Build an ArmResult with or without a prediction."""
    pred = _make_prediction(drug_name, diseases) if diseases else None
    return ArmResult(
        arm_id=arm_id,
        drug_name=drug_name,
        model_id=model_id,
        prediction=pred,
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "web_search_requests": 1,
        },
        wall_clock_seconds=1.234,
        error=error,
    )


def _make_drug_metrics(
    drug_name: str,
    arm_id: str,
    difficulty: DrugDifficulty | None = DrugDifficulty.MEDIUM,
) -> DrugMetrics:
    """Build a DrugMetrics with plausible values."""
    return DrugMetrics(
        drug_name=drug_name,
        arm_id=arm_id,
        difficulty=difficulty,
        n_ground_truth=5,
        n_predicted=3,
        precision_at_1=1.0,
        precision_at_10=0.6,
        recall_at_1=0.2,
        recall_at_10=0.6,
        roc_auc=0.75,
        true_positives_at_10=["disease A", "disease B", "disease C"],
    )


@pytest.fixture()
def mini_report() -> ExperimentReport:
    """Build a small ExperimentReport with 2 drugs and 2 arms."""
    drugs = ["aspirin", "ibuprofen"]
    arm_ids = ["pipeline-gpt-oss", "baseline-gpt5-nosearch"]
    model_ids = [
        "groq:openai/gpt-oss-120b",
        "openai:gpt-5.2-2025-12-11",
    ]

    drug_results: dict[str, DrugResult] = {}
    all_metrics: list[DrugMetrics] = []

    for drug in drugs:
        arm_results: dict[str, ArmResult] = {}
        metrics: dict[str, DrugMetrics] = {}

        for arm_id, model_id in zip(arm_ids, model_ids):
            ar = _make_arm_result(
                drug_name=drug,
                arm_id=arm_id,
                model_id=model_id,
                diseases=["disease A", "disease B"],
            )
            arm_results[arm_id] = ar

            dm = _make_drug_metrics(drug, arm_id)
            metrics[arm_id] = dm
            all_metrics.append(dm)

        drug_results[drug] = DrugResult(
            drug_name=drug,
            chembl_id=f"CHEMBL_{drug.upper()}",
            pubchem_cid=12345,
            ground_truth_diseases={"disease A", "disease B", "disease X"},
            difficulty=DrugDifficulty.MEDIUM,
            arm_results=arm_results,
            metrics=metrics,
            n_evidence_docs=10,
            n_chunks_indexed=30,
        )

    # Build aggregate metrics per arm
    from src.evaluation.accuracy import aggregate_metrics

    arm_aggregates: dict[str, AggregateMetrics] = {}
    for arm_id in arm_ids:
        arm_metrics = [m for m in all_metrics if m.arm_id == arm_id]
        arm_aggregates[arm_id] = aggregate_metrics(arm_metrics, arm_id)

    difficulty_map: dict[str, ClassifiedDrug] = {
        drug: ClassifiedDrug(
            drug_name=drug,
            difficulty=DrugDifficulty.MEDIUM,
            reference_p_at_10=0.4,
        )
        for drug in drugs
    }

    return ExperimentReport(
        n_drugs=2,
        n_arms=2,
        wall_clock_seconds=5.678,
        drug_results=drug_results,
        difficulty_map=difficulty_map,
        arm_aggregates=arm_aggregates,
        cached_results=1,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestExportResults:
    """Integration test: export_results produces all 4 artefacts."""

    def test_produces_four_files(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        assert set(paths.keys()) == {
            "arm_results",
            "metrics",
            "aggregate",
            "report_json",
        }
        for p in paths.values():
            assert Path(p).exists()

    def test_creates_output_directory(self, mini_report, tmp_path):
        new_dir = tmp_path / "sub" / "nested"
        export_results(mini_report, output_dir=new_dir)
        assert new_dir.exists()


class TestArmResultsCsv:
    """Verify arm_results.csv content."""

    def test_row_count(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["arm_results"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # 2 drugs x 2 arms = 4 rows
        assert len(rows) == 4

    def test_columns(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["arm_results"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = [
            "drug_name",
            "chembl_id",
            "arm_id",
            "model_id",
            "has_prediction",
            "n_associations",
            "n_evidence_chains",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "web_search_requests",
            "wall_clock_seconds",
            "error",
        ]
        assert headers == expected

    def test_token_counts_present(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["arm_results"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert int(row["input_tokens"]) == 100
            assert int(row["output_tokens"]) == 50
            assert int(row["total_tokens"]) == 150

    def test_error_arm_has_no_prediction(self, mini_report, tmp_path):
        """Add a failed arm result and verify has_prediction=False."""
        # Mutate report: add a failed arm to aspirin
        dr = mini_report.drug_results["aspirin"]
        failed_ar = _make_arm_result(
            drug_name="aspirin",
            arm_id="pipeline-failed",
            model_id="groq:broken",
            diseases=None,
            error="timeout",
        )
        dr.arm_results["pipeline-failed"] = failed_ar

        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["arm_results"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        failed_rows = [r for r in rows if r["arm_id"] == "pipeline-failed"]
        assert len(failed_rows) == 1
        assert failed_rows[0]["has_prediction"] == "False"
        assert failed_rows[0]["error"] == "timeout"
        assert failed_rows[0]["n_associations"] == "0"


class TestMetricsCsv:
    """Verify metrics.csv content."""

    def test_row_count(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["metrics"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_columns(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["metrics"], newline="", encoding="utf-8") as f:
            headers = csv.DictReader(f).fieldnames
        expected = [
            "drug_name",
            "arm_id",
            "difficulty",
            "n_ground_truth",
            "n_predicted",
            "precision_at_1",
            "precision_at_10",
            "recall_at_1",
            "recall_at_10",
            "roc_auc",
            "true_positives_at_10",
        ]
        assert headers == expected

    def test_metric_values(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["metrics"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        first = rows[0]
        assert first["difficulty"] == "medium"
        assert float(first["precision_at_1"]) == 1.0
        assert float(first["recall_at_10"]) == 0.6
        assert first["true_positives_at_10"] == "disease A; disease B; disease C"


class TestAggregateCsv:
    """Verify aggregate.csv content."""

    def test_row_count(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["aggregate"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # one row per arm
        assert len(rows) == 2

    def test_columns(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["aggregate"], newline="", encoding="utf-8") as f:
            headers = csv.DictReader(f).fieldnames
        assert "arm_id" in headers
        assert "mean_precision_at_10" in headers
        assert "easy_precision_at_10" in headers
        assert "hard_recall_at_10" in headers

    def test_aggregate_values(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        with open(paths["aggregate"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert int(row["n_drugs"]) == 2
            assert float(row["mean_precision_at_1"]) == 1.0
            # Only medium drugs in this fixture, so easy/hard should be empty
            assert row["easy_precision_at_10"] == ""
            assert row["hard_precision_at_10"] == ""
            assert row["medium_precision_at_10"] == "0.6"


class TestReportJson:
    """Verify report.json content."""

    def test_valid_json(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_top_level_keys(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        expected_keys = {
            "n_drugs",
            "n_arms",
            "wall_clock_seconds",
            "cached_results",
            "drug_results",
            "difficulty_map",
            "arm_aggregates",
        }
        assert set(data.keys()) == expected_keys

    def test_drug_results_structure(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        dr = data["drug_results"]["aspirin"]
        assert dr["drug_name"] == "aspirin"
        assert dr["chembl_id"] == "CHEMBL_ASPIRIN"
        assert "arm_results" in dr
        assert "metrics" in dr
        assert isinstance(dr["ground_truth_diseases"], list)

    def test_difficulty_map_structure(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        cd = data["difficulty_map"]["aspirin"]
        assert cd["difficulty"] == "medium"
        assert cd["reference_p_at_10"] == 0.4

    def test_n_drugs(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        assert data["n_drugs"] == 2
        assert data["n_arms"] == 2
        assert data["cached_results"] == 1

    def test_arm_aggregates_serialised(self, mini_report, tmp_path):
        paths = export_results(mini_report, output_dir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        agg = data["arm_aggregates"]
        assert "pipeline-gpt-oss" in agg
        assert "baseline-gpt5-nosearch" in agg
        assert agg["pipeline-gpt-oss"]["n_drugs"] == 2
