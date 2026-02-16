"""Export experiment results to CSV and JSON.

Produces four artefacts inside the output directory:

- ``arm_results.csv``   -- one row per (drug, arm): prediction counts,
  usage, timing, error flag.
- ``metrics.csv``       -- one row per (drug, arm): P@1, P@10, R@1,
  R@10, AUC, difficulty.
- ``aggregate.csv``     -- one row per arm: mean metrics across all drugs,
  plus per-difficulty breakdown.
- ``report.json``       -- full ``ExperimentReport`` serialised as JSON
  (lossless, machine-readable).

Usage::

    from src.viz.export import export_results
    export_results(report, output_dir="results/run_001")
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.experiment.orchestrator import DrugResult, ExperimentReport

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def export_results(
    report: ExperimentReport,
    output_dir: str | Path = "results",
) -> dict[str, Path]:
    """Write all CSV and JSON artefacts for *report*.

    Parameters
    ----------
    report:
        The ``ExperimentReport`` returned by ``ExperimentRunner.run()``.
    output_dir:
        Directory to write files into (created if it does not exist).

    Returns
    -------
    dict[str, Path]
        Mapping of artefact name to its absolute path on disk.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    paths["arm_results"] = _write_arm_results_csv(report, out / "arm_results.csv")
    paths["metrics"] = _write_metrics_csv(report, out / "metrics.csv")
    paths["aggregate"] = _write_aggregate_csv(report, out / "aggregate.csv")
    paths["report_json"] = _write_report_json(report, out / "report.json")

    logger.info(
        "Exported %d artefacts to %s",
        len(paths),
        out.resolve(),
    )
    return paths


# ------------------------------------------------------------------
# arm_results.csv
# ------------------------------------------------------------------

_ARM_RESULTS_COLUMNS = [
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


def _write_arm_results_csv(
    report: ExperimentReport,
    path: Path,
) -> Path:
    """Write one row per (drug, arm) with prediction summary and usage."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ARM_RESULTS_COLUMNS)
        writer.writeheader()

        for dr in _sorted_drug_results(report):
            for arm_id, ar in sorted(dr.arm_results.items()):
                pred = ar.prediction
                n_assoc = len(pred.associations) if pred else 0
                n_chains = (
                    sum(len(a.evidence_chains) for a in pred.associations)
                    if pred
                    else 0
                )
                writer.writerow({
                    "drug_name": ar.drug_name,
                    "chembl_id": dr.chembl_id or "",
                    "arm_id": ar.arm_id,
                    "model_id": ar.model_id,
                    "has_prediction": pred is not None,
                    "n_associations": n_assoc,
                    "n_evidence_chains": n_chains,
                    "input_tokens": ar.usage.get("input_tokens", 0),
                    "output_tokens": ar.usage.get("output_tokens", 0),
                    "total_tokens": ar.usage.get("total_tokens", 0),
                    "web_search_requests": ar.usage.get("web_search_requests", 0),
                    "wall_clock_seconds": round(ar.wall_clock_seconds, 3),
                    "error": ar.error or "",
                })

    logger.debug("Wrote %s", path)
    return path


# ------------------------------------------------------------------
# metrics.csv
# ------------------------------------------------------------------

_METRICS_COLUMNS = [
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


def _write_metrics_csv(
    report: ExperimentReport,
    path: Path,
) -> Path:
    """Write one row per (drug, arm) with accuracy metrics."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRICS_COLUMNS)
        writer.writeheader()

        for dr in _sorted_drug_results(report):
            for arm_id, dm in sorted(dr.metrics.items()):
                writer.writerow({
                    "drug_name": dm.drug_name,
                    "arm_id": dm.arm_id,
                    "difficulty": dm.difficulty.value if dm.difficulty else "",
                    "n_ground_truth": dm.n_ground_truth,
                    "n_predicted": dm.n_predicted,
                    "precision_at_1": round(dm.precision_at_1, 4),
                    "precision_at_10": round(dm.precision_at_10, 4),
                    "recall_at_1": round(dm.recall_at_1, 4),
                    "recall_at_10": round(dm.recall_at_10, 4),
                    "roc_auc": round(dm.roc_auc, 4) if dm.roc_auc is not None else "",
                    "true_positives_at_10": "; ".join(dm.true_positives_at_10),
                })

    logger.debug("Wrote %s", path)
    return path


# ------------------------------------------------------------------
# aggregate.csv
# ------------------------------------------------------------------

_AGGREGATE_COLUMNS = [
    "arm_id",
    "n_drugs",
    "mean_precision_at_1",
    "mean_precision_at_10",
    "mean_recall_at_1",
    "mean_recall_at_10",
    "mean_roc_auc",
    "easy_precision_at_10",
    "medium_precision_at_10",
    "hard_precision_at_10",
    "easy_recall_at_10",
    "medium_recall_at_10",
    "hard_recall_at_10",
]


def _write_aggregate_csv(
    report: ExperimentReport,
    path: Path,
) -> Path:
    """Write one row per arm with aggregate metrics (overall + per-difficulty)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_AGGREGATE_COLUMNS)
        writer.writeheader()

        for arm_id, agg in sorted(report.arm_aggregates.items()):
            by_diff = agg.by_difficulty if hasattr(agg, "by_difficulty") else {}
            row: dict[str, object] = {
                "arm_id": arm_id,
                "n_drugs": agg.n_drugs,
                "mean_precision_at_1": round(agg.mean_precision_at_1, 4),
                "mean_precision_at_10": round(agg.mean_precision_at_10, 4),
                "mean_recall_at_1": round(agg.mean_recall_at_1, 4),
                "mean_recall_at_10": round(agg.mean_recall_at_10, 4),
                "mean_roc_auc": (
                    round(agg.mean_roc_auc, 4)
                    if agg.mean_roc_auc is not None
                    else ""
                ),
            }

            # Per-difficulty breakdowns (may be absent)
            for diff in ("easy", "medium", "hard"):
                diff_agg = by_diff.get(diff)
                if diff_agg:
                    row[f"{diff}_precision_at_10"] = round(
                        diff_agg.mean_precision_at_10, 4
                    )
                    row[f"{diff}_recall_at_10"] = round(
                        diff_agg.mean_recall_at_10, 4
                    )
                else:
                    row[f"{diff}_precision_at_10"] = ""
                    row[f"{diff}_recall_at_10"] = ""

            writer.writerow(row)

    logger.debug("Wrote %s", path)
    return path


# ------------------------------------------------------------------
# report.json
# ------------------------------------------------------------------

def _write_report_json(
    report: ExperimentReport,
    path: Path,
) -> Path:
    """Serialise the full report to JSON (lossless round-trip)."""
    data = _report_to_dict(report)
    path.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Wrote %s", path)
    return path


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def _report_to_dict(report: ExperimentReport) -> dict:
    """Convert an ``ExperimentReport`` to a JSON-safe dict."""
    from dataclasses import asdict

    from src.experiment.orchestrator import DrugResult

    drug_results_out: dict[str, dict] = {}
    for drug_key, dr in report.drug_results.items():
        dr_dict: dict = {
            "drug_name": dr.drug_name,
            "chembl_id": dr.chembl_id,
            "pubchem_cid": dr.pubchem_cid,
            "ground_truth_diseases": sorted(dr.ground_truth_diseases),
            "difficulty": dr.difficulty.value if dr.difficulty else None,
            "n_evidence_docs": dr.n_evidence_docs,
            "n_chunks_indexed": dr.n_chunks_indexed,
            "arm_results": {
                arm_id: ar.model_dump(mode="json")
                for arm_id, ar in dr.arm_results.items()
            },
            "metrics": {
                arm_id: asdict(dm)
                for arm_id, dm in dr.metrics.items()
            },
        }
        drug_results_out[drug_key] = dr_dict

    difficulty_out: dict[str, dict] = {}
    for drug_key, cd in report.difficulty_map.items():
        difficulty_out[drug_key] = {
            "drug_name": cd.drug_name,
            "difficulty": cd.difficulty.value,
            "reference_p_at_10": cd.reference_p_at_10,
        }

    aggregate_out: dict[str, dict] = {}
    for arm_id, agg in report.arm_aggregates.items():
        agg_dict = asdict(agg)
        aggregate_out[arm_id] = agg_dict

    return {
        "n_drugs": report.n_drugs,
        "n_arms": report.n_arms,
        "wall_clock_seconds": round(report.wall_clock_seconds, 3),
        "cached_results": report.cached_results,
        "drug_results": drug_results_out,
        "difficulty_map": difficulty_out,
        "arm_aggregates": aggregate_out,
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _sorted_drug_results(report: ExperimentReport) -> list[DrugResult]:
    """Return drug results sorted alphabetically by drug name."""
    return sorted(report.drug_results.values(), key=lambda dr: dr.drug_name.lower())
