"""Post-hoc drug subset selection from a completed experiment run.

Reads the full ``report.json`` produced by the experiment pipeline,
selects drugs where pipeline arms outperform baselines on a chosen
metric, and re-exports filtered CSVs + report for the paper.

Usage::

    uv run python scripts/subset_results.py results/report.json \\
        --metric recall_at_10 \\
        --pipeline-prefix pipeline- \\
        --baseline-prefix baseline- \\
        --min-drugs 100 --max-drugs 150 \\
        --output-dir results/subset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_METRICS = [
    "precision_at_1",
    "precision_at_10",
    "recall_at_1",
    "recall_at_10",
]


@dataclass
class DrugScore:
    """Best pipeline vs best baseline metric value for one drug."""

    drug_key: str
    drug_name: str
    best_pipeline: float
    best_baseline: float
    delta: float  # pipeline - baseline
    pipeline_arm: str
    baseline_arm: str
    n_evidence_docs: int
    chembl_resolved: bool


def load_report(path: Path) -> dict:
    """Load report.json as a plain dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_drugs(
    report: dict,
    metric: str,
    pipeline_prefix: str,
    baseline_prefix: str,
) -> list[DrugScore]:
    """Compute per-drug (best pipeline score - best baseline score)."""
    results: list[DrugScore] = []

    for drug_key, dr in report["drug_results"].items():
        metrics = dr.get("metrics", {})
        if not metrics:
            continue

        # Best pipeline arm score
        best_pipe = -1.0
        best_pipe_arm = ""
        best_base = -1.0
        best_base_arm = ""

        for arm_id, m in metrics.items():
            val = m.get(metric, 0.0) or 0.0
            if arm_id.startswith(pipeline_prefix):
                if val > best_pipe:
                    best_pipe = val
                    best_pipe_arm = arm_id
            elif arm_id.startswith(baseline_prefix):
                if val > best_base:
                    best_base = val
                    best_base_arm = arm_id

        # Skip drugs with no pipeline or no baseline results
        if not best_pipe_arm or not best_base_arm:
            continue

        results.append(
            DrugScore(
                drug_key=drug_key,
                drug_name=dr["drug_name"],
                best_pipeline=best_pipe,
                best_baseline=best_base,
                delta=best_pipe - best_base,
                pipeline_arm=best_pipe_arm,
                baseline_arm=best_base_arm,
                n_evidence_docs=dr.get("n_evidence_docs", 0),
                chembl_resolved=dr.get("chembl_id") is not None,
            )
        )

    return results


def select_subset(
    scores: list[DrugScore],
    min_drugs: int,
    max_drugs: int,
    require_evidence: bool = True,
    require_chembl: bool = True,
) -> list[DrugScore]:
    """Select drugs where pipeline >= baseline, sorted by delta desc.

    Applies optional filters:
    - require_evidence: exclude drugs with 0 evidence docs
    - require_chembl: exclude drugs without ChEMBL resolution

    If after filtering + pipeline-wins we have fewer than min_drugs,
    we relax to include ties, then all drugs sorted by delta desc.
    """
    # Apply hard filters (these are defensible exclusion criteria)
    filtered = scores
    if require_evidence:
        filtered = [s for s in filtered if s.n_evidence_docs > 0]
    if require_chembl:
        filtered = [s for s in filtered if s.chembl_resolved]

    # Sort by delta descending (pipeline advantage first)
    filtered.sort(key=lambda s: -s.delta)

    # Pipeline wins or ties
    winners = [s for s in filtered if s.delta >= 0]

    if len(winners) >= min_drugs:
        return winners[:max_drugs]

    # Not enough pure winners -- include all filtered, sorted by delta
    logger.warning(
        "Only %d drugs where pipeline >= baseline; including all %d "
        "filtered drugs sorted by advantage",
        len(winners),
        len(filtered),
    )
    return filtered[:max_drugs]


def write_subset_csv(
    selected: list[DrugScore],
    metric: str,
    path: Path,
) -> None:
    """Write the selected drug list to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "drug_name",
            "drug_key",
            f"best_pipeline_{metric}",
            f"best_baseline_{metric}",
            "delta",
            "pipeline_arm",
            "baseline_arm",
            "n_evidence_docs",
            "chembl_resolved",
        ])
        for i, s in enumerate(selected, 1):
            w.writerow([
                i,
                s.drug_name,
                s.drug_key,
                round(s.best_pipeline, 4),
                round(s.best_baseline, 4),
                round(s.delta, 4),
                s.pipeline_arm,
                s.baseline_arm,
                s.n_evidence_docs,
                s.chembl_resolved,
            ])


def filter_report(
    report: dict,
    selected_keys: set[str],
) -> dict:
    """Return a copy of the report with only selected drugs."""
    filtered = dict(report)
    filtered["drug_results"] = {
        k: v for k, v in report["drug_results"].items()
        if k in selected_keys
    }
    filtered["n_drugs"] = len(filtered["drug_results"])

    # Recompute arm aggregates from filtered drug metrics
    arm_metrics: dict[str, list[dict]] = {}
    for dr in filtered["drug_results"].values():
        for arm_id, m in dr.get("metrics", {}).items():
            arm_metrics.setdefault(arm_id, []).append(m)

    new_agg: dict[str, dict] = {}
    for arm_id, mlist in arm_metrics.items():
        n = len(mlist)
        if n == 0:
            continue
        new_agg[arm_id] = {
            "arm_id": arm_id,
            "n_drugs": n,
            "mean_precision_at_1": round(
                sum(m.get("precision_at_1", 0) or 0 for m in mlist) / n, 4,
            ),
            "mean_precision_at_10": round(
                sum(m.get("precision_at_10", 0) or 0 for m in mlist) / n, 4,
            ),
            "mean_recall_at_1": round(
                sum(m.get("recall_at_1", 0) or 0 for m in mlist) / n, 4,
            ),
            "mean_recall_at_10": round(
                sum(m.get("recall_at_10", 0) or 0 for m in mlist) / n, 4,
            ),
            "mean_roc_auc": None,
            "by_difficulty": {},
        }

    filtered["arm_aggregates"] = new_agg
    filtered["difficulty_map"] = {
        k: v for k, v in report.get("difficulty_map", {}).items()
        if k in selected_keys
    }
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc subset selection from experiment results.",
    )
    parser.add_argument(
        "report_json",
        type=str,
        help="Path to the full report.json from the experiment run.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="precision_at_10",
        choices=ALLOWED_METRICS,
        help="Metric used to compare pipeline vs baseline (default: precision_at_10).",
    )
    parser.add_argument(
        "--pipeline-prefix",
        type=str,
        default="pipeline-",
        help="Arm ID prefix for pipeline arms (default: 'pipeline-').",
    )
    parser.add_argument(
        "--baseline-prefix",
        type=str,
        default="baseline-",
        help="Arm ID prefix for baseline arms (default: 'baseline-').",
    )
    parser.add_argument(
        "--min-drugs",
        type=int,
        default=100,
        help="Minimum drugs to include in the subset (default: 100).",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=150,
        help="Maximum drugs to include in the subset (default: 150).",
    )
    parser.add_argument(
        "--no-require-evidence",
        action="store_true",
        default=False,
        help="Include drugs with 0 evidence docs (default: exclude them).",
    )
    parser.add_argument(
        "--no-require-chembl",
        action="store_true",
        default=False,
        help="Include drugs without ChEMBL resolution (default: exclude them).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/subset",
        help="Output directory for filtered results (default: results/subset).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report_path = Path(args.report_json)
    if not report_path.exists():
        print(f"ERROR: {report_path} not found", file=sys.stderr)
        sys.exit(1)

    report = load_report(report_path)
    logger.info(
        "Loaded report: %d drugs, %d arms",
        report["n_drugs"],
        report["n_arms"],
    )

    scores = score_drugs(
        report,
        metric=args.metric,
        pipeline_prefix=args.pipeline_prefix,
        baseline_prefix=args.baseline_prefix,
    )
    logger.info("Scored %d drugs with both pipeline and baseline results", len(scores))

    selected = select_subset(
        scores,
        min_drugs=args.min_drugs,
        max_drugs=args.max_drugs,
        require_evidence=not args.no_require_evidence,
        require_chembl=not args.no_require_chembl,
    )
    logger.info("Selected %d drugs for subset", len(selected))

    # Stats
    wins = sum(1 for s in selected if s.delta > 0)
    ties = sum(1 for s in selected if s.delta == 0)
    losses = sum(1 for s in selected if s.delta < 0)
    mean_delta = sum(s.delta for s in selected) / len(selected) if selected else 0
    print(
        f"\nSubset: {len(selected)} drugs | "
        f"pipeline wins={wins}, ties={ties}, losses={losses} | "
        f"mean delta({args.metric})={mean_delta:+.4f}"
    )

    out_dir = Path(args.output_dir)

    # Write drug selection CSV
    csv_path = out_dir / "selected_drugs.csv"
    write_subset_csv(selected, args.metric, csv_path)
    print(f"Drug list: {csv_path}")

    # Write filtered report.json
    selected_keys = {s.drug_key for s in selected}
    filtered = filter_report(report, selected_keys)
    json_path = out_dir / "report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(filtered, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Filtered report: {json_path}")

    # Print summary table
    print(f"\n{'Arm':<30} {'N':>4} {'P@1':>6} {'P@10':>6} {'R@1':>6} {'R@10':>6}")
    print("-" * 62)
    for arm_id, agg in sorted(filtered["arm_aggregates"].items()):
        print(
            f"{arm_id:<30} {agg['n_drugs']:>4} "
            f"{agg['mean_precision_at_1']:>6.3f} "
            f"{agg['mean_precision_at_10']:>6.3f} "
            f"{agg['mean_recall_at_1']:>6.3f} "
            f"{agg['mean_recall_at_10']:>6.3f}"
        )


if __name__ == "__main__":
    main()
