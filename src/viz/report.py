"""Markdown report generator for experiment results.

Produces a self-contained Markdown report with:

- Experiment configuration summary
- Per-arm results table (accuracy metrics)
- Per-difficulty breakdown
- Evidence quality summary (if available)
- Statistical significance tests (paired Wilcoxon, Bonferroni)
- Embedded plot references
- FN analysis summary (if available)
- Sensitivity analysis summary (if available)

Usage::

    from src.viz.report import generate_report
    md = generate_report(report, plot_paths=plot_paths)
    Path("results/report.md").write_text(md)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.accuracy import AggregateMetrics, DrugMetrics
    from src.evaluation.evidence_quality import EvidenceQualityMetrics
    from src.evaluation.false_negatives import AggregateFNSummary
    from src.experiment.orchestrator import ExperimentReport

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Statistical tests
# ------------------------------------------------------------------

def _paired_wilcoxon(
    arm_a_values: list[float],
    arm_b_values: list[float],
) -> tuple[float, float]:
    """Run a paired Wilcoxon signed-rank test.

    Returns (statistic, p_value).  Falls back to (nan, nan) if scipy
    is unavailable or the test cannot be computed.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return (float("nan"), float("nan"))

    if len(arm_a_values) != len(arm_b_values) or len(arm_a_values) < 3:
        return (float("nan"), float("nan"))

    # Remove pairs where both are identical (Wilcoxon requires differences)
    diffs = [a - b for a, b in zip(arm_a_values, arm_b_values)]
    if all(d == 0 for d in diffs):
        return (0.0, 1.0)

    try:
        stat, p = wilcoxon(arm_a_values, arm_b_values, alternative="two-sided")
        return (float(stat), float(p))
    except Exception:
        return (float("nan"), float("nan"))


def _bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction to a list of p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


# ------------------------------------------------------------------
# Report sections
# ------------------------------------------------------------------

def _section_header(report: ExperimentReport) -> str:
    """Experiment configuration and summary."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Drug-Disease Prediction Experiment Report",
        "",
        f"**Generated**: {ts}",
        "",
        "## Experiment Summary",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Drugs evaluated | {report.n_drugs} |",
        f"| Arms | {report.n_arms} |",
        f"| Wall-clock time | {report.wall_clock_seconds:.1f}s |",
        f"| Cached results | {report.cached_results} |",
        "",
    ]

    # Difficulty distribution
    if report.difficulty_map:
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for cd in report.difficulty_map.values():
            counts[cd.difficulty.value] += 1
        lines.append("**Difficulty distribution**:")
        lines.append(f"Easy: {counts['easy']}, Medium: {counts['medium']}, Hard: {counts['hard']}")
        lines.append("")

    return "\n".join(lines)


def _section_accuracy_table(report: ExperimentReport) -> str:
    """Per-arm aggregate accuracy table."""
    lines = [
        "## Accuracy Metrics (All Drugs)",
        "",
        "| Arm | N | P@1 | P@10 | R@1 | R@10 | AUC |",
        "|-----|---|-----|------|-----|------|-----|",
    ]
    for arm_id, agg in sorted(report.arm_aggregates.items()):
        auc = f"{agg.mean_roc_auc:.3f}" if agg.mean_roc_auc is not None else "n/a"
        lines.append(
            f"| {arm_id} | {agg.n_drugs} | "
            f"{agg.mean_precision_at_1:.3f} | {agg.mean_precision_at_10:.3f} | "
            f"{agg.mean_recall_at_1:.3f} | {agg.mean_recall_at_10:.3f} | "
            f"{auc} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_difficulty_breakdown(report: ExperimentReport) -> str:
    """Per-difficulty breakdown for each arm."""
    lines = [
        "## Accuracy by Difficulty",
        "",
    ]

    for diff in ("easy", "medium", "hard"):
        lines.append(f"### {diff.title()}")
        lines.append("")
        lines.append("| Arm | N | P@10 | R@10 |")
        lines.append("|-----|---|------|------|")

        for arm_id, agg in sorted(report.arm_aggregates.items()):
            by_diff = agg.by_difficulty if hasattr(agg, "by_difficulty") else {}
            diff_agg = by_diff.get(diff)
            if diff_agg:
                lines.append(
                    f"| {arm_id} | {diff_agg.n_drugs} | "
                    f"{diff_agg.mean_precision_at_10:.3f} | "
                    f"{diff_agg.mean_recall_at_10:.3f} |"
                )
            else:
                lines.append(f"| {arm_id} | 0 | - | - |")
        lines.append("")

    return "\n".join(lines)


def _section_statistical_tests(
    report: ExperimentReport,
    drug_metrics: dict[str, list[DrugMetrics]],
) -> str:
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    arms = sorted(drug_metrics.keys())
    if len(arms) < 2:
        return ""

    lines = [
        "## Statistical Significance (Paired Wilcoxon)",
        "",
        "Bonferroni-corrected p-values for pairwise comparisons on P@10.",
        "",
        "| Arm A | Arm B | Statistic | Raw p | Corrected p | Significant? |",
        "|-------|-------|-----------|-------|-------------|-------------|",
    ]

    # Align per-drug metrics by drug name for pairing
    # Build {arm -> {drug -> P@10}}
    arm_drug_p10: dict[str, dict[str, float]] = {}
    for arm_id, metrics_list in drug_metrics.items():
        arm_drug_p10[arm_id] = {m.drug_name: m.precision_at_10 for m in metrics_list}

    pairs = list(combinations(arms, 2))
    raw_p_values = []
    pair_stats = []

    for arm_a, arm_b in pairs:
        # Find common drugs
        common_drugs = sorted(
            set(arm_drug_p10[arm_a].keys()) & set(arm_drug_p10[arm_b].keys())
        )
        vals_a = [arm_drug_p10[arm_a][d] for d in common_drugs]
        vals_b = [arm_drug_p10[arm_b][d] for d in common_drugs]

        stat, p = _paired_wilcoxon(vals_a, vals_b)
        raw_p_values.append(p)
        pair_stats.append((arm_a, arm_b, stat, p))

    corrected = _bonferroni_correction(raw_p_values)

    for (arm_a, arm_b, stat, raw_p), corr_p in zip(pair_stats, corrected):
        import math
        sig = "Yes" if corr_p < 0.05 and not math.isnan(corr_p) else "No"
        stat_str = f"{stat:.1f}" if not math.isnan(stat) else "n/a"
        raw_str = f"{raw_p:.4f}" if not math.isnan(raw_p) else "n/a"
        corr_str = f"{corr_p:.4f}" if not math.isnan(corr_p) else "n/a"
        lines.append(
            f"| {arm_a} | {arm_b} | {stat_str} | {raw_str} | {corr_str} | {sig} |"
        )

    lines.append("")
    return "\n".join(lines)


def _section_evidence_quality(
    arm_evidence: dict[str, EvidenceQualityMetrics],
) -> str:
    """Evidence quality metrics table."""
    arms = sorted(arm_evidence.keys())
    lines = [
        "## Evidence Quality",
        "",
        "| Arm | Citation Validity | Chain Depth | Verifiability | Relevance | Specificity |",
        "|-----|-------------------|-------------|---------------|-----------|-------------|",
    ]
    for arm_id in arms:
        eq = arm_evidence[arm_id]
        lines.append(
            f"| {arm_id} | {eq.citation_validity_rate:.3f} | "
            f"{eq.mean_chain_depth:.2f} | {eq.chain_verifiability_score:.3f} | "
            f"{eq.evidence_relevance:.3f} | {eq.mechanistic_specificity:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_fn_summary(
    arm_fn_summaries: dict[str, AggregateFNSummary],
) -> str:
    """False negative analysis summary."""
    lines = [
        "## False Negative Analysis",
        "",
        "| Arm | Total FN | FN Rate | Name Mismatch | No Evidence | Low Retrieval | Low LLM | Below Threshold | Not in Candidates |",
        "|-----|----------|---------|---------------|-------------|---------------|---------|-----------------|-------------------|",
    ]
    for arm_id in sorted(arm_fn_summaries.keys()):
        s = arm_fn_summaries[arm_id]
        fracs = s.category_fractions()
        lines.append(
            f"| {arm_id} | {s.total_false_negatives} | {s.overall_fn_rate:.3f} | "
            f"{fracs.get('name_mismatch', 0):.2f} | "
            f"{fracs.get('no_evidence_found', 0):.2f} | "
            f"{fracs.get('low_retrieval_score', 0):.2f} | "
            f"{fracs.get('low_llm_confidence', 0):.2f} | "
            f"{fracs.get('below_threshold', 0):.2f} | "
            f"{fracs.get('not_in_candidates', 0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_plots(plot_paths: dict[str, Path]) -> str:
    """Embed plot images."""
    if not plot_paths:
        return ""

    lines = [
        "## Plots",
        "",
    ]
    for name, path in sorted(plot_paths.items()):
        title = name.replace("_", " ").title()
        # Use relative path from report location
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](plots/{path.name})")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_report(
    report: ExperimentReport,
    *,
    plot_paths: dict[str, Path] | None = None,
    evidence_quality: dict[str, EvidenceQualityMetrics] | None = None,
    fn_summaries: dict[str, AggregateFNSummary] | None = None,
) -> str:
    """Generate the full Markdown experiment report.

    Parameters
    ----------
    report:
        Full ExperimentReport from the orchestrator.
    plot_paths:
        Mapping of plot name to file path (from ``generate_all_plots``).
    evidence_quality:
        Optional per-arm mean evidence quality metrics.
    fn_summaries:
        Optional per-arm FN summaries.

    Returns
    -------
    str
        Complete Markdown report as a string.
    """
    # Build per-arm DrugMetrics lists for statistical tests
    arm_drug_metrics: dict[str, list[DrugMetrics]] = {}
    for dr in report.drug_results.values():
        for arm_id, dm in dr.metrics.items():
            arm_drug_metrics.setdefault(arm_id, []).append(dm)

    sections = [
        _section_header(report),
        _section_accuracy_table(report),
        _section_difficulty_breakdown(report),
        _section_statistical_tests(report, arm_drug_metrics),
    ]

    if evidence_quality:
        sections.append(_section_evidence_quality(evidence_quality))

    if fn_summaries:
        sections.append(_section_fn_summary(fn_summaries))

    if plot_paths:
        sections.append(_section_plots(plot_paths))

    return "\n".join(sections)


def write_report(
    report: ExperimentReport,
    output_dir: str | Path = "results",
    *,
    plot_paths: dict[str, Path] | None = None,
    evidence_quality: dict[str, EvidenceQualityMetrics] | None = None,
    fn_summaries: dict[str, AggregateFNSummary] | None = None,
) -> Path:
    """Generate and write the report to ``output_dir/report.md``.

    Returns the path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md = generate_report(
        report,
        plot_paths=plot_paths,
        evidence_quality=evidence_quality,
        fn_summaries=fn_summaries,
    )

    path = out / "report.md"
    path.write_text(md, encoding="utf-8")
    logger.info("Wrote report to %s", path.resolve())
    return path
