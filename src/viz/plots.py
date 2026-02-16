"""Publication-quality plot generation for experiment results.

Produces 9 plot types as PNG files inside the output directory:

1. **precision_recall_bars** -- Grouped bar chart: P@1, P@10, R@1, R@10 per arm.
2. **roc_curves** -- ROC curves per arm (macro-averaged from per-drug data).
3. **pr_curves** -- Precision-Recall curves per arm.
4. **evidence_radar** -- Radar chart: 5 evidence-quality metrics per arm.
5. **cost_bars** -- Stacked bar chart: input/output tokens per arm.
6. **sensitivity_heatmap** -- 2D heatmap: w_retrieval x threshold -> P@10.
7. **fn_distribution** -- Stacked bar chart: FN categories per arm.
8. **score_distribution** -- Histograms: TP vs FP confidence per arm.
9. **ablation_delta** -- Horizontal bar chart: ablation -> delta P@10.

All functions follow the same pattern::

    fig = plot_xxx(data, ...)  # returns matplotlib Figure
    fig.savefig(path, ...)     # caller controls output

The top-level ``generate_all_plots`` helper writes all available plots
to an output directory and returns a mapping of plot name to file path.

Usage::

    from src.viz.plots import generate_all_plots
    paths = generate_all_plots(report, output_dir="results/plots")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.evaluation.accuracy import AggregateMetrics, DrugMetrics
    from src.evaluation.evidence_quality import EvidenceQualityMetrics
    from src.evaluation.false_negatives import AggregateFNSummary
    from src.evaluation.sensitivity import (
        AggregateAblationResult,
        HeatmapCell,
    )
    from src.experiment.orchestrator import ExperimentReport

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Colour palette (colour-blind-friendly Okabe-Ito)
# ------------------------------------------------------------------

_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
]


def _arm_colour(idx: int) -> str:
    """Return a colour from the palette, cycling if needed."""
    return _PALETTE[idx % len(_PALETTE)]


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

_FIG_DPI = 150
_FIG_SIZE = (10, 6)


def _save(fig: Figure, path: Path) -> Path:
    """Save figure and close it."""
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved plot: %s", path)
    return path


# ------------------------------------------------------------------
# 1. Precision / Recall bar chart
# ------------------------------------------------------------------

def plot_precision_recall_bars(
    arm_aggregates: dict[str, AggregateMetrics],
) -> Figure:
    """Grouped bar chart of P@1, P@10, R@1, R@10 per arm."""
    arms = sorted(arm_aggregates.keys())
    metrics = ["P@1", "P@10", "R@1", "R@10"]
    n_arms = len(arms)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    x = np.arange(n_arms)
    width = 0.8 / n_metrics

    for i, metric_label in enumerate(metrics):
        values = []
        for arm_id in arms:
            agg = arm_aggregates[arm_id]
            if metric_label == "P@1":
                values.append(agg.mean_precision_at_1)
            elif metric_label == "P@10":
                values.append(agg.mean_precision_at_10)
            elif metric_label == "R@1":
                values.append(agg.mean_recall_at_1)
            else:
                values.append(agg.mean_recall_at_10)

        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric_label,
            color=_PALETTE[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall by Arm")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2. ROC curves (macro-averaged from per-drug binary predictions)
# ------------------------------------------------------------------

def plot_roc_curves(
    drug_metrics: dict[str, list[DrugMetrics]],
) -> Figure:
    """ROC-like summary: plot mean AUC as a bar chart per arm.

    True multi-threshold ROC curves require per-sample confidence scores.
    Since we store aggregate AUC per drug, this plots the distribution of
    per-drug AUC values as a box-and-whisker.
    """
    arms = sorted(drug_metrics.keys())
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    data_for_box = []
    labels = []
    for arm_id in arms:
        aucs = [
            m.roc_auc for m in drug_metrics[arm_id]
            if m.roc_auc is not None
        ]
        if aucs:
            data_for_box.append(aucs)
            labels.append(arm_id)

    if data_for_box:
        bp = ax.boxplot(
            data_for_box,
            patch_artist=True,
            tick_labels=labels,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 5},
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_arm_colour(i))
            patch.set_alpha(0.7)

    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC Distribution by Arm")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 3. PR curves (similar approach -- per-drug P@10 vs R@10 scatter)
# ------------------------------------------------------------------

def plot_pr_scatter(
    drug_metrics: dict[str, list[DrugMetrics]],
) -> Figure:
    """Precision vs Recall scatter per arm (one point per drug).

    Each drug's (R@10, P@10) is plotted, colour-coded by arm.
    """
    arms = sorted(drug_metrics.keys())
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    for idx, arm_id in enumerate(arms):
        recalls = [m.recall_at_10 for m in drug_metrics[arm_id]]
        precisions = [m.precision_at_10 for m in drug_metrics[arm_id]]
        ax.scatter(
            recalls,
            precisions,
            color=_arm_colour(idx),
            label=arm_id,
            alpha=0.5,
            s=25,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Precision@10")
    ax.set_title("Precision vs Recall per Drug")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 4. Evidence quality radar chart
# ------------------------------------------------------------------

_RADAR_METRICS = [
    "citation_validity_rate",
    "mean_chain_depth",
    "chain_verifiability_score",
    "evidence_relevance",
    "mechanistic_specificity",
]

_RADAR_LABELS = [
    "Citation\nValidity",
    "Chain\nDepth",
    "Chain\nVerifiability",
    "Evidence\nRelevance",
    "Mechanistic\nSpecificity",
]


def plot_evidence_radar(
    arm_evidence: dict[str, EvidenceQualityMetrics],
) -> Figure:
    """Radar chart of 5 evidence-quality metrics per arm.

    Parameters
    ----------
    arm_evidence:
        Mapping of arm_id to mean EvidenceQualityMetrics across all drugs.
    """
    arms = sorted(arm_evidence.keys())
    n_metrics = len(_RADAR_METRICS)

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for idx, arm_id in enumerate(arms):
        eq = arm_evidence[arm_id]
        values = []
        for metric in _RADAR_METRICS:
            v = getattr(eq, metric, 0.0)
            # Normalise chain depth to 0-1 range (assume max ~5)
            if metric == "mean_chain_depth":
                v = min(v / 5.0, 1.0)
            values.append(v)
        values += values[:1]  # close polygon

        ax.plot(angles, values, "-o", color=_arm_colour(idx), label=arm_id,
                markersize=4, linewidth=1.5)
        ax.fill(angles, values, color=_arm_colour(idx), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_RADAR_LABELS, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Evidence Quality Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 5. Cost / token bars
# ------------------------------------------------------------------

def plot_cost_bars(
    arm_usage: dict[str, dict[str, float]],
) -> Figure:
    """Stacked bar chart: input vs output tokens per arm.

    Parameters
    ----------
    arm_usage:
        arm_id -> {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
    """
    arms = sorted(arm_usage.keys())
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    input_vals = [arm_usage[a].get("input_tokens", 0) for a in arms]
    output_vals = [arm_usage[a].get("output_tokens", 0) for a in arms]

    x = np.arange(len(arms))
    ax.bar(x, input_vals, label="Input tokens", color=_PALETTE[0], edgecolor="white")
    ax.bar(x, output_vals, bottom=input_vals, label="Output tokens",
           color=_PALETTE[1], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Token count")
    ax.set_title("Token Usage by Arm")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 6. Sensitivity heatmap
# ------------------------------------------------------------------

def plot_sensitivity_heatmap(
    cells: list[HeatmapCell],
) -> Figure:
    """2D heatmap: w_retrieval (y) x threshold (x) -> P@10 value.

    Parameters
    ----------
    cells:
        Output from ``sensitivity.build_heatmap_data()``.
    """
    if not cells:
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
        return fig

    # Build 2D grid
    w_vals = sorted(set(c.w_retrieval for c in cells))
    t_vals = sorted(set(c.threshold for c in cells))
    grid = np.full((len(w_vals), len(t_vals)), np.nan)

    w_idx = {w: i for i, w in enumerate(w_vals)}
    t_idx = {t: i for i, t in enumerate(t_vals)}

    for c in cells:
        grid[w_idx[c.w_retrieval], t_idx[c.threshold]] = c.value

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(t_vals)))
    ax.set_xticklabels([f"{t:.2f}" for t in t_vals], rotation=45, fontsize=7)
    ax.set_yticks(range(len(w_vals)))
    ax.set_yticklabels([f"{w:.1f}" for w in w_vals], fontsize=8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("w_retrieval")
    ax.set_title("Weight Sensitivity: P@10")
    fig.colorbar(im, ax=ax, label="P@10")

    # Annotate cells
    for i in range(len(w_vals)):
        for j in range(len(t_vals)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 7. FN category distribution
# ------------------------------------------------------------------

def plot_fn_distribution(
    arm_fn_summaries: dict[str, AggregateFNSummary],
) -> Figure:
    """Stacked bar chart: FN category counts per arm.

    Parameters
    ----------
    arm_fn_summaries:
        arm_id -> AggregateFNSummary.
    """
    arms = sorted(arm_fn_summaries.keys())

    # Get all unique categories
    all_categories: list[str] = []
    for summ in arm_fn_summaries.values():
        for cat in summ.category_counts:
            if cat not in all_categories:
                all_categories.append(cat)
    all_categories.sort()

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    x = np.arange(len(arms))

    bottoms = np.zeros(len(arms))
    for cat_idx, cat in enumerate(all_categories):
        values = np.array([
            arm_fn_summaries[a].category_counts.get(cat, 0) for a in arms
        ], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=cat.replace("_", " ").title(),
            color=_arm_colour(cat_idx),
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("False Negative Count")
    ax.set_title("False Negative Categories by Arm")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 8. Score distribution (TP vs FP histograms)
# ------------------------------------------------------------------

def plot_score_distribution(
    drug_metrics: dict[str, list[DrugMetrics]],
) -> Figure:
    """Histograms of P@10 scores split by arm.

    Shows distribution of per-drug P@10 values for each arm.
    """
    arms = sorted(drug_metrics.keys())
    n_arms = len(arms)
    if n_arms == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig
    n_cols = min(n_arms, 3)
    n_rows = math.ceil(n_arms / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_arms == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, arm_id in enumerate(arms):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        p10_vals = [m.precision_at_10 for m in drug_metrics[arm_id]]
        ax.hist(
            p10_vals,
            bins=np.linspace(0, 1, 11),
            color=_arm_colour(idx),
            edgecolor="white",
            alpha=0.8,
        )
        ax.set_title(arm_id, fontsize=8)
        ax.set_xlabel("P@10", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.set_xlim(0, 1)

    # Hide unused subplots
    for idx in range(n_arms, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("P@10 Score Distribution per Arm", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 9. Ablation delta chart
# ------------------------------------------------------------------

def plot_ablation_delta(
    ablation_results: list[AggregateAblationResult],
) -> Figure:
    """Horizontal bar chart: ablation type -> mean delta P@10.

    Negative deltas (component removal hurts) are coloured red;
    positive deltas (component removal helps) are green.
    """
    ablation_results = sorted(ablation_results, key=lambda r: r.mean_delta_p_at_10)
    labels = [r.ablation_type.value.replace("_", " ").title() for r in ablation_results]
    deltas = [r.mean_delta_p_at_10 for r in ablation_results]

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    colours = ["#D55E00" if d < 0 else "#009E73" for d in deltas]

    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colours, edgecolor="white", height=0.6)

    # Value labels
    for i, d in enumerate(deltas):
        offset = -0.01 if d < 0 else 0.01
        ha = "right" if d < 0 else "left"
        ax.text(d + offset, i, f"{d:+.3f}", va="center", ha=ha, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Delta P@10 (ablated - baseline)")
    ax.set_title("Ablation Impact on P@10")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------

def generate_all_plots(
    report: ExperimentReport,
    output_dir: str | Path = "results/plots",
    *,
    heatmap_data: list[HeatmapCell] | None = None,
    fn_summaries: dict[str, AggregateFNSummary] | None = None,
    evidence_quality: dict[str, EvidenceQualityMetrics] | None = None,
    ablation_results: list[AggregateAblationResult] | None = None,
) -> dict[str, Path]:
    """Generate all available plots and save to *output_dir*.

    Parameters
    ----------
    report:
        Full ExperimentReport from the orchestrator.
    output_dir:
        Directory for plot PNGs.
    heatmap_data:
        Optional pre-computed heatmap cells from ``sensitivity.build_heatmap_data()``.
    fn_summaries:
        Optional per-arm FN summaries (requires false_negatives evaluation).
    evidence_quality:
        Optional per-arm mean evidence quality metrics.
    ablation_results:
        Optional aggregate ablation results.

    Returns
    -------
    dict[str, Path]
        Mapping of plot name to file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Build per-arm DrugMetrics lists
    arm_drug_metrics: dict[str, list[DrugMetrics]] = {}
    for dr in report.drug_results.values():
        for arm_id, dm in dr.metrics.items():
            arm_drug_metrics.setdefault(arm_id, []).append(dm)

    # Build per-arm usage totals
    arm_usage: dict[str, dict[str, float]] = {}
    for dr in report.drug_results.values():
        for arm_id, ar in dr.arm_results.items():
            if arm_id not in arm_usage:
                arm_usage[arm_id] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            arm_usage[arm_id]["input_tokens"] += ar.usage.get("input_tokens", 0)
            arm_usage[arm_id]["output_tokens"] += ar.usage.get("output_tokens", 0)
            arm_usage[arm_id]["total_tokens"] += ar.usage.get("total_tokens", 0)

    # 1. Precision / Recall bars (always available)
    try:
        fig = plot_precision_recall_bars(report.arm_aggregates)
        paths["precision_recall_bars"] = _save(fig, out / "precision_recall_bars.png")
    except Exception:
        logger.exception("Failed to generate precision_recall_bars")

    # 2. ROC-AUC boxplot
    try:
        fig = plot_roc_curves(arm_drug_metrics)
        paths["roc_curves"] = _save(fig, out / "roc_curves.png")
    except Exception:
        logger.exception("Failed to generate roc_curves")

    # 3. PR scatter
    try:
        fig = plot_pr_scatter(arm_drug_metrics)
        paths["pr_curves"] = _save(fig, out / "pr_curves.png")
    except Exception:
        logger.exception("Failed to generate pr_curves")

    # 4. Evidence quality radar (requires evidence_quality data)
    if evidence_quality:
        try:
            fig = plot_evidence_radar(evidence_quality)
            paths["evidence_radar"] = _save(fig, out / "evidence_radar.png")
        except Exception:
            logger.exception("Failed to generate evidence_radar")

    # 5. Cost bars (always available via usage stats)
    try:
        fig = plot_cost_bars(arm_usage)
        paths["cost_bars"] = _save(fig, out / "cost_bars.png")
    except Exception:
        logger.exception("Failed to generate cost_bars")

    # 6. Sensitivity heatmap (optional)
    if heatmap_data:
        try:
            fig = plot_sensitivity_heatmap(heatmap_data)
            paths["sensitivity_heatmap"] = _save(fig, out / "sensitivity_heatmap.png")
        except Exception:
            logger.exception("Failed to generate sensitivity_heatmap")

    # 7. FN distribution (optional)
    if fn_summaries:
        try:
            fig = plot_fn_distribution(fn_summaries)
            paths["fn_distribution"] = _save(fig, out / "fn_distribution.png")
        except Exception:
            logger.exception("Failed to generate fn_distribution")

    # 8. Score distribution (always available)
    try:
        fig = plot_score_distribution(arm_drug_metrics)
        paths["score_distribution"] = _save(fig, out / "score_distribution.png")
    except Exception:
        logger.exception("Failed to generate score_distribution")

    # 9. Ablation delta (optional)
    if ablation_results:
        try:
            fig = plot_ablation_delta(ablation_results)
            paths["ablation_delta"] = _save(fig, out / "ablation_delta.png")
        except Exception:
            logger.exception("Failed to generate ablation_delta")

    logger.info("Generated %d plots in %s", len(paths), out.resolve())
    return paths
