"""Sensitivity analysis and ablation study utilities.

Provides post-hoc weight sweeps, threshold sweeps, and ablation
configuration helpers for evaluating robustness of pipeline scoring.

Weight sweep: re-scores predictions with ``w_retrieval`` from 0.0 to 1.0
(step 0.1), where ``w_llm = 1 - w_retrieval``.  For each weight pair the
top-K metrics are recomputed from cached (retrieval_score, llm_confidence)
pairs.

Threshold sweep: varies the ``threshold_final`` from 0.3 to 0.9 (step 0.05)
and recomputes binary predicted/not-predicted labels, then recalculates
precision and recall at K.

Both sweeps operate purely on cached data -- no additional API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from src.evaluation.accuracy import (
    _normalize_disease,
    _precision_at_k,
    _recall_at_k,
)
from src.schemas.prediction import DrugDiseasePrediction, ScoredAssociation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Weight sweep
# ------------------------------------------------------------------

# Default sweep parameters
WEIGHT_SWEEP_START = 0.0
WEIGHT_SWEEP_END = 1.0
WEIGHT_SWEEP_STEP = 0.1

THRESHOLD_SWEEP_START = 0.3
THRESHOLD_SWEEP_END = 0.9
THRESHOLD_SWEEP_STEP = 0.05


@dataclass
class SweepPoint:
    """Metrics at a single (w_retrieval, threshold) coordinate."""

    w_retrieval: float
    w_llm: float
    threshold: float
    precision_at_1: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0
    n_predicted: int = 0
    n_ground_truth: int = 0


@dataclass
class WeightSweepResult:
    """Full weight sweep for one drug across all weight values."""

    drug_name: str
    arm_id: str
    threshold: float
    points: list[SweepPoint] = field(default_factory=list)


@dataclass
class ThresholdSweepResult:
    """Full threshold sweep for one drug across all threshold values."""

    drug_name: str
    arm_id: str
    w_retrieval: float
    w_llm: float
    points: list[SweepPoint] = field(default_factory=list)


# ------------------------------------------------------------------
# Cached score container
# ------------------------------------------------------------------

class CachedScore(BaseModel):
    """Per-association cached scores used for post-hoc resweep.

    When running the pipeline arm, both the retrieval score and the LLM
    confidence are available.  For baselines only LLM confidence exists,
    so ``retrieval_score`` defaults to 0.0.
    """

    disease_name: str
    retrieval_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Retrieval subsystem score (0 for baselines)",
    )
    llm_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="LLM-assigned confidence",
    )

    def combined_score(self, w_retrieval: float) -> float:
        """Compute weighted combination: w_r * retrieval + (1 - w_r) * llm."""
        w_llm = 1.0 - w_retrieval
        return w_retrieval * self.retrieval_score + w_llm * self.llm_confidence


def extract_cached_scores(
    prediction: DrugDiseasePrediction,
    retrieval_scores: dict[str, float] | None = None,
) -> list[CachedScore]:
    """Extract ``CachedScore`` list from a prediction.

    Parameters
    ----------
    prediction:
        Agent output with associations and their LLM confidence.
    retrieval_scores:
        Optional mapping of normalised disease name to retrieval score.
        When ``None`` (baseline arms), retrieval_score defaults to 0.0.

    Returns
    -------
    list[CachedScore]
        One entry per association in the prediction.
    """
    scores: list[CachedScore] = []
    ret_map = retrieval_scores or {}
    for assoc in prediction.associations:
        norm = _normalize_disease(assoc.disease_name)
        scores.append(
            CachedScore(
                disease_name=assoc.disease_name,
                retrieval_score=ret_map.get(norm, 0.0),
                llm_confidence=assoc.confidence,
            )
        )
    return scores


# ------------------------------------------------------------------
# Recompute metrics at a given (w_retrieval, threshold) point
# ------------------------------------------------------------------

def _recompute_metrics(
    cached_scores: list[CachedScore],
    ground_truth: set[str],
    w_retrieval: float,
    threshold: float,
) -> SweepPoint:
    """Re-score and recompute P@K / R@K at one (weight, threshold) point.

    1. Compute combined score for each association.
    2. Sort by descending combined score.
    3. Apply threshold to produce the ``predicted`` flag.
    4. Compute P@1, P@10, R@1, R@10 on the predicted subset.
    """
    w_llm = 1.0 - w_retrieval
    norm_gt = {_normalize_disease(d) for d in ground_truth}

    # Score and sort
    scored = [
        (cs.disease_name, cs.combined_score(w_retrieval))
        for cs in cached_scores
    ]
    scored.sort(key=lambda x: -x[1])

    # Apply threshold
    predicted_names = [name for name, score in scored if score >= threshold]

    return SweepPoint(
        w_retrieval=round(w_retrieval, 4),
        w_llm=round(w_llm, 4),
        threshold=round(threshold, 4),
        precision_at_1=_precision_at_k(predicted_names, norm_gt, 1),
        precision_at_10=_precision_at_k(predicted_names, norm_gt, 10),
        recall_at_1=_recall_at_k(predicted_names, norm_gt, 1),
        recall_at_10=_recall_at_k(predicted_names, norm_gt, 10),
        n_predicted=len(predicted_names),
        n_ground_truth=len(norm_gt),
    )


# ------------------------------------------------------------------
# Weight sweep (fixed threshold, vary w_retrieval)
# ------------------------------------------------------------------

def weight_sweep(
    cached_scores: list[CachedScore],
    ground_truth: set[str],
    arm_id: str,
    drug_name: str,
    threshold: float = 0.5,
    *,
    start: float = WEIGHT_SWEEP_START,
    end: float = WEIGHT_SWEEP_END,
    step: float = WEIGHT_SWEEP_STEP,
) -> WeightSweepResult:
    """Sweep ``w_retrieval`` from *start* to *end* (inclusive) at fixed threshold.

    Parameters
    ----------
    cached_scores:
        Cached (retrieval_score, llm_confidence) per association.
    ground_truth:
        Normalised ground-truth disease names.
    arm_id / drug_name:
        Metadata for the result container.
    threshold:
        Fixed threshold for the predicted / not-predicted cutoff.
    start / end / step:
        Sweep range (defaults 0.0 .. 1.0, step 0.1 -> 11 points).

    Returns
    -------
    WeightSweepResult
    """
    points: list[SweepPoint] = []
    w = start
    while w <= end + 1e-9:
        pt = _recompute_metrics(cached_scores, ground_truth, w, threshold)
        points.append(pt)
        w += step
        w = round(w, 6)  # avoid floating point drift

    return WeightSweepResult(
        drug_name=drug_name,
        arm_id=arm_id,
        threshold=threshold,
        points=points,
    )


# ------------------------------------------------------------------
# Threshold sweep (fixed weights, vary threshold)
# ------------------------------------------------------------------

def threshold_sweep(
    cached_scores: list[CachedScore],
    ground_truth: set[str],
    arm_id: str,
    drug_name: str,
    w_retrieval: float = 0.5,
    *,
    start: float = THRESHOLD_SWEEP_START,
    end: float = THRESHOLD_SWEEP_END,
    step: float = THRESHOLD_SWEEP_STEP,
) -> ThresholdSweepResult:
    """Sweep ``threshold_final`` from *start* to *end* at fixed weights.

    Parameters
    ----------
    cached_scores:
        Cached (retrieval_score, llm_confidence) per association.
    ground_truth:
        Normalised ground-truth disease names.
    arm_id / drug_name:
        Metadata for the result container.
    w_retrieval:
        Fixed retrieval weight.
    start / end / step:
        Sweep range (defaults 0.3 .. 0.9, step 0.05 -> 13 points).

    Returns
    -------
    ThresholdSweepResult
    """
    w_llm = 1.0 - w_retrieval
    points: list[SweepPoint] = []
    t = start
    while t <= end + 1e-9:
        pt = _recompute_metrics(cached_scores, ground_truth, w_retrieval, t)
        points.append(pt)
        t += step
        t = round(t, 6)

    return ThresholdSweepResult(
        drug_name=drug_name,
        arm_id=arm_id,
        w_retrieval=w_retrieval,
        w_llm=w_llm,
        points=points,
    )


# ------------------------------------------------------------------
# Aggregate sweep results across drugs
# ------------------------------------------------------------------

@dataclass
class AggregateSweepPoint:
    """Mean metrics at one (w_retrieval, threshold) across N drugs."""

    w_retrieval: float
    w_llm: float
    threshold: float
    mean_precision_at_1: float = 0.0
    mean_precision_at_10: float = 0.0
    mean_recall_at_1: float = 0.0
    mean_recall_at_10: float = 0.0
    n_drugs: int = 0


def aggregate_weight_sweeps(
    sweeps: list[WeightSweepResult],
) -> list[AggregateSweepPoint]:
    """Average weight sweep results across multiple drugs.

    All sweeps must share the same set of ``w_retrieval`` values (i.e.
    generated with the same start/end/step).

    Returns
    -------
    list[AggregateSweepPoint]
        One entry per weight value, ordered ascending by ``w_retrieval``.
    """
    if not sweeps:
        return []

    n = len(sweeps)
    n_points = len(sweeps[0].points)

    agg: list[AggregateSweepPoint] = []
    for i in range(n_points):
        pts = [s.points[i] for s in sweeps if i < len(s.points)]
        if not pts:
            continue
        agg.append(
            AggregateSweepPoint(
                w_retrieval=pts[0].w_retrieval,
                w_llm=pts[0].w_llm,
                threshold=pts[0].threshold,
                mean_precision_at_1=sum(p.precision_at_1 for p in pts) / n,
                mean_precision_at_10=sum(p.precision_at_10 for p in pts) / n,
                mean_recall_at_1=sum(p.recall_at_1 for p in pts) / n,
                mean_recall_at_10=sum(p.recall_at_10 for p in pts) / n,
                n_drugs=len(pts),
            )
        )
    return agg


def aggregate_threshold_sweeps(
    sweeps: list[ThresholdSweepResult],
) -> list[AggregateSweepPoint]:
    """Average threshold sweep results across multiple drugs.

    Returns
    -------
    list[AggregateSweepPoint]
        One entry per threshold value, ordered ascending by ``threshold``.
    """
    if not sweeps:
        return []

    n = len(sweeps)
    n_points = len(sweeps[0].points)

    agg: list[AggregateSweepPoint] = []
    for i in range(n_points):
        pts = [s.points[i] for s in sweeps if i < len(s.points)]
        if not pts:
            continue
        agg.append(
            AggregateSweepPoint(
                w_retrieval=pts[0].w_retrieval,
                w_llm=pts[0].w_llm,
                threshold=pts[0].threshold,
                mean_precision_at_1=sum(p.precision_at_1 for p in pts) / n,
                mean_precision_at_10=sum(p.precision_at_10 for p in pts) / n,
                mean_recall_at_1=sum(p.recall_at_1 for p in pts) / n,
                mean_recall_at_10=sum(p.recall_at_10 for p in pts) / n,
                n_drugs=len(pts),
            )
        )
    return agg


# ------------------------------------------------------------------
# Ablation configuration
# ------------------------------------------------------------------

class AblationType(str, Enum):
    """Predefined ablation studies from the project plan."""

    NO_DGIDB = "no_dgidb"
    NO_PUBCHEM = "no_pubchem"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    NO_TOOLS = "no_tools"


@dataclass
class AblationConfig:
    """Configuration for one ablation study.

    Each ablation removes a specific component from the pipeline
    to measure its marginal contribution.
    """

    ablation_type: AblationType
    description: str
    disabled_sources: list[str] = field(default_factory=list)
    search_mode: str = "hybrid"  # "hybrid", "dense", "sparse"
    tools_enabled: bool = True

    @property
    def label(self) -> str:
        return self.ablation_type.value


# Pre-defined ablation configurations per project plan
ABLATION_CONFIGS: dict[AblationType, AblationConfig] = {
    AblationType.NO_DGIDB: AblationConfig(
        ablation_type=AblationType.NO_DGIDB,
        description="Remove DGIdb from evidence sources",
        disabled_sources=["dgidb"],
    ),
    AblationType.NO_PUBCHEM: AblationConfig(
        ablation_type=AblationType.NO_PUBCHEM,
        description="Remove PubChem from evidence sources",
        disabled_sources=["pubchem"],
    ),
    AblationType.DENSE_ONLY: AblationConfig(
        ablation_type=AblationType.DENSE_ONLY,
        description="Remove sparse SPLADE, use dense-only search",
        search_mode="dense",
    ),
    AblationType.SPARSE_ONLY: AblationConfig(
        ablation_type=AblationType.SPARSE_ONLY,
        description="Remove dense MedCPT, use SPLADE-only search",
        search_mode="sparse",
    ),
    AblationType.NO_TOOLS: AblationConfig(
        ablation_type=AblationType.NO_TOOLS,
        description="Remove agent tools, prompt-only",
        tools_enabled=False,
    ),
}


# ------------------------------------------------------------------
# Ablation result container
# ------------------------------------------------------------------

@dataclass
class AblationResult:
    """Comparison of baseline vs. ablated arm for one ablation study."""

    ablation_type: AblationType
    arm_id: str
    drug_name: str

    # Baseline (full pipeline) metrics
    baseline_p_at_1: float = 0.0
    baseline_p_at_10: float = 0.0
    baseline_r_at_1: float = 0.0
    baseline_r_at_10: float = 0.0

    # Ablated metrics
    ablated_p_at_1: float = 0.0
    ablated_p_at_10: float = 0.0
    ablated_r_at_1: float = 0.0
    ablated_r_at_10: float = 0.0

    @property
    def delta_p_at_10(self) -> float:
        """Change in P@10 when component is removed (negative = component helps)."""
        return self.ablated_p_at_10 - self.baseline_p_at_10

    @property
    def delta_r_at_10(self) -> float:
        """Change in R@10 when component is removed."""
        return self.ablated_r_at_10 - self.baseline_r_at_10


@dataclass
class AggregateAblationResult:
    """Mean ablation deltas across all drugs for one ablation type."""

    ablation_type: AblationType
    n_drugs: int = 0
    mean_delta_p_at_1: float = 0.0
    mean_delta_p_at_10: float = 0.0
    mean_delta_r_at_1: float = 0.0
    mean_delta_r_at_10: float = 0.0


def aggregate_ablation_results(
    results: list[AblationResult],
) -> AggregateAblationResult:
    """Compute mean ablation deltas across drugs.

    All results must share the same ``ablation_type``.
    """
    if not results:
        raise ValueError("Cannot aggregate empty result list")

    atype = results[0].ablation_type
    n = len(results)

    return AggregateAblationResult(
        ablation_type=atype,
        n_drugs=n,
        mean_delta_p_at_1=sum(
            r.ablated_p_at_1 - r.baseline_p_at_1 for r in results
        ) / n,
        mean_delta_p_at_10=sum(
            r.ablated_p_at_10 - r.baseline_p_at_10 for r in results
        ) / n,
        mean_delta_r_at_1=sum(
            r.ablated_r_at_1 - r.baseline_r_at_1 for r in results
        ) / n,
        mean_delta_r_at_10=sum(
            r.ablated_r_at_10 - r.baseline_r_at_10 for r in results
        ) / n,
    )


# ------------------------------------------------------------------
# 2D heatmap data (w_retrieval x threshold -> metric)
# ------------------------------------------------------------------

@dataclass
class HeatmapCell:
    """Single cell in a 2D weight-threshold heatmap."""

    w_retrieval: float
    threshold: float
    value: float


def build_heatmap_data(
    cached_scores: list[CachedScore],
    ground_truth: set[str],
    metric: str = "precision_at_10",
    *,
    w_start: float = WEIGHT_SWEEP_START,
    w_end: float = WEIGHT_SWEEP_END,
    w_step: float = WEIGHT_SWEEP_STEP,
    t_start: float = THRESHOLD_SWEEP_START,
    t_end: float = THRESHOLD_SWEEP_END,
    t_step: float = THRESHOLD_SWEEP_STEP,
) -> list[HeatmapCell]:
    """Build 2D grid of metric values over (w_retrieval, threshold).

    Parameters
    ----------
    cached_scores:
        Per-association cached scores.
    ground_truth:
        Normalised ground-truth disease set.
    metric:
        Which metric to extract from each ``SweepPoint``.  Must be one of
        ``precision_at_1``, ``precision_at_10``, ``recall_at_1``,
        ``recall_at_10``.
    w_start, w_end, w_step:
        Weight sweep range.
    t_start, t_end, t_step:
        Threshold sweep range.

    Returns
    -------
    list[HeatmapCell]
        Flat list of cells (row-major: weight outer, threshold inner).
    """
    valid_metrics = {"precision_at_1", "precision_at_10", "recall_at_1", "recall_at_10"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

    cells: list[HeatmapCell] = []
    w = w_start
    while w <= w_end + 1e-9:
        t = t_start
        while t <= t_end + 1e-9:
            pt = _recompute_metrics(cached_scores, ground_truth, w, t)
            cells.append(
                HeatmapCell(
                    w_retrieval=round(w, 4),
                    threshold=round(t, 4),
                    value=getattr(pt, metric),
                )
            )
            t += t_step
            t = round(t, 6)
        w += w_step
        w = round(w, 6)

    return cells
