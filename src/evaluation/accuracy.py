"""Accuracy metrics for drug-disease prediction evaluation.

Computes Precision@K, Recall@K, and ROC-AUC by comparing
``DrugDiseasePrediction`` output against CTD ground truth.

Matching pipeline (applied to both predicted and ground-truth names):

1. **Deep normalisation** -- strip parenthetical suffixes, de-invert
   MeSH comma syntax (``"Leukemia, Myeloid"`` -> ``"myeloid leukemia"``),
   lowercase, collapse whitespace.
2. **Fuzzy fallback** -- when exact normalised match fails, use
   ``rapidfuzz.fuzz.token_sort_ratio`` with a configurable threshold
   (default 80) to catch remaining synonym / plural differences.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz

from src.schemas.prediction import DrugDiseasePrediction, DrugDifficulty

logger = logging.getLogger(__name__)

# Minimum token-sort-ratio score (0-100) to count as a fuzzy match.
_FUZZY_THRESHOLD: int = 80


# ------------------------------------------------------------------
# Per-drug metric container
# ------------------------------------------------------------------

@dataclass
class DrugMetrics:
    """Evaluation metrics for one drug against its ground-truth diseases."""

    drug_name: str
    arm_id: str
    difficulty: DrugDifficulty | None = None
    n_ground_truth: int = 0
    n_predicted: int = 0

    # Precision / Recall at various K
    precision_at_1: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0

    # ROC-AUC (may be None if only one class present)
    roc_auc: float | None = None

    # Matched diseases (for debugging)
    true_positives_at_10: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Disease-name normalisation
# ------------------------------------------------------------------

# Regex to strip parenthetical suffixes:  "narcolepsy (excessive ...)" -> "narcolepsy"
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*")
# Regex to strip trailing qualifiers after " -- " or " - " dash:
_DASH_QUALIFIER_RE = re.compile(r"\s+[-\u2013\u2014]{1,2}\s+.*$")
# Regex for slash-separated alternatives: keep only the first part
_SLASH_RE = re.compile(r"\s*/\s*")


def _normalize_disease(name: str) -> str:
    """Backward-compatible lightweight normalisation.

    Kept for any external callers that rely on the old signature.
    Internally the evaluation now uses ``_deep_normalize_disease``.
    """
    return " ".join(name.lower().strip().split())


@lru_cache(maxsize=4096)
def _deep_normalize_disease(name: str) -> str:
    """Aggressive normalisation for CTD <-> LLM name matching.

    Steps:
    1. Strip parenthetical annotations -- ``(MDS)``, ``(eczema)``, etc.
    2. De-invert MeSH comma syntax -- ``"Leukemia, Myeloid"``
       becomes ``"myeloid leukemia"``.
    3. Strip trailing dash qualifiers.
    4. Take only the first slash-alternative.
    5. Lowercase, collapse whitespace.
    """
    s = name

    # 1. Remove parenthetical content
    s = _PAREN_RE.sub(" ", s)

    # 2. De-invert MeSH "Primary, Qualifier1, Qualifier2" style
    #    "Leukemia, Myelomonocytic, Chronic" -> "chronic myelomonocytic leukemia"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            # Reverse qualifier order and append the primary term
            s = " ".join(parts[1:]) + " " + parts[0]

    # 3. Strip trailing dash qualifiers
    s = _DASH_QUALIFIER_RE.sub("", s)

    # 4. First slash-alternative only
    if "/" in s:
        s = _SLASH_RE.split(s)[0]

    # 5. Lowercase + collapse whitespace
    return " ".join(s.lower().split())


def _matches_ground_truth(
    predicted_name: str,
    ground_truth_set: set[str],
    gt_deep_norms: set[str] | None = None,
) -> bool:
    """Return True if *predicted_name* matches any entry in *ground_truth_set*.

    Tries exact deep-normalised match first; falls back to fuzzy
    token-sort-ratio if no exact hit.

    Parameters
    ----------
    predicted_name:
        A single disease name produced by the LLM.
    ground_truth_set:
        The full set of deep-normalised ground-truth names.
    gt_deep_norms:
        Pre-computed ``{_deep_normalize_disease(g) for g in ...}``.
        If ``None``, *ground_truth_set* is assumed to already be
        deep-normalised.
    """
    norm_pred = _deep_normalize_disease(predicted_name)
    gt_normed = gt_deep_norms if gt_deep_norms is not None else ground_truth_set

    # Fast path: exact normalised match
    if norm_pred in gt_normed:
        return True

    # Slow path: fuzzy match against each ground-truth entry
    for gt in gt_normed:
        score = fuzz.token_sort_ratio(norm_pred, gt)
        if score >= _FUZZY_THRESHOLD:
            return True

    return False


def _precision_at_k(
    predicted_diseases: list[str],
    ground_truth_deep: set[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K predictions that are correct.

    *ground_truth_deep* must already be deep-normalised.
    """
    if k <= 0 or not predicted_diseases:
        return 0.0
    top_k = predicted_diseases[:k]
    hits = sum(
        1 for d in top_k
        if _matches_ground_truth(d, ground_truth_deep)
    )
    return hits / len(top_k)


def _recall_at_k(
    predicted_diseases: list[str],
    ground_truth_deep: set[str],
    k: int,
) -> float:
    """Recall@K: fraction of ground-truth diseases found in top-K.

    *ground_truth_deep* must already be deep-normalised.
    """
    if k <= 0 or not ground_truth_deep:
        return 0.0
    top_k = predicted_diseases[:k]
    top_k_deep = {_deep_normalize_disease(d) for d in top_k}

    def _gt_found(gt_norm: str) -> bool:
        if gt_norm in top_k_deep:
            return True
        for pred_norm in top_k_deep:
            if fuzz.token_sort_ratio(gt_norm, pred_norm) >= _FUZZY_THRESHOLD:
                return True
        return False

    hits = sum(1 for gt in ground_truth_deep if _gt_found(gt))
    return hits / len(ground_truth_deep)


def _roc_auc_manual(
    scored_diseases: list[tuple[str, float]],
    ground_truth_deep: set[str],
) -> float | None:
    """Compute ROC-AUC from (disease, confidence) pairs.

    Uses the trapezoidal rule on the ROC curve.  Returns ``None``
    when there is only one class (all positives or all negatives).

    *ground_truth_deep* must already be deep-normalised.
    """
    if not scored_diseases or not ground_truth_deep:
        return None

    # Sort by descending confidence (stable sort)
    sorted_pairs = sorted(scored_diseases, key=lambda x: -x[1])
    labels = [
        1 if _matches_ground_truth(d, ground_truth_deep) else 0
        for d, _ in sorted_pairs
    ]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # undefined AUC

    # Wilcoxon-Mann-Whitney statistic
    tp = 0
    fp = 0
    auc = 0.0
    tp_prev = 0
    fp_prev = 0
    prev_score = -math.inf

    for i, label in enumerate(labels):
        score = sorted_pairs[i][1]
        if score != prev_score:
            # trapezoidal step
            auc += (fp - fp_prev) * (tp + tp_prev) / 2
            tp_prev = tp
            fp_prev = fp
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1
    auc += (fp - fp_prev) * (tp + tp_prev) / 2

    if n_pos * n_neg == 0:
        return None
    return auc / (n_pos * n_neg)


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def evaluate_prediction(
    prediction: DrugDiseasePrediction,
    ground_truth_diseases: set[str],
    arm_id: str,
    difficulty: DrugDifficulty | None = None,
) -> DrugMetrics:
    """Score a single drug's prediction against CTD ground truth.

    Parameters
    ----------
    prediction:
        Structured output from any experimental arm.
    ground_truth_diseases:
        Set of disease names from CTD (already lowercased + normalised).
    arm_id:
        Identifier of the arm that produced this prediction.
    difficulty:
        Optional difficulty label for stratification.

    Returns
    -------
    DrugMetrics
        All computed metrics for one (drug, arm) pair.
    """
    # Extract predicted disease names sorted by descending confidence
    associations = sorted(
        prediction.associations,
        key=lambda a: a.confidence,
        reverse=True,
    )
    predicted_names = [a.disease_name for a in associations if a.predicted]
    scored_pairs = [(a.disease_name, a.confidence) for a in associations]

    # Deep-normalise ground truth for matching
    norm_gt = {_deep_normalize_disease(d) for d in ground_truth_diseases}

    # Compute metrics
    p1 = _precision_at_k(predicted_names, norm_gt, 1)
    p10 = _precision_at_k(predicted_names, norm_gt, 10)
    r1 = _recall_at_k(predicted_names, norm_gt, 1)
    r10 = _recall_at_k(predicted_names, norm_gt, 10)
    auc = _roc_auc_manual(scored_pairs, norm_gt)

    # Debug: which diseases matched at k=10
    top_10 = predicted_names[:10]
    tp_names = [
        d for d in top_10
        if _matches_ground_truth(d, norm_gt)
    ]

    return DrugMetrics(
        drug_name=prediction.drug_name,
        arm_id=arm_id,
        difficulty=difficulty,
        n_ground_truth=len(norm_gt),
        n_predicted=len(predicted_names),
        precision_at_1=p1,
        precision_at_10=p10,
        recall_at_1=r1,
        recall_at_10=r10,
        roc_auc=auc,
        true_positives_at_10=tp_names,
    )


# ------------------------------------------------------------------
# Aggregate metrics across drugs
# ------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Summary statistics across all drugs for one arm."""

    arm_id: str
    n_drugs: int = 0
    mean_precision_at_1: float = 0.0
    mean_precision_at_10: float = 0.0
    mean_recall_at_1: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_roc_auc: float | None = None

    # Per-difficulty breakdowns (key = difficulty name)
    by_difficulty: dict[str, "AggregateMetrics"] = field(default_factory=dict)


def aggregate_metrics(
    drug_metrics: list[DrugMetrics],
    arm_id: str,
    *,
    _include_difficulty_breakdown: bool = True,
) -> AggregateMetrics:
    """Compute mean metrics across all drugs for one arm.

    Also produces per-difficulty breakdowns using the
    ``DrugDifficulty`` covariate on each ``DrugMetrics``.
    """
    if not drug_metrics:
        return AggregateMetrics(arm_id=arm_id)

    n = len(drug_metrics)
    agg = AggregateMetrics(
        arm_id=arm_id,
        n_drugs=n,
        mean_precision_at_1=sum(m.precision_at_1 for m in drug_metrics) / n,
        mean_precision_at_10=sum(m.precision_at_10 for m in drug_metrics) / n,
        mean_recall_at_1=sum(m.recall_at_1 for m in drug_metrics) / n,
        mean_recall_at_10=sum(m.recall_at_10 for m in drug_metrics) / n,
    )

    # Mean AUC (ignoring drugs where AUC is undefined)
    auc_values = [m.roc_auc for m in drug_metrics if m.roc_auc is not None]
    if auc_values:
        agg.mean_roc_auc = sum(auc_values) / len(auc_values)

    # Per-difficulty breakdown (one level only -- no recursion)
    if _include_difficulty_breakdown:
        by_diff: dict[str, list[DrugMetrics]] = {}
        for m in drug_metrics:
            key = m.difficulty.value if m.difficulty else "unknown"
            by_diff.setdefault(key, []).append(m)

        for diff_name, diff_metrics in by_diff.items():
            agg.by_difficulty[diff_name] = aggregate_metrics(
                diff_metrics, arm_id, _include_difficulty_breakdown=False
            )

    return agg
