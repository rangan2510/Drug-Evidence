"""False negative analysis for drug-disease predictions.

Categorises each ground-truth association that the pipeline missed into
a taxonomy of failure modes, enabling targeted improvements.

Six FN categories (from project plan Phase 15):

1. NO_EVIDENCE_FOUND -- no documents retrieved for this association
2. LOW_RETRIEVAL_SCORE -- docs found but retrieval score below gate
3. LOW_LLM_CONFIDENCE -- good retrieval but LLM not convinced
4. BELOW_THRESHOLD -- combined score below threshold
5. NOT_IN_CANDIDATES -- association not in candidate set at all
6. NAME_MISMATCH -- disease name differs (fuzzy / embedding similarity)

Near-miss detection uses MedCPT query embeddings to compute cosine
similarity between ground-truth disease names and predicted names.
Similarities above 0.7 suggest name normalisation issues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from src.evaluation.accuracy import (
    _deep_normalize_disease,
    _matches_ground_truth,
    _normalize_disease,
)
from src.schemas.prediction import DrugDiseasePrediction

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# FN taxonomy
# ------------------------------------------------------------------

class FalseNegativeCategory(str, Enum):
    """Taxonomy of why a ground-truth association was missed."""

    NO_EVIDENCE_FOUND = "no_evidence_found"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    LOW_LLM_CONFIDENCE = "low_llm_confidence"
    BELOW_THRESHOLD = "below_threshold"
    NOT_IN_CANDIDATES = "not_in_candidates"
    NAME_MISMATCH = "name_mismatch"


# ------------------------------------------------------------------
# FN record
# ------------------------------------------------------------------

class FalseNegativeRecord(BaseModel):
    """Single false negative for one drug-disease pair."""

    drug: str
    disease_gt: str = Field(description="Ground-truth disease name")
    category: FalseNegativeCategory
    closest_prediction: str | None = Field(
        default=None, description="Nearest predicted disease by embedding similarity"
    )
    similarity_to_closest: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Cosine similarity to closest prediction",
    )
    retrieval_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Retrieval score (if available)",
    )
    llm_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="LLM confidence (if available)",
    )
    evidence_sources_checked: list[str] = Field(
        default_factory=list,
        description="Which databases had data for this association",
    )


# ------------------------------------------------------------------
# FN analysis summary
# ------------------------------------------------------------------

@dataclass
class FalseNegativeSummary:
    """Aggregate FN analysis for one drug across one arm."""

    drug_name: str
    arm_id: str
    n_ground_truth: int = 0
    n_true_positives: int = 0
    n_false_negatives: int = 0
    records: list[FalseNegativeRecord] = field(default_factory=list)

    @property
    def fn_rate(self) -> float:
        """Fraction of ground-truth diseases that were missed."""
        if self.n_ground_truth == 0:
            return 0.0
        return self.n_false_negatives / self.n_ground_truth

    def category_counts(self) -> dict[str, int]:
        """Count FNs by category."""
        counts: dict[str, int] = {cat.value: 0 for cat in FalseNegativeCategory}
        for rec in self.records:
            counts[rec.category.value] += 1
        return counts


# ------------------------------------------------------------------
# Near-miss detection helpers
# ------------------------------------------------------------------

NEAR_MISS_THRESHOLD = 0.7


def _cosine_similarity_matrix(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute pairwise cosine similarity between row vectors in *a* and *b*.

    Parameters
    ----------
    a : (M, D) array
    b : (N, D) array

    Returns
    -------
    (M, N) cosine similarity matrix.
    """
    # Normalise
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def find_nearest_prediction(
    gt_name: str,
    predicted_names: list[str],
    gt_embeddings: NDArray[np.float32] | None = None,
    pred_embeddings: NDArray[np.float32] | None = None,
    gt_index: int | None = None,
) -> tuple[str | None, float | None]:
    """Find the predicted disease most similar to a ground-truth disease.

    Uses MedCPT embeddings if provided, otherwise falls back to
    normalised string matching (exact match -> 1.0, else 0.0).

    Parameters
    ----------
    gt_name:
        The ground-truth disease name.
    predicted_names:
        All predicted disease names.
    gt_embeddings:
        Pre-computed MedCPT embeddings for all GT diseases (M, D).
    pred_embeddings:
        Pre-computed MedCPT embeddings for all predicted diseases (N, D).
    gt_index:
        Index of ``gt_name`` in ``gt_embeddings``.

    Returns
    -------
    (closest_name, similarity) or (None, None) if no predictions exist.
    """
    if not predicted_names:
        return None, None

    # Embedding-based similarity
    if (
        gt_embeddings is not None
        and pred_embeddings is not None
        and gt_index is not None
        and gt_index < gt_embeddings.shape[0]
        and pred_embeddings.shape[0] > 0
    ):
        gt_vec = gt_embeddings[gt_index : gt_index + 1]  # (1, D)
        sims = _cosine_similarity_matrix(gt_vec, pred_embeddings)[0]  # (N,)
        best_idx = int(np.argmax(sims))
        return predicted_names[best_idx], float(sims[best_idx])

    # Fallback: deep-normalised + fuzzy match
    norm_gt = _deep_normalize_disease(gt_name)
    for pn in predicted_names:
        if _deep_normalize_disease(pn) == norm_gt:
            return pn, 1.0
    # Fuzzy fallback
    from rapidfuzz import fuzz as _fuzz
    best_name, best_score = predicted_names[0], 0.0
    for pn in predicted_names:
        score = _fuzz.token_sort_ratio(norm_gt, _deep_normalize_disease(pn)) / 100.0
        if score > best_score:
            best_name, best_score = pn, score
    return best_name, best_score


# ------------------------------------------------------------------
# FN categorisation logic
# ------------------------------------------------------------------

def _categorise_fn(
    gt_disease: str,
    prediction: DrugDiseasePrediction,
    retrieval_scores: dict[str, float] | None = None,
    evidence_sources: dict[str, list[str]] | None = None,
    threshold: float = 0.5,
    retrieval_gate: float = 0.3,
    *,
    predicted_names: list[str] | None = None,
    gt_embeddings: NDArray[np.float32] | None = None,
    pred_embeddings: NDArray[np.float32] | None = None,
    gt_index: int | None = None,
    near_miss_threshold: float = NEAR_MISS_THRESHOLD,
) -> FalseNegativeRecord:
    """Categorise a single false negative.

    The categorisation follows a decision tree:
    1. Is the disease in the candidate set at all?
       No -> NOT_IN_CANDIDATES
    2. Does any retrieved evidence exist?
       No -> NO_EVIDENCE_FOUND
    3. Is the retrieval score above the gate?
       No -> LOW_RETRIEVAL_SCORE
    4. Is the LLM confidence reasonable (above gate)?
       No -> LOW_LLM_CONFIDENCE
    5. Is the combined score above threshold?
       No -> BELOW_THRESHOLD
    6. Otherwise -> NAME_MISMATCH (present but under a different name)
    """
    norm_gt = _deep_normalize_disease(gt_disease)
    ret_map = retrieval_scores or {}
    src_map = evidence_sources or {}

    # All predicted names for nearest-match
    all_predicted = predicted_names or [
        a.disease_name for a in prediction.associations
    ]

    # Find nearest prediction
    closest, sim = find_nearest_prediction(
        gt_disease, all_predicted,
        gt_embeddings, pred_embeddings, gt_index,
    )

    # Check if match exists in predictions (deep-normalised + fuzzy)
    norm_predictions = {
        _deep_normalize_disease(a.disease_name): a
        for a in prediction.associations
    }

    # Sources checked for this disease
    sources_checked = src_map.get(norm_gt, [])

    # --- Decision tree ---

    # 1. Check if in candidate set at all (exact deep-norm or fuzzy)
    matched_key = norm_gt if norm_gt in norm_predictions else None
    if matched_key is None:
        # Try fuzzy match against deep-normalised prediction keys
        from rapidfuzz import fuzz as _fuzz
        for pkey in norm_predictions:
            if _fuzz.token_sort_ratio(norm_gt, pkey) >= 80:
                matched_key = pkey
                break

    if matched_key is None:
        # Might be a name mismatch if high embedding similarity
        if sim is not None and sim >= near_miss_threshold:
            matched_assoc = norm_predictions.get(
                _deep_normalize_disease(closest) if closest else "", None
            )
            return FalseNegativeRecord(
                drug=prediction.drug_name,
                disease_gt=gt_disease,
                category=FalseNegativeCategory.NAME_MISMATCH,
                closest_prediction=closest,
                similarity_to_closest=sim,
                retrieval_score=matched_assoc.confidence if matched_assoc else None,
                llm_confidence=matched_assoc.confidence if matched_assoc else None,
                evidence_sources_checked=sources_checked,
            )

        return FalseNegativeRecord(
            drug=prediction.drug_name,
            disease_gt=gt_disease,
            category=FalseNegativeCategory.NOT_IN_CANDIDATES,
            closest_prediction=closest,
            similarity_to_closest=sim,
            evidence_sources_checked=sources_checked,
        )

    # Disease is in predictions -- get its association record
    assoc = norm_predictions[matched_key]
    ret_score = ret_map.get(norm_gt) or ret_map.get(matched_key)

    # 2. No evidence found
    if not sources_checked and ret_score is None:
        return FalseNegativeRecord(
            drug=prediction.drug_name,
            disease_gt=gt_disease,
            category=FalseNegativeCategory.NO_EVIDENCE_FOUND,
            closest_prediction=assoc.disease_name,
            similarity_to_closest=1.0,
            llm_confidence=assoc.confidence,
            evidence_sources_checked=sources_checked,
        )

    # 3. Low retrieval score
    if ret_score is not None and ret_score < retrieval_gate:
        return FalseNegativeRecord(
            drug=prediction.drug_name,
            disease_gt=gt_disease,
            category=FalseNegativeCategory.LOW_RETRIEVAL_SCORE,
            closest_prediction=assoc.disease_name,
            similarity_to_closest=1.0,
            retrieval_score=ret_score,
            llm_confidence=assoc.confidence,
            evidence_sources_checked=sources_checked,
        )

    # 4. Low LLM confidence
    if assoc.confidence < retrieval_gate:
        return FalseNegativeRecord(
            drug=prediction.drug_name,
            disease_gt=gt_disease,
            category=FalseNegativeCategory.LOW_LLM_CONFIDENCE,
            closest_prediction=assoc.disease_name,
            similarity_to_closest=1.0,
            retrieval_score=ret_score,
            llm_confidence=assoc.confidence,
            evidence_sources_checked=sources_checked,
        )

    # 5. Below combined threshold (predicted=False or low confidence)
    if not assoc.predicted or assoc.confidence < threshold:
        return FalseNegativeRecord(
            drug=prediction.drug_name,
            disease_gt=gt_disease,
            category=FalseNegativeCategory.BELOW_THRESHOLD,
            closest_prediction=assoc.disease_name,
            similarity_to_closest=1.0,
            retrieval_score=ret_score,
            llm_confidence=assoc.confidence,
            evidence_sources_checked=sources_checked,
        )

    # 6. Must be a name mismatch or edge case
    return FalseNegativeRecord(
        drug=prediction.drug_name,
        disease_gt=gt_disease,
        category=FalseNegativeCategory.NAME_MISMATCH,
        closest_prediction=closest,
        similarity_to_closest=sim,
        retrieval_score=ret_score,
        llm_confidence=assoc.confidence,
        evidence_sources_checked=sources_checked,
    )


# ------------------------------------------------------------------
# FN Analyzer -- main entry point
# ------------------------------------------------------------------

def analyse_false_negatives(
    prediction: DrugDiseasePrediction,
    ground_truth_diseases: set[str],
    arm_id: str,
    *,
    retrieval_scores: dict[str, float] | None = None,
    evidence_sources: dict[str, list[str]] | None = None,
    threshold: float = 0.5,
    retrieval_gate: float = 0.3,
    gt_embeddings: NDArray[np.float32] | None = None,
    pred_embeddings: NDArray[np.float32] | None = None,
) -> FalseNegativeSummary:
    """Analyse all false negatives for a single drug prediction.

    Parameters
    ----------
    prediction:
        Agent output (``DrugDiseasePrediction``).
    ground_truth_diseases:
        Set of ground-truth disease names (normalised or raw).
    arm_id:
        Arm identifier.
    retrieval_scores:
        Mapping of normalised disease name to retrieval score.
    evidence_sources:
        Mapping of normalised disease name to list of source names.
    threshold:
        Combined score threshold for positive prediction.
    retrieval_gate:
        Minimum retrieval score to be considered "retrieved".
    gt_embeddings:
        Pre-computed MedCPT query embeddings for all GT diseases.
    pred_embeddings:
        Pre-computed MedCPT query embeddings for all predicted diseases.

    Returns
    -------
    FalseNegativeSummary
    """
    norm_gt = {_normalize_disease(d) for d in ground_truth_diseases}

    # Determine which GT diseases are true positives
    predicted_positive = set()
    for assoc in prediction.associations:
        if assoc.predicted:
            predicted_positive.add(_normalize_disease(assoc.disease_name))

    true_positives = norm_gt & predicted_positive
    false_negatives = norm_gt - predicted_positive

    # Collect all predicted names for near-miss
    all_predicted = [a.disease_name for a in prediction.associations]

    # Map ground-truth names (original) to their normalised forms
    gt_orig_to_norm: dict[str, str] = {}
    gt_list: list[str] = []
    for d in ground_truth_diseases:
        norm = _normalize_disease(d)
        gt_orig_to_norm[d] = norm
        gt_list.append(d)

    records: list[FalseNegativeRecord] = []
    for i, gt_disease in enumerate(gt_list):
        norm = gt_orig_to_norm[gt_disease]
        if norm in false_negatives:
            rec = _categorise_fn(
                gt_disease,
                prediction,
                retrieval_scores=retrieval_scores,
                evidence_sources=evidence_sources,
                threshold=threshold,
                retrieval_gate=retrieval_gate,
                predicted_names=all_predicted,
                gt_embeddings=gt_embeddings,
                pred_embeddings=pred_embeddings,
                gt_index=i,
            )
            records.append(rec)

    return FalseNegativeSummary(
        drug_name=prediction.drug_name,
        arm_id=arm_id,
        n_ground_truth=len(norm_gt),
        n_true_positives=len(true_positives),
        n_false_negatives=len(false_negatives),
        records=records,
    )


# ------------------------------------------------------------------
# Aggregate FN analysis across drugs
# ------------------------------------------------------------------

@dataclass
class AggregateFNSummary:
    """Aggregated FN statistics across multiple drugs for one arm."""

    arm_id: str
    n_drugs: int = 0
    total_ground_truth: int = 0
    total_true_positives: int = 0
    total_false_negatives: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    near_miss_count: int = 0
    near_miss_records: list[FalseNegativeRecord] = field(default_factory=list)

    @property
    def overall_fn_rate(self) -> float:
        """Overall FN rate across all drugs."""
        if self.total_ground_truth == 0:
            return 0.0
        return self.total_false_negatives / self.total_ground_truth

    def category_fractions(self) -> dict[str, float]:
        """FN category distribution as fractions."""
        total = self.total_false_negatives
        if total == 0:
            return {cat.value: 0.0 for cat in FalseNegativeCategory}
        return {k: v / total for k, v in self.category_counts.items()}


def aggregate_fn_summaries(
    summaries: list[FalseNegativeSummary],
    arm_id: str,
    *,
    near_miss_threshold: float = NEAR_MISS_THRESHOLD,
) -> AggregateFNSummary:
    """Aggregate FN summaries across multiple drugs.

    Parameters
    ----------
    summaries:
        Per-drug FN summaries.
    arm_id:
        Arm identifier.
    near_miss_threshold:
        Similarity threshold for flagging near misses.

    Returns
    -------
    AggregateFNSummary
    """
    cat_counts: dict[str, int] = {cat.value: 0 for cat in FalseNegativeCategory}
    near_misses: list[FalseNegativeRecord] = []

    total_gt = 0
    total_tp = 0
    total_fn = 0

    for s in summaries:
        total_gt += s.n_ground_truth
        total_tp += s.n_true_positives
        total_fn += s.n_false_negatives

        for rec in s.records:
            cat_counts[rec.category.value] += 1
            if (
                rec.similarity_to_closest is not None
                and rec.similarity_to_closest >= near_miss_threshold
                and rec.category != FalseNegativeCategory.NAME_MISMATCH
            ):
                near_misses.append(rec)

    # Also include explicit NAME_MISMATCH records as near misses
    for s in summaries:
        for rec in s.records:
            if rec.category == FalseNegativeCategory.NAME_MISMATCH:
                near_misses.append(rec)

    return AggregateFNSummary(
        arm_id=arm_id,
        n_drugs=len(summaries),
        total_ground_truth=total_gt,
        total_true_positives=total_tp,
        total_false_negatives=total_fn,
        category_counts=cat_counts,
        near_miss_count=len(near_misses),
        near_miss_records=near_misses,
    )


# ------------------------------------------------------------------
# Source coverage analysis
# ------------------------------------------------------------------

def source_coverage_analysis(
    summaries: list[FalseNegativeSummary],
) -> dict[str, int]:
    """Count how many FN records each data source covers.

    Returns a dict mapping source name to the count of FN records
    where that source had evidence.  Higher counts suggest the source
    is relevant but the pipeline still missed the association.
    """
    coverage: dict[str, int] = {}
    for s in summaries:
        for rec in s.records:
            for src in rec.evidence_sources_checked:
                coverage[src] = coverage.get(src, 0) + 1
    return dict(sorted(coverage.items(), key=lambda x: -x[1]))
