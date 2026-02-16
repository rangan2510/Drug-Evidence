"""Drug difficulty classifier based on baseline-gpt5-nosearch P@10.

Classifies each drug into ``DrugDifficulty.EASY``, ``MEDIUM``, or
``HARD`` based on how well the strongest no-search baseline performs.

This is a **covariate for stratified analysis**, NOT a selection gate.
All drugs passing the association-count filter are included regardless
of difficulty.

Thresholds
----------
* EASY:   P@10 > 0.4   (baseline already gets many right)
* MEDIUM: 0.1 <= P@10 <= 0.4
* HARD:   P@10 < 0.1   (baseline struggles)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.evaluation.accuracy import DrugMetrics, evaluate_prediction
from src.experiment.runner import ArmResult
from src.schemas.prediction import DrugDifficulty

logger = logging.getLogger(__name__)

# Arm used as the difficulty reference
_REFERENCE_ARM = "baseline-gpt5-nosearch"

# P@10 thresholds
_EASY_THRESHOLD = 0.4
_HARD_THRESHOLD = 0.1


@dataclass
class ClassifiedDrug:
    """A drug with its assigned difficulty label."""

    drug_name: str
    difficulty: DrugDifficulty
    reference_p_at_10: float


def classify_difficulty(
    result: ArmResult,
    ground_truth: set[str],
) -> DrugDifficulty:
    """Classify one drug's difficulty from its no-search baseline result.

    Parameters
    ----------
    result:
        ``ArmResult`` from the ``baseline-gpt5-nosearch`` arm.
    ground_truth:
        Normalised disease names from CTD.

    Returns
    -------
    DrugDifficulty
        EASY, MEDIUM, or HARD.
    """
    if result.prediction is None:
        # If the baseline errored, treat as HARD
        return DrugDifficulty.HARD

    metrics = evaluate_prediction(
        result.prediction,
        ground_truth,
        arm_id=result.arm_id,
    )
    return _difficulty_from_p10(metrics.precision_at_10)


def _difficulty_from_p10(p_at_10: float) -> DrugDifficulty:
    """Map a P@10 score to a difficulty bucket."""
    if p_at_10 > _EASY_THRESHOLD:
        return DrugDifficulty.EASY
    if p_at_10 >= _HARD_THRESHOLD:
        return DrugDifficulty.MEDIUM
    return DrugDifficulty.HARD


def classify_batch(
    nosearch_results: dict[str, ArmResult],
    ground_truths: dict[str, set[str]],
) -> dict[str, ClassifiedDrug]:
    """Classify difficulty for a batch of drugs.

    Parameters
    ----------
    nosearch_results:
        ``{drug_name_lower: ArmResult}`` from the ``baseline-gpt5-nosearch`` arm.
    ground_truths:
        ``{drug_name_lower: set[disease]}`` from CTD.

    Returns
    -------
    dict[str, ClassifiedDrug]
        Keyed by lowercased drug name.
    """
    classified: dict[str, ClassifiedDrug] = {}

    for drug_lower, result in nosearch_results.items():
        gt = ground_truths.get(drug_lower, set())
        diff = classify_difficulty(result, gt)

        if result.prediction is not None:
            metrics = evaluate_prediction(result.prediction, gt, arm_id=result.arm_id)
            ref_p10 = metrics.precision_at_10
        else:
            ref_p10 = 0.0

        classified[drug_lower] = ClassifiedDrug(
            drug_name=drug_lower,
            difficulty=diff,
            reference_p_at_10=ref_p10,
        )

    # Log distribution
    counts = {d.value: 0 for d in DrugDifficulty}
    for cd in classified.values():
        counts[cd.difficulty.value] += 1
    logger.info(
        "Difficulty distribution: easy=%d, medium=%d, hard=%d",
        counts["easy"],
        counts["medium"],
        counts["hard"],
    )

    return classified
