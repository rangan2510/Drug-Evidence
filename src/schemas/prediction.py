"""Prediction schemas -- structured output produced by PydanticAI agents.

``DrugDiseasePrediction`` is the ``output_type`` for every experimental arm,
so pipeline and baseline agents are directly comparable.
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Mechanistic chain components
# ------------------------------------------------------------------

class EdgeType(str, Enum):
    """Relationship type within a mechanistic chain.

    Covers the standard biomedical relationship types that LLMs
    produce when describing drug-target-pathway-disease connections.
    """

    BINDS = "binds"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"
    MODULATES = "modulates"
    TRANSPORTS = "transports"
    METABOLIZES = "metabolizes"
    PARTICIPATES_IN = "participates_in"
    ASSOCIATED_WITH = "associated_with"
    PROMOTES = "promotes"
    SUPPRESSES = "suppresses"
    REGULATES = "regulates"
    CONTRIBUTES_TO = "contributes_to"
    INTERACTS_WITH = "interacts_with"
    COACTIVATES = "coactivates"


# Synonym map: common LLM-generated edge type strings -> valid EdgeType values.
# This handles the many plausible relationship terms that frontier models
# produce but are not in the canonical 16-member enum.
_EDGE_TYPE_SYNONYMS: dict[str, str] = {
    # Binding variants
    "agonizes": "binds",
    "agonist_of": "binds",
    "partial_agonist_of": "binds",
    "antagonizes": "inhibits",
    "antagonist_of": "inhibits",
    "inverse_agonist_of": "inhibits",
    "blocks": "inhibits",
    "targets": "binds",
    "ligand_of": "binds",
    "substrate_of": "binds",
    "allosteric_modulator_of": "modulates",
    # Regulation variants
    "increases": "upregulates",
    "decreases": "downregulates",
    "induces": "activates",
    "stimulates": "activates",
    "enhances": "promotes",
    "potentiates": "promotes",
    "attenuates": "suppresses",
    "reduces": "suppresses",
    "represses": "suppresses",
    "inhibits_activity_of": "inhibits",
    "up_regulates": "upregulates",
    "down_regulates": "downregulates",
    "up-regulates": "upregulates",
    "down-regulates": "downregulates",
    # Pathway/process variants
    "involved_in": "participates_in",
    "part_of": "participates_in",
    "mediates": "participates_in",
    "catalyzes": "metabolizes",
    "converts": "metabolizes",
    "metabolized_by": "metabolizes",
    "is_metabolized_by": "metabolizes",
    "expressed_in": "associated_with",
    "located_in": "associated_with",
    "implicated_in": "associated_with",
    "linked_to": "associated_with",
    "correlates_with": "associated_with",
    "co-expressed_with": "associated_with",
    "affects": "modulates",
    # Disease-relation variants
    "treats": "associated_with",
    "therapeutic_for": "associated_with",
    "indicated_for": "associated_with",
    "alleviates": "suppresses",
    "ameliorates": "suppresses",
    "exacerbates": "promotes",
    "causes": "contributes_to",
    "predisposes_to": "contributes_to",
    "risk_factor_for": "contributes_to",
}


class MechanisticEdge(BaseModel):
    """Single directed edge in a drug -> target -> pathway -> disease chain."""

    source_entity: str = Field(..., description="e.g. drug name, gene symbol, pathway")
    target_entity: str = Field(..., description="e.g. gene symbol, pathway, disease")
    relationship: EdgeType
    evidence_snippet: str = Field(
        ..., description="Supporting text from literature or database"
    )
    pmid: str | None = None
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional edge-level confidence (some models produce this)",
    )

    @field_validator("relationship", mode="before")
    @classmethod
    def _coerce_edge_type(cls, v: object) -> object:
        """Map common LLM-generated synonyms to canonical EdgeType values.

        If the raw value is not in the enum but matches a known synonym,
        silently remap it.  If truly unknown, fall back to 'associated_with'
        and log a warning so the experiment runner can track coverage.
        """
        if isinstance(v, EdgeType):
            return v
        if not isinstance(v, str):
            return v  # let Pydantic raise a type error

        normalized = v.strip().lower().replace(" ", "_").replace("-", "_")

        # Direct enum match (handles case-insensitive)
        try:
            return EdgeType(normalized)
        except ValueError:
            pass

        # Synonym lookup
        if normalized in _EDGE_TYPE_SYNONYMS:
            mapped = _EDGE_TYPE_SYNONYMS[normalized]
            logger.debug("EdgeType coercion: '%s' -> '%s'", v, mapped)
            return EdgeType(mapped)

        # Last resort: fall back to associated_with
        logger.warning(
            "Unknown EdgeType '%s' -- falling back to 'associated_with'. "
            "Consider adding it to _EDGE_TYPE_SYNONYMS.",
            v,
        )
        return EdgeType.ASSOCIATED_WITH


class EvidenceChain(BaseModel):
    """Ordered chain of mechanistic edges linking a drug to a disease."""

    edges: list[MechanisticEdge] = Field(
        ..., min_length=1, description="At least one edge required"
    )
    summary: str = Field(
        ..., description="One-sentence plain-language summary of the full chain"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Agent-assigned confidence in this chain"
    )


# ------------------------------------------------------------------
# Per-association prediction
# ------------------------------------------------------------------

class ScoredAssociation(BaseModel):
    """Single drug-disease association with evidence chains."""

    disease_name: str
    disease_id: str | None = None
    predicted: bool = Field(
        ..., description="Whether the agent predicts a therapeutic association"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score"
    )
    evidence_chains: list[EvidenceChain] = Field(
        default_factory=list,
        description="Mechanistic chains supporting this prediction",
    )


# ------------------------------------------------------------------
# Top-level structured output (shared by ALL arms)
# ------------------------------------------------------------------

class DrugDifficulty(str, Enum):
    """Stratification covariate, NOT a filter."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DrugDiseasePrediction(BaseModel):
    """Top-level output produced by every experimental arm.

    This is the ``output_type`` passed to ``pydantic_ai.Agent``.
    """

    drug_name: str
    drug_chembl_id: str | None = None
    associations: list[ScoredAssociation] = Field(
        ..., description="All evaluated disease associations"
    )
    reasoning: str = Field(
        default="No reasoning provided.",
        description="Agent's high-level reasoning summary",
    )
