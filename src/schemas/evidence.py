"""Evidence schemas -- uniform types returned by every data client.

Every data source (OpenTargets, DGIdb, PubChem, PharmGKB, ChEMBL, PubMed)
maps its response into ``list[EvidenceDocument]`` so downstream consumers
never need to know which API the data came from.
"""

from __future__ import annotations

from enum import Enum
from datetime import UTC, datetime

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class EvidenceSource(str, Enum):
    """Canonical data source identifiers."""

    OPENTARGETS = "opentargets"
    DGIDB = "dgidb"
    PUBCHEM = "pubchem"
    PHARMGKB = "pharmgkb"
    CHEMBL = "chembl"
    PUBMED = "pubmed"
    REACTOME = "reactome"


class EvidenceType(str, Enum):
    """Broad category of evidence."""

    LITERATURE = "literature"
    CLINICAL_TRIAL = "clinical_trial"
    PATHWAY = "pathway"
    BINDING_ASSAY = "binding_assay"
    PHARMACOLOGICAL_ACTION = "pharmacological_action"
    CLINICAL_ANNOTATION = "clinical_annotation"
    MECHANISM_OF_ACTION = "mechanism_of_action"
    DRUG_GENE_INTERACTION = "drug_gene_interaction"


# ------------------------------------------------------------------
# Core models
# ------------------------------------------------------------------

class Citation(BaseModel):
    """A single literature or database reference."""

    pmid: str | None = None
    doi: str | None = None
    url: str | None = None
    title: str | None = None
    year: int | None = None


class EvidenceDocument(BaseModel):
    """Uniform evidence record returned by every data client."""

    text: str = Field(..., description="Human-readable evidence snippet or abstract")
    source: EvidenceSource
    evidence_type: EvidenceType
    citation: Citation = Field(default_factory=Citation)

    drug_name: str
    drug_chembl_id: str | None = None
    drug_pubchem_cid: int | None = None

    target_symbol: str | None = None
    target_ensembl_id: str | None = None
    disease_name: str | None = None
    disease_id: str | None = None  # e.g. EFO, MONDO, MeSH

    score: float | None = Field(
        default=None,
        description="Source-provided relevance or association score (0-1 where available)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Extra source-specific fields that do not fit the common schema",
    )

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
