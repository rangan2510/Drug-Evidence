"""Dependency container for PydanticAI evidence agents.

Injected via ``RunContext[EvidenceDeps]`` into every agent tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.settings import Settings
from src.data.aggregator import EvidenceAggregator
from src.vector.store import HybridVectorStore

if TYPE_CHECKING:
    from src.data.evidence_store import EvidenceStore


@dataclass
class EvidenceDeps:
    """Runtime dependencies passed to PydanticAI agents via ``RunContext``.

    Attributes
    ----------
    settings:
        Application configuration.
    vector_store:
        Qdrant hybrid dense+sparse store for evidence retrieval.
    aggregator:
        Fan-out client for fetching fresh evidence from biomedical APIs.
    evidence_store:
        MongoDB staged evidence store (preferred over live API calls).
        When set, pipeline agents use MongoDB tools instead of live lookups.
    drug_name:
        Current drug being analysed.
    chembl_id:
        ChEMBL identifier for the drug (may be ``None``).
    pubchem_cid:
        PubChem compound ID for the drug (may be ``None``).
    """

    settings: Settings
    vector_store: HybridVectorStore | None
    aggregator: EvidenceAggregator
    evidence_store: EvidenceStore | None = None
    drug_name: str = ""
    chembl_id: str | None = None
    pubchem_cid: int | None = None
