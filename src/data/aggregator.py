"""Evidence aggregator -- fan-out to all data sources, dedupe, return unified list.

Orchestrates:
  1. Normalize drug name -> ChEMBL ID, PubChem CID, synonyms
  2. Fan-out to OpenTargets, DGIdb, PubChem, PharmGKB, ChEMBL
  3. Collect PMIDs from all sources
  4. Batch-fetch PubMed abstracts for collected PMIDs
  5. Deduplicate and return ``list[EvidenceDocument]``
"""

from __future__ import annotations

import asyncio
import logging

from src.config.settings import Settings
from src.data.chembl import ChEMBLClient
from src.data.dgidb import DGIdbClient
from src.data.normalizer import DrugNormalizer, NormalizedDrug
from src.data.opentargets import OpenTargetsClient
from src.data.pharmgkb import PharmGKBClient
from src.data.pubchem import PubChemClient
from src.data.pubmed import PubMedClient
from src.data.reactome import ReactomeClient
from src.schemas.evidence import EvidenceDocument

logger = logging.getLogger(__name__)


async def _safe_fetch(
    coro: asyncio.coroutines,
    source_name: str,
) -> list[EvidenceDocument]:
    """Run a data-source fetch, returning [] on any failure."""
    try:
        return await coro
    except Exception:
        logger.exception("Data source '%s' failed; returning empty list", source_name)
        return []


class EvidenceAggregator:
    """Fan-out evidence retrieval across all data sources for a single drug."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._normalizer = DrugNormalizer(self._settings)
        self._opentargets = OpenTargetsClient(self._settings)
        self._dgidb = DGIdbClient(self._settings)
        self._pubchem = PubChemClient(self._settings)
        self._pharmgkb = PharmGKBClient(self._settings)
        self._chembl = ChEMBLClient(self._settings)
        self._pubmed = PubMedClient(self._settings)
        self._reactome = ReactomeClient(self._settings)

    async def gather(
        self,
        drug_name: str,
        *,
        chembl_id: str | None = None,
        pubchem_cid: int | None = None,
        skip_normalize: bool = False,
    ) -> list[EvidenceDocument]:
        """Gather evidence from all sources for a single drug.

        Parameters
        ----------
        drug_name
            Drug name to query.
        chembl_id / pubchem_cid
            Pre-resolved identifiers (skips normalization step).
        skip_normalize
            If True and identifiers are provided, skip the normalizer call.
        """
        # Step 1: normalize
        norm: NormalizedDrug | None = None
        if not skip_normalize or (chembl_id is None and pubchem_cid is None):
            norm = await self._normalizer.normalize(drug_name)
            chembl_id = chembl_id or (norm.chembl_id if norm else None)
            pubchem_cid = pubchem_cid or (norm.pubchem_cid if norm else None)
            drug_name = norm.preferred_name if norm and norm.preferred_name else drug_name

        synonyms = list(norm.synonyms) if norm and norm.synonyms else []

        # Step 2: fan-out to data sources (concurrent)
        all_docs: list[EvidenceDocument] = []
        tasks: list[asyncio.Task] = []

        async with asyncio.TaskGroup() as tg:
            # OpenTargets requires ChEMBL ID
            if chembl_id:
                tasks.append(
                    tg.create_task(
                        _safe_fetch(
                            self._opentargets.fetch(drug_name, chembl_id),
                            "opentargets",
                        ),
                        name="opentargets",
                    )
                )
                tasks.append(
                    tg.create_task(
                        _safe_fetch(
                            self._chembl.fetch(drug_name, chembl_id),
                            "chembl",
                        ),
                        name="chembl",
                    )
                )

            # DGIdb works with drug names + synonyms
            tasks.append(
                tg.create_task(
                    _safe_fetch(
                        self._dgidb.fetch(drug_name, chembl_id, synonyms=synonyms),
                        "dgidb",
                    ),
                    name="dgidb",
                )
            )

            # PubChem uses CID
            tasks.append(
                tg.create_task(
                    _safe_fetch(
                        self._pubchem.fetch(drug_name, chembl_id, pubchem_cid),
                        "pubchem",
                    ),
                    name="pubchem",
                )
            )

            # PharmGKB uses drug name
            tasks.append(
                tg.create_task(
                    _safe_fetch(
                        self._pharmgkb.fetch(drug_name, chembl_id),
                        "pharmgkb",
                    ),
                    name="pharmgkb",
                )
            )

        # Collect results (all tasks succeed due to _safe_fetch wrapper)
        for task in tasks:
            all_docs.extend(task.result())

        # Step 3: enrich target-level docs with pathway mappings from Reactome
        ensembl_to_symbol: dict[str, str] = {}
        for doc in all_docs:
            if doc.target_ensembl_id:
                ensembl_to_symbol[doc.target_ensembl_id] = doc.target_symbol or ""

        if ensembl_to_symbol:
            reactome_docs = await _safe_fetch(
                self._reactome.fetch(drug_name, chembl_id, ensembl_to_symbol),
                "reactome",
            )
            all_docs.extend(reactome_docs)

        # Step 4: collect all PMIDs from fetched docs for PubMed enrichment
        seen_pmids: set[str] = set()
        for doc in all_docs:
            if doc.citation.pmid:
                seen_pmids.add(doc.citation.pmid)
            for pmid in doc.metadata.get("all_pmids", []):
                if pmid:
                    seen_pmids.add(str(pmid))

        # Fetch PubMed abstracts for PMIDs we haven't already got full text for
        existing_pubmed_pmids = {
            doc.citation.pmid
            for doc in all_docs
            if doc.source.value == "pubmed" and doc.citation.pmid
        }
        new_pmids = list(seen_pmids - existing_pubmed_pmids)

        if new_pmids:
            pubmed_docs = await self._pubmed.fetch_by_pmids(
                new_pmids, drug_name, chembl_id
            )
            all_docs.extend(pubmed_docs)

        # Step 5: deduplicate
        deduped = self._deduplicate(all_docs)

        logger.info(
            "Aggregator: %d total docs (%d after dedup) for %s",
            len(all_docs),
            len(deduped),
            drug_name,
        )
        return deduped

    @staticmethod
    def _deduplicate(docs: list[EvidenceDocument]) -> list[EvidenceDocument]:
        """Remove exact duplicate evidence documents.

        Uses (source, text[:200], drug_name) as a dedup key.
        """
        seen: set[tuple[str, str, str]] = set()
        unique: list[EvidenceDocument] = []
        for doc in docs:
            key = (doc.source.value, doc.text[:200], doc.drug_name.lower())
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique
