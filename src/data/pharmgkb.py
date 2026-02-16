"""PharmGKB REST client.

Fetches clinical annotations and drug-gene-disease relationships from
the PharmGKB REST API.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings
from src.data.http import cached_async_client
from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)

logger = logging.getLogger(__name__)

_BASE = "https://api.pharmgkb.org/v1/data"


class PharmGKBClient:
    """Fetch clinical annotations from PharmGKB."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(20.0)
        self._headers = {"Accept": "application/json"}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict | list:
        resp = await client.get(url, timeout=self._timeout, headers=self._headers)
        resp.raise_for_status()
        return resp.json()

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str | None = None,
    ) -> list[EvidenceDocument]:
        """Retrieve clinical annotation evidence from PharmGKB.

        Parameters
        ----------
        drug_name
            Canonical drug name.
        chembl_id
            Optional ChEMBL ID (metadata).
        """
        docs: list[EvidenceDocument] = []

        async with cached_async_client(self._settings) as client:
            # Step 1: search for the drug in PharmGKB
            drug_id = await self._resolve_drug(client, drug_name)
            if drug_id is None:
                logger.info("PharmGKB: drug not found: %s", drug_name)
                return docs

            # Step 2: fetch clinical annotations for the drug
            ann_docs = await self._fetch_clinical_annotations(
                client, drug_name, chembl_id, drug_id
            )
            docs.extend(ann_docs)

            # Step 3: fetch drug-label annotations
            label_docs = await self._fetch_label_annotations(
                client, drug_name, chembl_id, drug_id
            )
            docs.extend(label_docs)

        logger.info(
            "PharmGKB: fetched %d evidence docs for %s",
            len(docs),
            drug_name,
        )
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_drug(
        self, client: httpx.AsyncClient, drug_name: str
    ) -> str | None:
        """Search PharmGKB for a drug, return its PharmGKB accession ID."""
        url = f"{_BASE}/chemical?name={drug_name}"
        try:
            result = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return None

        # Result is usually {"data": [...]} or a list directly
        items = result if isinstance(result, list) else result.get("data", [])
        if not items:
            return None

        # Take best match
        for item in items:
            if item.get("name", "").lower() == drug_name.lower():
                return item.get("id")
        # Fallback: first result
        return items[0].get("id") if items else None

    async def _fetch_clinical_annotations(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str | None,
        drug_id: str,
    ) -> list[EvidenceDocument]:
        """Fetch clinical annotations linked to the drug."""
        url = f"{_BASE}/clinicalAnnotation?relatedChemicals.accessionId={drug_id}"
        try:
            result = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return []

        items = result if isinstance(result, list) else result.get("data", [])

        docs: list[EvidenceDocument] = []
        for ann in items[:30]:  # cap
            ann_id = ann.get("id", "")
            level = ann.get("evidenceLevel", "")
            phenotypes = ann.get("relatedDiseases", [])
            genes = ann.get("relatedGenes", [])
            gene_symbols = [g.get("symbol", "") for g in genes if g.get("symbol")]
            disease_names = [p.get("name", "") for p in phenotypes if p.get("name")]

            summary = ann.get("summary", "") or ann.get("text", "") or ""
            if not summary:
                summary = (
                    f"Clinical annotation for {drug_name} "
                    f"involving {', '.join(gene_symbols) or 'unknown gene(s)'}"
                )

            text = (
                f"PharmGKB clinical annotation (level {level}): {summary}"
            )

            docs.append(
                EvidenceDocument(
                    text=text,
                    source=EvidenceSource.PHARMGKB,
                    evidence_type=EvidenceType.CLINICAL_ANNOTATION,
                    citation=Citation(
                        url=f"https://www.pharmgkb.org/clinicalAnnotation/{ann_id}",
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                    target_symbol=gene_symbols[0] if gene_symbols else None,
                    disease_name=disease_names[0] if disease_names else None,
                    metadata={
                        "pharmgkb_id": ann_id,
                        "evidence_level": level,
                        "all_genes": gene_symbols,
                        "all_diseases": disease_names,
                    },
                )
            )
        return docs

    async def _fetch_label_annotations(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str | None,
        drug_id: str,
    ) -> list[EvidenceDocument]:
        """Fetch drug label annotations (FDA, EMA, etc.)."""
        url = f"{_BASE}/drugLabel?relatedChemicals.accessionId={drug_id}"
        try:
            result = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return []

        items = result if isinstance(result, list) else result.get("data", [])

        docs: list[EvidenceDocument] = []
        for label in items[:10]:
            label_id = label.get("id", "")
            label_name = label.get("name", "")
            source = label.get("source", "")
            summary = label.get("summary", "") or ""

            text = (
                f"PharmGKB drug label ({source}): {label_name}. {summary}".strip()
            )
            docs.append(
                EvidenceDocument(
                    text=text,
                    source=EvidenceSource.PHARMGKB,
                    evidence_type=EvidenceType.CLINICAL_ANNOTATION,
                    citation=Citation(
                        url=f"https://www.pharmgkb.org/drugLabel/{label_id}",
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                    metadata={
                        "label_id": label_id,
                        "label_source": source,
                    },
                )
            )
        return docs
