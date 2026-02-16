"""ChEMBL REST client.

Fetches mechanism-of-action data and target binding assays from the
ChEMBL REST API (EBI).
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

_BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLClient:
    """Fetch mechanism-of-action and assay evidence from ChEMBL."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(20.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict:
        resp = await client.get(
            url,
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str,
    ) -> list[EvidenceDocument]:
        """Retrieve mechanism-of-action and assay evidence from ChEMBL.

        Parameters
        ----------
        drug_name
            Canonical drug name.
        chembl_id
            ChEMBL molecule ID (e.g. ``"CHEMBL25"``).
        """
        docs: list[EvidenceDocument] = []

        async with cached_async_client(self._settings) as client:
            # Mechanism of action
            moa_docs = await self._fetch_mechanisms(
                client, drug_name, chembl_id
            )
            docs.extend(moa_docs)

            # Target binding activities
            activity_docs = await self._fetch_activities(
                client, drug_name, chembl_id
            )
            docs.extend(activity_docs)

        logger.info(
            "ChEMBL: fetched %d evidence docs for %s (%s)",
            len(docs),
            drug_name,
            chembl_id,
        )
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_mechanisms(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str,
    ) -> list[EvidenceDocument]:
        """Fetch mechanism-of-action records for a molecule."""
        url = (
            f"{_BASE}/mechanism.json"
            f"?molecule_chembl_id={chembl_id}&limit=20"
        )
        try:
            data = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return []

        mechanisms = data.get("mechanisms", [])
        docs: list[EvidenceDocument] = []

        for mech in mechanisms:
            action_type = mech.get("action_type", "")
            target_name = mech.get("target_pref_name", "") or mech.get("target_name", "")
            target_chembl = mech.get("target_chembl_id", "")
            disease_name = mech.get("disease_efficacy") or None
            mech_desc = mech.get("mechanism_of_action", "")
            mech_refs = mech.get("mechanism_refs") or []

            # Extract PMIDs from references
            pmids = [
                ref.get("ref_id", "")
                for ref in mech_refs
                if ref.get("ref_type") == "PubMed"
            ]

            text = (
                f"ChEMBL mechanism of action: {drug_name} "
                f"{action_type} {target_name}. {mech_desc}".strip()
            )

            docs.append(
                EvidenceDocument(
                    text=text,
                    source=EvidenceSource.CHEMBL,
                    evidence_type=EvidenceType.MECHANISM_OF_ACTION,
                    citation=Citation(
                        pmid=pmids[0] if pmids else None,
                        url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                    target_symbol=target_name or None,
                    metadata={
                        "action_type": action_type,
                        "target_chembl_id": target_chembl,
                        "all_pmids": pmids,
                    },
                )
            )
        return docs

    async def _fetch_activities(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str,
    ) -> list[EvidenceDocument]:
        """Fetch top binding/functional activities for a molecule."""
        url = (
            f"{_BASE}/activity.json"
            f"?molecule_chembl_id={chembl_id}"
            f"&standard_type__in=IC50,EC50,Ki,Kd"
            f"&pchembl_value__isnull=false"
            f"&order_by=-pchembl_value"
            f"&limit=20"
        )
        try:
            data = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return []

        activities = data.get("activities", [])
        docs: list[EvidenceDocument] = []

        for act in activities:
            target_name = act.get("target_pref_name", "")
            target_chembl = act.get("target_chembl_id", "")
            std_type = act.get("standard_type", "")
            std_value = act.get("standard_value", "")
            std_units = act.get("standard_units", "")
            pchembl = act.get("pchembl_value")
            assay_chembl = act.get("assay_chembl_id", "")
            doc_chembl = act.get("document_chembl_id", "")

            text = (
                f"ChEMBL activity: {drug_name} vs {target_name}: "
                f"{std_type}={std_value} {std_units} "
                f"(pChEMBL={pchembl})."
            )

            docs.append(
                EvidenceDocument(
                    text=text,
                    source=EvidenceSource.CHEMBL,
                    evidence_type=EvidenceType.BINDING_ASSAY,
                    citation=Citation(
                        url=f"https://www.ebi.ac.uk/chembl/assay_report_card/{assay_chembl}/"
                        if assay_chembl
                        else None,
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                    target_symbol=target_name or None,
                    score=float(pchembl) / 10.0 if pchembl else None,  # normalize ~0-1
                    metadata={
                        "standard_type": std_type,
                        "standard_value": std_value,
                        "standard_units": std_units,
                        "pchembl_value": pchembl,
                        "target_chembl_id": target_chembl,
                        "assay_chembl_id": assay_chembl,
                        "document_chembl_id": doc_chembl,
                    },
                )
            )
        return docs
