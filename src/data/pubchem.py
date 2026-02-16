"""PubChem PUG-REST evidence client.

Fetches pharmacological actions and bioassay summaries for a compound
from the PubChem PUG-REST API.
"""

from __future__ import annotations

import json
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

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_COMPOUND = f"{_BASE}/compound"


class PubChemClient:
    """Fetch pharmacological and bioassay evidence from PubChem."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        # PubChem classification endpoint for popular compounds can return
        # very large JSON payloads (70+ MB for aspirin).  Use a generous
        # timeout so reads finish reliably.
        self._timeout = httpx.Timeout(60.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict:
        resp = await client.get(url, timeout=self._timeout)
        resp.raise_for_status()
        # PubChem classification responses may contain non-UTF-8 bytes
        # (e.g. 0xAE = registered-trademark in latin-1).  Decoding the
        # raw bytes as latin-1 is lossless for all 0x00-0xFF values and
        # keeps the JSON structure intact, unlike errors="ignore" which
        # strips bytes and can shift structural delimiters.
        try:
            return resp.json()
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = resp.content.decode("latin-1")
            return json.loads(text)

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str | None = None,
        pubchem_cid: int | None = None,
    ) -> list[EvidenceDocument]:
        """Retrieve evidence from PubChem for a compound.

        Parameters
        ----------
        drug_name
            Canonical drug name.
        chembl_id
            Optional ChEMBL ID (metadata).
        pubchem_cid
            PubChem Compound ID. If None, resolves via name search.
        """
        docs: list[EvidenceDocument] = []

        async with cached_async_client(self._settings) as client:
            # Resolve CID if not provided
            if pubchem_cid is None:
                pubchem_cid = await self._resolve_cid(client, drug_name)
                if pubchem_cid is None:
                    return docs

            # Fetch pharmacological actions
            pharm_docs = await self._fetch_pharmacological_actions(
                client, drug_name, chembl_id, pubchem_cid
            )
            docs.extend(pharm_docs)

            # Fetch bioassay summaries
            assay_docs = await self._fetch_bioassay_summary(
                client, drug_name, chembl_id, pubchem_cid
            )
            docs.extend(assay_docs)

        logger.info(
            "PubChem: fetched %d evidence docs for %s (CID=%s)",
            len(docs),
            drug_name,
            pubchem_cid,
        )
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_cid(
        self, client: httpx.AsyncClient, drug_name: str
    ) -> int | None:
        """Resolve a drug name to a PubChem CID."""
        url = f"{_COMPOUND}/name/{drug_name}/cids/JSON"
        try:
            data = await self._get_json(client, url)
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
        except httpx.HTTPStatusError:
            return None

    async def _fetch_pharmacological_actions(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str | None,
        cid: int,
    ) -> list[EvidenceDocument]:
        """Fetch pharmacological action classifications."""
        url = f"{_COMPOUND}/cid/{cid}/classification/JSON"
        try:
            data = await self._get_json(client, url)
        except (httpx.HTTPStatusError, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(
                "PubChem classification unavailable for CID %s", cid
            )
            return []

        docs: list[EvidenceDocument] = []
        hierarchies = data.get("Hierarchies", {}).get("Hierarchy", [])
        for hierarchy in hierarchies[:20]:  # cap to avoid huge responses
            h_name = hierarchy.get("SourceName", "")
            nodes = hierarchy.get("Node", [])
            for node in nodes[:10]:
                info = node.get("Information", {})
                name = info.get("Name", "")
                description = info.get("Description", "")
                if not name:
                    continue
                text = (
                    f"PubChem pharmacological classification for {drug_name}: "
                    f"{name}. {description}".strip()
                )
                docs.append(
                    EvidenceDocument(
                        text=text,
                        source=EvidenceSource.PUBCHEM,
                        evidence_type=EvidenceType.PHARMACOLOGICAL_ACTION,
                        citation=Citation(
                            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                        ),
                        drug_name=drug_name,
                        drug_chembl_id=chembl_id,
                        drug_pubchem_cid=cid,
                        metadata={
                            "source_name": h_name,
                            "classification_name": name,
                        },
                    )
                )
        return docs

    async def _fetch_bioassay_summary(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        chembl_id: str | None,
        cid: int,
    ) -> list[EvidenceDocument]:
        """Fetch concise bioassay summary data."""
        url = f"{_COMPOUND}/cid/{cid}/assaysummary/JSON"
        try:
            data = await self._get_json(client, url)
        except httpx.HTTPStatusError:
            return []

        docs: list[EvidenceDocument] = []
        table = data.get("Table", {})
        columns = table.get("Columns", {}).get("Column", [])
        rows = table.get("Row", [])

        # Find column indices
        col_map: dict[str, int] = {}
        for i, col in enumerate(columns):
            col_map[col] = i

        aid_idx = col_map.get("AID")
        target_name_idx = col_map.get("TargetName")
        target_gi_idx = col_map.get("TargetGI")
        activity_idx = col_map.get("Activity Outcome")

        for row in rows[:30]:  # cap
            cells = row.get("Cell", [])
            if not cells:
                continue

            def _cell(idx: int | None) -> str:
                if idx is None or idx >= len(cells):
                    return ""
                val = cells[idx]
                # PubChem may return cells as plain values (str/int)
                # or as dicts with a "Value" key.
                if isinstance(val, dict):
                    return str(val.get("Value", ""))
                return str(val)

            aid = _cell(aid_idx)
            target_name = _cell(target_name_idx)
            activity = _cell(activity_idx)

            if not target_name or activity.lower() not in ("active", "probe"):
                continue

            text = (
                f"PubChem bioassay: {drug_name} is {activity.lower()} against "
                f"{target_name} (AID {aid})."
            )
            docs.append(
                EvidenceDocument(
                    text=text,
                    source=EvidenceSource.PUBCHEM,
                    evidence_type=EvidenceType.BINDING_ASSAY,
                    citation=Citation(
                        url=f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                    drug_pubchem_cid=cid,
                    target_symbol=target_name,
                    metadata={
                        "aid": aid,
                        "activity_outcome": activity,
                    },
                )
            )
        return docs
