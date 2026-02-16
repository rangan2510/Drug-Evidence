"""Drug name normalizer -- deterministic resolution via PubChem + ChEMBL.

Replaces the legacy Tavily web-search + regex approach (R2-D).

Flow
----
1. **PubChem**: drug name -> PubChem CID -> synonyms + cross-references
2. **ChEMBL**: drug name -> molecule search -> CHEMBL ID, synonyms, pref_name
3. Cross-reference: PubChem xrefs may already contain a ChEMBL ID.
4. No web search involved -- fully deterministic and reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import Settings
from src.data.http import cached_async_client

logger = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass
class NormalizedDrug:
    """Result of normalizing a free-text drug name."""

    query: str
    preferred_name: str | None = None
    chembl_id: str | None = None
    pubchem_cid: int | None = None
    synonyms: list[str] = field(default_factory=list)
    inchi_key: str | None = None


class DrugNormalizer:
    """Resolve a drug name to canonical identifiers using PubChem + ChEMBL."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = cached_async_client(
                self._settings,
                timeout=httpx.Timeout(30.0),
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def normalize(self, drug_name: str) -> NormalizedDrug:
        """Resolve *drug_name* to canonical IDs, synonyms, and preferred name."""
        result = NormalizedDrug(query=drug_name)

        # Run PubChem and ChEMBL in parallel
        pubchem_ok = await self._resolve_pubchem(drug_name, result)
        chembl_ok = await self._resolve_chembl(drug_name, result)

        # If PubChem gave us xref ChEMBL ID but direct ChEMBL search didn't
        if not result.chembl_id and pubchem_ok:
            await self._pubchem_xref_chembl(result)

        # Dedupe synonyms
        seen: set[str] = set()
        unique: list[str] = []
        for syn in result.synonyms:
            low = syn.lower()
            if low not in seen:
                seen.add(low)
                unique.append(syn)
        result.synonyms = unique[:20]  # cap at 20

        if not result.preferred_name:
            result.preferred_name = drug_name

        return result

    # ------------------------------------------------------------------
    # PubChem
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def _resolve_pubchem(
        self, drug_name: str, result: NormalizedDrug
    ) -> bool:
        """Query PubChem for CID + synonyms."""
        client = await self._get_client()
        # Step 1: name -> CID
        url = f"{_PUBCHEM_BASE}/compound/name/{httpx.URL(drug_name).raw_path.decode() if False else drug_name}/cids/JSON"
        try:
            resp = await client.get(
                f"{_PUBCHEM_BASE}/compound/name/{drug_name}/cids/JSON"
            )
            resp.raise_for_status()
            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if not cids:
                return False
            result.pubchem_cid = cids[0]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("PubChem: no compound found for %r", drug_name)
                return False
            raise

        # Step 2: CID -> synonyms
        try:
            resp = await client.get(
                f"{_PUBCHEM_BASE}/compound/cid/{result.pubchem_cid}/synonyms/JSON"
            )
            resp.raise_for_status()
            syn_data = resp.json()
            syns = (
                syn_data.get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", [])
            )
            result.synonyms.extend(syns[:30])
        except httpx.HTTPStatusError:
            logger.debug("PubChem: failed to fetch synonyms for CID %s", result.pubchem_cid)

        return True

    # ------------------------------------------------------------------
    # ChEMBL
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def _resolve_chembl(
        self, drug_name: str, result: NormalizedDrug
    ) -> bool:
        """Query ChEMBL molecule search for CHEMBL ID + preferred name."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{_CHEMBL_BASE}/molecule/search.json",
                params={"q": drug_name, "limit": 5},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("ChEMBL: no molecule found for %r", drug_name)
                return False
            raise

        molecules = data.get("molecules", [])
        if not molecules:
            return False

        # Pick best match: prefer exact pref_name match
        best = molecules[0]
        for mol in molecules:
            pref = (mol.get("pref_name") or "").lower()
            if pref == drug_name.lower():
                best = mol
                break

        result.chembl_id = best.get("molecule_chembl_id")
        result.preferred_name = best.get("pref_name") or result.preferred_name
        result.inchi_key = (
            best.get("molecule_structures", {}) or {}
        ).get("standard_inchi_key")

        # Collect synonyms from ChEMBL
        for syn_entry in best.get("molecule_synonyms", []):
            syn_val = syn_entry.get("molecule_synonym")
            if syn_val:
                result.synonyms.append(syn_val)

        return True

    # ------------------------------------------------------------------
    # PubChem xref -> ChEMBL fallback
    # ------------------------------------------------------------------

    async def _pubchem_xref_chembl(self, result: NormalizedDrug) -> None:
        """Check PubChem cross-references for a ChEMBL ID."""
        if result.pubchem_cid is None:
            return
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{_PUBCHEM_BASE}/compound/cid/{result.pubchem_cid}/xrefs/RegistryID/JSON"
            )
            resp.raise_for_status()
            data = resp.json()
            reg_ids = (
                data.get("InformationList", {})
                .get("Information", [{}])[0]
                .get("RegistryID", [])
            )
            for rid in reg_ids:
                if rid.startswith("CHEMBL"):
                    result.chembl_id = rid
                    return
        except httpx.HTTPStatusError:
            logger.debug(
                "PubChem xref lookup failed for CID %s", result.pubchem_cid
            )
