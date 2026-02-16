"""Reactome bulk pathway client.

Downloads and caches Reactome bulk mapping files, then maps target Ensembl IDs
to Homo sapiens pathways for pathway-level evidence enrichment.
"""

from __future__ import annotations

import logging
from asyncio import Lock
from pathlib import Path

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

_ENSEMBL_FILE = "Ensembl2Reactome.txt"


class ReactomeClient:
    """Map Ensembl targets to Reactome pathways using bulk download files."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(60.0)
        self._load_lock = Lock()
        self._loaded = False
        self._pathways_by_ensembl: dict[str, list[dict[str, str]]] = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _download_file(self, url: str, dest: Path) -> None:
        async with cached_async_client(self._settings) as client:
            response = await client.get(url, timeout=self._timeout)
            response.raise_for_status()
            dest.write_bytes(response.content)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        async with self._load_lock:
            if self._loaded:
                return

            cache_dir = Path(self._settings.reactome_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            ensembl_path = cache_dir / _ENSEMBL_FILE
            if not ensembl_path.exists() or ensembl_path.stat().st_size == 0:
                url = f"{self._settings.reactome_download_base}/{_ENSEMBL_FILE}"
                logger.info("Reactome: downloading %s", url)
                await self._download_file(url, ensembl_path)

            mapping: dict[str, list[dict[str, str]]] = {}
            with ensembl_path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 6:
                        continue

                    ensembl_id = parts[0].strip()
                    pathway_id = parts[1].strip()
                    pathway_url = parts[2].strip()
                    pathway_name = parts[3].strip()
                    evidence_code = parts[4].strip()
                    species = parts[5].strip()

                    if species.lower() != "homo sapiens":
                        continue

                    if not ensembl_id or not pathway_id:
                        continue

                    mapping.setdefault(ensembl_id, []).append(
                        {
                            "pathway_id": pathway_id,
                            "pathway_name": pathway_name,
                            "pathway_url": pathway_url,
                            "evidence_code": evidence_code,
                        }
                    )

            self._pathways_by_ensembl = mapping
            self._loaded = True
            logger.info(
                "Reactome: loaded pathway mappings for %d Ensembl IDs",
                len(self._pathways_by_ensembl),
            )

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str | None,
        ensembl_to_symbol: dict[str, str],
    ) -> list[EvidenceDocument]:
        """Return pathway evidence for targets linked to *drug_name*.

        Parameters
        ----------
        drug_name
            Canonical drug name.
        chembl_id
            Optional ChEMBL ID for metadata propagation.
        ensembl_to_symbol
            Mapping of Ensembl gene IDs to human-readable target symbols.
        """
        if not ensembl_to_symbol:
            return []

        await self._ensure_loaded()

        docs: list[EvidenceDocument] = []
        seen_pairs: set[tuple[str, str]] = set()

        for ensembl_id, target_symbol in ensembl_to_symbol.items():
            pathways = self._pathways_by_ensembl.get(ensembl_id, [])
            for pathway in pathways:
                pair = (ensembl_id, pathway["pathway_id"])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                text = (
                    f"Reactome pathway mapping: target {target_symbol or ensembl_id} "
                    f"participates in pathway '{pathway['pathway_name']}' "
                    f"({pathway['pathway_id']})."
                )

                docs.append(
                    EvidenceDocument(
                        text=text,
                        source=EvidenceSource.REACTOME,
                        evidence_type=EvidenceType.PATHWAY,
                        citation=Citation(
                            url=pathway["pathway_url"] or "https://reactome.org",
                            title=pathway["pathway_name"] or None,
                        ),
                        drug_name=drug_name,
                        drug_chembl_id=chembl_id,
                        target_symbol=target_symbol or None,
                        target_ensembl_id=ensembl_id,
                        metadata={
                            "pathway_id": pathway["pathway_id"],
                            "pathway_name": pathway["pathway_name"],
                            "evidence_code": pathway["evidence_code"],
                        },
                    )
                )

        logger.info(
            "Reactome: fetched %d pathway docs for %s",
            len(docs),
            drug_name,
        )
        return docs
