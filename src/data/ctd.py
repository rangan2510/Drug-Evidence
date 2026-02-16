"""CTD (Comparative Toxicogenomics Database) ground-truth loader.

Downloads the CTD chemicals-diseases association file and parses it into
a lookup of drug -> set[disease_name] for evaluation.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

_CTD_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class CTDRecord:
    """Single CTD chemical-disease association."""

    chemical_name: str
    chemical_id: str  # e.g. "MESH:D001241"
    disease_name: str
    disease_id: str  # e.g. "MESH:D003920"
    direct_evidence: str  # "marker/mechanism" or "therapeutic"
    inference_gene_symbol: str = ""
    inference_score: float | None = None
    pmids: list[str] = field(default_factory=list)


class CTDClient:
    """Load and query CTD ground-truth associations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._records: list[CTDRecord] = []
        self._therapeutic: dict[str, set[str]] = {}  # drug_lower -> {disease_name}

    @property
    def is_loaded(self) -> bool:
        return len(self._records) > 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load(self, cache_path: Path | None = None) -> None:
        """Download (or load cached) CTD data and index it.

        Parameters
        ----------
        cache_path
            Local ``.tsv.gz`` path.  If it exists, skip download.
        """
        cache_path = cache_path or (_DATA_DIR / "CTD_chemicals_diseases.tsv.gz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not cache_path.exists():
            await self._download(cache_path)

        self._parse(cache_path)
        self._index()
        logger.info(
            "CTD loaded: %d records, %d drugs with therapeutic evidence",
            len(self._records),
            len(self._therapeutic),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _download(self, dest: Path) -> None:
        logger.info("Downloading CTD chemicals-diseases file ...")
        async with cached_async_client(
            self._settings,
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
        ) as client:
            resp = await client.get(_CTD_URL)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        logger.info("CTD file saved to %s (%d bytes)", dest, dest.stat().st_size)

    def _parse(self, path: Path) -> None:
        """Parse the gzipped TSV into CTDRecord objects."""
        records: list[CTDRecord] = []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
                # Columns: ChemicalName, ChemicalID, CasRN, DiseaseName,
                #          DiseaseID, DirectEvidence, InferenceGeneSymbol,
                #          InferenceScore, OmimIDs, PubMedIDs
                direct_ev = parts[5].strip()
                if not direct_ev:
                    continue  # skip inferred-only rows

                pmids_raw = parts[9] if len(parts) > 9 else ""
                pmids = [p.strip() for p in pmids_raw.split("|") if p.strip()]

                inf_score: float | None = None
                if parts[7].strip():
                    try:
                        inf_score = float(parts[7])
                    except ValueError:
                        pass

                records.append(
                    CTDRecord(
                        chemical_name=parts[0].strip(),
                        chemical_id=parts[1].strip(),
                        disease_name=parts[3].strip(),
                        disease_id=parts[4].strip(),
                        direct_evidence=direct_ev,
                        inference_gene_symbol=parts[6].strip(),
                        inference_score=inf_score,
                        pmids=pmids,
                    )
                )
        self._records = records

    def _index(self) -> None:
        """Build fast lookup: drug_lower -> set of therapeutic disease names."""
        therapeutic: dict[str, set[str]] = {}
        for rec in self._records:
            if "therapeutic" in rec.direct_evidence.lower():
                key = rec.chemical_name.lower()
                therapeutic.setdefault(key, set()).add(rec.disease_name)
        self._therapeutic = therapeutic

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_therapeutic_diseases(self, drug_name: str) -> set[str]:
        """Return set of disease names with *therapeutic* evidence for *drug_name*."""
        return self._therapeutic.get(drug_name.lower(), set())

    def count_associations(self, drug_name: str) -> int:
        """Number of therapeutic disease associations."""
        return len(self.get_therapeutic_diseases(drug_name))

    def get_candidate_drugs(
        self,
        min_assoc: int | None = None,
        max_assoc: int | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return drug names filtered by association count.

        Parameters
        ----------
        min_assoc / max_assoc
            Inclusive bounds on the number of therapeutic associations.
        limit
            Max drugs to return (after filtering).
        """
        min_a = min_assoc if min_assoc is not None else self._settings.min_associations
        max_a = max_assoc if max_assoc is not None else self._settings.max_associations

        candidates = [
            name
            for name, diseases in self._therapeutic.items()
            if min_a <= len(diseases) <= max_a
        ]
        candidates.sort()
        if limit is not None:
            candidates = candidates[:limit]
        return candidates
