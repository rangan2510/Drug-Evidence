"""PubMed eFetch / eSearch client.

Batch-fetches PubMed abstracts by PMID or by search query, returning
uniform ``list[EvidenceDocument]``.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried."""
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False

from src.config.settings import Settings
from src.data.http import cached_async_client
from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)

logger = logging.getLogger(__name__)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_BATCH_SIZE = 50  # PMIDs per eFetch request (NCBI recommends <= 200)


class PubMedClient:
    """Batch-fetch PubMed abstracts and search for drug-related literature."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(30.0)

    async def fetch_by_pmids(
        self,
        pmids: list[str],
        drug_name: str,
        chembl_id: str | None = None,
    ) -> list[EvidenceDocument]:
        """Fetch abstracts for a list of PMIDs.

        Parameters
        ----------
        pmids
            PubMed IDs to fetch.
        drug_name
            Drug name for metadata tagging.
        chembl_id
            Optional ChEMBL ID.
        """
        unique_pmids = list(dict.fromkeys(pmids))  # dedupe, preserve order
        if not unique_pmids:
            return []

        docs: list[EvidenceDocument] = []
        async with cached_async_client(self._settings) as client:
            for i in range(0, len(unique_pmids), _BATCH_SIZE):
                batch = unique_pmids[i : i + _BATCH_SIZE]
                batch_docs = await self._efetch_batch(
                    client, batch, drug_name, chembl_id
                )
                docs.extend(batch_docs)

        logger.info(
            "PubMed: fetched %d abstracts from %d PMIDs for %s",
            len(docs),
            len(unique_pmids),
            drug_name,
        )
        return docs

    async def search_and_fetch(
        self,
        drug_name: str,
        chembl_id: str | None = None,
        *,
        extra_terms: str = "",
        max_results: int = 20,
    ) -> list[EvidenceDocument]:
        """Search PubMed for drug-related articles and fetch abstracts.

        Parameters
        ----------
        drug_name
            Drug name for the search query.
        chembl_id
            Optional ChEMBL ID (metadata).
        extra_terms
            Additional search terms (e.g. disease name).
        max_results
            Maximum number of PMIDs from the search.
        """
        query = f"{drug_name} AND (mechanism OR therapeutic OR treatment)"
        if extra_terms:
            query = f"{drug_name} AND ({extra_terms})"

        async with cached_async_client(self._settings) as client:
            pmids = await self._esearch(client, query, max_results)

        if not pmids:
            return []

        return await self.fetch_by_pmids(pmids, drug_name, chembl_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=_is_retryable,
        reraise=True,
    )
    async def _esearch(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[str]:
        """Run an eSearch query, return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if self._settings.entrez_email:
            params["email"] = self._settings.entrez_email

        resp = await client.get(_ESEARCH_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=_is_retryable,
        reraise=True,
    )
    async def _efetch_batch(
        self,
        client: httpx.AsyncClient,
        pmids: list[str],
        drug_name: str,
        chembl_id: str | None,
    ) -> list[EvidenceDocument]:
        """Fetch abstracts for a batch of PMIDs via eFetch XML."""
        params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self._settings.entrez_email:
            params["email"] = self._settings.entrez_email

        resp = await client.get(_EFETCH_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()

        return self._parse_efetch_xml(resp.text, drug_name, chembl_id)

    def _parse_efetch_xml(
        self,
        xml_text: str,
        drug_name: str,
        chembl_id: str | None,
    ) -> list[EvidenceDocument]:
        """Parse eFetch XML into EvidenceDocuments."""
        docs: list[EvidenceDocument] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.warning("PubMed: failed to parse XML response")
            return docs

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else None

            # Title
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None else ""

            # Abstract
            abstract_parts = []
            for abs_text in article.findall(".//AbstractText"):
                label = abs_text.get("Label", "")
                text = abs_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            if not abstract:
                continue

            # Year
            year = None
            year_el = article.find(".//PubDate/Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    pass

            # DOI
            doi = None
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text
                    break

            docs.append(
                EvidenceDocument(
                    text=abstract,
                    source=EvidenceSource.PUBMED,
                    evidence_type=EvidenceType.LITERATURE,
                    citation=Citation(
                        pmid=pmid,
                        doi=doi,
                        title=title,
                        year=year,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        if pmid
                        else None,
                    ),
                    drug_name=drug_name,
                    drug_chembl_id=chembl_id,
                )
            )

        return docs
