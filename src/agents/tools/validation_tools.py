"""Agent tools for citation and entity validation.

Provides PMID/DOI resolution checks so the agent can verify that cited
references actually exist before including them in evidence chains.
"""

from __future__ import annotations

import logging

import httpx
from pydantic_ai import RunContext

from src.agents.deps import EvidenceDeps
from src.data.http import cached_async_client

logger = logging.getLogger(__name__)

_PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_DOI_RESOLVER = "https://doi.org"


async def validate_pmids(
    ctx: RunContext[EvidenceDeps],
    pmids: list[str],
) -> list[dict]:
    """Validate a list of PubMed IDs by checking them against the NCBI API.

    Args:
        ctx: Agent run context with dependencies.
        pmids: List of PMID strings to validate (e.g. ["12345678", "87654321"]).

    Returns:
        A list of dicts, each containing ``pmid``, ``valid`` (bool), and
        ``title`` if the PMID exists.
    """
    logger.info("[TOOL] validate_pmids | count=%d | sample=%s", len(pmids), pmids[:3])
    if not pmids:
        return []

    # Deduplicate and cap at 50 to avoid overloading
    unique_pmids = list(dict.fromkeys(pmids))[:50]

    settings = ctx.deps.settings
    params = {
        "db": "pubmed",
        "id": ",".join(unique_pmids),
        "retmode": "json",
    }
    if settings.entrez_email:
        params["email"] = settings.entrez_email

    results: list[dict] = []

    try:
        async with cached_async_client(settings) as client:
            resp = await client.get(
                _PUBMED_ESUMMARY,
                params=params,
                timeout=httpx.Timeout(20.0),
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("PMID validation request failed: %s", exc)
        return [{"pmid": p, "valid": False, "error": str(exc)} for p in unique_pmids]

    esummary_result = data.get("result", {})
    for pmid in unique_pmids:
        info = esummary_result.get(pmid, {})
        if "error" in info:
            results.append({"pmid": pmid, "valid": False})
        elif info.get("title"):
            results.append({
                "pmid": pmid,
                "valid": True,
                "title": info["title"],
                "year": info.get("pubdate", "")[:4],
            })
        else:
            results.append({"pmid": pmid, "valid": False})

    valid_count = sum(1 for r in results if r.get("valid"))
    logger.info("[TOOL] validate_pmids -> %d/%d valid", valid_count, len(results))
    return results


async def validate_doi(
    ctx: RunContext[EvidenceDeps],
    doi: str,
) -> dict:
    """Validate a DOI by performing a HEAD request to doi.org.

    Args:
        ctx: Agent run context with dependencies.
        doi: The DOI string (e.g. "10.1038/s41586-020-2649-2").

    Returns:
        A dict with ``doi``, ``valid`` (bool), and optionally
        ``resolved_url``.
    """
    logger.info("[TOOL] validate_doi | doi=%s", doi)
    url = f"{_DOI_RESOLVER}/{doi}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(15.0),
        ) as client:
            resp = await client.head(url)
            if resp.status_code < 400:
                return {
                    "doi": doi,
                    "valid": True,
                    "resolved_url": str(resp.url),
                }
            return {"doi": doi, "valid": False, "status": resp.status_code}
    except httpx.HTTPError as exc:
        logger.warning("DOI validation failed for %s: %s", doi, exc)
        return {"doi": doi, "valid": False, "error": str(exc)}
