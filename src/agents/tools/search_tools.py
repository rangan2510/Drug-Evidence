"""Agent tools for vector-store and PubMed search.

Registered on the evidence agent via ``@agent.tool``.  Each tool receives
``RunContext[EvidenceDeps]`` and returns plain dicts/lists (not Pydantic
models) for the LLM to consume.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import RunContext

from src.agents.deps import EvidenceDeps

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Qdrant hybrid search (sync underneath, wrapped for async agent)
# ------------------------------------------------------------------


async def search_evidence(
    ctx: RunContext[EvidenceDeps],
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search the evidence vector store with hybrid dense+sparse retrieval.

    Args:
        ctx: Agent run context with dependencies.
        query: Natural-language query describing the mechanism or association
            to look up (e.g. "aspirin COX-2 inhibition colorectal cancer").
        limit: Maximum number of results to return (default 10).

    Returns:
        A list of dicts, each containing ``text``, ``score``, ``source``,
        ``drug_name``, ``target_symbol``, ``disease_name``, ``pmid``.
    """
    logger.info("[TOOL] search_evidence | drug=%s | query=%r | limit=%d", ctx.deps.drug_name, query, limit)
    store = ctx.deps.vector_store
    if store is None:
        return [{"error": "Vector store is not available"}]
    drug = ctx.deps.drug_name

    # hybrid_search is synchronous -- run in thread to avoid blocking
    results = await asyncio.to_thread(
        store.hybrid_search,
        query,
        limit=min(limit, 20),
        drug_filter=drug or None,
    )
    logger.info("[TOOL] search_evidence -> %d results", len(results))

    return [
        {
            "text": r.text,
            "score": round(r.score, 4),
            "source": r.payload.get("source", ""),
            "drug_name": r.payload.get("drug_name", ""),
            "target_symbol": r.payload.get("target_symbol", ""),
            "disease_name": r.payload.get("disease_name", ""),
            "pmid": r.payload.get("pmid", ""),
            "doi": r.payload.get("doi", ""),
        }
        for r in results
    ]


# ------------------------------------------------------------------
# PubMed literature search (async)
# ------------------------------------------------------------------


async def search_pubmed(
    ctx: RunContext[EvidenceDeps],
    query_terms: str,
    max_results: int = 10,
) -> list[dict]:
    """Search PubMed for recent literature related to the current drug.

    Args:
        ctx: Agent run context with dependencies.
        query_terms: Additional search terms to combine with the drug name
            (e.g. "mechanism of action breast cancer").
        max_results: Maximum number of abstracts to return (default 10).

    Returns:
        A list of dicts, each containing ``title``, ``abstract``, ``pmid``,
        ``year``.
    """
    logger.info("[TOOL] search_pubmed | drug=%s | query=%r | max=%d", ctx.deps.drug_name, query_terms, max_results)
    pubmed = ctx.deps.aggregator._pubmed
    drug = ctx.deps.drug_name
    chembl_id = ctx.deps.chembl_id

    docs = await pubmed.search_and_fetch(
        drug_name=drug,
        chembl_id=chembl_id,
        extra_terms=query_terms,
        max_results=min(max_results, 20),
    )
    logger.info("[TOOL] search_pubmed -> %d abstracts", len(docs))

    return [
        {
            "title": d.citation.title or "",
            "abstract": d.text[:1500],  # truncate very long abstracts
            "pmid": d.citation.pmid or "",
            "year": d.citation.year,
            "source": d.source.value,
        }
        for d in docs
    ]
