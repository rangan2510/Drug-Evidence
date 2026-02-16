"""Agent tools backed by MongoDB staged evidence.

These replace the live-API ``db_tools`` and Qdrant ``search_tools`` with
queries against the pre-populated MongoDB ``evidence`` collection.  Every
tool operates within the scope of the current drug (``ctx.deps.drug_name``).
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from src.agents.deps import EvidenceDeps

logger = logging.getLogger(__name__)

# Cap per tool call to keep the LLM context manageable.
_MAX_DOCS = 30
_TEXT_TRUNC = 800


def _fmt_doc(d: dict, *, include_text: bool = True) -> dict:
    """Normalise a MongoDB evidence record for LLM consumption."""
    out: dict = {
        "source": d.get("source", ""),
        "drug_name": d.get("drug_name", ""),
        "target_symbol": d.get("target_symbol") or "",
        "disease_name": d.get("disease_name") or "",
        "evidence_type": d.get("evidence_type", ""),
        "score": d.get("score"),
        "pmid": (d.get("citation") or {}).get("pmid") or "",
        "doi": (d.get("citation") or {}).get("doi") or "",
    }
    if include_text:
        out["text"] = (d.get("text") or "")[:_TEXT_TRUNC]
    return out


# ------------------------------------------------------------------
# Full-text search
# ------------------------------------------------------------------


async def search_staged_evidence(
    ctx: RunContext[EvidenceDeps],
    query: str,
    limit: int = 15,
) -> list[dict]:
    """Search staged evidence for the current drug using full-text search.

    Args:
        ctx: Agent run context with dependencies.
        query: Natural-language query describing the mechanism, target, or
            disease association to look up (e.g. "COX-2 inhibition colorectal
            cancer").
        limit: Maximum results to return (default 15, max 30).

    Returns:
        Matching evidence records ranked by text relevance.
    """
    logger.info("[TOOL] search_staged_evidence | drug=%s | query=%r | limit=%d", ctx.deps.drug_name, query, limit)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    limit = min(limit, _MAX_DOCS)
    results = await store.search_text(ctx.deps.drug_name, query, limit=limit)
    logger.info("[TOOL] search_staged_evidence -> %d results", len(results))
    return [_fmt_doc(r) for r in results]


# ------------------------------------------------------------------
# Source-specific queries
# ------------------------------------------------------------------


async def lookup_staged_dgidb(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch DGIdb drug-gene interactions from staged evidence.

    Returns a list of records with ``target_symbol``, ``interaction_type``,
    ``text``, ``pmid``, and ``score``.
    """
    logger.info("[TOOL] lookup_staged_dgidb | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(ctx.deps.drug_name, "dgidb", limit=_MAX_DOCS)
    logger.info("[TOOL] lookup_staged_dgidb -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def lookup_staged_opentargets(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch OpenTargets disease associations from staged evidence.

    Returns records with ``target_symbol``, ``disease_name``,
    ``disease_id``, ``score``, ``text``, ``pmid``.
    """
    logger.info("[TOOL] lookup_staged_opentargets | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(
        ctx.deps.drug_name, "opentargets", limit=_MAX_DOCS
    )
    logger.info("[TOOL] lookup_staged_opentargets -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def lookup_staged_pubchem(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch PubChem pharmacological data from staged evidence."""
    logger.info("[TOOL] lookup_staged_pubchem | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(ctx.deps.drug_name, "pubchem", limit=_MAX_DOCS)
    logger.info("[TOOL] lookup_staged_pubchem -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def lookup_staged_chembl(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch ChEMBL mechanism-of-action and binding data from staged evidence."""
    logger.info("[TOOL] lookup_staged_chembl | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(ctx.deps.drug_name, "chembl", limit=_MAX_DOCS)
    logger.info("[TOOL] lookup_staged_chembl -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def lookup_staged_pharmgkb(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch PharmGKB clinical annotations from staged evidence."""
    logger.info("[TOOL] lookup_staged_pharmgkb | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(ctx.deps.drug_name, "pharmgkb", limit=_MAX_DOCS)
    logger.info("[TOOL] lookup_staged_pharmgkb -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def lookup_staged_pubmed(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch PubMed abstracts from staged evidence.

    Unlike the live ``search_pubmed`` tool, this returns only abstracts
    that were pre-fetched during the staging phase.
    """
    logger.info("[TOOL] lookup_staged_pubmed | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_source(ctx.deps.drug_name, "pubmed", limit=_MAX_DOCS)
    result = [
        {
            "title": (d.get("citation") or {}).get("title", ""),
            "abstract": (d.get("text") or "")[:1500],
            "pmid": (d.get("citation") or {}).get("pmid", ""),
            "year": (d.get("citation") or {}).get("year"),
            "source": "pubmed",
        }
        for d in docs
    ]
    logger.info("[TOOL] lookup_staged_pubmed -> %d abstracts", len(result))
    return result


# ------------------------------------------------------------------
# Target / disease filtering
# ------------------------------------------------------------------


async def filter_by_target(
    ctx: RunContext[EvidenceDeps],
    target_symbol: str,
    limit: int = 20,
) -> list[dict]:
    """Retrieve staged evidence mentioning a specific gene/target symbol.

    Args:
        ctx: Agent run context.
        target_symbol: Gene symbol to filter on (e.g. "PTGS2", "EGFR").
        limit: Max results (default 20).

    Returns:
        Evidence records referencing the given target.
    """
    logger.info("[TOOL] filter_by_target | drug=%s | target=%s", ctx.deps.drug_name, target_symbol)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_target(
        ctx.deps.drug_name,
        target_symbol,
        limit=min(limit, _MAX_DOCS),
    )
    logger.info("[TOOL] filter_by_target -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


async def filter_by_disease(
    ctx: RunContext[EvidenceDeps],
    disease_name: str,
    limit: int = 20,
) -> list[dict]:
    """Retrieve staged evidence mentioning a specific disease.

    Args:
        ctx: Agent run context.
        disease_name: Disease name to filter on (e.g. "breast cancer").
        limit: Max results (default 20).

    Returns:
        Evidence records referencing the given disease.
    """
    logger.info("[TOOL] filter_by_disease | drug=%s | disease=%s", ctx.deps.drug_name, disease_name)
    store = ctx.deps.evidence_store
    if store is None:
        return [{"error": "MongoDB evidence store not available"}]

    docs = await store.find_by_disease(
        ctx.deps.drug_name,
        disease_name,
        limit=min(limit, _MAX_DOCS),
    )
    logger.info("[TOOL] filter_by_disease -> %d docs", len(docs))
    return [_fmt_doc(d) for d in docs]


# ------------------------------------------------------------------
# Evidence overview
# ------------------------------------------------------------------


async def get_evidence_overview(
    ctx: RunContext[EvidenceDeps],
) -> dict:
    """Get a summary of all staged evidence for the current drug.

    Returns a dict with counts per source, distinct targets, distinct
    diseases, and total document count -- useful for planning which
    tools to call next.
    """
    logger.info("[TOOL] get_evidence_overview | drug=%s", ctx.deps.drug_name)
    store = ctx.deps.evidence_store
    if store is None:
        return {"error": "MongoDB evidence store not available"}

    summary = await store.get_evidence_summary(ctx.deps.drug_name)
    logger.info("[TOOL] get_evidence_overview -> %s", {k: v for k, v in summary.items() if k != 'error'})
    return summary
