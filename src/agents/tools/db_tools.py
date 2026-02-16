"""Agent tools for querying biomedical databases.

Each tool wraps one async data client, exposing a simplified dict interface
for the LLM.  Registered on the evidence agent via ``@agent.tool``.
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from src.agents.deps import EvidenceDeps

logger = logging.getLogger(__name__)

# Max evidence documents to return per tool call (keeps context small)
_MAX_DOCS = 30


# ------------------------------------------------------------------
# DGIdb -- drug-gene interactions
# ------------------------------------------------------------------

async def lookup_dgidb(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch drug-gene interactions from DGIdb for the current drug.

    Returns a list of dicts with ``target_symbol``, ``interaction_type``,
    ``text``, ``pmid``, and ``score``.
    """
    logger.info("[TOOL] lookup_dgidb | drug=%s", ctx.deps.drug_name)
    client = ctx.deps.aggregator._dgidb
    docs = await client.fetch(
        drug_name=ctx.deps.drug_name,
        chembl_id=ctx.deps.chembl_id,
    )
    logger.info("[TOOL] lookup_dgidb -> %d docs", len(docs))
    return [
        {
            "target_symbol": d.target_symbol or "",
            "interaction_type": d.metadata.get("interaction_type", ""),
            "text": d.text[:800],
            "pmid": d.citation.pmid or "",
            "score": d.score,
        }
        for d in docs[:_MAX_DOCS]
    ]


# ------------------------------------------------------------------
# OpenTargets -- disease associations via target
# ------------------------------------------------------------------

async def lookup_opentargets(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch disease associations from OpenTargets for the current drug.

    Requires a ChEMBL ID. Returns a list of dicts with
    ``target_symbol``, ``disease_name``, ``disease_id``, ``score``,
    ``text``, and ``pmid``.
    """
    logger.info("[TOOL] lookup_opentargets | drug=%s | chembl=%s", ctx.deps.drug_name, ctx.deps.chembl_id)
    if not ctx.deps.chembl_id:
        return [{"error": "ChEMBL ID not available for this drug"}]

    client = ctx.deps.aggregator._opentargets
    docs = await client.fetch(
        drug_name=ctx.deps.drug_name,
        chembl_id=ctx.deps.chembl_id,
    )
    logger.info("[TOOL] lookup_opentargets -> %d docs", len(docs))
    return [
        {
            "target_symbol": d.target_symbol or "",
            "disease_name": d.disease_name or "",
            "disease_id": d.disease_id or "",
            "score": d.score,
            "text": d.text[:800],
            "pmid": d.citation.pmid or "",
        }
        for d in docs[:_MAX_DOCS]
    ]


# ------------------------------------------------------------------
# PubChem -- pharmacology / bioassay data
# ------------------------------------------------------------------

async def lookup_pubchem(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch pharmacological data from PubChem for the current drug.

    Returns a list of dicts with ``text``, ``evidence_type``, ``pmid``,
    and ``score``.
    """
    logger.info("[TOOL] lookup_pubchem | drug=%s", ctx.deps.drug_name)
    client = ctx.deps.aggregator._pubchem
    docs = await client.fetch(
        drug_name=ctx.deps.drug_name,
        chembl_id=ctx.deps.chembl_id,
        pubchem_cid=ctx.deps.pubchem_cid,
    )
    logger.info("[TOOL] lookup_pubchem -> %d docs", len(docs))
    return [
        {
            "text": d.text[:800],
            "evidence_type": d.evidence_type.value,
            "pmid": d.citation.pmid or "",
            "score": d.score,
        }
        for d in docs[:_MAX_DOCS]
    ]


# ------------------------------------------------------------------
# ChEMBL -- mechanism of action + binding activities
# ------------------------------------------------------------------

async def lookup_chembl(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch mechanism-of-action and binding data from ChEMBL.

    Requires a ChEMBL ID. Returns a list of dicts with
    ``target_symbol``, ``text``, ``evidence_type``, ``pmid``, and ``score``.
    """
    logger.info("[TOOL] lookup_chembl | drug=%s | chembl=%s", ctx.deps.drug_name, ctx.deps.chembl_id)
    if not ctx.deps.chembl_id:
        return [{"error": "ChEMBL ID not available for this drug"}]

    client = ctx.deps.aggregator._chembl
    docs = await client.fetch(
        drug_name=ctx.deps.drug_name,
        chembl_id=ctx.deps.chembl_id,
    )
    logger.info("[TOOL] lookup_chembl -> %d docs", len(docs))
    return [
        {
            "target_symbol": d.target_symbol or "",
            "text": d.text[:800],
            "evidence_type": d.evidence_type.value,
            "pmid": d.citation.pmid or "",
            "score": d.score,
        }
        for d in docs[:_MAX_DOCS]
    ]


# ------------------------------------------------------------------
# PharmGKB -- clinical annotations + drug labels
# ------------------------------------------------------------------

async def lookup_pharmgkb(
    ctx: RunContext[EvidenceDeps],
) -> list[dict]:
    """Fetch clinical annotations from PharmGKB for the current drug.

    Returns a list of dicts with ``text``, ``evidence_type``, ``pmid``,
    ``disease_name``, and ``score``.
    """
    logger.info("[TOOL] lookup_pharmgkb | drug=%s", ctx.deps.drug_name)
    client = ctx.deps.aggregator._pharmgkb
    docs = await client.fetch(
        drug_name=ctx.deps.drug_name,
        chembl_id=ctx.deps.chembl_id,
    )
    logger.info("[TOOL] lookup_pharmgkb -> %d docs", len(docs))
    return [
        {
            "text": d.text[:800],
            "evidence_type": d.evidence_type.value,
            "disease_name": d.disease_name or "",
            "pmid": d.citation.pmid or "",
            "score": d.score,
        }
        for d in docs[:_MAX_DOCS]
    ]
