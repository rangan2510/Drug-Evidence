"""Structured evidence extraction agent -- PydanticAI.

This agent receives a drug name + identifiers, queries biomedical databases
and a pre-indexed vector store, then returns a ``DrugDiseasePrediction`` with
mechanistic evidence chains linking the drug to every disease it considers
therapeutically relevant.

Two modes of operation:

* **Staged (MongoDB)**: when ``use_staged=True`` (default), tools query
  pre-populated MongoDB collections -- no live API calls, fully reproducible.
* **Live**: when ``use_staged=False``, tools hit the live data APIs and
  Qdrant vector store (original v2 behaviour).

Usage
-----
::

    from src.agents.evidence_agent import build_evidence_agent

    # Staged (preferred for experiments)
    agent = build_evidence_agent("openai:gpt-5.2-2025-12-11", use_staged=True)

    # Live (for debugging or ad-hoc queries)
    agent = build_evidence_agent("openai:gpt-5.2-2025-12-11", use_staged=False)
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from src.agents.deps import EvidenceDeps
from src.agents.tools.db_tools import (
    lookup_chembl,
    lookup_dgidb,
    lookup_opentargets,
    lookup_pharmgkb,
    lookup_pubchem,
)
from src.agents.tools.mongo_tools import (
    filter_by_disease,
    filter_by_target,
    get_evidence_overview,
    lookup_staged_chembl,
    lookup_staged_dgidb,
    lookup_staged_opentargets,
    lookup_staged_pharmgkb,
    lookup_staged_pubchem,
    lookup_staged_pubmed,
    search_staged_evidence,
)
from src.agents.tools.search_tools import search_evidence, search_pubmed
from src.agents.tools.validation_tools import validate_doi, validate_pmids
from src.schemas.prediction import DrugDiseasePrediction

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# System instructions
# ------------------------------------------------------------------

_SYSTEM_PROMPT_STAGED = """\
You are a biomedical evidence extraction agent.  Your task is to predict
which diseases a given drug may be therapeutically relevant for, and to
support each prediction with mechanistic evidence chains.

WORKFLOW
--------
1. Start by calling get_evidence_overview to understand the available staged
   evidence (sources, targets, diseases, document counts).
2. Use search_staged_evidence with broad queries about the drug's mechanism
   of action, targets, and pathways.
3. Optionally use search_evidence for hybrid vector retrieval when a staged
    Qdrant index is available.
4. Use the source-specific lookup tools (lookup_staged_dgidb,
   lookup_staged_opentargets, lookup_staged_chembl, lookup_staged_pubchem,
   lookup_staged_pharmgkb, lookup_staged_pubmed) to gather structured
   evidence about drug-gene interactions, disease associations, mechanism
   of action, and clinical annotations.
5. Use filter_by_target and filter_by_disease to drill into specific
   hypotheses linking genes to diseases.
6. Validate key PMIDs (validate_pmids) to ensure citations are real.

EVIDENCE CHAINS
---------------
For each disease association you predict, construct one or more mechanistic
evidence chains of the form:

    drug -> [binds/inhibits/activates] -> target_gene
          -> [participates_in/modulates] -> pathway
          -> [associated_with] -> disease

Each edge must have:
  - source_entity and target_entity (specific names, not generic terms)
  - relationship (one of: binds, inhibits, activates, upregulates,
    downregulates, modulates, transports, metabolizes, participates_in,
    associated_with, promotes, suppresses, regulates, contributes_to,
    interacts_with, coactivates)
  - evidence_snippet (a direct quote or close paraphrase from a source)
  - pmid (if available)

SCORING
-------
- confidence: 0.0-1.0 reflecting the strength of evidence
- predicted: true if you believe the association is therapeutically relevant

OUTPUT RULES
------------
- Include ALL diseases with non-trivial evidence, even if confidence is low.
- Do NOT fabricate PMIDs or evidence snippets.  If you cannot find a PMID,
  leave it as null.
- The reasoning field should be a concise (2-4 sentence) summary of your
  overall analysis strategy and key findings.
- Prefer specific gene symbols (e.g. PTGS2) over vague terms.
- Do not use emojis anywhere in the output.
"""

_SYSTEM_PROMPT_LIVE = """\
You are a biomedical evidence extraction agent.  Your task is to predict
which diseases a given drug may be therapeutically relevant for, and to
support each prediction with mechanistic evidence chains.

WORKFLOW
--------
1. Start by querying the vector store (search_evidence) with broad queries
   about the drug's mechanism of action, targets, and pathways.
2. Use the database lookup tools (lookup_dgidb, lookup_opentargets,
   lookup_chembl, lookup_pubchem, lookup_pharmgkb) to gather structured
   evidence about drug-gene interactions, disease associations, mechanism
   of action, and clinical annotations.
3. Search PubMed (search_pubmed) for recent literature supporting or
   refuting specific mechanistic hypotheses.
4. Validate key PMIDs (validate_pmids) to ensure citations are real.

EVIDENCE CHAINS
---------------
For each disease association you predict, construct one or more mechanistic
evidence chains of the form:

    drug -> [binds/inhibits/activates] -> target_gene
          -> [participates_in/modulates] -> pathway
          -> [associated_with] -> disease

Each edge must have:
  - source_entity and target_entity (specific names, not generic terms)
  - relationship (one of: binds, inhibits, activates, upregulates,
    downregulates, modulates, transports, metabolizes, participates_in,
    associated_with, promotes, suppresses, regulates, contributes_to,
    interacts_with, coactivates)
  - evidence_snippet (a direct quote or close paraphrase from a source)
  - pmid (if available)

SCORING
-------
- confidence: 0.0-1.0 reflecting the strength of evidence
- predicted: true if you believe the association is therapeutically relevant

OUTPUT RULES
------------
- Include ALL diseases with non-trivial evidence, even if confidence is low.
- Do NOT fabricate PMIDs or evidence snippets.  If you cannot find a PMID,
  leave it as null.
- The reasoning field should be a concise (2-4 sentence) summary of your
  overall analysis strategy and key findings.
- Prefer specific gene symbols (e.g. PTGS2) over vague terms.
- Do not use emojis anywhere in the output.
"""


# ------------------------------------------------------------------
# Agent factories
# ------------------------------------------------------------------


def build_evidence_agent(
    model_id: str | Model,
    *,
    retries: int = 2,
    use_staged: bool = True,
) -> Agent[EvidenceDeps, DrugDiseasePrediction]:
    """Build a PydanticAI evidence extraction agent.

    Parameters
    ----------
    model_id:
        PydanticAI model string (e.g. ``"openai:gpt-5.2-2025-12-11"``)
        or a ``Model`` instance (e.g. ``TestModel`` for unit tests).
    retries:
        Number of retries for failed LLM calls (default 2).
    use_staged:
        If ``True`` (default), register MongoDB tools that query staged
        evidence.  If ``False``, register live-API + Qdrant tools.

    Returns
    -------
    Agent[EvidenceDeps, DrugDiseasePrediction]
        A configured agent ready to be run with ``await agent.run(...)``.
    """
    prompt = _SYSTEM_PROMPT_STAGED if use_staged else _SYSTEM_PROMPT_LIVE

    agent: Agent[EvidenceDeps, DrugDiseasePrediction] = Agent(
        model=model_id,
        output_type=DrugDiseasePrediction,
        deps_type=EvidenceDeps,
        retries=retries,
        instructions=prompt,
    )

    if use_staged:
        # MongoDB-backed tools -- no live API calls
        agent.tool(get_evidence_overview)
        agent.tool(search_evidence)
        agent.tool(search_staged_evidence)
        agent.tool(lookup_staged_dgidb)
        agent.tool(lookup_staged_opentargets)
        agent.tool(lookup_staged_pubchem)
        agent.tool(lookup_staged_chembl)
        agent.tool(lookup_staged_pharmgkb)
        agent.tool(lookup_staged_pubmed)
        agent.tool(filter_by_target)
        agent.tool(filter_by_disease)
    else:
        # Live API + Qdrant tools (original behaviour)
        agent.tool(search_evidence)
        agent.tool(search_pubmed)
        agent.tool(lookup_dgidb)
        agent.tool(lookup_opentargets)
        agent.tool(lookup_pubchem)
        agent.tool(lookup_chembl)
        agent.tool(lookup_pharmgkb)

    # Always register validation tools
    agent.tool(validate_pmids)
    agent.tool(validate_doi)

    return agent


# ------------------------------------------------------------------
# Dynamic system prompt with drug context
# ------------------------------------------------------------------


def _add_drug_context_prompt(agent: Agent[EvidenceDeps, DrugDiseasePrediction]) -> None:
    """Register a dynamic system prompt that injects drug-specific context."""

    @agent.system_prompt
    async def drug_context(ctx: RunContext[EvidenceDeps]) -> str:
        parts = [f"Drug under analysis: {ctx.deps.drug_name}"]
        if ctx.deps.chembl_id:
            parts.append(f"ChEMBL ID: {ctx.deps.chembl_id}")
        if ctx.deps.pubchem_cid:
            parts.append(f"PubChem CID: {ctx.deps.pubchem_cid}")
        return "\n".join(parts)


def build_evidence_agent_with_context(
    model_id: str | Model,
    *,
    retries: int = 2,
    use_staged: bool = True,
) -> Agent[EvidenceDeps, DrugDiseasePrediction]:
    """Build an evidence agent with a dynamic drug-context system prompt.

    Identical to ``build_evidence_agent`` but also adds a dynamic system
    prompt that injects the current drug name and identifiers into the
    conversation.
    """
    agent = build_evidence_agent(model_id, retries=retries, use_staged=use_staged)
    _add_drug_context_prompt(agent)
    return agent
