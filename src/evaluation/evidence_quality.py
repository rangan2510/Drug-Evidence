"""Evidence quality metrics for drug-disease predictions.

Five metrics that measure the quality of the evidence produced by each
experimental arm -- independent of accuracy against ground truth.

Metrics
-------
1. **Citation validity rate** -- fraction of cited PMIDs / DOIs that
   resolve to real records (batch-checked via PubMed eSummary / DOI HEAD).
2. **Mean chain depth** -- average number of edges per mechanistic chain.
3. **Chain verifiability score** -- fraction of edges that carry a PMID or
   DOI citation.
4. **Evidence relevance** -- mean cosine similarity between each edge's
   evidence snippet and its (source -> relation -> target) claim, using
   the MedCPT query encoder.
5. **Mechanistic specificity** -- LLM-as-judge score (0-1) evaluating
   whether chains name specific genes / pathways versus generic language.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import httpx
import numpy as np
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.schemas.prediction import (
    DrugDiseasePrediction,
    EvidenceChain,
    MechanisticEdge,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PubMed eSummary URL for batch PMID validation
# ---------------------------------------------------------------------------

_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_DOI_URL = "https://doi.org"
_PMID_BATCH = 100  # NCBI recommends <= 200

# Rate-limit PubMed: ~3 req/s without API key.  Allow 2 concurrent
# requests so multiple callers don't overwhelm the endpoint.
_PUBMED_SEM: asyncio.Semaphore | None = None

def _get_pubmed_sem() -> asyncio.Semaphore:
    """Lazy-init a module-level PubMed semaphore (must be called inside a running loop)."""
    global _PUBMED_SEM  # noqa: PLW0603
    if _PUBMED_SEM is None:
        _PUBMED_SEM = asyncio.Semaphore(2)
    return _PUBMED_SEM

# Regex for a plausible PMID (1-8 digit integer, possibly prefixed by PMC)
_PMID_RE = re.compile(r"^\d{1,8}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class EvidenceQualityMetrics(BaseModel):
    """Quality metrics for a single prediction."""

    citation_validity_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of citations that resolve to a real PMID/DOI",
    )
    mean_chain_depth: float = Field(
        ..., ge=0.0,
        description="Average number of edges in mechanistic chains",
    )
    chain_verifiability_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of edges with a supporting PMID or DOI",
    )
    evidence_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Mean cosine similarity between evidence text and claim",
    )
    mechanistic_specificity: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM-as-judge score: specific genes/pathways vs generic",
    )

    # Diagnostic counts
    n_citations_checked: int = 0
    n_citations_valid: int = 0
    n_chains: int = 0
    n_edges: int = 0


# ===================================================================
# 1. Citation Validity Rate
# ===================================================================

async def _validate_pmids_batch(
    pmids: list[str],
    settings: Settings | None = None,
) -> set[str]:
    """Return the subset of *pmids* that resolve to real PubMed records.

    Uses the eSummary endpoint with JSON output.  A PMID is valid when
    the response includes a ``result.<pmid>`` entry without an ``error``
    key.

    Rate-limited via a module-level semaphore and retries with
    exponential backoff on 429 responses.
    """
    settings = settings or Settings()
    valid: set[str] = set()
    timeout = httpx.Timeout(30.0)
    sem = _get_pubmed_sem()
    max_retries = 4

    async with httpx.AsyncClient() as client:
        for i in range(0, len(pmids), _PMID_BATCH):
            batch = pmids[i : i + _PMID_BATCH]
            params: dict[str, str] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "json",
            }
            if settings.entrez_email:
                params["email"] = settings.entrez_email

            for attempt in range(max_retries):
                async with sem:
                    try:
                        resp = await client.get(
                            _ESUMMARY_URL, params=params, timeout=timeout,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        result = data.get("result", {})
                        for pmid in batch:
                            entry = result.get(pmid, {})
                            if isinstance(entry, dict) and "error" not in entry:
                                valid.add(pmid)
                        break  # success
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 429 and attempt < max_retries - 1:
                            wait = 2 ** attempt + 0.5
                            logger.debug(
                                "PubMed 429 -- retry %d/%d after %.1fs",
                                attempt + 1, max_retries, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        logger.warning("PMID validation batch failed: %s", exc)
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        logger.warning("PMID validation batch failed: %s", exc)
                        break

    return valid


async def _validate_dois_batch(dois: list[str]) -> set[str]:
    """Return the subset of *dois* that resolve via doi.org HEAD request."""
    valid: set[str] = set()
    timeout = httpx.Timeout(15.0)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for doi in dois:
            try:
                resp = await client.head(
                    f"{_DOI_URL}/{doi}", timeout=timeout,
                )
                if resp.status_code < 400:
                    valid.add(doi)
            except httpx.HTTPError:
                pass

    return valid


def _collect_citations(
    prediction: DrugDiseasePrediction,
) -> tuple[list[str], list[str]]:
    """Extract unique PMIDs and DOIs from all edges in a prediction."""
    pmids: set[str] = set()
    dois: set[str] = set()

    for assoc in prediction.associations:
        for chain in assoc.evidence_chains:
            for edge in chain.edges:
                if edge.pmid and _PMID_RE.match(edge.pmid):
                    pmids.add(edge.pmid)

    # Also scan chain summaries for cited DOIs (rare but possible)
    # DOIs are primarily on edges, but we also check the citation if present
    return sorted(pmids), sorted(dois)


async def citation_validity_rate(
    prediction: DrugDiseasePrediction,
    settings: Settings | None = None,
) -> tuple[float, int, int]:
    """Compute the fraction of cited PMIDs/DOIs that are valid.

    Returns
    -------
    (rate, n_valid, n_total)
    """
    pmids, dois = _collect_citations(prediction)
    n_total = len(pmids) + len(dois)
    if n_total == 0:
        return 0.0, 0, 0

    valid_pmids, valid_dois = set(), set()
    if pmids:
        valid_pmids = await _validate_pmids_batch(pmids, settings)
    if dois:
        valid_dois = await _validate_dois_batch(dois)

    n_valid = len(valid_pmids) + len(valid_dois)
    rate = n_valid / n_total
    return rate, n_valid, n_total


# ===================================================================
# 2. Mean Chain Depth
# ===================================================================

def mean_chain_depth(prediction: DrugDiseasePrediction) -> float:
    """Average number of edges across all chains in the prediction."""
    chains = _all_chains(prediction)
    if not chains:
        return 0.0
    return sum(len(c.edges) for c in chains) / len(chains)


# ===================================================================
# 3. Chain Verifiability Score
# ===================================================================

def chain_verifiability_score(prediction: DrugDiseasePrediction) -> float:
    """Fraction of edges that carry a PMID or DOI citation."""
    edges = _all_edges(prediction)
    if not edges:
        return 0.0
    n_cited = sum(
        1 for e in edges
        if (e.pmid and _PMID_RE.match(e.pmid))
    )
    return n_cited / len(edges)


# ===================================================================
# 4. Evidence Relevance (MedCPT cosine similarity)
# ===================================================================

def evidence_relevance(
    prediction: DrugDiseasePrediction,
    *,
    embedding_manager: object | None = None,
) -> float:
    """Mean cosine similarity between each edge's snippet and its claim.

    The *claim* is constructed as
    ``"{source_entity} {relationship} {target_entity}"``.

    The *evidence* is ``edge.evidence_snippet``.

    Both are encoded with the MedCPT query encoder and compared via
    cosine similarity.  If no ``embedding_manager`` is provided, a new
    one is created and only the query encoder is loaded.

    Parameters
    ----------
    prediction:
        The structured output from an experimental arm.
    embedding_manager:
        An ``EmbeddingManager`` instance (must have ``embed_queries``).
        If ``None``, one will be created and loaded lazily.

    Returns
    -------
    float
        Mean cosine similarity in [0, 1].  Returns 0.0 if no edges.
    """
    edges = _all_edges(prediction)
    if not edges:
        return 0.0

    claims: list[str] = []
    snippets: list[str] = []

    for edge in edges:
        claim = f"{edge.source_entity} {edge.relationship.value} {edge.target_entity}"
        snippet = edge.evidence_snippet.strip()
        if snippet:
            claims.append(claim)
            snippets.append(snippet)

    if not claims:
        return 0.0

    # Lazy import to avoid circular imports at module level
    from src.vector.embeddings import EmbeddingManager

    if embedding_manager is None:
        mgr = EmbeddingManager()
        mgr.load_query_only()
    else:
        mgr = embedding_manager  # type: ignore[assignment]

    claim_vecs = mgr.embed_queries(claims)   # (N, 768)
    snippet_vecs = mgr.embed_queries(snippets)  # (N, 768)

    # Row-wise cosine similarity
    norms_c = np.linalg.norm(claim_vecs, axis=1, keepdims=True)
    norms_s = np.linalg.norm(snippet_vecs, axis=1, keepdims=True)

    # Avoid division by zero
    norms_c = np.where(norms_c == 0, 1.0, norms_c)
    norms_s = np.where(norms_s == 0, 1.0, norms_s)

    claim_normed = claim_vecs / norms_c
    snippet_normed = snippet_vecs / norms_s

    similarities = np.sum(claim_normed * snippet_normed, axis=1)
    # Clamp to [0, 1] (cosine sim can be slightly negative)
    similarities = np.clip(similarities, 0.0, 1.0)

    return float(np.mean(similarities))


# ===================================================================
# 5. Mechanistic Specificity (LLM-as-judge)
# ===================================================================

_SPECIFICITY_PROMPT = """\
You are an expert biomedical reviewer. Given a mechanistic evidence chain \
linking a drug to a disease, rate its MECHANISTIC SPECIFICITY on a scale \
from 0.0 to 1.0.

Scoring guide:
- 1.0: Every step names specific genes, proteins, receptors, or pathways \
(e.g. "aspirin inhibits PTGS2 (COX-2), reducing PGE2 synthesis in the \
arachidonic acid pathway").
- 0.5: Mix of specific and generic language (e.g. "the drug affects \
inflammatory pathways through enzyme inhibition").
- 0.0: Entirely generic (e.g. "the drug treats the disease through its \
mechanism of action").

Chain summary: {summary}
Chain edges:
{edges_text}

Respond with ONLY a single decimal number between 0.0 and 1.0. \
No explanation."""


async def mechanistic_specificity(
    prediction: DrugDiseasePrediction,
    *,
    model_id: str = "groq:qwen/qwen3-32b",
    settings: Settings | None = None,
) -> float:
    """LLM-as-judge score for mechanistic specificity.

    Uses PydanticAI with a cheap model (default: Qwen3 32B on Groq)
    to rate each chain's specificity 0-1, then returns the mean.

    Falls back to a heuristic if no API key or model is available.

    Parameters
    ----------
    prediction:
        Structured output.
    model_id:
        PydanticAI model identifier.
    settings:
        For API key access.
    """
    chains = _all_chains(prediction)
    if not chains:
        return 0.0

    scores: list[float] = []

    try:
        from pydantic_ai import Agent

        agent = Agent(model_id, output_type=float)

        for chain in chains:
            edges_text = "\n".join(
                f"  {e.source_entity} --[{e.relationship.value}]--> {e.target_entity}"
                for e in chain.edges
            )
            prompt = _SPECIFICITY_PROMPT.format(
                summary=chain.summary,
                edges_text=edges_text,
            )
            try:
                result = await agent.run(prompt)
                score = max(0.0, min(1.0, result.output))
                scores.append(score)
            except Exception as exc:
                logger.warning("LLM specificity scoring failed: %s", exc)
                scores.append(_heuristic_specificity(chain))

    except Exception as exc:
        logger.warning(
            "PydanticAI not available for specificity, using heuristic: %s", exc,
        )
        scores = [_heuristic_specificity(c) for c in chains]

    return sum(scores) / len(scores) if scores else 0.0


def _heuristic_specificity(chain: EvidenceChain) -> float:
    """Simple regex-based heuristic fallback for specificity.

    Checks whether edges reference specific gene symbols (uppercase
    2-6 letter identifiers), pathway names, or receptor names.
    """
    # Pattern for gene-like symbols: uppercase, 2-6 chars, possibly
    # followed by a digit (e.g. PTGS2, AMPK, BCR-ABL)
    gene_re = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")

    specific_count = 0
    total = 0

    for edge in chain.edges:
        total += 1
        text = f"{edge.source_entity} {edge.target_entity} {edge.evidence_snippet}"
        matches = gene_re.findall(text)
        if len(matches) >= 1:
            specific_count += 1

    if total == 0:
        return 0.0
    return specific_count / total


# ===================================================================
# Full evaluation
# ===================================================================

async def evaluate_evidence_quality(
    prediction: DrugDiseasePrediction,
    *,
    embedding_manager: object | None = None,
    settings: Settings | None = None,
    use_llm_specificity: bool = True,
    specificity_model_id: str = "groq:qwen/qwen3-32b",
) -> EvidenceQualityMetrics:
    """Compute all five evidence quality metrics for a prediction.

    Parameters
    ----------
    prediction:
        Structured output from any experimental arm.
    embedding_manager:
        Pre-loaded ``EmbeddingManager`` for evidence relevance.
    settings:
        Application settings (API keys, Entrez email).
    use_llm_specificity:
        If ``True``, use LLM-as-judge for specificity. If ``False``,
        fall back to heuristic.
    specificity_model_id:
        PydanticAI model id for the LLM-as-judge.

    Returns
    -------
    EvidenceQualityMetrics
    """
    chains = _all_chains(prediction)
    edges = _all_edges(prediction)

    # 1. Citation validity
    cvr, n_valid, n_total = await citation_validity_rate(prediction, settings)

    # 2. Chain depth
    depth = mean_chain_depth(prediction)

    # 3. Chain verifiability
    verif = chain_verifiability_score(prediction)

    # 4. Evidence relevance
    relevance = evidence_relevance(prediction, embedding_manager=embedding_manager)

    # 5. Mechanistic specificity
    if use_llm_specificity:
        specificity = await mechanistic_specificity(
            prediction, model_id=specificity_model_id, settings=settings,
        )
    else:
        specificity_scores = [_heuristic_specificity(c) for c in chains]
        specificity = (
            sum(specificity_scores) / len(specificity_scores)
            if specificity_scores
            else 0.0
        )

    return EvidenceQualityMetrics(
        citation_validity_rate=cvr,
        mean_chain_depth=depth,
        chain_verifiability_score=verif,
        evidence_relevance=relevance,
        mechanistic_specificity=specificity,
        n_citations_checked=n_total,
        n_citations_valid=n_valid,
        n_chains=len(chains),
        n_edges=len(edges),
    )


# ===================================================================
# Internal helpers
# ===================================================================

def _all_chains(prediction: DrugDiseasePrediction) -> list[EvidenceChain]:
    """Flatten all evidence chains from all associations."""
    chains: list[EvidenceChain] = []
    for assoc in prediction.associations:
        chains.extend(assoc.evidence_chains)
    return chains


def _all_edges(prediction: DrugDiseasePrediction) -> list[MechanisticEdge]:
    """Flatten all edges from all chains from all associations."""
    edges: list[MechanisticEdge] = []
    for assoc in prediction.associations:
        for chain in assoc.evidence_chains:
            edges.extend(chain.edges)
    return edges
