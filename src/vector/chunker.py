"""Evidence document chunker with rich metadata.

Splits ``EvidenceDocument`` texts into token-sized chunks suitable for
indexing in the hybrid vector store.  Each chunk carries the full
provenance metadata as a Qdrant-filterable payload.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from src.schemas.evidence import EvidenceDocument

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A chunk of evidence text with metadata payload.

    The ``payload`` dict is stored verbatim as the Qdrant point payload,
    so every key is filterable via Qdrant conditions.
    """

    chunk_id: str
    text: str
    payload: dict = field(default_factory=dict)


def _deterministic_id(text: str, doc_index: int, chunk_index: int) -> str:
    """SHA-256 hex digest truncated to 32 chars for a stable point id."""
    raw = f"{doc_index}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* on whitespace into chunks of roughly *chunk_size* words.

    Uses a simple word-level sliding window.  For biomedical text the
    word count is a reasonable proxy for token count (MedCPT tokeniser
    averages ~1.3 tokens per whitespace word on PubMed abstracts).

    Returns at least one chunk even if the text is shorter than
    *chunk_size*.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_evidence(
    documents: list[EvidenceDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[EvidenceChunk]:
    """Split a list of evidence documents into indexable chunks.

    Parameters
    ----------
    documents:
        Evidence records from the aggregator.
    chunk_size:
        Target chunk size in whitespace-separated words.
    chunk_overlap:
        Overlap between consecutive chunks (in words).

    Returns
    -------
    list[EvidenceChunk]
        Chunks ready for embedding and indexing.
    """
    chunks: list[EvidenceChunk] = []

    for doc_idx, doc in enumerate(documents):
        text = doc.text.strip()
        if not text:
            continue

        parts = _split_text(text, chunk_size, chunk_overlap)

        for chunk_idx, part in enumerate(parts):
            chunk_id = _deterministic_id(part, doc_idx, chunk_idx)

            payload: dict = {
                "source": doc.source.value,
                "evidence_type": doc.evidence_type.value,
                "drug_name": doc.drug_name,
                "drug_chembl_id": doc.drug_chembl_id or "",
                "target_symbol": doc.target_symbol or "",
                "target_ensembl_id": doc.target_ensembl_id or "",
                "disease_name": doc.disease_name or "",
                "disease_id": doc.disease_id or "",
                "score": doc.score if doc.score is not None else 0.0,
                "pmid": doc.citation.pmid or "",
                "doi": doc.citation.doi or "",
                "year": doc.citation.year or 0,
                "chunk_index": chunk_idx,
                "total_chunks": len(parts),
                "text": part,
            }

            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    text=part,
                    payload=payload,
                )
            )

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
