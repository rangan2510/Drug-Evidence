"""Embedding manager -- dense MedCPT + sparse SPLADE.

Dense embeddings use the asymmetric MedCPT encoder pair from NCBI:
  - ``ncbi/MedCPT-Query-Encoder``  for queries  (768-d, max_length=64)
  - ``ncbi/MedCPT-Article-Encoder`` for documents (768-d, max_length=512)

Sparse embeddings use SPLADE via ``sentence-transformers`` SparseEncoder:
  - ``naver/splade-cocondenser-ensembledistil`` (default; Apache-2.0)
  - Learned sparse representations with automatic term expansion
  - Pure Python/PyTorch -- no fastembed or Rust stemmers required
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SparseEncoder
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sparse embedding result container
# ---------------------------------------------------------------------------

@dataclass
class SparseVector:
    """Indices + values pair for a sparse embedding."""

    indices: NDArray[np.int64]
    values: NDArray[np.float64]


def _sparse_tensor_to_vectors(tensor: torch.Tensor) -> list[SparseVector]:
    """Convert a SPLADE output tensor to a list of SparseVector.

    ``SparseEncoder.encode()`` returns a sparse COO tensor of shape
    ``(batch, vocab_size)`` when ``convert_to_sparse_tensor=True``.
    We coalesce it, then extract per-row indices and values directly
    from the COO representation -- no ``nonzero()`` call needed.
    """
    results: list[SparseVector] = []

    if tensor.is_sparse:
        t = tensor.coalesce().cpu()
        idx = t.indices()      # shape (2, nnz): [row_indices, col_indices]
        vals = t.values()      # shape (nnz,)
        batch_size = t.shape[0]
        for row_idx in range(batch_size):
            mask = idx[0] == row_idx
            col_ids = idx[1][mask]
            row_vals = vals[mask]
            results.append(
                SparseVector(
                    indices=col_ids.numpy().astype(np.int64),
                    values=row_vals.numpy().astype(np.float64),
                )
            )
    else:
        # Dense fallback (e.g. convert_to_sparse_tensor=False)
        t = tensor.cpu()
        for row in t:
            nz = row.nonzero(as_tuple=True)[0]
            results.append(
                SparseVector(
                    indices=nz.numpy().astype(np.int64),
                    values=row[nz].numpy().astype(np.float64),
                )
            )
    return results


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingManager:
    """Wraps MedCPT (dense, 768-d) and SPLADE (sparse) encoders.

    Parameters
    ----------
    query_model_name:
        HuggingFace model id for the MedCPT query encoder.
    doc_model_name:
        HuggingFace model id for the MedCPT article/document encoder.
    sparse_model_name:
        HuggingFace model id for the SPLADE sparse encoder.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
    max_query_length:
        Maximum token length for queries (MedCPT default is 64).
    max_doc_length:
        Maximum token length for documents.
    """

    query_model_name: str = "ncbi/MedCPT-Query-Encoder"
    doc_model_name: str = "ncbi/MedCPT-Article-Encoder"
    sparse_model_name: str = "naver/splade-cocondenser-ensembledistil"
    device: str = "cpu"
    max_query_length: int = 64
    max_doc_length: int = 512

    # Private -- populated by ``load()``
    _query_tokenizer: AutoTokenizer | None = field(default=None, init=False, repr=False)
    _query_model: AutoModel | None = field(default=None, init=False, repr=False)
    _article_tokenizer: AutoTokenizer | None = field(default=None, init=False, repr=False)
    _article_model: AutoModel | None = field(default=None, init=False, repr=False)
    _sparse_encoder: SparseEncoder | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingManager:
        """Create an ``EmbeddingManager`` from application settings."""
        return cls(
            query_model_name=getattr(settings, "dense_model_query", "ncbi/MedCPT-Query-Encoder"),
            doc_model_name=getattr(settings, "dense_model_doc", "ncbi/MedCPT-Article-Encoder"),
            sparse_model_name=getattr(settings, "sparse_model", "naver/splade-cocondenser-ensembledistil"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download / load all encoder models into memory."""
        self._load_dense_query()
        self._load_dense_article()
        self._load_sparse()

    def load_query_only(self) -> None:
        """Load only the query-side models (dense query + sparse)."""
        self._load_dense_query()
        self._load_sparse()

    def _load_dense_query(self) -> None:
        if self._query_model is not None:
            return
        logger.info("Loading MedCPT query encoder: %s", self.query_model_name)
        self._query_tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
        self._query_model = AutoModel.from_pretrained(self.query_model_name).to(self.device)
        self._query_model.eval()

    def _load_dense_article(self) -> None:
        if self._article_model is not None:
            return
        logger.info("Loading MedCPT article encoder: %s", self.doc_model_name)
        self._article_tokenizer = AutoTokenizer.from_pretrained(self.doc_model_name)
        self._article_model = AutoModel.from_pretrained(self.doc_model_name).to(self.device)
        self._article_model.eval()

    def _load_sparse(self) -> None:
        if self._sparse_encoder is not None:
            return
        logger.info("Loading SPLADE sparse encoder: %s", self.sparse_model_name)
        self._sparse_encoder = SparseEncoder(
            self.sparse_model_name,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Dense embeddings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_queries(self, queries: list[str]) -> NDArray[np.float32]:
        """Encode search queries with MedCPT-Query-Encoder.

        Returns array of shape ``(len(queries), 768)``.
        """
        self._load_dense_query()
        assert self._query_tokenizer is not None
        assert self._query_model is not None

        encoded = self._query_tokenizer(
            queries,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_query_length,
        ).to(self.device)
        embeds = self._query_model(**encoded).last_hidden_state[:, 0, :]
        return embeds.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_documents(self, documents: list[str]) -> NDArray[np.float32]:
        """Encode documents with MedCPT-Article-Encoder.

        Returns array of shape ``(len(documents), 768)``.
        """
        self._load_dense_article()
        assert self._article_tokenizer is not None
        assert self._article_model is not None

        encoded = self._article_tokenizer(
            documents,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_doc_length,
        ).to(self.device)
        embeds = self._article_model(**encoded).last_hidden_state[:, 0, :]
        return embeds.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Sparse embeddings (SPLADE via sentence-transformers)
    # ------------------------------------------------------------------

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts with SPLADE sparse encoder.

        Returns a list of ``SparseVector`` objects (one per text).
        """
        self._load_sparse()
        assert self._sparse_encoder is not None

        tensor = self._sparse_encoder.encode(texts, convert_to_tensor=True)
        return _sparse_tensor_to_vectors(tensor)

    def embed_queries_sparse(self, queries: list[str]) -> list[SparseVector]:
        """Encode queries with SPLADE (uses encode_query for prompt handling)."""
        self._load_sparse()
        assert self._sparse_encoder is not None

        tensor = self._sparse_encoder.encode_query(queries, convert_to_tensor=True)
        return _sparse_tensor_to_vectors(tensor)

    def embed_documents_sparse(
        self,
        documents: list[str],
        batch_size: int = 16,
    ) -> list[SparseVector]:
        """Encode documents with SPLADE (uses encode_document for prompt handling).

        Uses sub-batching to avoid long stalls on CPU.
        """
        self._load_sparse()
        assert self._sparse_encoder is not None

        all_vecs: list[SparseVector] = []
        for i in range(0, len(documents), batch_size):
            sub = documents[i : i + batch_size]
            logger.info(
                "SPLADE encoding docs %d-%d / %d",
                i + 1,
                min(i + len(sub), len(documents)),
                len(documents),
            )
            tensor = self._sparse_encoder.encode_document(sub, convert_to_tensor=True)
            all_vecs.extend(_sparse_tensor_to_vectors(tensor))
        return all_vecs

    # ------------------------------------------------------------------
    # Convenience -- both dense + sparse at once
    # ------------------------------------------------------------------

    def embed_documents_hybrid(
        self,
        documents: list[str],
    ) -> tuple[NDArray[np.float32], list[SparseVector]]:
        """Return ``(dense_array, sparse_list)`` for a batch of documents."""
        dense = self.embed_documents(documents)
        sparse = self.embed_documents_sparse(documents)
        return dense, sparse

    def embed_query_hybrid(
        self,
        query: str,
    ) -> tuple[NDArray[np.float32], SparseVector]:
        """Return ``(dense_vector, sparse_vector)`` for a single query."""
        dense = self.embed_queries([query])
        sparse = self.embed_queries_sparse([query])
        return dense[0], sparse[0]
