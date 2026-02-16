"""Qdrant hybrid vector store -- dense MedCPT + sparse SPLADE + RRF fusion.

Collection layout
-----------------
* Named vector  ``"dense"``   -- ``VectorParams(size=768, distance=Cosine)``
* Sparse vector ``"splade"``  -- ``SparseVectorParams(modifier=Modifier.IDF)``

Query strategy
--------------
1. ``Prefetch`` dense top-K  (e.g. 50)
2. ``Prefetch`` sparse top-K (e.g. 50)
3. ``FusionQuery(fusion=Fusion.RRF)`` merges and re-ranks

Key improvement over v1:
  - v1: dense-only MedCPT, wiped collection per drug.
  - v2: hybrid dense+sparse, persistent across the experiment run,
    metadata filtering by drug / source / evidence_type.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from src.vector.chunker import EvidenceChunk
from src.vector.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result from hybrid search."""

    chunk_id: str
    score: float
    text: str
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HybridVectorStore
# ---------------------------------------------------------------------------


class HybridVectorStore:
    """Qdrant-backed hybrid vector store with dense + sparse + RRF fusion.

    Parameters
    ----------
    client:
        A ``QdrantClient`` instance (remote or ``:memory:``).
    embeddings:
        An ``EmbeddingManager`` with loaded models.
    collection_name:
        Name of the Qdrant collection to use.
    dense_size:
        Dimensionality of the dense vector (768 for MedCPT).
    rrf_k:
        Reciprocal Rank Fusion constant (default 60).
    prefetch_limit:
        Number of candidates to prefetch from each index before fusion.
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: EmbeddingManager,
        collection_name: str = "evidence_v2",
        dense_size: int = 768,
        rrf_k: int = 60,
        prefetch_limit: int = 50,
    ) -> None:
        self._client = client
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._dense_size = dense_size
        self._rrf_k = rrf_k
        self._prefetch_limit = prefetch_limit

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        embeddings: EmbeddingManager,
        *,
        in_memory: bool = False,
    ) -> HybridVectorStore:
        """Create a store from application ``Settings``.

        Client resolution order (cascade):
        1. ``in_memory=True`` or ``settings.qdrant_in_memory`` -- use ``:memory:``
        2. Try localhost Qdrant at ``settings.qdrant_url`` (healthcheck)
        3. Fall back to **on-disk** Qdrant at ``settings.qdrant_on_disk_path``
        4. If all else fails and ``qdrant_fallback_to_in_memory`` is True,
           use ``:memory:`` as a last resort.

        Parameters
        ----------
        settings:
            ``src.config.settings.Settings`` instance.
        embeddings:
            A loaded ``EmbeddingManager``.
        in_memory:
            If *True*, use an in-memory Qdrant client (useful for tests).
        """
        use_in_memory = in_memory or bool(getattr(settings, "qdrant_in_memory", False))
        client: QdrantClient | None = None

        if use_in_memory:
            client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client (explicit).")
        else:
            # Try localhost first
            qdrant_url = str(
                getattr(settings, "qdrant_url", "http://localhost:6333")
            ).rstrip("/")
            health_path = str(getattr(settings, "qdrant_healthcheck_path", "/healthz"))
            health_url = f"{qdrant_url}{health_path}"
            try:
                resp = httpx.get(health_url, timeout=httpx.Timeout(2.0))
                if resp.status_code < 400:
                    client = QdrantClient(url=settings.qdrant_url)
                    logger.info("Connected to Qdrant at %s.", settings.qdrant_url)
                else:
                    logger.warning(
                        "Qdrant healthcheck returned %s (%s).",
                        resp.status_code,
                        health_url,
                    )
            except httpx.HTTPError:
                logger.warning(
                    "Qdrant healthcheck failed (%s).",
                    health_url,
                )

            # Fall back to on-disk persistent storage
            if client is None:
                on_disk_path = str(
                    getattr(settings, "qdrant_on_disk_path", ".qdrant_data")
                )
                if on_disk_path:
                    import os

                    os.makedirs(on_disk_path, exist_ok=True)
                    client = QdrantClient(path=on_disk_path)
                    logger.info(
                        "Using on-disk Qdrant at '%s' (persistent).",
                        on_disk_path,
                    )

            # Last resort: in-memory fallback
            if client is None and bool(
                getattr(settings, "qdrant_fallback_to_in_memory", True)
            ):
                client = QdrantClient(":memory:")
                logger.warning(
                    "All Qdrant backends unavailable; using in-memory fallback."
                )

            if client is None:
                msg = "No Qdrant backend available and in-memory fallback is disabled."
                raise RuntimeError(msg)

        store = cls(
            client=client,
            embeddings=embeddings,
            collection_name=settings.qdrant_collection,
            dense_size=settings.dense_vector_size,
            rrf_k=settings.rrf_k,
        )
        return store

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the hybrid collection if it does not already exist."""
        if self._client.collection_exists(self._collection_name):
            logger.info("Collection '%s' already exists.", self._collection_name)
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "splade": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        logger.info(
            "Created collection '%s' (dense=%d-d Cosine, sparse=SPLADE+IDF).",
            self._collection_name,
            self._dense_size,
        )

    def delete_collection(self) -> None:
        """Delete the collection (irreversible)."""
        self._client.delete_collection(self._collection_name)
        logger.info("Deleted collection '%s'.", self._collection_name)

    def collection_info(self) -> dict:
        """Return basic collection stats."""
        info = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        chunks: list[EvidenceChunk],
        batch_size: int = 32,
    ) -> int:
        """Embed and upsert evidence chunks into the collection.

        Returns the number of points upserted.
        """
        self.ensure_collection()
        total = 0
        n_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(
            "Indexing %d chunks in %d batches (batch_size=%d)",
            len(chunks),
            n_batches,
            batch_size,
        )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            texts = [c.text for c in batch]
            logger.info(
                "  Batch %d/%d: dense encoding %d texts ...",
                batch_num,
                n_batches,
                len(texts),
            )

            # Compute embeddings
            dense_vecs = self._embeddings.embed_documents(texts)
            logger.info(
                "  Batch %d/%d: sparse (SPLADE) encoding ...",
                batch_num,
                n_batches,
            )
            sparse_vecs = self._embeddings.embed_documents_sparse(texts)

            # Build Qdrant points
            points = []
            for j, chunk in enumerate(batch):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

                sv = sparse_vecs[j]
                sparse_vector = models.SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                )

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vecs[j].tolist(),
                        "splade": sparse_vector,
                    },
                    payload=chunk.payload,
                )
                points.append(point)

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            total += len(points)
            logger.debug(
                "Upserted batch %d-%d (%d points).", i, i + len(batch), len(points)
            )

        logger.info("Indexed %d chunks into '%s'.", total, self._collection_name)
        return total

    def upsert_points(
        self,
        points: list[models.PointStruct],
        batch_size: int = 256,
    ) -> int:
        """Upsert precomputed Qdrant points into the collection.

        This is used by Phase 0 flows that precompute vectors/payloads,
        persist them as point objects, and later bulk-load into Qdrant.
        """
        if not points:
            return 0

        self.ensure_collection()
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )
            total += len(batch)
        logger.info(
            "Upserted %d precomputed points into '%s'.",
            total,
            self._collection_name,
        )
        return total

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 10,
        drug_filter: str | None = None,
        source_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Run hybrid dense+sparse search with RRF fusion.

        Parameters
        ----------
        query:
            Natural-language search query.
        limit:
            Maximum number of results to return.
        drug_filter:
            If given, restrict to chunks where ``payload.drug_name == drug_filter``.
        source_filter:
            If given, restrict to chunks where ``payload.source == source_filter``.
        score_threshold:
            Minimum fusion score to include a result.
        """
        # Encode the query with both encoders
        dense_vec, sparse_vec = self._embeddings.embed_query_hybrid(query)

        # Build optional Qdrant filter
        conditions: list[models.Condition] = []
        if drug_filter:
            conditions.append(
                models.FieldCondition(
                    key="drug_name",
                    match=models.MatchValue(value=drug_filter),
                )
            )
        if source_filter:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source_filter),
                )
            )
        qdrant_filter = models.Filter(must=conditions) if conditions else None

        # Prefetch from each index, then fuse with RRF
        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec.tolist(),
                    using="dense",
                    limit=self._prefetch_limit,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using="splade",
                    limit=self._prefetch_limit,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            SearchResult(
                chunk_id=str(pt.id),
                score=pt.score,
                text=pt.payload.get("text", "") if pt.payload else "",
                payload=pt.payload or {},
            )
            for pt in results.points
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self, drug_filter: str | None = None) -> int:
        """Return the number of indexed points, optionally filtered by drug."""
        if drug_filter:
            result = self._client.count(
                collection_name=self._collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="drug_name",
                            match=models.MatchValue(value=drug_filter),
                        )
                    ]
                ),
            )
        else:
            result = self._client.count(collection_name=self._collection_name)
        return result.count

    def scroll_all(
        self,
        limit: int = 100,
        drug_filter: str | None = None,
    ) -> list[dict]:
        """Scroll through all points, returning payloads."""
        qdrant_filter = None
        if drug_filter:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="drug_name",
                        match=models.MatchValue(value=drug_filter),
                    )
                ]
            )

        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return [p.payload for p in points if p.payload]
