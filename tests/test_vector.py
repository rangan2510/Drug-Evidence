"""Tests for hybrid vector search (embeddings, chunker, store).

Uses Qdrant in-memory mode so no external services are needed.
MedCPT and SPLADE model downloads are required on first run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)
from src.vector.chunker import EvidenceChunk, chunk_evidence
from src.vector.embeddings import EmbeddingManager, SparseVector
from src.vector.store import HybridVectorStore, SearchResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_manager() -> EmbeddingManager:
    """Module-scoped embedding manager (models loaded once)."""
    mgr = EmbeddingManager()
    mgr.load()
    return mgr


@pytest.fixture()
def sample_documents() -> list[EvidenceDocument]:
    """A small set of evidence documents for testing."""
    return [
        EvidenceDocument(
            text=(
                "Aspirin irreversibly inhibits cyclooxygenase-1 (COX-1) and "
                "cyclooxygenase-2 (COX-2) enzymes, blocking prostaglandin "
                "synthesis. This mechanism underlies its analgesic, "
                "antipyretic, and anti-inflammatory effects."
            ),
            source=EvidenceSource.PUBMED,
            evidence_type=EvidenceType.MECHANISM_OF_ACTION,
            drug_name="aspirin",
            drug_chembl_id="CHEMBL25",
            target_symbol="PTGS1",
            citation=Citation(pmid="12345678", title="Aspirin and COX"),
        ),
        EvidenceDocument(
            text=(
                "Metformin activates AMP-activated protein kinase (AMPK) "
                "in hepatocytes, leading to reduced hepatic glucose "
                "production. It is the first-line treatment for type 2 "
                "diabetes mellitus."
            ),
            source=EvidenceSource.OPENTARGETS,
            evidence_type=EvidenceType.PHARMACOLOGICAL_ACTION,
            drug_name="metformin",
            drug_chembl_id="CHEMBL1431",
            target_symbol="PRKAA1",
            disease_name="type 2 diabetes mellitus",
        ),
        EvidenceDocument(
            text=(
                "Imatinib selectively inhibits BCR-ABL tyrosine kinase, "
                "the constitutively active fusion protein in chronic "
                "myeloid leukemia (CML). It also inhibits KIT and PDGFRA."
            ),
            source=EvidenceSource.CHEMBL,
            evidence_type=EvidenceType.BINDING_ASSAY,
            drug_name="imatinib",
            drug_chembl_id="CHEMBL941",
            target_symbol="ABL1",
            disease_name="chronic myeloid leukemia",
        ),
    ]


# ===================================================================
# Chunker tests
# ===================================================================

class TestChunker:
    def test_single_short_doc_produces_one_chunk(self, sample_documents: list[EvidenceDocument]) -> None:
        chunks = chunk_evidence([sample_documents[0]], chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == sample_documents[0].text.strip()

    def test_payload_fields(self, sample_documents: list[EvidenceDocument]) -> None:
        chunks = chunk_evidence([sample_documents[0]], chunk_size=512)
        p = chunks[0].payload
        assert p["source"] == "pubmed"
        assert p["evidence_type"] == "mechanism_of_action"
        assert p["drug_name"] == "aspirin"
        assert p["drug_chembl_id"] == "CHEMBL25"
        assert p["target_symbol"] == "PTGS1"
        assert p["pmid"] == "12345678"
        assert p["chunk_index"] == 0
        assert p["total_chunks"] == 1

    def test_long_text_splits(self) -> None:
        long_text = " ".join(["word"] * 200)
        doc = EvidenceDocument(
            text=long_text,
            source=EvidenceSource.PUBMED,
            evidence_type=EvidenceType.LITERATURE,
            drug_name="testdrug",
        )
        chunks = chunk_evidence([doc], chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 2
        # Last chunk payload records correct total
        assert chunks[-1].payload["total_chunks"] == len(chunks)

    def test_multiple_documents(self, sample_documents: list[EvidenceDocument]) -> None:
        chunks = chunk_evidence(sample_documents, chunk_size=512)
        assert len(chunks) == 3  # each doc is short => 1 chunk each
        drugs = {c.payload["drug_name"] for c in chunks}
        assert drugs == {"aspirin", "metformin", "imatinib"}

    def test_empty_text_skipped(self) -> None:
        doc = EvidenceDocument(
            text="   ",
            source=EvidenceSource.DGIDB,
            evidence_type=EvidenceType.DRUG_GENE_INTERACTION,
            drug_name="empty",
        )
        chunks = chunk_evidence([doc])
        assert len(chunks) == 0

    def test_deterministic_ids(self, sample_documents: list[EvidenceDocument]) -> None:
        chunks_a = chunk_evidence(sample_documents)
        chunks_b = chunk_evidence(sample_documents)
        ids_a = [c.chunk_id for c in chunks_a]
        ids_b = [c.chunk_id for c in chunks_b]
        assert ids_a == ids_b


# ===================================================================
# Embedding tests
# ===================================================================

class TestEmbeddingManager:
    def test_dense_query_shape(self, embedding_manager: EmbeddingManager) -> None:
        vecs = embedding_manager.embed_queries(["aspirin mechanism of action"])
        assert vecs.shape == (1, 768)
        assert vecs.dtype == np.float32

    def test_dense_query_batch(self, embedding_manager: EmbeddingManager) -> None:
        queries = ["aspirin", "metformin", "imatinib"]
        vecs = embedding_manager.embed_queries(queries)
        assert vecs.shape == (3, 768)

    def test_dense_document_shape(self, embedding_manager: EmbeddingManager) -> None:
        vecs = embedding_manager.embed_documents(["Aspirin inhibits COX enzymes."])
        assert vecs.shape == (1, 768)
        assert vecs.dtype == np.float32

    def test_sparse_returns_sparse_vectors(self, embedding_manager: EmbeddingManager) -> None:
        results = embedding_manager.embed_sparse(["aspirin mechanism of action"])
        assert len(results) == 1
        sv = results[0]
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == len(sv.values)
        assert len(sv.indices) > 0

    def test_hybrid_query(self, embedding_manager: EmbeddingManager) -> None:
        dense, sparse = embedding_manager.embed_query_hybrid("aspirin COX inhibitor")
        assert dense.shape == (768,)
        assert isinstance(sparse, SparseVector)
        assert len(sparse.indices) > 0

    def test_hybrid_documents(self, embedding_manager: EmbeddingManager) -> None:
        texts = ["Aspirin blocks COX.", "Metformin activates AMPK."]
        dense, sparse = embedding_manager.embed_documents_hybrid(texts)
        assert dense.shape == (2, 768)
        assert len(sparse) == 2

    def test_dense_vectors_normalized_range(self, embedding_manager: EmbeddingManager) -> None:
        vecs = embedding_manager.embed_queries(["test query"])
        norms = np.linalg.norm(vecs, axis=1)
        # MedCPT embeddings should have reasonable norms (not zero, not huge)
        assert all(0.1 < n < 100.0 for n in norms)


# ===================================================================
# HybridVectorStore tests (Qdrant in-memory)
# ===================================================================

class TestHybridVectorStore:
    @pytest.fixture()
    def store(self, embedding_manager: EmbeddingManager) -> HybridVectorStore:
        """Fresh in-memory Qdrant store per test."""
        from qdrant_client import QdrantClient

        client = QdrantClient(":memory:")
        store = HybridVectorStore(
            client=client,
            embeddings=embedding_manager,
            collection_name="test_evidence",
        )
        return store

    def test_ensure_collection_creates(self, store: HybridVectorStore) -> None:
        store.ensure_collection()
        info = store.collection_info()
        assert info["name"] == "test_evidence"
        assert info["points_count"] == 0

    def test_ensure_collection_idempotent(self, store: HybridVectorStore) -> None:
        store.ensure_collection()
        store.ensure_collection()  # should not raise
        info = store.collection_info()
        assert info["points_count"] == 0

    def test_index_chunks(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        n = store.index_chunks(chunks)
        assert n == 3
        assert store.count() == 3

    def test_hybrid_search_returns_results(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        results = store.hybrid_search("aspirin COX inhibitor", limit=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # The aspirin doc should rank highest
        assert "aspirin" in results[0].text.lower() or "cox" in results[0].text.lower()

    def test_drug_filter(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        results = store.hybrid_search("enzyme inhibitor", drug_filter="imatinib", limit=10)
        assert len(results) > 0
        for r in results:
            assert r.payload["drug_name"] == "imatinib"

    def test_source_filter(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        results = store.hybrid_search("drug mechanism", source_filter="pubmed", limit=10)
        for r in results:
            assert r.payload["source"] == "pubmed"

    def test_count_with_drug_filter(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        assert store.count(drug_filter="aspirin") == 1
        assert store.count(drug_filter="metformin") == 1
        assert store.count(drug_filter="nonexistent") == 0

    def test_scroll_all(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        payloads = store.scroll_all(limit=100)
        assert len(payloads) == 3
        drugs = {p["drug_name"] for p in payloads}
        assert drugs == {"aspirin", "metformin", "imatinib"}

    def test_delete_collection(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)
        assert store.count() == 3

        store.delete_collection()
        # After deletion, the collection should not exist
        assert not store._client.collection_exists("test_evidence")

    def test_search_result_has_score(
        self,
        store: HybridVectorStore,
        sample_documents: list[EvidenceDocument],
    ) -> None:
        chunks = chunk_evidence(sample_documents)
        store.index_chunks(chunks)

        results = store.hybrid_search("diabetes treatment metformin", limit=3)
        assert len(results) > 0
        for r in results:
            assert isinstance(r.score, float)
            assert r.score > 0
