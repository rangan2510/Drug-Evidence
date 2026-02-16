"""MongoDB evidence store -- stage, query, and retrieve evidence documents.

Replaces the live-API-per-arm pattern with a pre-populated MongoDB
collection.  Evidence is staged once (via ``scripts/stage_evidence.py``)
and queried during arm execution through agent tools.

Collections
-----------
``evidence``
    One document per ``EvidenceDocument``.  Indexed on
    ``drug_name``, ``source``, ``target_symbol``, ``disease_name``,
    and a text index on ``text`` for $text queries.

``drugs``
    One document per normalised drug with staging metadata
    (chembl_id, pubchem_cid, n_docs, staged_at, sources_status).

Uses pymongo 4.x native async via ``pymongo.asynchronous``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pymongo import AsyncMongoClient, TEXT
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from src.config.settings import Settings
from src.schemas.evidence import EvidenceDocument

logger = logging.getLogger(__name__)

_EVIDENCE_COLLECTION = "evidence"
_DRUGS_COLLECTION = "drugs"


class EvidenceStore:
    """Async MongoDB client for staged evidence storage and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: AsyncMongoClient | None = None
        self._db: AsyncDatabase | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the MongoDB connection and ensure indexes exist."""
        uri = self._settings.mongo_uri
        db_name = self._settings.mongo_db_name
        logger.info("Connecting to MongoDB: %s / %s", uri, db_name)

        self._client = AsyncMongoClient(uri)
        self._db = self._client[db_name]
        await self._ensure_indexes()

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._db = None

    @property
    def db(self) -> AsyncDatabase:
        assert self._db is not None, "Call connect() first"
        return self._db

    @property
    def evidence(self) -> AsyncCollection:
        return self.db[_EVIDENCE_COLLECTION]

    @property
    def drugs(self) -> AsyncCollection:
        return self.db[_DRUGS_COLLECTION]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def _ensure_indexes(self) -> None:
        """Create indexes for efficient querying."""
        ev = self.evidence

        # Compound index for drug-scoped queries (most common pattern)
        await ev.create_index(
            [("drug_name_lower", 1), ("source", 1)],
            name="drug_source",
        )
        await ev.create_index(
            [("drug_name_lower", 1), ("target_symbol_lower", 1)],
            name="drug_target",
        )
        await ev.create_index(
            [("drug_name_lower", 1), ("disease_name_lower", 1)],
            name="drug_disease",
        )
        await ev.create_index(
            [("drug_name_lower", 1), ("citation.pmid", 1)],
            name="drug_pmid",
        )
        # Text index for full-text search within a drug's evidence
        await ev.create_index(
            [("text", TEXT)],
            name="text_search",
            default_language="english",
        )

        # Drugs collection
        await self.drugs.create_index("drug_name_lower", unique=True, name="drug_pk")

        logger.info("MongoDB indexes ensured.")

    # ------------------------------------------------------------------
    # Staging: write evidence
    # ------------------------------------------------------------------

    async def stage_drug(
        self,
        drug_name: str,
        chembl_id: str | None,
        pubchem_cid: int | None,
        docs: list[EvidenceDocument],
        sources_status: dict[str, str] | None = None,
    ) -> int:
        """Store all evidence documents for a single drug.

        Replaces any previously staged evidence for this drug (idempotent).

        Parameters
        ----------
        drug_name:
            Canonical drug name.
        chembl_id / pubchem_cid:
            Resolved identifiers.
        docs:
            Evidence documents from all 6 sources.
        sources_status:
            Per-source status dict, e.g. ``{"chembl": "ok", "dgidb": "error"}``.

        Returns
        -------
        int
            Number of documents inserted.
        """
        drug_lower = drug_name.lower()

        # Remove old evidence for this drug
        delete_result = await self.evidence.delete_many({"drug_name_lower": drug_lower})
        if delete_result.deleted_count > 0:
            logger.info(
                "Cleared %d old docs for '%s'",
                delete_result.deleted_count,
                drug_name,
            )

        # Insert new evidence
        if docs:
            records = [self._doc_to_record(d) for d in docs]
            result = await self.evidence.insert_many(records)
            n_inserted = len(result.inserted_ids)
        else:
            n_inserted = 0

        # Upsert drug metadata
        await self.drugs.update_one(
            {"drug_name_lower": drug_lower},
            {
                "$set": {
                    "drug_name": drug_name,
                    "drug_name_lower": drug_lower,
                    "chembl_id": chembl_id,
                    "pubchem_cid": pubchem_cid,
                    "n_docs": n_inserted,
                    "staged_at": datetime.now(UTC),
                    "sources_status": sources_status or {},
                },
            },
            upsert=True,
        )

        logger.info(
            "Staged %d docs for '%s' (chembl=%s, cid=%s)",
            n_inserted,
            drug_name,
            chembl_id,
            pubchem_cid,
        )
        return n_inserted

    # ------------------------------------------------------------------
    # Querying: read evidence
    # ------------------------------------------------------------------

    async def get_drug_meta(self, drug_name: str) -> dict | None:
        """Return staging metadata for a drug, or None if not staged."""
        return await self.drugs.find_one(
            {"drug_name_lower": drug_name.lower()},
            {"_id": 0},
        )

    async def is_staged(self, drug_name: str) -> bool:
        """Check if evidence for a drug is already staged."""
        meta = await self.get_drug_meta(drug_name)
        return meta is not None and meta.get("n_docs", 0) > 0

    async def get_all_evidence(
        self,
        drug_name: str,
        limit: int = 500,
    ) -> list[dict]:
        """Return all evidence documents for a drug."""
        cursor = self.evidence.find(
            {"drug_name_lower": drug_name.lower()},
            {"_id": 0},
        ).limit(limit)
        return await cursor.to_list()

    async def search_text(
        self,
        drug_name: str,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """Full-text search within a drug's evidence using MongoDB $text.

        Parameters
        ----------
        drug_name:
            Scope search to this drug's evidence only.
        query:
            Natural-language query (MongoDB text index tokenizes and stems).
        limit:
            Max results to return.

        Returns
        -------
        list[dict]
            Matching evidence records with ``text_score`` for ranking.
        """
        cursor = (
            self.evidence.find(
                {
                    "drug_name_lower": drug_name.lower(),
                    "$text": {"$search": query},
                },
                {"_id": 0, "text_score": {"$meta": "textScore"}},
            )
            .sort([("text_score", {"$meta": "textScore"})])
            .limit(limit)
        )
        return await cursor.to_list()

    async def find_by_source(
        self,
        drug_name: str,
        source: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return evidence from a specific source for a drug."""
        cursor = self.evidence.find(
            {"drug_name_lower": drug_name.lower(), "source": source},
            {"_id": 0},
        ).limit(limit)
        return await cursor.to_list()

    async def find_by_target(
        self,
        drug_name: str,
        target_symbol: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return evidence mentioning a specific gene/target."""
        cursor = self.evidence.find(
            {
                "drug_name_lower": drug_name.lower(),
                "target_symbol_lower": target_symbol.lower(),
            },
            {"_id": 0},
        ).limit(limit)
        return await cursor.to_list()

    async def find_by_disease(
        self,
        drug_name: str,
        disease_name: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return evidence mentioning a specific disease."""
        cursor = self.evidence.find(
            {
                "drug_name_lower": drug_name.lower(),
                "disease_name_lower": disease_name.lower(),
            },
            {"_id": 0},
        ).limit(limit)
        return await cursor.to_list()

    async def get_distinct_targets(self, drug_name: str) -> list[str]:
        """Return distinct target symbols for a drug."""
        return await self.evidence.distinct(
            "target_symbol",
            {"drug_name_lower": drug_name.lower(), "target_symbol": {"$ne": None}},
        )

    async def get_distinct_diseases(self, drug_name: str) -> list[str]:
        """Return distinct disease names mentioned in evidence for a drug."""
        return await self.evidence.distinct(
            "disease_name",
            {"drug_name_lower": drug_name.lower(), "disease_name": {"$ne": None}},
        )

    async def get_distinct_sources(self, drug_name: str) -> list[str]:
        """Return distinct evidence sources for a drug."""
        return await self.evidence.distinct(
            "source",
            {"drug_name_lower": drug_name.lower()},
        )

    async def get_evidence_summary(self, drug_name: str) -> dict:
        """Return a summary of staged evidence for a drug.

        Useful as a quick overview tool for the agent.
        """
        meta = await self.get_drug_meta(drug_name)
        if meta is None:
            return {"error": f"No staged evidence for '{drug_name}'"}

        targets = await self.get_distinct_targets(drug_name)
        diseases = await self.get_distinct_diseases(drug_name)
        sources = await self.get_distinct_sources(drug_name)

        return {
            "drug_name": meta.get("drug_name", drug_name),
            "chembl_id": meta.get("chembl_id"),
            "pubchem_cid": meta.get("pubchem_cid"),
            "n_evidence_docs": meta.get("n_docs", 0),
            "staged_at": str(meta.get("staged_at", "")),
            "sources": sources,
            "sources_status": meta.get("sources_status", {}),
            "n_distinct_targets": len(targets),
            "targets": targets[:30],  # cap for LLM context
            "n_distinct_diseases": len(diseases),
            "diseases": diseases[:30],
        }

    async def count_staged_drugs(self) -> int:
        """Return the number of drugs with staged evidence."""
        return await self.drugs.count_documents({})

    async def list_staged_drugs(self) -> list[dict]:
        """Return metadata for all staged drugs."""
        cursor = self.drugs.find({}, {"_id": 0}).sort("drug_name_lower", 1)
        return await cursor.to_list()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_record(doc: EvidenceDocument) -> dict[str, Any]:
        """Convert an EvidenceDocument to a MongoDB-friendly dict."""
        d = doc.model_dump(mode="json")
        # Add lowercased fields for case-insensitive lookups
        d["drug_name_lower"] = doc.drug_name.lower()
        d["target_symbol_lower"] = (doc.target_symbol or "").lower() or None
        d["disease_name_lower"] = (doc.disease_name or "").lower() or None
        return d
