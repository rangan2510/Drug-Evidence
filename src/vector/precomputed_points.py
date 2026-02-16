"""Precomputed Qdrant point objects for staged evidence.

This module supports a lightweight Phase 0 flow:
1) Read staged evidence records (no chunking)
2) Embed each record text as-is (dense + sparse)
3) Build ``qdrant_client.models.PointStruct`` objects
4) Save/load points as ``.pkl`` shards
5) Bulk upsert into Qdrant (including in-memory clients)
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from qdrant_client import models

from src.vector.embeddings import EmbeddingManager


def _stable_point_id(record: dict) -> str:
    """Return a deterministic point ID for one evidence record."""
    raw = "|".join(
        [
            str(record.get("drug_name", "")),
            str(record.get("source", "")),
            str(record.get("target_symbol", "")),
            str(record.get("disease_name", "")),
            str((record.get("citation") or {}).get("pmid", "")),
            str((record.get("citation") or {}).get("doi", "")),
            str(record.get("text", "")),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _record_to_payload(record: dict) -> dict:
    citation = record.get("citation") or {}
    return {
        "source": record.get("source", ""),
        "evidence_type": record.get("evidence_type", ""),
        "drug_name": record.get("drug_name", ""),
        "drug_chembl_id": record.get("drug_chembl_id") or "",
        "drug_pubchem_cid": record.get("drug_pubchem_cid") or 0,
        "target_symbol": record.get("target_symbol") or "",
        "target_ensembl_id": record.get("target_ensembl_id") or "",
        "disease_name": record.get("disease_name") or "",
        "disease_id": record.get("disease_id") or "",
        "score": record.get("score") if record.get("score") is not None else 0.0,
        "pmid": citation.get("pmid") or "",
        "doi": citation.get("doi") or "",
        "year": citation.get("year") or 0,
        "text": (record.get("text") or "").strip(),
    }


def build_points_from_records(
    records: list[dict],
    embeddings: EmbeddingManager,
) -> list[models.PointStruct]:
    """Embed records as-is and return Qdrant ``PointStruct`` objects."""
    if not records:
        return []

    payloads = [_record_to_payload(r) for r in records]
    texts = [p["text"] for p in payloads]

    dense_vecs = embeddings.embed_documents(texts)
    sparse_vecs = embeddings.embed_documents_sparse(texts)

    points: list[models.PointStruct] = []
    for idx, payload in enumerate(payloads):
        sv = sparse_vecs[idx]
        sparse_vector = models.SparseVector(
            indices=sv.indices.tolist(),
            values=sv.values.tolist(),
        )
        points.append(
            models.PointStruct(
                id=_stable_point_id(records[idx]),
                vector={
                    "dense": dense_vecs[idx].tolist(),
                    "splade": sparse_vector,
                },
                payload=payload,
            )
        )

    return points


def save_points_shard(points: list[models.PointStruct], output_path: Path) -> None:
    """Serialize a list of point objects into one ``.pkl`` file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(points, fh)


def load_points_shard(path: Path) -> list[models.PointStruct]:
    """Load one ``.pkl`` shard containing point objects."""
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        msg = f"Point shard must contain a list, got {type(data).__name__}: {path}"
        raise TypeError(msg)
    return data


def load_points_from_dir(points_dir: Path) -> list[models.PointStruct]:
    """Load all ``points_*.pkl`` files from a directory."""
    files = sorted(points_dir.glob("points_*.pkl"))
    all_points: list[models.PointStruct] = []
    for file in files:
        all_points.extend(load_points_shard(file))
    return all_points
