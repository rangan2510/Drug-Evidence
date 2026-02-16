"""Phase 0 end-to-end prep: CTD candidates -> Mongo staging -> PKL points -> in-memory Qdrant.

Usage
-----

    # Full Phase 0 for default target_drugs (200)
    uv run python -m scripts.phase0_prepare

    # Smaller smoke run
    uv run python -m scripts.phase0_prepare --drugs 10 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from src.config.settings import Settings
from src.data.aggregator import EvidenceAggregator
from src.data.ctd import CTDClient
from src.data.evidence_store import EvidenceStore
from src.data.normalizer import DrugNormalizer
from src.vector.embeddings import EmbeddingManager
from src.vector.precomputed_points import (
    build_points_from_records,
    load_points_from_dir,
    save_points_shard,
)
from src.vector.store import HybridVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hishel").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_EXPECTED_SOURCES = [
    "opentargets",
    "dgidb",
    "pubchem",
    "pharmgkb",
    "chembl",
    "pubmed",
    "reactome",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Phase 0 preparation flow")
    parser.add_argument("--drugs", type=int, default=200)
    parser.add_argument("--min-assoc", type=int, default=10)
    parser.add_argument("--max-assoc", type=int, default=250)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--skip-stage", action="store_true", default=False)
    parser.add_argument("--skip-vectorize", action="store_true", default=False)
    parser.add_argument("--skip-qdrant-load", action="store_true", default=False)
    parser.add_argument("--vector-batch", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--points-dir", type=str, default="data/phase0_points")
    parser.add_argument(
        "--ctd-out", type=str, default="results/phase0/ctd_candidates.json"
    )
    parser.add_argument(
        "--smoke-query", type=str, default="mechanism of action disease association"
    )
    return parser


async def _select_candidates(settings: Settings, args: argparse.Namespace) -> list[str]:
    ctd = CTDClient(settings)
    await ctd.load()
    candidates = ctd.get_candidate_drugs(
        min_assoc=args.min_assoc,
        max_assoc=args.max_assoc,
        limit=args.drugs,
    )
    out_path = Path(args.ctd_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    logger.info("Selected %d CTD candidates (saved to %s)", len(candidates), out_path)
    return candidates


async def _stage_one_drug(
    drug_name: str,
    normalizer: DrugNormalizer,
    aggregator: EvidenceAggregator,
    store: EvidenceStore,
) -> int:
    chembl_id = None
    pubchem_cid = None
    sources_status: dict[str, str] = {}

    try:
        norm = await normalizer.normalize(drug_name)
        chembl_id = norm.chembl_id
        pubchem_cid = norm.pubchem_cid
        drug_name = norm.preferred_name or drug_name
    except Exception:
        logger.exception("Normalisation failed for '%s'", drug_name)

    docs = []
    try:
        docs = await aggregator.gather(
            drug_name,
            chembl_id=chembl_id,
            pubchem_cid=pubchem_cid,
            skip_normalize=True,
        )
    except Exception:
        logger.exception("Aggregation failed for '%s'", drug_name)

    source_counts: dict[str, int] = {}
    for doc in docs:
        source_counts[doc.source.value] = source_counts.get(doc.source.value, 0) + 1

    for src_name in _EXPECTED_SOURCES:
        if source_counts.get(src_name, 0) > 0:
            sources_status[src_name] = "ok"
        elif src_name in ("opentargets", "chembl") and not chembl_id:
            sources_status[src_name] = "skipped_no_chembl"
        else:
            sources_status[src_name] = "empty_or_error"

    return await store.stage_drug(
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=pubchem_cid,
        docs=docs,
        sources_status=sources_status,
    )


async def _stage_candidates(
    candidates: list[str],
    settings: Settings,
    args: argparse.Namespace,
) -> None:
    store = EvidenceStore(settings)
    await store.connect()

    normalizer = DrugNormalizer(settings)
    aggregator = EvidenceAggregator(settings)

    try:
        pending = list(candidates)
        if args.resume:
            pending = []
            for name in candidates:
                is_staged = await store.is_staged(name)
                if not is_staged:
                    try:
                        norm = await normalizer.normalize(name)
                        if norm.preferred_name and norm.preferred_name != name:
                            is_staged = await store.is_staged(norm.preferred_name)
                    except Exception:
                        logger.exception("Resume normalisation failed for '%s'", name)

                if not is_staged:
                    pending.append(name)
            logger.info(
                "Resume mode: %d staged, %d pending",
                len(candidates) - len(pending),
                len(pending),
            )

        total_docs = 0
        t0 = time.perf_counter()
        for idx, name in enumerate(pending, 1):
            logger.info("Staging %d/%d: %s", idx, len(pending), name)
            inserted = await _stage_one_drug(name, normalizer, aggregator, store)
            total_docs += inserted
            logger.info("[%s] staged docs=%d cumulative=%d", name, inserted, total_docs)
            if idx < len(pending) and args.delay > 0:
                await asyncio.sleep(args.delay)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Staging complete: %d drugs, inserted_docs=%d, elapsed=%.1fs",
            len(pending),
            total_docs,
            elapsed,
        )
    finally:
        await store.close()
        await normalizer.close()


async def _vectorize_to_pickles(
    candidates: list[str],
    settings: Settings,
    args: argparse.Namespace,
) -> int:
    points_dir = Path(args.points_dir)
    points_dir.mkdir(parents=True, exist_ok=True)

    for f in points_dir.glob("points_*.pkl"):
        f.unlink()

    store = EvidenceStore(settings)
    await store.connect()

    embeddings = EmbeddingManager.from_settings(settings)
    embeddings.load()

    try:
        n_staged = await store.count_staged_drugs()
        logger.info(
            "Vectorizing all staged evidence documents from MongoDB (%d staged drugs)",
            n_staged,
        )
        cursor = store.evidence.find({}, {"_id": 0})

        records_batch: list[dict] = []
        shard_points = []
        shard_idx = 1
        total_records = 0
        total_points = 0

        async for record in cursor:
            records_batch.append(record)
            if len(records_batch) >= args.vector_batch:
                points = build_points_from_records(records_batch, embeddings)
                shard_points.extend(points)
                total_records += len(records_batch)
                total_points += len(points)
                records_batch.clear()

                if len(shard_points) >= args.shard_size:
                    out = points_dir / f"points_{shard_idx:05d}.pkl"
                    save_points_shard(shard_points, out)
                    logger.info("Saved %d points to %s", len(shard_points), out)
                    shard_points = []
                    shard_idx += 1

        if records_batch:
            points = build_points_from_records(records_batch, embeddings)
            shard_points.extend(points)
            total_records += len(records_batch)
            total_points += len(points)

        if shard_points:
            out = points_dir / f"points_{shard_idx:05d}.pkl"
            save_points_shard(shard_points, out)
            logger.info("Saved %d points to %s", len(shard_points), out)

        logger.info(
            "Vectorization complete: records=%d points=%d shards=%d",
            total_records,
            total_points,
            len(list(points_dir.glob("points_*.pkl"))),
        )
        return total_points
    finally:
        await store.close()


def _load_into_in_memory_qdrant(
    settings: Settings,
    args: argparse.Namespace,
) -> tuple[int, int]:
    points_dir = Path(args.points_dir)
    points = load_points_from_dir(points_dir)

    embeddings = EmbeddingManager.from_settings(settings)
    embeddings.load_query_only()

    store = HybridVectorStore.from_settings(settings, embeddings, in_memory=True)
    store.ensure_collection()
    loaded = store.upsert_points(points)

    results = store.hybrid_search(args.smoke_query, limit=3)
    logger.info(
        "In-memory Qdrant smoke search returned %d hits for query='%s'",
        len(results),
        args.smoke_query,
    )
    return loaded, len(results)


async def main() -> None:
    args = _build_parser().parse_args()
    settings = Settings(target_drugs=args.drugs)

    t0 = time.perf_counter()
    candidates = await _select_candidates(settings, args)

    if not args.skip_stage:
        await _stage_candidates(candidates, settings, args)

    total_points = 0
    if not args.skip_vectorize:
        total_points = await _vectorize_to_pickles(candidates, settings, args)

    loaded = 0
    smoke_hits = 0
    if not args.skip_qdrant_load:
        loaded, smoke_hits = _load_into_in_memory_qdrant(settings, args)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Phase 0 complete: candidates=%d points_built=%d points_loaded=%d smoke_hits=%d elapsed=%.1fs",
        len(candidates),
        total_points,
        loaded,
        smoke_hits,
        elapsed,
    )


if __name__ == "__main__":
    asyncio.run(main())
