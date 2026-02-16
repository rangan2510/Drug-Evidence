from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from src.config.settings import Settings
from src.data.aggregator import EvidenceAggregator
from src.data.evidence_store import EvidenceStore
from src.data.normalizer import DrugNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SOURCES = [
    "opentargets",
    "dgidb",
    "pubchem",
    "pharmgkb",
    "chembl",
    "pubmed",
    "reactome",
]


async def _stage_one(
    raw_name: str,
    normalizer: DrugNormalizer,
    aggregator: EvidenceAggregator,
    store: EvidenceStore,
) -> tuple[str, int]:
    drug_name = raw_name
    chembl_id = None
    pubchem_cid = None

    try:
        norm = await normalizer.normalize(raw_name)
        chembl_id = norm.chembl_id
        pubchem_cid = norm.pubchem_cid
        drug_name = norm.preferred_name or raw_name
    except Exception:
        logger.exception("Normalization failed for '%s'", raw_name)

    docs = []
    try:
        docs = await aggregator.gather(
            drug_name,
            chembl_id=chembl_id,
            pubchem_cid=pubchem_cid,
            skip_normalize=True,
        )
    except Exception:
        logger.exception("Aggregation failed for '%s'", raw_name)

    source_counts: dict[str, int] = {}
    for doc in docs:
        source_counts[doc.source.value] = source_counts.get(doc.source.value, 0) + 1

    sources_status: dict[str, str] = {}
    for source_name in EXPECTED_SOURCES:
        if source_counts.get(source_name, 0) > 0:
            sources_status[source_name] = "ok"
        elif source_name in ("opentargets", "chembl") and not chembl_id:
            sources_status[source_name] = "skipped_no_chembl"
        else:
            sources_status[source_name] = "empty_or_error"

    n_docs = await store.stage_drug(
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=pubchem_cid,
        docs=docs,
        sources_status=sources_status,
    )
    return drug_name, n_docs


async def main() -> None:
    settings = Settings()
    target_names = [
        line.strip()
        for line in Path("data/target_set.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    store = EvidenceStore(settings)
    await store.connect()

    missing: list[str] = []
    for name in target_names:
        meta = await store.get_drug_meta(name)
        if not (meta and meta.get("n_docs", 0) > 0):
            missing.append(name)

    print(f"Target drugs: {len(target_names)}")
    print(f"Missing (to stage): {len(missing)}")

    if not missing:
        await store.close()
        print("Nothing to stage.")
        return

    normalizer = DrugNormalizer(settings)
    aggregator = EvidenceAggregator(settings)

    t0 = time.perf_counter()
    staged_ok = 0
    staged_fail = 0
    total_docs = 0

    for idx, raw_name in enumerate(missing, 1):
        print(f"=== [{idx}/{len(missing)}] {raw_name} ===")
        try:
            staged_name, n_docs = await _stage_one(raw_name, normalizer, aggregator, store)
            staged_ok += 1
            total_docs += n_docs
            print(f"staged as '{staged_name}' with {n_docs} docs")
        except Exception:
            staged_fail += 1
            logger.exception("Failed to stage '%s'", raw_name)

    elapsed = time.perf_counter() - t0
    await store.close()

    print("\n=== Staging complete ===")
    print(f"Success: {staged_ok}")
    print(f"Failed: {staged_fail}")
    print(f"Total docs inserted: {total_docs}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
