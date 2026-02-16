"""Stage evidence from 6 biomedical APIs into MongoDB.

This is Phase 0 -- run ONCE before any experiment.  It downloads evidence
for every candidate drug, stores the results in MongoDB, and tracks
per-source status so partial failures can be resumed.

Usage
-----
::

    # Stage evidence for 200 drugs (default)
    uv run python -m scripts.stage_evidence

    # Stage a small batch for testing
    uv run python -m scripts.stage_evidence --drugs 10

    # Resume an interrupted staging run (skips already-staged drugs)
    uv run python -m scripts.stage_evidence --resume

    # Re-stage specific drugs
    uv run python -m scripts.stage_evidence --drug-names "aspirin,ibuprofen"

    # Check staging status
    uv run python -m scripts.stage_evidence --status-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from src.config.settings import Settings
from src.data.aggregator import EvidenceAggregator
from src.data.ctd import CTDClient
from src.data.evidence_store import EvidenceStore
from src.data.normalizer import DrugNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quieten chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hishel").setLevel(logging.WARNING)

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
    parser = argparse.ArgumentParser(
        description="Stage evidence from biomedical APIs into MongoDB"
    )
    parser.add_argument(
        "--drugs",
        type=int,
        default=200,
        help="Number of candidate drugs to stage (default: 200)",
    )
    parser.add_argument(
        "--drug-names",
        type=str,
        default="",
        help="Comma-separated drug names to stage (overrides --drugs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip drugs that already have staged evidence",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between drugs (rate limiting, default: 1.0)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        default=False,
        help="Only print staging status, do not fetch anything",
    )
    parser.add_argument(
        "--min-assoc",
        type=int,
        default=10,
        help="Min disease associations for CTD candidate filtering",
    )
    parser.add_argument(
        "--max-assoc",
        type=int,
        default=250,
        help="Max disease associations for CTD candidate filtering",
    )
    return parser


async def _print_status(store: EvidenceStore) -> None:
    """Print a summary of what has been staged."""
    n = await store.count_staged_drugs()
    logger.info("=== Staging Status: %d drugs in MongoDB ===", n)

    if n == 0:
        logger.info("No drugs staged yet.  Run without --status-only to begin.")
        return

    drugs = await store.list_staged_drugs()
    total_docs = 0
    for d in drugs:
        n_docs = d.get("n_docs", 0)
        total_docs += n_docs
        status = d.get("sources_status", {})
        ok_sources = sum(1 for v in status.values() if v == "ok")
        logger.info(
            "  %-30s  docs=%4d  sources_ok=%d/%d  chembl=%s",
            d.get("drug_name", "?"),
            n_docs,
            ok_sources,
            len(_EXPECTED_SOURCES),
            d.get("chembl_id") or "n/a",
        )
    logger.info("Total evidence documents: %d", total_docs)


async def _stage_drug(
    drug_name: str,
    normalizer: DrugNormalizer,
    aggregator: EvidenceAggregator,
    store: EvidenceStore,
) -> int:
    """Normalise and stage evidence for one drug.  Returns doc count."""
    # Normalise
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

    # Gather evidence
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

    # Determine per-source status from what we got
    source_counts: dict[str, int] = {}
    for d in docs:
        source_counts[d.source.value] = source_counts.get(d.source.value, 0) + 1

    for src_name in _EXPECTED_SOURCES:
        if source_counts.get(src_name, 0) > 0:
            sources_status[src_name] = "ok"
        elif src_name in ("opentargets", "chembl") and not chembl_id:
            sources_status[src_name] = "skipped_no_chembl"
        else:
            sources_status[src_name] = "empty_or_error"

    # Store in MongoDB
    n = await store.stage_drug(
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=pubchem_cid,
        docs=docs,
        sources_status=sources_status,
    )
    return n


async def main() -> None:
    args = _build_parser().parse_args()
    settings = Settings()

    store = EvidenceStore(settings)
    await store.connect()

    if args.status_only:
        await _print_status(store)
        await store.close()
        return

    # Determine drug list
    if args.drug_names:
        candidates = [n.strip() for n in args.drug_names.split(",") if n.strip()]
        logger.info("Staging %d user-specified drugs", len(candidates))
    else:
        ctd = CTDClient(settings)
        await ctd.load()
        candidates = ctd.get_candidate_drugs(
            min_assoc=args.min_assoc,
            max_assoc=args.max_assoc,
            limit=args.drugs,
        )
        logger.info("Selected %d candidates from CTD", len(candidates))

    # Filter already-staged if --resume
    if args.resume:
        pending = []
        for name in candidates:
            if await store.is_staged(name):
                logger.info("Skipping '%s' (already staged)", name)
            else:
                pending.append(name)
        logger.info(
            "Resume: %d already staged, %d remaining",
            len(candidates) - len(pending),
            len(pending),
        )
        candidates = pending

    if not candidates:
        logger.info("Nothing to stage.")
        await store.close()
        return

    # Stage
    normalizer = DrugNormalizer(settings)
    aggregator = EvidenceAggregator(settings)

    t0 = time.perf_counter()
    total_docs = 0
    successes = 0
    failures = 0

    for idx, drug_name in enumerate(candidates, 1):
        logger.info(
            "=== Staging %d / %d: %s ===",
            idx,
            len(candidates),
            drug_name,
        )
        try:
            n = await _stage_drug(drug_name, normalizer, aggregator, store)
            total_docs += n
            successes += 1
            logger.info(
                "[%s] Staged %d docs (cumulative: %d)",
                drug_name,
                n,
                total_docs,
            )
        except Exception:
            logger.exception("Failed to stage '%s'", drug_name)
            failures += 1

        # Rate-limit between drugs
        if idx < len(candidates) and args.delay > 0:
            await asyncio.sleep(args.delay)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Staging complete: %d drugs (%d ok, %d failed), %d total docs in %.1f s",
        len(candidates),
        successes,
        failures,
        total_docs,
        elapsed,
    )

    # Print final status
    await _print_status(store)
    await store.close()
    await normalizer.close()


if __name__ == "__main__":
    asyncio.run(main())
