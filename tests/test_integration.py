"""Integration tests -- hit REAL APIs with a well-known drug (aspirin).

These tests validate that:
  1. Each data-source API is reachable.
  2. The response JSON/XML schema matches what our parsing code expects.
  3. Parsed ``EvidenceDocument`` objects have sensible field values.

Run separately from fast unit tests:

    uv run pytest tests/test_integration.py -v --timeout=120

The test drug is **aspirin** -- guaranteed to exist in every biomedical
database.  Identifiers:
    ChEMBL ID : CHEMBL25
    PubChem CID: 2244
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from src.config.settings import Settings
from src.data.chembl import ChEMBLClient
from src.data.dgidb import DGIdbClient
from src.data.normalizer import DrugNormalizer, NormalizedDrug
from src.data.opentargets import OpenTargetsClient
from src.data.pharmgkb import PharmGKBClient
from src.data.pubchem import PubChemClient
from src.data.pubmed import PubMedClient
from src.schemas.evidence import EvidenceDocument, EvidenceSource

logger = logging.getLogger(__name__)

# -- Constants for the test drug -------------------------------------------
DRUG_NAME = "aspirin"
CHEMBL_ID = "CHEMBL25"
PUBCHEM_CID = 2244

# Mark every test in this module as async + integration
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# -- Shared fixtures -------------------------------------------------------

@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


# -- Helper -----------------------------------------------------------------

def _assert_docs(
    docs: list[EvidenceDocument],
    expected_source: EvidenceSource,
    *,
    min_count: int = 1,
) -> None:
    """Common assertions for a list of evidence documents."""
    assert len(docs) >= min_count, (
        f"Expected at least {min_count} doc(s) from {expected_source.value}, "
        f"got {len(docs)}"
    )
    for doc in docs:
        assert doc.source == expected_source, (
            f"Wrong source: expected {expected_source.value}, got {doc.source.value}"
        )
        assert doc.text, "EvidenceDocument.text must not be empty"
        assert doc.drug_name.lower() == DRUG_NAME.lower(), (
            f"drug_name mismatch: {doc.drug_name!r}"
        )


# ==========================================================================
# 1. DrugNormalizer -- PubChem + ChEMBL resolution
# ==========================================================================

class TestNormalizer:
    """Real API: resolve 'aspirin' via PubChem + ChEMBL."""

    async def test_normalize_aspirin(self, settings: Settings) -> None:
        norm = DrugNormalizer(settings)
        try:
            result: NormalizedDrug = await norm.normalize(DRUG_NAME)
        finally:
            await norm.close()

        # PubChem CID must resolve
        assert result.pubchem_cid is not None, "PubChem CID not resolved"
        assert result.pubchem_cid == PUBCHEM_CID, (
            f"Expected CID {PUBCHEM_CID}, got {result.pubchem_cid}"
        )

        # ChEMBL ID must resolve
        assert result.chembl_id is not None, "ChEMBL ID not resolved"
        assert result.chembl_id.upper() == CHEMBL_ID, (
            f"Expected {CHEMBL_ID}, got {result.chembl_id}"
        )

        # Must have synonyms
        assert len(result.synonyms) > 0, "Expected at least one synonym"
        logger.info(
            "Normalizer OK: CID=%s, ChEMBL=%s, synonyms=%d",
            result.pubchem_cid,
            result.chembl_id,
            len(result.synonyms),
        )

    async def test_normalize_unknown_drug(self, settings: Settings) -> None:
        """A garbage string should not crash, just return sparse results."""
        norm = DrugNormalizer(settings)
        try:
            result = await norm.normalize("xyzzy_not_a_drug_12345")
        finally:
            await norm.close()

        assert result.query == "xyzzy_not_a_drug_12345"
        # CID / ChEMBL may or may not be None, but it must not raise


# ==========================================================================
# 2. OpenTargets -- GraphQL knownDrugs + evidence
# ==========================================================================

class TestOpenTargets:
    """Real API: fetch aspirin evidence from OpenTargets Platform."""

    async def test_fetch_aspirin(self, settings: Settings) -> None:
        client = OpenTargetsClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID)

        _assert_docs(docs, EvidenceSource.OPENTARGETS, min_count=1)

        # Spot-check: at least one doc should have a target symbol
        has_target = any(d.target_symbol for d in docs)
        assert has_target, "Expected at least one doc with target_symbol"

        # At least one doc should have a disease name
        has_disease = any(d.disease_name for d in docs)
        assert has_disease, "Expected at least one doc with disease_name"

        logger.info("OpenTargets OK: %d docs", len(docs))


# ==========================================================================
# 3. DGIdb -- drug-gene interactions
# ==========================================================================

class TestDGIdb:
    """Real API: fetch aspirin gene interactions from DGIdb."""

    async def test_fetch_aspirin(self, settings: Settings) -> None:
        client = DGIdbClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID)

        _assert_docs(docs, EvidenceSource.DGIDB, min_count=1)

        # Every doc should reference a gene
        for doc in docs:
            assert doc.target_symbol, (
                f"DGIdb doc missing target_symbol: {doc.text[:80]}"
            )

        logger.info("DGIdb OK: %d docs", len(docs))


# ==========================================================================
# 4. PubChem -- pharmacological actions + bioassays
# ==========================================================================

class TestPubChem:
    """Real API: fetch aspirin pharmacology + bioassays from PubChem."""

    async def test_fetch_aspirin_with_cid(self, settings: Settings) -> None:
        """Provide CID directly -- skips name resolution step."""
        client = PubChemClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID, pubchem_cid=PUBCHEM_CID)

        _assert_docs(docs, EvidenceSource.PUBCHEM, min_count=1)
        logger.info("PubChem OK: %d docs (with CID)", len(docs))

    async def test_fetch_aspirin_by_name(self, settings: Settings) -> None:
        """Let the client resolve CID from name."""
        client = PubChemClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID)

        # Should still succeed (CID resolution + pharmacology)
        _assert_docs(docs, EvidenceSource.PUBCHEM, min_count=1)
        logger.info("PubChem OK: %d docs (by name)", len(docs))


# ==========================================================================
# 5. PharmGKB -- clinical annotations
# ==========================================================================

class TestPharmGKB:
    """Real API: fetch aspirin clinical annotations from PharmGKB."""

    async def test_fetch_aspirin(self, settings: Settings) -> None:
        client = PharmGKBClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID)

        _assert_docs(docs, EvidenceSource.PHARMGKB, min_count=1)

        # Spot-check: aspirin has well-known pharmacogenomic annotations
        has_gene = any(d.target_symbol for d in docs)
        logger.info(
            "PharmGKB OK: %d docs, has_gene=%s", len(docs), has_gene
        )


# ==========================================================================
# 6. ChEMBL -- mechanism of action + activities
# ==========================================================================

class TestChEMBL:
    """Real API: fetch aspirin MoA + binding data from ChEMBL."""

    async def test_fetch_aspirin(self, settings: Settings) -> None:
        client = ChEMBLClient(settings)
        docs = await client.fetch(DRUG_NAME, CHEMBL_ID)

        _assert_docs(docs, EvidenceSource.CHEMBL, min_count=1)

        # Aspirin has a well-known mechanism: COX inhibitor
        moa_docs = [
            d for d in docs
            if "mechanism" in d.text.lower() or "action" in d.text.lower()
        ]
        assert len(moa_docs) >= 1, (
            "Expected at least one mechanism-of-action doc for aspirin"
        )

        logger.info("ChEMBL OK: %d docs (%d MoA)", len(docs), len(moa_docs))


# ==========================================================================
# 7. PubMed -- eSearch + eFetch
# ==========================================================================

class TestPubMed:
    """Real API: search + fetch PubMed abstracts about aspirin."""

    async def test_search_and_fetch(self, settings: Settings) -> None:
        client = PubMedClient(settings)
        docs = await client.search_and_fetch(DRUG_NAME, CHEMBL_ID, max_results=5)

        _assert_docs(docs, EvidenceSource.PUBMED, min_count=1)

        # Each doc should have a valid PMID
        for doc in docs:
            assert doc.citation is not None, "PubMed doc must have a citation"
            assert doc.citation.pmid, f"PubMed doc missing PMID: {doc.text[:80]}"

        logger.info("PubMed OK: %d abstracts", len(docs))

    async def test_fetch_by_known_pmids(self, settings: Settings) -> None:
        """Fetch specific PMIDs for classic aspirin papers."""
        known_pmids = [
            "14592543",  # "The mechanism of action of aspirin" (2003)
            "9263351",   # "Aspirin and platelets" (1997)
        ]
        client = PubMedClient(settings)
        docs = await client.fetch_by_pmids(known_pmids, DRUG_NAME, CHEMBL_ID)

        assert len(docs) >= 1, (
            f"Expected at least 1 abstract for known PMIDs, got {len(docs)}"
        )
        for doc in docs:
            assert doc.citation.pmid in known_pmids

        logger.info("PubMed known-PMID OK: %d abstracts", len(docs))


# ==========================================================================
# 8. Smoke test: all clients together (mini aggregation)
# ==========================================================================

class TestAllClientsTogether:
    """Call every client in parallel, verify combined results."""

    async def test_parallel_fetch_all(self, settings: Settings) -> None:
        """Fan out to all 6 evidence sources concurrently."""
        ot = OpenTargetsClient(settings)
        dgi = DGIdbClient(settings)
        pc = PubChemClient(settings)
        pgkb = PharmGKBClient(settings)
        chembl = ChEMBLClient(settings)
        pm = PubMedClient(settings)

        results = await asyncio.gather(
            ot.fetch(DRUG_NAME, CHEMBL_ID),
            dgi.fetch(DRUG_NAME, CHEMBL_ID),
            pc.fetch(DRUG_NAME, CHEMBL_ID, pubchem_cid=PUBCHEM_CID),
            pgkb.fetch(DRUG_NAME, CHEMBL_ID),
            chembl.fetch(DRUG_NAME, CHEMBL_ID),
            pm.search_and_fetch(DRUG_NAME, CHEMBL_ID, max_results=5),
            return_exceptions=True,
        )

        source_names = [
            "OpenTargets", "DGIdb", "PubChem", "PharmGKB", "ChEMBL", "PubMed",
        ]

        total = 0
        for name, res in zip(source_names, results):
            if isinstance(res, Exception):
                logger.error("%s FAILED: %s", name, res)
                pytest.fail(f"{name} raised {type(res).__name__}: {res}")
            else:
                count = len(res)
                total += count
                logger.info("%s: %d docs", name, count)

        assert total >= 10, (
            f"Expected at least 10 total docs across all sources for aspirin, "
            f"got {total}"
        )
        logger.info("All-clients parallel OK: %d total docs", total)
