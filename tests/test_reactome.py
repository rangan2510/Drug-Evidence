"""Smoke tests for the Reactome pathway client and its aggregator integration.

Uses a minimal in-memory Ensembl2Reactome.txt fixture (no network calls).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.config.settings import Settings
from src.data.reactome import ReactomeClient
from src.schemas.evidence import EvidenceDocument, EvidenceSource, EvidenceType

# ---------------------------------------------------------------
# Fixture: minimal Ensembl2Reactome bulk file
# ---------------------------------------------------------------

# Format: Ensembl_ID \t Pathway_ID \t URL \t Pathway_Name \t Evidence_Code \t Species
_BULK_FIXTURE = textwrap.dedent("""\
    ENSG00000073756\tR-HSA-2162123\thttps://reactome.org/PathwayBrowser/#/R-HSA-2162123\tSynthesis of Prostaglandins (PG) and Thromboxanes (TX)\tIEA\tHomo sapiens
    ENSG00000073756\tR-HSA-211859\thttps://reactome.org/PathwayBrowser/#/R-HSA-211859\tBiological oxidations\tTAS\tHomo sapiens
    ENSG00000073756\tR-HSA-9999999\thttps://reactome.org/PathwayBrowser/#/R-HSA-9999999\tMouse-only pathway\tIEA\tMus musculus
    ENSG00000169083\tR-HSA-76002\thttps://reactome.org/PathwayBrowser/#/R-HSA-76002\tPlatelet activation, signaling and aggregation\tTAS\tHomo sapiens
    ENSG00000169083\tR-HSA-2162123\thttps://reactome.org/PathwayBrowser/#/R-HSA-2162123\tSynthesis of Prostaglandins (PG) and Thromboxanes (TX)\tIEA\tHomo sapiens
    BADLINE_ONLY_THREE_COLUMNS\tR-HSA-0000\thttps://x
""")


@pytest.fixture()
def reactome_cache(tmp_path: Path) -> Path:
    """Write the fixture bulk file and return its parent directory."""
    cache_dir = tmp_path / "reactome"
    cache_dir.mkdir()
    (cache_dir / "Ensembl2Reactome.txt").write_text(_BULK_FIXTURE, encoding="utf-8")
    return cache_dir


def _make_settings(cache_dir: Path) -> Settings:
    """Return Settings pointing at the tmp cache directory."""
    return Settings(
        reactome_cache_dir=str(cache_dir),
        reactome_download_base="https://reactome.org/download/current",
    )


# =====================================================================
# ReactomeClient -- unit tests
# =====================================================================


class TestReactomeClientParsing:
    """Ensure the bulk TSV is parsed, species-filtered, and mapped correctly."""

    @pytest.mark.asyncio
    async def test_loads_human_only(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        await client._ensure_loaded()

        assert client._loaded is True
        # Mus musculus row should be excluded
        assert "ENSG00000073756" in client._pathways_by_ensembl
        pathways = client._pathways_by_ensembl["ENSG00000073756"]
        pathway_ids = {p["pathway_id"] for p in pathways}
        assert "R-HSA-2162123" in pathway_ids
        assert "R-HSA-211859" in pathway_ids
        # Mouse-only pathway must NOT appear
        assert "R-HSA-9999999" not in pathway_ids

    @pytest.mark.asyncio
    async def test_skips_malformed_lines(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        await client._ensure_loaded()

        # The bad line has only 3 columns -- it should be silently dropped
        assert "BADLINE_ONLY_THREE_COLUMNS" not in client._pathways_by_ensembl

    @pytest.mark.asyncio
    async def test_second_gene_present(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        await client._ensure_loaded()

        assert "ENSG00000169083" in client._pathways_by_ensembl
        pathways = client._pathways_by_ensembl["ENSG00000169083"]
        assert len(pathways) == 2


# =====================================================================
# ReactomeClient.fetch() -- returns valid EvidenceDocument list
# =====================================================================


class TestReactomeFetch:
    """Verify .fetch() returns properly structured EvidenceDocuments."""

    @pytest.mark.asyncio
    async def test_returns_evidence_documents(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        ensembl_map = {"ENSG00000073756": "PTGS2"}

        docs = await client.fetch("ASPIRIN", "CHEMBL25", ensembl_map)

        # Should produce 2 pathway docs (2 human pathways for that gene)
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, EvidenceDocument)
            assert doc.source == EvidenceSource.REACTOME
            assert doc.evidence_type == EvidenceType.PATHWAY
            assert doc.drug_name == "ASPIRIN"
            assert doc.drug_chembl_id == "CHEMBL25"
            assert doc.target_symbol == "PTGS2"
            assert doc.target_ensembl_id == "ENSG00000073756"
            assert doc.citation.url is not None
            assert "pathway_id" in doc.metadata
            assert "pathway_name" in doc.metadata
            assert "evidence_code" in doc.metadata

    @pytest.mark.asyncio
    async def test_multiple_targets(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        ensembl_map = {
            "ENSG00000073756": "PTGS2",
            "ENSG00000169083": "AR",
        }
        docs = await client.fetch("ASPIRIN", "CHEMBL25", ensembl_map)

        # PTGS2 -> 2 pathways, AR -> 2 pathways = 4 docs total
        assert len(docs) == 4
        pathway_ids = {d.metadata["pathway_id"] for d in docs}
        assert "R-HSA-2162123" in pathway_ids
        assert "R-HSA-211859" in pathway_ids
        assert "R-HSA-76002" in pathway_ids

    @pytest.mark.asyncio
    async def test_deduplicates_same_gene_pathway_pair(self, reactome_cache: Path) -> None:
        """Passing the same Ensembl ID twice should not duplicate docs."""
        client = ReactomeClient(_make_settings(reactome_cache))
        ensembl_map = {
            "ENSG00000073756": "PTGS2",
        }
        docs = await client.fetch("ASPIRIN", "CHEMBL25", ensembl_map)
        assert len(docs) == 2  # still 2, no duplicates

    @pytest.mark.asyncio
    async def test_empty_ensembl_map_returns_nothing(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        docs = await client.fetch("ASPIRIN", "CHEMBL25", {})
        assert docs == []

    @pytest.mark.asyncio
    async def test_unknown_ensembl_id_returns_nothing(self, reactome_cache: Path) -> None:
        client = ReactomeClient(_make_settings(reactome_cache))
        docs = await client.fetch("ASPIRIN", "CHEMBL25", {"ENSG99999999999": "FAKE"})
        assert docs == []

    @pytest.mark.asyncio
    async def test_schema_validation(self, reactome_cache: Path) -> None:
        """Every doc must round-trip through Pydantic model_validate."""
        client = ReactomeClient(_make_settings(reactome_cache))
        docs = await client.fetch(
            "ASPIRIN", "CHEMBL25", {"ENSG00000073756": "PTGS2"}
        )
        for doc in docs:
            validated = EvidenceDocument.model_validate(doc.model_dump())
            assert validated.source == EvidenceSource.REACTOME


# =====================================================================
# ReactomeClient -- download trigger
# =====================================================================


class TestReactomeDownload:
    """Verify the download path is invoked when the cache file is missing."""

    @pytest.mark.asyncio
    async def test_downloads_when_cache_missing(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "reactome_empty"
        cache_dir.mkdir()
        settings = _make_settings(cache_dir)
        client = ReactomeClient(settings)

        # Patch _download_file to write the fixture instead of hitting the network
        async def fake_download(url: str, dest: Path) -> None:
            dest.write_text(_BULK_FIXTURE, encoding="utf-8")

        with patch.object(client, "_download_file", side_effect=fake_download) as mock_dl:
            docs = await client.fetch("ASPIRIN", None, {"ENSG00000073756": "PTGS2"})
            mock_dl.assert_called_once()
            assert len(docs) == 2
            # Verify the cache file now exists
            assert (cache_dir / "Ensembl2Reactome.txt").exists()


# =====================================================================
# Aggregator integration -- Reactome wiring
# =====================================================================


class TestAggregatorReactomeWiring:
    """Verify the aggregator calls ReactomeClient with Ensembl IDs from upstream docs."""

    @pytest.mark.asyncio
    async def test_reactome_called_when_ensembl_targets_present(self) -> None:
        """Mock all data clients; confirm Reactome receives the Ensembl map."""
        from src.data.aggregator import EvidenceAggregator

        agg = EvidenceAggregator()

        # Build a fake upstream doc that carries an Ensembl target ID
        upstream_doc = EvidenceDocument(
            text="PTGS2 is a target of Aspirin",
            source=EvidenceSource.OPENTARGETS,
            evidence_type=EvidenceType.LITERATURE,
            drug_name="ASPIRIN",
            drug_chembl_id="CHEMBL25",
            target_symbol="PTGS2",
            target_ensembl_id="ENSG00000073756",
        )

        reactome_doc = EvidenceDocument(
            text="Pathway: prostaglandin synthesis",
            source=EvidenceSource.REACTOME,
            evidence_type=EvidenceType.PATHWAY,
            drug_name="ASPIRIN",
            drug_chembl_id="CHEMBL25",
            target_symbol="PTGS2",
            target_ensembl_id="ENSG00000073756",
            metadata={"pathway_id": "R-HSA-2162123", "pathway_name": "PG synthesis", "evidence_code": "IEA"},
        )

        # Stub the normalizer
        from src.data.normalizer import NormalizedDrug

        agg._normalizer.normalize = AsyncMock(
            return_value=NormalizedDrug(
                query="ASPIRIN",
                preferred_name="ASPIRIN",
                chembl_id="CHEMBL25",
                pubchem_cid=2244,
                synonyms=["aspirin"],
            )
        )

        # Stub all core sources -- only OpenTargets returns a doc with Ensembl ID
        agg._opentargets.fetch = AsyncMock(return_value=[upstream_doc])
        agg._dgidb.fetch = AsyncMock(return_value=[])
        agg._pubchem.fetch = AsyncMock(return_value=[])
        agg._pharmgkb.fetch = AsyncMock(return_value=[])
        agg._chembl.fetch = AsyncMock(return_value=[])
        agg._pubmed.fetch_by_pmids = AsyncMock(return_value=[])

        # Stub Reactome -- verify it receives the correct Ensembl map
        agg._reactome.fetch = AsyncMock(return_value=[reactome_doc])

        docs = await agg.gather("ASPIRIN")

        # Reactome should have been called with the Ensembl ID from the upstream doc
        agg._reactome.fetch.assert_called_once()
        call_args = agg._reactome.fetch.call_args
        ensembl_map_arg = call_args[0][2]  # third positional arg
        assert "ENSG00000073756" in ensembl_map_arg
        assert ensembl_map_arg["ENSG00000073756"] == "PTGS2"

        # Final doc list should contain both the upstream and reactome docs
        sources = {d.source for d in docs}
        assert EvidenceSource.OPENTARGETS in sources
        assert EvidenceSource.REACTOME in sources

    @pytest.mark.asyncio
    async def test_reactome_skipped_when_no_ensembl_targets(self) -> None:
        """If no upstream docs carry Ensembl IDs, Reactome is not called."""
        from src.data.aggregator import EvidenceAggregator

        agg = EvidenceAggregator()

        # Doc WITHOUT Ensembl ID
        upstream_doc = EvidenceDocument(
            text="Some DGIdb interaction",
            source=EvidenceSource.DGIDB,
            evidence_type=EvidenceType.DRUG_GENE_INTERACTION,
            drug_name="FAKECHEM",
        )

        from src.data.normalizer import NormalizedDrug

        agg._normalizer.normalize = AsyncMock(
            return_value=NormalizedDrug(
                query="FAKECHEM",
                preferred_name="FAKECHEM",
                chembl_id=None,
                pubchem_cid=None,
                synonyms=[],
            )
        )

        agg._opentargets.fetch = AsyncMock(return_value=[])
        agg._dgidb.fetch = AsyncMock(return_value=[upstream_doc])
        agg._pubchem.fetch = AsyncMock(return_value=[])
        agg._pharmgkb.fetch = AsyncMock(return_value=[])
        agg._chembl.fetch = AsyncMock(return_value=[])
        agg._pubmed.fetch_by_pmids = AsyncMock(return_value=[])
        agg._reactome.fetch = AsyncMock(return_value=[])

        docs = await agg.gather("FAKECHEM")

        # Reactome should NOT have been called -- empty ensembl_to_symbol dict
        agg._reactome.fetch.assert_not_called()

        # Only the DGIdb doc should appear
        assert len(docs) == 1
        assert docs[0].source == EvidenceSource.DGIDB
