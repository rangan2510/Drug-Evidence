"""Tests for all data-source clients and the evidence aggregator.

Uses MockTransport to intercept HTTP calls with canned JSON responses.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import httpx
import pytest

from src.schemas.evidence import EvidenceDocument, EvidenceSource, EvidenceType

# =====================================================================
# Shared mock transport
# =====================================================================


class MockTransport(httpx.AsyncBaseTransport):
    """Route requests to canned JSON/text/bytes responses."""

    def __init__(self, routes: dict[str, dict | list | int | str | bytes]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, payload in self._routes.items():
            if pattern in url:
                if isinstance(payload, int):
                    return httpx.Response(status_code=payload)
                if isinstance(payload, bytes):
                    return httpx.Response(200, content=payload)
                if isinstance(payload, str):
                    return httpx.Response(200, text=payload)
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "Not found"})


# =====================================================================
# CTD client tests
# =====================================================================


class TestCTDClient:
    """Tests for CTDClient using a small in-memory TSV.gz fixture."""

    @staticmethod
    def _make_tsv_gz(tmp_path: Path) -> Path:
        """Create a minimal CTD-format TSV.gz file."""
        lines = [
            "# CTD test header\n",
            "# Columns: ChemicalName\tChemicalID\tCasRN\tDiseaseName\t"
            "DiseaseID\tDirectEvidence\tInferenceGeneSymbol\tInferenceScore\t"
            "OmimIDs\tPubMedIDs\n",
            "Aspirin\tMESH:D001241\t50-78-2\tDiabetes Mellitus, Type 2\t"
            "MESH:D003924\ttherapeutic\tPTGS2\t\t\t12345678|23456789\n",
            "Aspirin\tMESH:D001241\t50-78-2\tMyocardial Infarction\t"
            "MESH:D009203\ttherapeutic\t\t\t\t34567890\n",
            "Aspirin\tMESH:D001241\t50-78-2\tStroke\t"
            "MESH:D020521\tmarker/mechanism\tPTGS1\t15.5\t\t\n",
            "Metformin\tMESH:D008687\t657-24-9\tDiabetes Mellitus, Type 2\t"
            "MESH:D003924\ttherapeutic\t\t\t\t45678901\n",
        ]
        dest = tmp_path / "CTD_chemicals_diseases.tsv.gz"
        with gzip.open(dest, "wt", encoding="utf-8") as f:
            f.writelines(lines)
        return dest

    @pytest.mark.asyncio
    async def test_load_and_query(self, tmp_path: Path) -> None:
        from src.data.ctd import CTDClient

        tsv_path = self._make_tsv_gz(tmp_path)
        client = CTDClient()
        await client.load(cache_path=tsv_path)

        assert client.is_loaded

        diseases = client.get_therapeutic_diseases("Aspirin")
        assert "Diabetes Mellitus, Type 2" in diseases
        assert "Myocardial Infarction" in diseases
        # marker/mechanism row should NOT appear in therapeutic set
        assert "Stroke" not in diseases
        assert client.count_associations("Aspirin") == 2

    @pytest.mark.asyncio
    async def test_get_candidate_drugs(self, tmp_path: Path) -> None:
        from src.data.ctd import CTDClient

        tsv_path = self._make_tsv_gz(tmp_path)
        client = CTDClient()
        await client.load(cache_path=tsv_path)

        # min=1, max=3 should include both aspirin (2) and metformin (1)
        candidates = client.get_candidate_drugs(min_assoc=1, max_assoc=3)
        assert "aspirin" in candidates
        assert "metformin" in candidates

        # min=2, max=2 should include only aspirin
        candidates = client.get_candidate_drugs(min_assoc=2, max_assoc=2)
        assert "aspirin" in candidates
        assert "metformin" not in candidates


# =====================================================================
# OpenTargets client tests
# =====================================================================


OPENTARGETS_KNOWN_DRUGS_RESP = {
    "data": {
        "drug": {
            "name": "ASPIRIN",
            "knownDrugs": {
                "rows": [
                    {
                        "target": {"id": "ENSG00000073756", "approvedSymbol": "PTGS2"},
                        "disease": {"id": "EFO_0003060", "name": "inflammatory disease"},
                        "phase": 4,
                        "status": "Approved",
                        "references": [],
                    }
                ]
            },
        }
    }
}

OPENTARGETS_EVIDENCE_RESP = {
    "data": {
        "disease": {
            "evidences": {
                "rows": [
                    {
                        "score": 0.85,
                        "literature": ["12345678"],
                        "textMiningSentences": [
                            {"text": "Aspirin inhibits PTGS2 in inflammatory disease."}
                        ],
                    }
                ]
            }
        }
    }
}


class TestOpenTargetsClient:
    @pytest.mark.asyncio
    async def test_fetch_returns_evidence_docs(self) -> None:
        from src.data.opentargets import OpenTargetsClient

        routes = {
            "api.platform.opentargets.org": OPENTARGETS_KNOWN_DRUGS_RESP,
        }
        # Need separate routes for the two GraphQL calls
        call_count = {"n": 0}

        class OTTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return httpx.Response(200, json=OPENTARGETS_KNOWN_DRUGS_RESP)
                return httpx.Response(200, json=OPENTARGETS_EVIDENCE_RESP)

        client = OpenTargetsClient()
        # Monkey-patch to use mock transport
        orig_fetch = client.fetch

        async def patched_fetch(drug_name: str, chembl_id: str, **kw):
            async with httpx.AsyncClient(transport=OTTransport()) as hc:
                # Step 1: associations
                result1 = await hc.post("https://api.platform.opentargets.org/api/v4/graphql", json={})
                assocs = (
                    result1.json()
                    .get("data", {}).get("drug", {}).get("knownDrugs", {}).get("rows", [])
                )
                docs = []
                for row in assocs:
                    target = row.get("target", {})
                    disease = row.get("disease", {})
                    # Step 2: evidence
                    result2 = await hc.post("https://api.platform.opentargets.org/api/v4/graphql", json={})
                    ev_rows = (
                        result2.json()
                        .get("data", {}).get("disease", {}).get("evidences", {}).get("rows", [])
                    )
                    for ev in ev_rows:
                        sentences = ev.get("textMiningSentences") or []
                        text = " ".join(s["text"] for s in sentences if s.get("text"))
                        pmids = [str(p) for p in (ev.get("literature") or [])]
                        for pmid in pmids[:3]:
                            docs.append(
                                EvidenceDocument(
                                    text=text or "evidence",
                                    source=EvidenceSource.OPENTARGETS,
                                    evidence_type=EvidenceType.LITERATURE,
                                    citation={"pmid": pmid},
                                    drug_name=drug_name,
                                    drug_chembl_id=chembl_id,
                                    target_symbol=target.get("approvedSymbol"),
                                    disease_name=disease.get("name"),
                                )
                            )
                return docs

        docs = await patched_fetch("aspirin", "CHEMBL25")

        assert len(docs) >= 1
        assert all(isinstance(d, EvidenceDocument) for d in docs)
        assert docs[0].source == EvidenceSource.OPENTARGETS
        assert docs[0].citation.pmid == "12345678"
        assert docs[0].target_symbol == "PTGS2"


# =====================================================================
# DGIdb client tests
# =====================================================================


DGIDB_RESPONSE = {
    "data": {
        "drugs": {
            "nodes": [
                {
                    "name": "ASPIRIN",
                    "conceptId": "chembl:CHEMBL25",
                    "interactions": [
                        {
                            "interactionScore": 8.5,
                            "interactionTypes": [
                                {"type": "inhibitor", "directionality": "inhibitory"}
                            ],
                            "interactionAttributes": [],
                            "gene": {"name": "PTGS2", "conceptId": "entrez:5743"},
                            "publications": [{"pmid": "11700954"}],
                            "sources": [
                                {"sourceDbName": "ChEMBL", "fullName": "ChEMBL"}
                            ],
                        }
                    ],
                }
            ]
        }
    }
}


class TestDGIdbClient:
    @pytest.mark.asyncio
    async def test_fetch_returns_interactions(self) -> None:
        from src.data.dgidb import DGIdbClient

        transport = MockTransport({"dgidb.org": DGIDB_RESPONSE})
        client = DGIdbClient()

        # Patch the client's method to use mock transport
        async def mock_post(hc, variables):
            resp = await hc.post(
                "https://dgidb.org/api/graphql",
                json={"query": "", "variables": variables},
            )
            resp.raise_for_status()
            return resp.json()

        async with httpx.AsyncClient(transport=transport) as hc:
            client._post_graphql = lambda c, v: mock_post(hc, v)
            docs = await client.fetch("aspirin", chembl_id="CHEMBL25")

        assert len(docs) == 1
        assert docs[0].source == EvidenceSource.DGIDB
        assert docs[0].evidence_type == EvidenceType.DRUG_GENE_INTERACTION
        assert docs[0].target_symbol == "PTGS2"
        assert "inhibitor" in docs[0].metadata["interaction_types"]


# =====================================================================
# PubChem client tests
# =====================================================================


PUBCHEM_CID_RESP = {"IdentifierList": {"CID": [2244]}}
PUBCHEM_CLASSIFICATION_RESP = {
    "Hierarchies": {
        "Hierarchy": [
            {
                "SourceName": "MeSH",
                "Node": [
                    {
                        "Information": {
                            "Name": "Anti-Inflammatory Agents, Non-Steroidal",
                            "Description": "NSAID classification",
                        }
                    }
                ],
            }
        ]
    }
}
PUBCHEM_ASSAY_RESP = {
    "Table": {
        "Columns": {"Column": ["AID", "TargetName", "TargetGI", "Activity Outcome"]},
        "Row": [
            {"Cell": [{"Value": "12345"}, {"Value": "COX-2"}, {"Value": "5743"}, {"Value": "Active"}]},
            {"Cell": [{"Value": "12346"}, {"Value": "COX-1"}, {"Value": "5742"}, {"Value": "Inactive"}]},
        ],
    }
}


class TestPubChemClient:
    @pytest.mark.asyncio
    async def test_fetch_returns_evidence(self) -> None:
        from src.data.pubchem import PubChemClient

        routes = {
            "/compound/name/aspirin/cids/JSON": PUBCHEM_CID_RESP,
            "/compound/cid/2244/classification/JSON": PUBCHEM_CLASSIFICATION_RESP,
            "/compound/cid/2244/assaysummary/JSON": PUBCHEM_ASSAY_RESP,
        }
        transport = MockTransport(routes)
        client = PubChemClient()

        # Override internal methods to use mock transport
        async with httpx.AsyncClient(transport=transport) as hc:
            original_get = client._get_json

            async def mock_get(c, url):
                return await original_get(hc, url)

            client._get_json = mock_get
            docs = await client.fetch("aspirin")

        # Should have classification docs + 1 active assay (inactive filtered out)
        pharm_docs = [d for d in docs if d.evidence_type == EvidenceType.PHARMACOLOGICAL_ACTION]
        assay_docs = [d for d in docs if d.evidence_type == EvidenceType.BINDING_ASSAY]

        assert len(pharm_docs) >= 1
        assert "Anti-Inflammatory" in pharm_docs[0].text
        assert len(assay_docs) == 1
        assert assay_docs[0].target_symbol == "COX-2"


# =====================================================================
# PharmGKB client tests
# =====================================================================


PHARMGKB_DRUG_RESP = {
    "data": [{"id": "PA448497", "name": "aspirin"}]
}
PHARMGKB_ANNOTATIONS_RESP = {
    "data": [
        {
            "id": "1234",
            "evidenceLevel": "1A",
            "summary": "Aspirin response is associated with PTGS2 variants.",
            "relatedGenes": [{"symbol": "PTGS2"}],
            "relatedDiseases": [{"name": "Cardiovascular Diseases"}],
        }
    ]
}
PHARMGKB_LABELS_RESP: dict = {"data": []}


class TestPharmGKBClient:
    @pytest.mark.asyncio
    async def test_fetch_returns_annotations(self) -> None:
        from src.data.pharmgkb import PharmGKBClient

        routes = {
            "/chemical?name=aspirin": PHARMGKB_DRUG_RESP,
            "clinicalAnnotation?relatedChemicals": PHARMGKB_ANNOTATIONS_RESP,
            "drugLabel?relatedChemicals": PHARMGKB_LABELS_RESP,
        }
        transport = MockTransport(routes)
        client = PharmGKBClient()

        async with httpx.AsyncClient(transport=transport) as hc:
            orig = client._get_json

            async def mock_get(c, url):
                return await orig(hc, url)

            client._get_json = mock_get
            docs = await client.fetch("aspirin")

        assert len(docs) >= 1
        assert docs[0].source == EvidenceSource.PHARMGKB
        assert docs[0].evidence_type == EvidenceType.CLINICAL_ANNOTATION
        assert docs[0].target_symbol == "PTGS2"
        assert "1A" in docs[0].metadata["evidence_level"]


# =====================================================================
# ChEMBL client tests
# =====================================================================


CHEMBL_MECHANISMS_RESP = {
    "mechanisms": [
        {
            "action_type": "INHIBITOR",
            "target_pref_name": "Cyclooxygenase-2",
            "target_chembl_id": "CHEMBL230",
            "mechanism_of_action": "Cyclooxygenase-2 inhibitor",
            "mechanism_refs": [{"ref_type": "PubMed", "ref_id": "10441473"}],
            "disease_efficacy": None,
        }
    ]
}
CHEMBL_ACTIVITIES_RESP = {
    "activities": [
        {
            "target_pref_name": "Cyclooxygenase-2",
            "target_chembl_id": "CHEMBL230",
            "standard_type": "IC50",
            "standard_value": "100",
            "standard_units": "nM",
            "pchembl_value": 7.0,
            "assay_chembl_id": "CHEMBL12345",
            "document_chembl_id": "CHEMBL_DOC1",
        }
    ]
}


class TestChEMBLClient:
    @pytest.mark.asyncio
    async def test_fetch_returns_mechanisms_and_activities(self) -> None:
        from src.data.chembl import ChEMBLClient

        routes = {
            "mechanism.json": CHEMBL_MECHANISMS_RESP,
            "activity.json": CHEMBL_ACTIVITIES_RESP,
        }
        transport = MockTransport(routes)
        client = ChEMBLClient()

        async with httpx.AsyncClient(transport=transport) as hc:
            orig = client._get_json

            async def mock_get(c, url):
                return await orig(hc, url)

            client._get_json = mock_get
            docs = await client.fetch("aspirin", "CHEMBL25")

        moa_docs = [d for d in docs if d.evidence_type == EvidenceType.MECHANISM_OF_ACTION]
        assay_docs = [d for d in docs if d.evidence_type == EvidenceType.BINDING_ASSAY]

        assert len(moa_docs) == 1
        assert "INHIBITOR" in moa_docs[0].text
        assert moa_docs[0].citation.pmid == "10441473"

        assert len(assay_docs) == 1
        assert assay_docs[0].target_symbol == "Cyclooxygenase-2"
        assert assay_docs[0].score == pytest.approx(0.7, abs=0.01)


# =====================================================================
# PubMed client tests
# =====================================================================


PUBMED_ESEARCH_RESP = {
    "esearchresult": {"idlist": ["12345678", "23456789"]}
}
PUBMED_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Aspirin and Inflammation</ArticleTitle>
        <Abstract>
          <AbstractText>Aspirin inhibits cyclooxygenase enzymes involved in inflammation.</AbstractText>
        </Abstract>
        <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/test.2020</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>23456789</PMID>
      <Article>
        <ArticleTitle>Aspirin in Cardiovascular Prevention</ArticleTitle>
        <Abstract>
          <AbstractText>Aspirin reduces cardiovascular events through platelet inhibition.</AbstractText>
        </Abstract>
        <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList></ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedClient:
    @pytest.mark.asyncio
    async def test_fetch_by_pmids(self) -> None:
        from src.data.pubmed import PubMedClient

        transport = MockTransport({"eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch": PUBMED_EFETCH_XML})
        client = PubMedClient()

        # Override _efetch_batch to use our transport
        async with httpx.AsyncClient(transport=transport) as hc:
            orig = client._efetch_batch

            async def mock_efetch(c, pmids, drug, cid):
                return await orig(hc, pmids, drug, cid)

            client._efetch_batch = mock_efetch
            docs = await client.fetch_by_pmids(["12345678", "23456789"], "aspirin", "CHEMBL25")

        assert len(docs) == 2
        assert all(d.source == EvidenceSource.PUBMED for d in docs)
        assert docs[0].citation.pmid == "12345678"
        assert docs[0].citation.year == 2020
        assert docs[0].citation.doi == "10.1234/test.2020"
        assert "cyclooxygenase" in docs[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_and_fetch(self) -> None:
        from src.data.pubmed import PubMedClient

        routes = {
            "esearch.fcgi": PUBMED_ESEARCH_RESP,
            "efetch.fcgi": PUBMED_EFETCH_XML,
        }
        transport = MockTransport(routes)
        client = PubMedClient()

        async with httpx.AsyncClient(transport=transport) as hc:
            async def mock_esearch(c, query, max_results):
                resp = await hc.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("esearchresult", {}).get("idlist", [])

            async def mock_efetch(c, pmids, drug, cid):
                orig = PubMedClient._parse_efetch_xml
                resp = await hc.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(pmids)},
                )
                return orig(client, resp.text, drug, cid)

            client._esearch = mock_esearch
            client._efetch_batch = mock_efetch
            docs = await client.search_and_fetch("aspirin", "CHEMBL25", max_results=5)

        assert len(docs) == 2
        assert all(d.drug_name == "aspirin" for d in docs)

    @pytest.mark.asyncio
    async def test_empty_pmids_returns_empty(self) -> None:
        from src.data.pubmed import PubMedClient

        client = PubMedClient()
        docs = await client.fetch_by_pmids([], "aspirin")
        assert docs == []

    @pytest.mark.asyncio
    async def test_deduplication_of_pmids(self) -> None:
        from src.data.pubmed import PubMedClient

        transport = MockTransport({"efetch": PUBMED_EFETCH_XML})
        client = PubMedClient()

        async with httpx.AsyncClient(transport=transport) as hc:
            orig = client._efetch_batch

            async def mock_efetch(c, pmids, drug, cid):
                return await orig(hc, pmids, drug, cid)

            client._efetch_batch = mock_efetch
            # Pass duplicate PMIDs
            docs = await client.fetch_by_pmids(
                ["12345678", "12345678", "23456789"], "aspirin"
            )

        # Should only get 2 unique docs despite 3 input PMIDs
        assert len(docs) == 2


# =====================================================================
# Aggregator dedup test (unit)
# =====================================================================


class TestAggregatorDedup:
    def test_deduplication_removes_exact_duplicates(self) -> None:
        from src.data.aggregator import EvidenceAggregator

        doc1 = EvidenceDocument(
            text="Aspirin inhibits COX-2",
            source=EvidenceSource.CHEMBL,
            evidence_type=EvidenceType.MECHANISM_OF_ACTION,
            drug_name="aspirin",
        )
        doc2 = EvidenceDocument(
            text="Aspirin inhibits COX-2",
            source=EvidenceSource.CHEMBL,
            evidence_type=EvidenceType.MECHANISM_OF_ACTION,
            drug_name="aspirin",
        )
        doc3 = EvidenceDocument(
            text="Different text about aspirin",
            source=EvidenceSource.PUBMED,
            evidence_type=EvidenceType.LITERATURE,
            drug_name="aspirin",
        )

        result = EvidenceAggregator._deduplicate([doc1, doc2, doc3])
        assert len(result) == 2

    def test_deduplication_keeps_different_sources(self) -> None:
        from src.data.aggregator import EvidenceAggregator

        doc1 = EvidenceDocument(
            text="Same text",
            source=EvidenceSource.CHEMBL,
            evidence_type=EvidenceType.MECHANISM_OF_ACTION,
            drug_name="aspirin",
        )
        doc2 = EvidenceDocument(
            text="Same text",
            source=EvidenceSource.OPENTARGETS,
            evidence_type=EvidenceType.LITERATURE,
            drug_name="aspirin",
        )
        result = EvidenceAggregator._deduplicate([doc1, doc2])
        assert len(result) == 2
