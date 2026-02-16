"""Tests for DrugNormalizer -- mocked PubChem + ChEMBL responses."""

from __future__ import annotations

import pytest
import httpx

from src.data.normalizer import DrugNormalizer, NormalizedDrug


# ------------------------------------------------------------------
# Fixtures: mock HTTP responses
# ------------------------------------------------------------------

PUBCHEM_CID_RESPONSE = {
    "IdentifierList": {"CID": [2244]}
}

PUBCHEM_SYNONYMS_RESPONSE = {
    "InformationList": {
        "Information": [
            {
                "CID": 2244,
                "Synonym": [
                    "aspirin",
                    "Acetylsalicylic acid",
                    "ASA",
                    "2-Acetoxybenzoic acid",
                ],
            }
        ]
    }
}

PUBCHEM_XREF_RESPONSE = {
    "InformationList": {
        "Information": [
            {"CID": 2244, "RegistryID": ["CHEMBL25", "DB00945"]}
        ]
    }
}

CHEMBL_SEARCH_RESPONSE = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL25",
            "pref_name": "ASPIRIN",
            "molecule_synonyms": [
                {"molecule_synonym": "Acetylsalicylic acid"},
                {"molecule_synonym": "ASA"},
            ],
            "molecule_structures": {
                "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
            },
        }
    ]
}


class MockTransport(httpx.AsyncBaseTransport):
    """Route requests to canned JSON responses."""

    def __init__(self, routes: dict[str, dict | int]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, payload in self._routes.items():
            if pattern in url:
                if isinstance(payload, int):
                    return httpx.Response(status_code=payload)
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"Fault": {"Message": "Not found"}})


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.fixture
def normalizer_with_mocks() -> DrugNormalizer:
    """Return a DrugNormalizer whose HTTP client uses canned responses."""
    routes = {
        "/compound/name/aspirin/cids/JSON": PUBCHEM_CID_RESPONSE,
        "/compound/cid/2244/synonyms/JSON": PUBCHEM_SYNONYMS_RESPONSE,
        "/compound/cid/2244/xrefs/RegistryID/JSON": PUBCHEM_XREF_RESPONSE,
        "/molecule/search.json": CHEMBL_SEARCH_RESPONSE,
    }
    transport = MockTransport(routes)
    norm = DrugNormalizer()
    norm._client = httpx.AsyncClient(transport=transport)
    return norm


@pytest.mark.asyncio
async def test_normalize_aspirin(normalizer_with_mocks: DrugNormalizer) -> None:
    result = await normalizer_with_mocks.normalize("aspirin")

    assert isinstance(result, NormalizedDrug)
    assert result.pubchem_cid == 2244
    assert result.chembl_id == "CHEMBL25"
    assert result.preferred_name is not None
    assert "aspirin" in [s.lower() for s in result.synonyms]
    assert result.inchi_key == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    await normalizer_with_mocks.close()


@pytest.mark.asyncio
async def test_normalize_unknown_drug() -> None:
    """Unknown drug should return query as preferred_name with no IDs."""
    routes: dict[str, dict | int] = {
        "/compound/name/": 404,
        "/molecule/search.json": {"molecules": []},
    }
    transport = MockTransport(routes)
    norm = DrugNormalizer()
    norm._client = httpx.AsyncClient(transport=transport)

    result = await norm.normalize("unknowndrug12345")
    assert result.chembl_id is None
    assert result.pubchem_cid is None
    assert result.preferred_name == "unknowndrug12345"
    await norm.close()


@pytest.mark.asyncio
async def test_pubchem_xref_fallback() -> None:
    """If ChEMBL search returns no molecules, PubChem xref should provide the ID."""
    routes = {
        "/compound/name/aspirin/cids/JSON": PUBCHEM_CID_RESPONSE,
        "/compound/cid/2244/synonyms/JSON": PUBCHEM_SYNONYMS_RESPONSE,
        "/compound/cid/2244/xrefs/RegistryID/JSON": PUBCHEM_XREF_RESPONSE,
        "/molecule/search.json": {"molecules": []},
    }
    transport = MockTransport(routes)
    norm = DrugNormalizer()
    norm._client = httpx.AsyncClient(transport=transport)

    result = await norm.normalize("aspirin")
    assert result.chembl_id == "CHEMBL25"
    assert result.pubchem_cid == 2244
    await norm.close()


@pytest.mark.asyncio
async def test_synonyms_deduplication(normalizer_with_mocks: DrugNormalizer) -> None:
    """Synonyms from PubChem + ChEMBL should be deduplicated."""
    result = await normalizer_with_mocks.normalize("aspirin")
    lowered = [s.lower() for s in result.synonyms]
    assert len(lowered) == len(set(lowered))
    await normalizer_with_mocks.close()
