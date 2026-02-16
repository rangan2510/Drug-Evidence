"""OpenTargets Platform GraphQL client.

Fetches drug-target-disease associations and linked PubMed evidence
from the OpenTargets Platform API (v4 GraphQL).
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings
from src.data.http import cached_async_client
from src.schemas.evidence import (
    Citation,
    EvidenceDocument,
    EvidenceSource,
    EvidenceType,
)

logger = logging.getLogger(__name__)

_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Query 1: drug -> knownDrugs -> target + disease pairs
_KNOWN_DRUGS_QUERY = """
query($chemblId: String!) {
  drug(chemblId: $chemblId) {
    name
    knownDrugs(size: 50) {
      rows {
        target { id approvedSymbol }
        disease { id name }
        phase
        status
        references {
          source
          ids
        }
      }
    }
  }
}
"""

# Query 2: evidence rows (europepmc datasource) for a target-disease pair
_EVIDENCE_QUERY = """
query($efoId: String!, $ensemblId: String!) {
  disease(efoId: $efoId) {
    evidences(ensemblIds: [$ensemblId], datasourceIds: ["europepmc"], size: 10) {
      rows {
        score
        literature
        textMiningSentences {
          dEnd
          dStart
          tEnd
          tStart
          section
          text
        }
      }
    }
  }
}
"""


class OpenTargetsClient:
    """Fetch drug-target-disease evidence from the OpenTargets Platform."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(15.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _post_graphql(
        self, client: httpx.AsyncClient, query: str, variables: dict
    ) -> dict:
        resp = await client.post(
            _GRAPHQL_URL,
            json={"query": query, "variables": variables},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str,
        *,
        max_targets: int = 50,
    ) -> list[EvidenceDocument]:
        """Retrieve evidence documents for a drug from OpenTargets.

        Parameters
        ----------
        drug_name
            Canonical drug name (for metadata).
        chembl_id
            ChEMBL identifier (e.g. ``"CHEMBL25"``).
        max_targets
            Cap on target-disease pairs to query evidence for.
        """
        docs: list[EvidenceDocument] = []

        async with cached_async_client(self._settings) as client:
            # Step 1: get known-drug associations
            associations = await self._fetch_associations(
                client, chembl_id, max_targets
            )
            if not associations:
                logger.info("OpenTargets: no knownDrugs for %s", chembl_id)
                return docs

            # Step 2: for each association, fetch evidence rows
            for assoc in associations:
                try:
                    result = await self._post_graphql(
                        client,
                        _EVIDENCE_QUERY,
                        {
                            "efoId": assoc["disease_id"],
                            "ensemblId": assoc["target_id"],
                        },
                    )
                except httpx.HTTPStatusError:
                    continue

                _data = result.get("data") or {}
                _disease = _data.get("disease") or {}
                _evidences = _disease.get("evidences") or {}
                rows = _evidences.get("rows") or []
                for row in rows:
                    # Build text from text-mining sentences if available
                    sentences = row.get("textMiningSentences") or []
                    text_parts = [s["text"] for s in sentences if s.get("text")]
                    text = " ".join(text_parts) if text_parts else (
                        f"Evidence linking {assoc['target_sym']} to "
                        f"{assoc['disease_name']} (score={row.get('score', 'N/A')})"
                    )

                    pmids = [str(p) for p in (row.get("literature") or [])]
                    for pmid in pmids[:3]:
                        docs.append(
                            EvidenceDocument(
                                text=text,
                                source=EvidenceSource.OPENTARGETS,
                                evidence_type=EvidenceType.LITERATURE,
                                citation=Citation(pmid=pmid),
                                drug_name=drug_name,
                                drug_chembl_id=chembl_id,
                                target_symbol=assoc["target_sym"],
                                target_ensembl_id=assoc["target_id"],
                                disease_name=assoc["disease_name"],
                                disease_id=assoc["disease_id"],
                                score=row.get("score"),
                                metadata={
                                    "phase": assoc.get("phase"),
                                    "status": assoc.get("status"),
                                },
                            )
                        )

        logger.info(
            "OpenTargets: fetched %d evidence docs for %s (%s)",
            len(docs),
            drug_name,
            chembl_id,
        )
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_associations(
        self,
        client: httpx.AsyncClient,
        chembl_id: str,
        max_targets: int,
    ) -> list[dict]:
        """Return list of {target_id, target_sym, disease_id, disease_name}."""
        try:
            result = await self._post_graphql(
                client, _KNOWN_DRUGS_QUERY, {"chemblId": chembl_id}
            )
        except httpx.HTTPStatusError:
            return []

        data = result.get("data") or {}
        drug = data.get("drug") or {}
        known = drug.get("knownDrugs") or {}
        rows = known.get("rows") or []

        associations: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            target = row.get("target") or {}
            disease = row.get("disease") or {}
            t_sym = target.get("approvedSymbol")
            d_name = disease.get("name")
            if not t_sym or not d_name:
                continue
            pair = (t_sym, d_name)
            if pair in seen:
                continue
            seen.add(pair)
            associations.append(
                {
                    "target_id": target["id"],
                    "target_sym": t_sym,
                    "disease_id": disease["id"],
                    "disease_name": d_name,
                    "phase": row.get("phase"),
                    "status": row.get("status"),
                }
            )
            if len(associations) >= max_targets:
                break

        return associations
