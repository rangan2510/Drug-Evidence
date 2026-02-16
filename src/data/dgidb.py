"""DGIdb (Drug Gene Interaction Database) GraphQL client.

Queries drug-gene interactions from DGIdb's GraphQL API, returning
uniform ``list[EvidenceDocument]``.
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

_GRAPHQL_URL = "https://dgidb.org/api/graphql"

_INTERACTIONS_QUERY = """
query($names: [String!]!) {
  drugs(names: $names) {
    nodes {
      name
      conceptId
      interactions {
        interactionScore
        interactionTypes {
          type
          directionality
        }
        interactionAttributes {
          name
          value
        }
        gene {
          name
          conceptId
        }
        publications {
          pmid
        }
        sources {
          sourceDbName
          fullName
        }
      }
    }
  }
}
"""


class DGIdbClient:
    """Fetch drug-gene interactions from DGIdb."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = httpx.Timeout(20.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _post_graphql(
        self, client: httpx.AsyncClient, variables: dict
    ) -> dict:
        resp = await client.post(
            _GRAPHQL_URL,
            json={"query": _INTERACTIONS_QUERY, "variables": variables},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch(
        self,
        drug_name: str,
        chembl_id: str | None = None,
        *,
        synonyms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        """Retrieve drug-gene interaction evidence from DGIdb.

        Parameters
        ----------
        drug_name
            Primary drug name to query.
        chembl_id
            Optional ChEMBL ID (stored in metadata).
        synonyms
            Additional names to try if the primary name yields no results.
        """
        names_to_try = [drug_name]
        if synonyms:
            names_to_try.extend(synonyms[:3])

        docs: list[EvidenceDocument] = []

        async with cached_async_client(self._settings) as client:
            result = await self._post_graphql(client, {"names": names_to_try})

        nodes = (
            result.get("data", {}).get("drugs", {}).get("nodes", [])
        )

        for drug_node in nodes:
            for interaction in drug_node.get("interactions", []):
                gene = interaction.get("gene") or {}
                gene_name = gene.get("name", "")
                gene_concept = gene.get("conceptId", "")

                # Interaction types
                int_types = interaction.get("interactionTypes") or []
                type_strs = [t.get("type", "") for t in int_types if t.get("type")]
                int_type_label = ", ".join(type_strs) if type_strs else "unknown"

                # Publications
                pubs = interaction.get("publications") or []
                pmids = [str(p["pmid"]) for p in pubs if p.get("pmid")]

                # Sources
                sources = interaction.get("sources") or []
                source_names = [s.get("sourceDbName", "") for s in sources]

                # Score
                score = interaction.get("interactionScore")

                text = (
                    f"Drug-gene interaction: {drug_name} {int_type_label} "
                    f"{gene_name}. Sources: {', '.join(source_names)}."
                )

                citation = Citation(
                    pmid=pmids[0] if pmids else None,
                    url=f"https://dgidb.org/genes/{gene_name}" if gene_name else None,
                )

                docs.append(
                    EvidenceDocument(
                        text=text,
                        source=EvidenceSource.DGIDB,
                        evidence_type=EvidenceType.DRUG_GENE_INTERACTION,
                        citation=citation,
                        drug_name=drug_name,
                        drug_chembl_id=chembl_id,
                        target_symbol=gene_name or None,
                        score=score,
                        metadata={
                            "interaction_types": type_strs,
                            "gene_concept_id": gene_concept,
                            "all_pmids": pmids,
                            "sources": source_names,
                        },
                    )
                )

        logger.info(
            "DGIdb: fetched %d interaction docs for %s",
            len(docs),
            drug_name,
        )
        return docs
