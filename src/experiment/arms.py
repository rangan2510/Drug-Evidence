"""Experimental arm definitions for the v2 frontier-model experiment.

Each arm pairs a MODEL_REGISTRY key with metadata describing how the arm
should be executed (pipeline with full evidence tooling vs web-search only).

The 8 arms form a 4x2 factorial design:

* **4 frontier models**: GPT-4.1, GPT-5.2, Sonnet 4.5, Opus 4.6
* **2 configurations per model**:
    - ``pipeline`` -- full evidence retrieval (6 biomedical APIs + Qdrant RAG)
    - ``websearch`` -- Tavily web search only (baseline comparison)

This design tests whether our structured evidence pipeline beats simple
web search across all frontier models.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.config.models import MODEL_REGISTRY, ModelSpec


# ------------------------------------------------------------------
# Arm type
# ------------------------------------------------------------------

class ArmType(str, Enum):
    """How evidence is gathered for the arm."""

    PIPELINE = "pipeline"    # full retrieval: 6 APIs + Qdrant RAG + tools
    WEBSEARCH = "websearch"  # Tavily web search only (baseline)


# ------------------------------------------------------------------
# Arm config
# ------------------------------------------------------------------

_BIOMEDICAL_DOMAINS: list[str] = [
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "clinicaltrials.gov",
    "opentargets.org",
    "dgidb.org",
    "pharmgkb.org",
    "ebi.ac.uk",
    "uniprot.org",
]


class ArmConfig(BaseModel):
    """Configuration for a single experimental arm."""

    arm_id: str = Field(..., description="Unique arm identifier, e.g. 'pipeline-gpt41'")
    model_key: str = Field(
        ...,
        description="Key into MODEL_REGISTRY (e.g. 'gpt_4_1')",
    )
    arm_type: ArmType
    description: str = Field(..., description="Human-readable purpose of this arm")
    uses_web_search: bool = Field(
        default=False,
        description="Whether the arm has access to Tavily web search",
    )
    allowed_search_domains: list[str] = Field(
        default_factory=list,
        description="Domain allow-list for web search (empty = unrestricted)",
    )

    def resolve_model_id(
        self,
        registry: dict[str, ModelSpec] | None = None,
    ) -> str:
        """Return the ``pydantic_ai_id`` for this arm's model.

        Parameters
        ----------
        registry:
            Model registry to look up the key in.
            Defaults to the module-level ``MODEL_REGISTRY``.

        Returns
        -------
        str
            e.g. ``"openai:gpt-4.1-2025-04-14"``

        Raises
        ------
        KeyError
            If ``model_key`` is not found in the registry.
        """
        reg = registry or MODEL_REGISTRY
        spec = reg[self.model_key]
        return spec.pydantic_ai_id


# ------------------------------------------------------------------
# 4 pipeline arms (evidence agent + 9 tools + Qdrant RAG)
# ------------------------------------------------------------------

PIPELINE_ARMS: dict[str, ArmConfig] = {
    "pipeline-gpt41": ArmConfig(
        arm_id="pipeline-gpt41",
        model_key="gpt_4_1",
        arm_type=ArmType.PIPELINE,
        description="Full evidence pipeline with GPT-4.1",
    ),
    "pipeline-gpt52": ArmConfig(
        arm_id="pipeline-gpt52",
        model_key="gpt_5_2",
        arm_type=ArmType.PIPELINE,
        description="Full evidence pipeline with GPT-5.2",
    ),
    "pipeline-sonnet45": ArmConfig(
        arm_id="pipeline-sonnet45",
        model_key="sonnet_4_5",
        arm_type=ArmType.PIPELINE,
        description="Full evidence pipeline with Sonnet 4.5",
    ),
    "pipeline-opus46": ArmConfig(
        arm_id="pipeline-opus46",
        model_key="claude_opus_4_6",
        arm_type=ArmType.PIPELINE,
        description="Full evidence pipeline with Opus 4.6",
    ),
}


# ------------------------------------------------------------------
# 4 websearch arms (Tavily only -- no vector store, no DB tools)
# ------------------------------------------------------------------

WEBSEARCH_ARMS: dict[str, ArmConfig] = {
    "websearch-gpt41": ArmConfig(
        arm_id="websearch-gpt41",
        model_key="gpt_4_1",
        arm_type=ArmType.WEBSEARCH,
        description="GPT-4.1 with Tavily web search only (baseline)",
        uses_web_search=True,
        allowed_search_domains=_BIOMEDICAL_DOMAINS,
    ),
    "websearch-gpt52": ArmConfig(
        arm_id="websearch-gpt52",
        model_key="gpt_5_2",
        arm_type=ArmType.WEBSEARCH,
        description="GPT-5.2 with Tavily web search only (baseline)",
        uses_web_search=True,
        allowed_search_domains=_BIOMEDICAL_DOMAINS,
    ),
    "websearch-sonnet45": ArmConfig(
        arm_id="websearch-sonnet45",
        model_key="sonnet_4_5",
        arm_type=ArmType.WEBSEARCH,
        description="Sonnet 4.5 with Tavily web search only (baseline)",
        uses_web_search=True,
        allowed_search_domains=_BIOMEDICAL_DOMAINS,
    ),
    "websearch-opus46": ArmConfig(
        arm_id="websearch-opus46",
        model_key="claude_opus_4_6",
        arm_type=ArmType.WEBSEARCH,
        description="Opus 4.6 with Tavily web search only (baseline)",
        uses_web_search=True,
        allowed_search_domains=_BIOMEDICAL_DOMAINS,
    ),
}


# ------------------------------------------------------------------
# Combined registry
# ------------------------------------------------------------------

ALL_ARMS: dict[str, ArmConfig] = {**PIPELINE_ARMS, **WEBSEARCH_ARMS}
