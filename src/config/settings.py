"""Centralised settings loaded from environment / .env file.

Uses Pydantic BaseSettings so every value can be overridden via env vars.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- API keys ----------------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # -- External service config -------------------------------------------
    entrez_email: str = ""
    qdrant_url: str = "http://localhost:6333"  # Qdrant running in Docker
    qdrant_on_disk_path: str = ".qdrant_data"  # On-disk fallback path
    qdrant_in_memory: bool = False
    qdrant_fallback_to_in_memory: bool = True
    qdrant_healthcheck_path: str = "/healthz"

    # -- MongoDB (staged evidence store) -----------------------------------
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "drug_discovery"
    staged_points_dir: str = "data/phase0_points"
    use_staged_vectors: bool = False

    # -- Model IDs (raw strings from provider websites) --------------------
    model_gpt_4_1: str = "gpt-4.1-2025-04-14"
    model_gpt_5_2: str = "gpt-5.2-2025-12-11"
    model_sonnet_4_5: str = "claude-sonnet-4-5-20250929"
    model_claude_opus_4_6: str = "claude-opus-4-6"
    # Legacy Groq models (kept for reference, no longer used in experiments)
    model_gpt_oss: str = "openai/gpt-oss-120b"
    model_llama_4: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    model_qwen3: str = "qwen/qwen3-32b"
    model_kimi_k2: str = "moonshotai/kimi-k2-instruct-0905"

    # -- Experiment knobs --------------------------------------------------
    target_drugs: int = Field(default=200, description="Number of drugs to evaluate")
    max_prescan: int = Field(default=600, description="CTD candidates to pre-scan")
    min_associations: int = Field(
        default=10, description="Min disease associations to include a drug"
    )
    max_associations: int = Field(
        default=250, description="Max disease associations to include a drug"
    )
    groq_workers: int = Field(
        default=1, ge=1, description="Max concurrent Groq arm workers"
    )
    openai_workers: int = Field(
        default=1, ge=1, description="Max concurrent OpenAI arm workers"
    )
    anthropic_workers: int = Field(
        default=1, ge=1, description="Max concurrent Anthropic arm workers"
    )
    pipeline_request_limit: int = Field(
        default=200,
        ge=20,
        description="PydanticAI request limit for pipeline arms",
    )

    # -- HTTP cache --------------------------------------------------------
    http_cache_dir: str = Field(
        default=".cache",
        description="Directory for the hishel SQLite HTTP cache",
    )
    http_cache_ttl: int = Field(
        default=86400,
        description="Default cache TTL in seconds (86 400 = 24 h)",
    )

    # -- Reactome bulk pathway mapping -------------------------------------
    reactome_download_base: str = "https://reactome.org/download/current"
    reactome_cache_dir: str = "data/reactome"

    # -- Vector store ------------------------------------------------------
    qdrant_collection: str = "evidence_v2"
    dense_model_query: str = "ncbi/MedCPT-Query-Encoder"
    dense_model_doc: str = "ncbi/MedCPT-Article-Encoder"
    dense_vector_size: int = 768
    sparse_model: str = "naver/splade-cocondenser-ensembledistil"
    chunk_size: int = Field(
        default=512, description="Token chunk size for evidence docs"
    )
    chunk_overlap: int = Field(default=50, description="Token overlap between chunks")
    rrf_k: int = Field(default=60, description="RRF fusion constant")
