"""Model registry -- maps logical model keys to provider-specific IDs and pricing.

The env file stores *raw* model strings from each provider website.
The provider prefix (``openai:``, ``anthropic:``, ``groq:``) is attached here
so the rest of the codebase only deals with ``ModelSpec.pydantic_ai_id``.
"""

from __future__ import annotations

from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, computed_field

from src.config.settings import Settings


class Provider(str, Enum):
    """Supported LLM API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class ModelSpec(BaseModel):
    """Immutable descriptor for a single LLM."""

    key: str
    provider: Provider
    raw_model_id: str  # value straight from the .env / provider website
    input_cost_per_mtok: Decimal  # USD per 1M input tokens
    output_cost_per_mtok: Decimal  # USD per 1M output tokens
    context_window: int  # max tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pydantic_ai_id(self) -> str:
        """String accepted by ``pydantic_ai.Agent(model=...)``."""
        return f"{self.provider.value}:{self.raw_model_id}"


def build_registry(settings: Settings) -> dict[str, ModelSpec]:
    """Construct the registry from current Settings (env-driven).

    The registry contains all 4 frontier models used in the v2 experiment:
    - GPT-4.1 and GPT-5.2 (OpenAI)
    - Sonnet 4.5 and Opus 4.6 (Anthropic)

    Legacy Groq models are retained for backward compatibility but are
    not used in the current experimental design.
    """

    return {
        # -- Frontier models (used in the 8-arm experiment) ----------------
        "gpt_4_1": ModelSpec(
            key="gpt_4_1",
            provider=Provider.OPENAI,
            raw_model_id=settings.model_gpt_4_1,
            input_cost_per_mtok=Decimal("2.00"),
            output_cost_per_mtok=Decimal("8.00"),
            context_window=1_047_576,
        ),
        "gpt_5_2": ModelSpec(
            key="gpt_5_2",
            provider=Provider.OPENAI,
            raw_model_id=settings.model_gpt_5_2,
            input_cost_per_mtok=Decimal("2.50"),
            output_cost_per_mtok=Decimal("10.00"),
            context_window=128_000,
        ),
        "sonnet_4_5": ModelSpec(
            key="sonnet_4_5",
            provider=Provider.ANTHROPIC,
            raw_model_id=settings.model_sonnet_4_5,
            input_cost_per_mtok=Decimal("3.00"),
            output_cost_per_mtok=Decimal("15.00"),
            context_window=200_000,
        ),
        "claude_opus_4_6": ModelSpec(
            key="claude_opus_4_6",
            provider=Provider.ANTHROPIC,
            raw_model_id=settings.model_claude_opus_4_6,
            input_cost_per_mtok=Decimal("15.00"),
            output_cost_per_mtok=Decimal("75.00"),
            context_window=200_000,
        ),
        # -- Legacy Groq models (kept for reference) -----------------------
        "gpt_oss": ModelSpec(
            key="gpt_oss",
            provider=Provider.GROQ,
            raw_model_id=settings.model_gpt_oss,
            input_cost_per_mtok=Decimal("0.30"),
            output_cost_per_mtok=Decimal("0.80"),
            context_window=131_072,
        ),
        "llama_4": ModelSpec(
            key="llama_4",
            provider=Provider.GROQ,
            raw_model_id=settings.model_llama_4,
            input_cost_per_mtok=Decimal("0.20"),
            output_cost_per_mtok=Decimal("0.60"),
            context_window=131_072,
        ),
        "qwen3": ModelSpec(
            key="qwen3",
            provider=Provider.GROQ,
            raw_model_id=settings.model_qwen3,
            input_cost_per_mtok=Decimal("0.20"),
            output_cost_per_mtok=Decimal("0.60"),
            context_window=131_072,
        ),
        "kimi_k2": ModelSpec(
            key="kimi_k2",
            provider=Provider.GROQ,
            raw_model_id=settings.model_kimi_k2,
            input_cost_per_mtok=Decimal("0.20"),
            output_cost_per_mtok=Decimal("0.60"),
            context_window=131_072,
        ),
    }


# Convenience: module-level singleton built from default Settings.
# Import as ``from src.config.models import MODEL_REGISTRY``.
MODEL_REGISTRY: dict[str, ModelSpec] = build_registry(Settings())
