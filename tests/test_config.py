"""Smoke tests for Settings and ModelRegistry."""

from __future__ import annotations

from src.config.settings import Settings
from src.config.models import ModelSpec, Provider, build_registry, MODEL_REGISTRY


def test_settings_loads_defaults() -> None:
    s = Settings()
    assert s.target_drugs == 200
    assert s.max_prescan == 600
    assert s.qdrant_collection == "evidence_v2"


def test_settings_reads_model_ids_from_env() -> None:
    s = Settings()
    assert s.model_gpt_5_2 == "gpt-5.2-2025-12-11"
    assert s.model_claude_opus_4_6 == "claude-opus-4-6"
    assert "llama-4-maverick" in s.model_llama_4


def test_model_registry_has_six_models() -> None:
    assert len(MODEL_REGISTRY) == 6
    expected_keys = {"gpt_5_2", "claude_opus_4_6", "gpt_oss", "llama_4", "qwen3", "kimi_k2"}
    assert set(MODEL_REGISTRY.keys()) == expected_keys


def test_pydantic_ai_id_format() -> None:
    spec = MODEL_REGISTRY["gpt_5_2"]
    assert spec.pydantic_ai_id == "openai:gpt-5.2-2025-12-11"

    spec = MODEL_REGISTRY["llama_4"]
    assert spec.pydantic_ai_id.startswith("groq:")

    spec = MODEL_REGISTRY["claude_opus_4_6"]
    assert spec.pydantic_ai_id == "anthropic:claude-opus-4-6"


def test_provider_assignment() -> None:
    assert MODEL_REGISTRY["gpt_5_2"].provider == Provider.OPENAI
    assert MODEL_REGISTRY["claude_opus_4_6"].provider == Provider.ANTHROPIC
    for key in ("gpt_oss", "llama_4", "qwen3", "kimi_k2"):
        assert MODEL_REGISTRY[key].provider == Provider.GROQ


def test_build_registry_uses_settings_overrides(monkeypatch: object) -> None:
    """Verify that env-var overrides flow through to the registry."""
    import os

    os.environ["MODEL_QWEN3"] = "qwen/qwen3-72b"
    try:
        s = Settings()
        reg = build_registry(s)
        assert reg["qwen3"].raw_model_id == "qwen/qwen3-72b"
        assert reg["qwen3"].pydantic_ai_id == "groq:qwen/qwen3-72b"
    finally:
        os.environ.pop("MODEL_QWEN3", None)
