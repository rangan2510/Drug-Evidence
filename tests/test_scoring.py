"""Tests for experiment arm configs and the pipeline runner.

Uses PydanticAI ``TestModel`` -- no live LLM API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.models.test import TestModel

from src.agents.deps import EvidenceDeps
from src.config.models import ModelSpec, build_registry
from src.config.settings import Settings
from src.experiment.arms import (
    ALL_ARMS,
    BASELINE_ARMS,
    PIPELINE_ARMS,
    ArmConfig,
    ArmType,
)
from src.experiment.runner import ArmResult, run_all_pipeline_arms, run_pipeline_arm
from src.schemas.prediction import DrugDiseasePrediction


# ======================================================================
# Helpers
# ======================================================================

def _make_settings() -> Settings:
    return Settings(_env_file=None, entrez_email="test@example.com")


def _make_prediction_args() -> dict[str, Any]:
    """Dict that ``TestModel.custom_output_args`` uses for ``DrugDiseasePrediction``."""
    return {
        "drug_name": "aspirin",
        "drug_chembl_id": "CHEMBL25",
        "associations": [
            {
                "disease_name": "colorectal cancer",
                "disease_id": "EFO_0000365",
                "predicted": True,
                "confidence": 0.82,
                "evidence_chains": [
                    {
                        "edges": [
                            {
                                "source_entity": "aspirin",
                                "target_entity": "PTGS2",
                                "relationship": "inhibits",
                                "evidence_snippet": "Aspirin inhibits COX-2",
                                "pmid": "12345678",
                            },
                        ],
                        "summary": "Aspirin inhibits COX-2, linked to colorectal cancer",
                        "confidence": 0.82,
                    }
                ],
            }
        ],
        "reasoning": "Found strong COX-2 inhibition evidence linking to colorectal cancer.",
    }


def _mock_deps(
    drug_name: str = "aspirin",
    chembl_id: str | None = "CHEMBL25",
    pubchem_cid: int | None = 2244,
) -> EvidenceDeps:
    """Create ``EvidenceDeps`` with mocked services."""
    settings = _make_settings()
    vector_store = MagicMock()
    aggregator = MagicMock()
    aggregator._dgidb = AsyncMock()
    aggregator._opentargets = AsyncMock()
    aggregator._pubchem = AsyncMock()
    aggregator._chembl = AsyncMock()
    aggregator._pharmgkb = AsyncMock()
    aggregator._pubmed = AsyncMock()

    return EvidenceDeps(
        settings=settings,
        vector_store=vector_store,
        aggregator=aggregator,
        drug_name=drug_name,
        chembl_id=chembl_id,
        pubchem_cid=pubchem_cid,
    )


def _make_test_registry() -> dict[str, ModelSpec]:
    """Build the real MODEL_REGISTRY from default Settings (no API calls)."""
    return build_registry(_make_settings())


# ======================================================================
# Arm config tests
# ======================================================================

class TestArmConfig:
    """Validate arm definitions and model key resolution."""

    def test_total_arm_count(self) -> None:
        assert len(ALL_ARMS) == 8

    def test_pipeline_arm_count(self) -> None:
        assert len(PIPELINE_ARMS) == 4

    def test_baseline_arm_count(self) -> None:
        assert len(BASELINE_ARMS) == 4

    def test_all_arms_is_union(self) -> None:
        assert set(ALL_ARMS) == set(PIPELINE_ARMS) | set(BASELINE_ARMS)

    def test_pipeline_arms_are_pipeline_type(self) -> None:
        for arm in PIPELINE_ARMS.values():
            assert arm.arm_type == ArmType.PIPELINE

    def test_baseline_arms_are_baseline_type(self) -> None:
        for arm in BASELINE_ARMS.values():
            assert arm.arm_type == ArmType.BASELINE

    def test_all_model_keys_exist_in_registry(self) -> None:
        registry = _make_test_registry()
        for arm in ALL_ARMS.values():
            assert arm.model_key in registry, (
                f"Arm '{arm.arm_id}' references model_key '{arm.model_key}' "
                f"not found in MODEL_REGISTRY"
            )

    def test_resolve_model_id_returns_valid_string(self) -> None:
        registry = _make_test_registry()
        for arm in ALL_ARMS.values():
            model_id = arm.resolve_model_id(registry)
            assert ":" in model_id, f"Expected 'provider:model' format, got '{model_id}'"

    def test_resolve_model_id_missing_key_raises(self) -> None:
        arm = ArmConfig(
            arm_id="test-bad",
            model_key="nonexistent_model",
            arm_type=ArmType.PIPELINE,
            description="Should fail",
        )
        with pytest.raises(KeyError):
            arm.resolve_model_id(_make_test_registry())

    def test_pipeline_arms_no_web_search(self) -> None:
        for arm in PIPELINE_ARMS.values():
            assert arm.uses_web_search is False

    def test_baseline_search_arms_have_domains(self) -> None:
        for arm_id in ("baseline-gpt5-search", "baseline-claude-search"):
            arm = BASELINE_ARMS[arm_id]
            assert arm.uses_web_search is True
            assert len(arm.allowed_search_domains) > 0

    def test_baseline_nosearch_arm(self) -> None:
        arm = BASELINE_ARMS["baseline-gpt5-nosearch"]
        assert arm.uses_web_search is False
        assert len(arm.allowed_search_domains) == 0

    @pytest.mark.parametrize(
        "arm_id,expected_provider",
        [
            ("pipeline-gpt-oss", "groq"),
            ("pipeline-llama4", "groq"),
            ("pipeline-qwen3", "groq"),
            ("pipeline-kimi-k2", "groq"),
            ("baseline-gpt5-nosearch", "openai"),
            ("baseline-gpt5-search", "openai"),
            ("baseline-claude-search", "anthropic"),
        ],
    )
    def test_arm_resolves_to_correct_provider(
        self, arm_id: str, expected_provider: str
    ) -> None:
        arm = ALL_ARMS[arm_id]
        model_id = arm.resolve_model_id(_make_test_registry())
        provider_prefix = model_id.split(":")[0]
        assert provider_prefix == expected_provider


# ======================================================================
# ArmResult model tests
# ======================================================================

class TestArmResult:
    """Validate ArmResult Pydantic model."""

    def test_success_result(self) -> None:
        prediction = DrugDiseasePrediction(**_make_prediction_args())
        result = ArmResult(
            arm_id="pipeline-gpt-oss",
            drug_name="aspirin",
            model_id="groq:openai/gpt-oss-120b",
            prediction=prediction,
            usage={"input_tokens": 500, "output_tokens": 200, "requests": 2},
            wall_clock_seconds=3.5,
        )
        assert result.error is None
        assert result.prediction is not None
        assert result.prediction.drug_name == "aspirin"

    def test_error_result(self) -> None:
        result = ArmResult(
            arm_id="pipeline-gpt-oss",
            drug_name="aspirin",
            model_id="groq:openai/gpt-oss-120b",
            error="UnexpectedModelBehavior: invalid JSON",
            wall_clock_seconds=1.2,
        )
        assert result.prediction is None
        assert result.error is not None

    def test_serialization_roundtrip(self) -> None:
        prediction = DrugDiseasePrediction(**_make_prediction_args())
        result = ArmResult(
            arm_id="pipeline-qwen3",
            drug_name="metformin",
            model_id="groq:qwen/qwen3-32b",
            prediction=prediction,
            usage={"input_tokens": 100, "output_tokens": 50},
            wall_clock_seconds=2.0,
        )
        data = result.model_dump()
        restored = ArmResult.model_validate(data)
        assert restored.arm_id == result.arm_id
        assert restored.prediction is not None
        assert restored.prediction.drug_name == prediction.drug_name


# ======================================================================
# run_pipeline_arm tests
# ======================================================================

class TestRunPipelineArm:
    """Test the single-arm runner with TestModel."""

    @pytest.mark.asyncio
    async def test_returns_valid_arm_result(self) -> None:
        """Basic run should produce a valid ArmResult with prediction."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )

        # Create a TestModel-based arm config
        arm = ArmConfig(
            arm_id="test-arm",
            model_key="gpt_oss",  # arbitrary, won't actually be used
            arm_type=ArmType.PIPELINE,
            description="Test arm",
        )

        deps = _mock_deps()

        # We override model resolution by passing model directly to the agent.
        # Since run_pipeline_arm builds via build_evidence_agent_with_context,
        # we patch it to accept our TestModel.
        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        assert isinstance(result, ArmResult)
        assert result.arm_id == "test-arm"
        assert result.drug_name == "aspirin"
        assert result.error is None
        assert result.prediction is not None
        assert isinstance(result.prediction, DrugDiseasePrediction)
        assert result.wall_clock_seconds > 0

    @pytest.mark.asyncio
    async def test_prediction_matches_expected_drug(self) -> None:
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        arm = ArmConfig(
            arm_id="test-arm",
            model_key="gpt_oss",
            arm_type=ArmType.PIPELINE,
            description="Test arm",
        )
        deps = _mock_deps(drug_name="aspirin", chembl_id="CHEMBL25")

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        assert result.prediction is not None
        assert result.prediction.drug_chembl_id == "CHEMBL25"
        assert len(result.prediction.associations) == 1

    @pytest.mark.asyncio
    async def test_usage_is_populated(self) -> None:
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        arm = ArmConfig(
            arm_id="test-arm",
            model_key="gpt_oss",
            arm_type=ArmType.PIPELINE,
            description="Test arm",
        )
        deps = _mock_deps()

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        assert isinstance(result.usage, dict)
        # TestModel reports some token usage
        assert "input_tokens" in result.usage
        assert "output_tokens" in result.usage

    @pytest.mark.asyncio
    async def test_model_id_is_resolved(self) -> None:
        """ArmResult.model_id should be the resolved pydantic_ai_id."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        arm = ArmConfig(
            arm_id="test-arm",
            model_key="llama_4",
            arm_type=ArmType.PIPELINE,
            description="Test arm",
        )
        deps = _mock_deps()

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        # Should resolve to "groq:meta-llama/llama-4-maverick-17b-128e-instruct"
        assert result.model_id.startswith("groq:")
        assert "llama-4" in result.model_id


# ======================================================================
# run_pipeline_arm error handling
# ======================================================================

class TestRunPipelineArmError:
    """Verify errors are captured in ArmResult, not raised."""

    @pytest.mark.asyncio
    async def test_model_error_captured(self) -> None:
        """If the agent run raises, ArmResult.error should be populated."""
        arm = ArmConfig(
            arm_id="test-error",
            model_key="gpt_oss",
            arm_type=ArmType.PIPELINE,
            description="Error test",
        )
        deps = _mock_deps()

        # Patch build_evidence_agent_with_context to raise on run
        async def _failing_run(*args, **kwargs):
            raise RuntimeError("Simulated model failure")

        mock_agent = MagicMock()
        mock_agent.run = _failing_run

        from unittest.mock import patch

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            return_value=mock_agent,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        assert result.error is not None
        assert "Simulated model failure" in result.error
        assert result.prediction is None
        assert result.wall_clock_seconds > 0

    @pytest.mark.asyncio
    async def test_key_error_on_bad_model_key(self) -> None:
        """If model_key is not in registry, error should be captured."""
        arm = ArmConfig(
            arm_id="test-bad-key",
            model_key="nonexistent",
            arm_type=ArmType.PIPELINE,
            description="Bad key test",
        )
        deps = _mock_deps()

        result = await run_pipeline_arm(arm=arm, deps=deps)
        assert result.error is not None
        assert "KeyError" in result.error


# ======================================================================
# Parametrised per-pipeline-arm test
# ======================================================================

class TestPerPipelineArm:
    """Run each of the 4 pipeline arms through the runner with TestModel."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arm_id", list(PIPELINE_ARMS.keys()))
    async def test_pipeline_arm_produces_valid_output(self, arm_id: str) -> None:
        arm = PIPELINE_ARMS[arm_id]
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        deps = _mock_deps()

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            result = await run_pipeline_arm(arm=arm, deps=deps)

        assert result.error is None, f"Arm '{arm_id}' failed: {result.error}"
        assert result.prediction is not None
        assert isinstance(result.prediction, DrugDiseasePrediction)
        assert result.arm_id == arm_id
        # Verify model_id resolves to the expected provider
        expected_provider = "groq"
        assert result.model_id.startswith(f"{expected_provider}:")


# ======================================================================
# run_all_pipeline_arms tests
# ======================================================================

class TestRunAllPipelineArms:
    """Test the multi-arm convenience runner."""

    @pytest.mark.asyncio
    async def test_runs_all_four_pipeline_arms(self) -> None:
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        deps = _mock_deps()

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            results = await run_all_pipeline_arms(deps=deps)

        assert len(results) == 4
        arm_ids = {r.arm_id for r in results}
        assert arm_ids == set(PIPELINE_ARMS.keys())

    @pytest.mark.asyncio
    async def test_skips_baseline_arms(self) -> None:
        """If baseline arms are passed, they should be skipped."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        deps = _mock_deps()
        baseline_arm = BASELINE_ARMS["baseline-gpt5-nosearch"]

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            results = await run_all_pipeline_arms(
                deps=deps,
                arms=[baseline_arm],
            )

        # Baseline arm should be skipped
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_custom_arm_subset(self) -> None:
        """Running with a subset of pipeline arms should work."""
        model = TestModel(
            custom_output_args=_make_prediction_args(),
            call_tools=[],
        )
        deps = _mock_deps()
        subset = [PIPELINE_ARMS["pipeline-gpt-oss"], PIPELINE_ARMS["pipeline-qwen3"]]

        from unittest.mock import patch

        from src.agents.evidence_agent import build_evidence_agent_with_context

        original_build = build_evidence_agent_with_context

        def _build_with_test_model(model_id, **kwargs):
            return original_build(model, **kwargs)

        with patch(
            "src.experiment.runner.build_evidence_agent_with_context",
            side_effect=_build_with_test_model,
        ):
            results = await run_all_pipeline_arms(deps=deps, arms=subset)

        assert len(results) == 2
        arm_ids = {r.arm_id for r in results}
        assert arm_ids == {"pipeline-gpt-oss", "pipeline-qwen3"}


# ======================================================================
# Prompt construction tests
# ======================================================================

class TestPromptConstruction:
    """Verify user prompt is built correctly from deps."""

    def test_default_prompt_with_chembl(self) -> None:
        from src.experiment.runner import _build_prompt

        deps = _mock_deps(drug_name="metformin", chembl_id="CHEMBL1431")
        prompt = _build_prompt(deps, override=None)
        assert "metformin" in prompt
        assert "CHEMBL1431" in prompt
        assert "predict disease associations" in prompt

    def test_default_prompt_without_chembl(self) -> None:
        from src.experiment.runner import _build_prompt

        deps = _mock_deps(drug_name="aspirin", chembl_id=None)
        prompt = _build_prompt(deps, override=None)
        assert "aspirin" in prompt
        assert "ChEMBL ID" not in prompt

    def test_custom_prompt_override(self) -> None:
        from src.experiment.runner import _build_prompt

        deps = _mock_deps()
        custom = "My custom analysis prompt"
        prompt = _build_prompt(deps, override=custom)
        assert prompt == custom
