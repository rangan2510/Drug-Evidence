"""Tests for frontier baseline agents (PydanticAI + Tavily) and runner dispatch.

Covers:
- Shared prompt builder
- BaselineResult dataclass
- System prompt sanity
- Unified agent factory (build_baseline_agent)
- Unified runner (run_baseline) via PydanticAI TestModel
- Runner dispatch (run_baseline_arm, run_all_baseline_arms)
- Arm config integration (BaselineProvider)
- LIVE integration tests (gated behind @pytest.mark.live)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from src.agents.baseline_agent import (
    BASELINE_SYSTEM_PROMPT,
    BaselineResult,
    _TavilyDeps,
    build_baseline_agent,
    build_baseline_prompt,
    extract_prediction,
    run_baseline,
)
from src.config.models import ModelSpec, build_registry
from src.config.settings import Settings
from src.experiment.arms import (
    BASELINE_ARMS,
    ArmConfig,
    ArmType,
    BaselineProvider,
)
from src.experiment.runner import ArmResult, run_all_baseline_arms, run_baseline_arm
from src.schemas.prediction import (
    DrugDiseasePrediction,
    EdgeType,
    EvidenceChain,
    MechanisticEdge,
    ScoredAssociation,
)


# ======================================================================
# Helpers / fixtures
# ======================================================================

def _make_settings(**overrides: Any) -> Settings:
    """Settings with dummy API keys for testing."""
    defaults: dict[str, Any] = {
        "_env_file": None,
        "entrez_email": "test@example.com",
        "openai_api_key": "sk-test-openai",
        "anthropic_api_key": "sk-test-anthropic",
        "groq_api_key": "sk-test-groq",
        "tavily_api_key": "tvly-test-tavily",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_prediction(**overrides: Any) -> DrugDiseasePrediction:
    """Minimal valid prediction for test fixtures."""
    defaults: dict[str, Any] = {
        "drug_name": "aspirin",
        "drug_chembl_id": "CHEMBL25",
        "associations": [
            ScoredAssociation(
                disease_name="colorectal cancer",
                disease_id="EFO_0000365",
                predicted=True,
                confidence=0.82,
                evidence_chains=[
                    EvidenceChain(
                        edges=[
                            MechanisticEdge(
                                source_entity="aspirin",
                                target_entity="PTGS2",
                                relationship=EdgeType.INHIBITS,
                                evidence_snippet="Aspirin inhibits COX-2",
                                pmid="12345678",
                            ),
                        ],
                        summary="COX-2 inhibition linked to colorectal cancer",
                        confidence=0.82,
                    ),
                ],
            ),
        ],
        "reasoning": "Found strong COX-2 inhibition evidence.",
    }
    defaults.update(overrides)
    return DrugDiseasePrediction(**defaults)


def _prediction_dict() -> dict[str, Any]:
    """Dict form suitable for TestModel.custom_output_args or JSON parsing."""
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
                        "summary": "COX-2 inhibition linked to colorectal cancer",
                        "confidence": 0.82,
                    }
                ],
            }
        ],
        "reasoning": "Found strong COX-2 inhibition evidence.",
    }


def _make_test_registry() -> dict[str, ModelSpec]:
    """Build the real MODEL_REGISTRY from default Settings."""
    return build_registry(_make_settings())


# ======================================================================
# 1. Prompt builder
# ======================================================================

class TestBuildBaselinePrompt:
    """Tests for build_baseline_prompt()."""

    def test_basic_drug_name(self) -> None:
        result = build_baseline_prompt("aspirin")
        assert "aspirin" in result
        assert "ChEMBL" not in result

    def test_with_chembl_id(self) -> None:
        result = build_baseline_prompt("aspirin", chembl_id="CHEMBL25")
        assert "aspirin" in result
        assert "CHEMBL25" in result

    def test_override_ignores_drug(self) -> None:
        result = build_baseline_prompt("aspirin", override="custom prompt")
        assert result == "custom prompt"
        assert "aspirin" not in result

    def test_override_with_chembl_still_uses_override(self) -> None:
        result = build_baseline_prompt("aspirin", "CHEMBL25", "custom")
        assert result == "custom"

    def test_none_chembl_id(self) -> None:
        result = build_baseline_prompt("ibuprofen", chembl_id=None)
        assert "ibuprofen" in result
        assert "ChEMBL" not in result


# ======================================================================
# 2. BaselineResult
# ======================================================================

class TestBaselineResult:
    """Tests for the BaselineResult dataclass."""

    def test_defaults(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred)
        assert br.input_tokens == 0
        assert br.output_tokens == 0
        assert br.total_tokens == 0
        assert br.web_search_requests == 0
        assert br.raw_model_id == ""
        assert br.extra == {}

    def test_all_fields(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(
            prediction=pred,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            web_search_requests=3,
            raw_model_id="gpt-5.2-2025-12-11",
            extra={"foo": "bar"},
        )
        assert br.input_tokens == 100
        assert br.output_tokens == 200
        assert br.total_tokens == 300
        assert br.web_search_requests == 3
        assert br.raw_model_id == "gpt-5.2-2025-12-11"

    def test_serialisable(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=10, output_tokens=20)
        d = asdict(br)
        assert d["input_tokens"] == 10
        assert "prediction" in d


# ======================================================================
# 3. System prompt
# ======================================================================

class TestSystemPrompt:
    """Sanity checks for the shared baseline system prompt."""

    def test_no_emojis(self) -> None:
        import re

        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff\U00002702-\U000027b0]"
        )
        assert not emoji_pattern.search(BASELINE_SYSTEM_PROMPT)

    def test_mentions_evidence_chains(self) -> None:
        assert "evidence" in BASELINE_SYSTEM_PROMPT.lower()
        assert "chain" in BASELINE_SYSTEM_PROMPT.lower()

    def test_mentions_pmid(self) -> None:
        assert "pmid" in BASELINE_SYSTEM_PROMPT.lower()


# ======================================================================
# 4. Agent factory (build_baseline_agent)
# ======================================================================

class TestBuildBaselineAgent:
    """Tests for the unified build_baseline_agent factory.

    Uses monkeypatch to set fake API keys so PydanticAI can initialise
    the provider objects without hitting the network.
    """

    @pytest.fixture(autouse=True)
    def _set_fake_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")

    def test_agent_output_type(self) -> None:
        agent = build_baseline_agent(
            "openai:gpt-5.2-2025-12-11", uses_web_search=False,
        )
        assert agent.output_type is str

    def test_agent_has_web_search_tool(self) -> None:
        agent = build_baseline_agent(
            "openai:gpt-5.2-2025-12-11", uses_web_search=True,
        )
        tool_names = list(agent._function_toolset.tools.keys())
        assert "web_search" in tool_names

    def test_agent_no_tools_when_search_disabled(self) -> None:
        agent = build_baseline_agent(
            "openai:gpt-5.2-2025-12-11", uses_web_search=False,
        )
        assert len(agent._function_toolset.tools) == 0

    def test_system_prompt_set(self) -> None:
        agent = build_baseline_agent("openai:gpt-5.2-2025-12-11")
        prompts = agent._instructions
        prompt_text = " ".join(prompts) if isinstance(prompts, (list, tuple)) else prompts
        assert "biomedical" in prompt_text.lower()

    def test_custom_retries(self) -> None:
        agent = build_baseline_agent(
            "openai:gpt-5.2-2025-12-11", retries=5,
        )
        assert agent._max_result_retries == 5

    def test_anthropic_model(self) -> None:
        agent = build_baseline_agent(
            "anthropic:claude-opus-4-6", uses_web_search=True,
        )
        assert agent.output_type is str
        tool_names = list(agent._function_toolset.tools.keys())
        assert "web_search" in tool_names


# ======================================================================
# 5. Unified runner (run_baseline) via TestModel
# ======================================================================

class TestRunBaseline:
    """Tests for run_baseline using PydanticAI TestModel.

    The runner creates a PydanticAI agent with ``output_type=str``,
    gets free-form text, and delegates to ``extract_prediction()``
    for structured extraction.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_success_with_test_model(self) -> None:
        """Verify round-trip via TestModel (no real LLM)."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ),
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="Aspirin inhibits COX-2."),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            result = await run_baseline(
                "test prompt",
                _make_settings(),
                pydantic_ai_model_id="openai:gpt-5.2-2025-12-11",
                uses_web_search=True,
                allowed_domains=["pubmed.ncbi.nlm.nih.gov"],
            )

        assert isinstance(result, BaselineResult)
        assert result.prediction.drug_name == "aspirin"
        assert result.raw_text != ""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_extract_prediction_called_with_openai_model(self) -> None:
        """Verify Instructor extraction uses the correct OpenAI model string."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ) as mock_extract,
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="Aspirin inhibits COX-2."),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            await run_baseline(
                "test prompt",
                _make_settings(openai_api_key="sk-test-key"),
                pydantic_ai_model_id="openai:gpt-5.2-2025-12-11",
                uses_web_search=False,
            )

        mock_extract.assert_awaited_once()
        call_kwargs = mock_extract.call_args
        assert call_kwargs[1]["provider_model"] == "openai/gpt-5.2-2025-12-11"
        assert call_kwargs[1]["api_key"] == "sk-test-key"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_extract_prediction_called_with_anthropic_model(self) -> None:
        """Verify Instructor extraction uses the correct Anthropic model string."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ) as mock_extract,
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="Aspirin inhibits COX-2."),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            await run_baseline(
                "test prompt",
                _make_settings(anthropic_api_key="sk-anthro-test"),
                pydantic_ai_model_id="anthropic:claude-opus-4-6",
                uses_web_search=False,
            )

        mock_extract.assert_awaited_once()
        call_kwargs = mock_extract.call_args
        assert call_kwargs[1]["provider_model"] == "anthropic/claude-opus-4-6"
        assert call_kwargs[1]["api_key"] == "sk-anthro-test"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_tavily_deps_populated(self) -> None:
        """Verify _TavilyDeps are set correctly from settings."""
        pred = _make_prediction()
        captured_deps: list[_TavilyDeps] = []

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ),
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="Aspirin inhibits COX-2."),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            # Patch the agent.run to capture deps
            original_run = test_agent.run

            async def capturing_run(prompt, *, deps=None, **kwargs):
                if deps:
                    captured_deps.append(deps)
                return await original_run(prompt, deps=deps, **kwargs)

            test_agent.run = capturing_run

            settings = _make_settings(tavily_api_key="tvly-captured")
            await run_baseline(
                "test prompt",
                settings,
                pydantic_ai_model_id="openai:gpt-5.2-2025-12-11",
                uses_web_search=True,
                allowed_domains=["opentargets.org"],
            )

        assert len(captured_deps) == 1
        assert captured_deps[0].tavily_api_key == "tvly-captured"
        assert captured_deps[0].allowed_domains == ["opentargets.org"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_raw_text_captured(self) -> None:
        """Verify the free-form text from the LLM is stored in raw_text."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ),
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="Detailed aspirin analysis..."),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            result = await run_baseline(
                "test prompt",
                _make_settings(),
                pydantic_ai_model_id="openai:gpt-5.2-2025-12-11",
                uses_web_search=False,
            )

        assert "Detailed aspirin analysis..." in result.raw_text

    @pytest.mark.asyncio(loop_scope="session")
    async def test_model_id_stored(self) -> None:
        """Verify raw_model_id records the PydanticAI model string."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ),
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="text"),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            result = await run_baseline(
                "test prompt",
                _make_settings(),
                pydantic_ai_model_id="anthropic:claude-opus-4-6",
                uses_web_search=False,
            )

        assert result.raw_model_id == "anthropic:claude-opus-4-6"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_search_does_not_register_tool(self) -> None:
        """When uses_web_search=False, build_baseline_agent should get that flag."""
        pred = _make_prediction()

        with (
            patch(
                "src.agents.baseline_agent.build_baseline_agent"
            ) as mock_factory,
            patch(
                "src.agents.baseline_agent.extract_prediction",
                new_callable=AsyncMock,
                return_value=pred,
            ),
        ):
            test_agent = Agent(
                model=TestModel(custom_output_text="text"),
                output_type=str,
                deps_type=_TavilyDeps,
            )
            mock_factory.return_value = test_agent

            await run_baseline(
                "test prompt",
                _make_settings(),
                pydantic_ai_model_id="openai:gpt-5.2-2025-12-11",
                uses_web_search=False,
            )

        mock_factory.assert_called_once_with(
            model_id="openai:gpt-5.2-2025-12-11",
            uses_web_search=False,
            retries=2,
        )


# ======================================================================
# 6. Arm config integration
# ======================================================================

class TestBaselineArmConfigs:
    """Tests for baseline arm configurations."""

    def test_all_baseline_arms_have_provider(self) -> None:
        for arm_id, arm in BASELINE_ARMS.items():
            assert arm.baseline_provider is not None, (
                f"Arm '{arm_id}' is missing baseline_provider"
            )

    def test_arm_types_are_baseline(self) -> None:
        for arm in BASELINE_ARMS.values():
            assert arm.arm_type == ArmType.BASELINE

    def test_four_baseline_arms(self) -> None:
        assert len(BASELINE_ARMS) == 4

    def test_gpt5_nosearch_is_openai(self) -> None:
        arm = BASELINE_ARMS["baseline-gpt5-nosearch"]
        assert arm.baseline_provider == BaselineProvider.OPENAI
        assert arm.uses_web_search is False

    def test_gpt5_search_is_openai(self) -> None:
        arm = BASELINE_ARMS["baseline-gpt5-search"]
        assert arm.baseline_provider == BaselineProvider.OPENAI
        assert arm.uses_web_search is True
        assert len(arm.allowed_search_domains) > 0

    def test_claude_nosearch_is_anthropic(self) -> None:
        arm = BASELINE_ARMS["baseline-claude-nosearch"]
        assert arm.baseline_provider == BaselineProvider.ANTHROPIC
        assert arm.uses_web_search is False

    def test_claude_search_is_anthropic(self) -> None:
        arm = BASELINE_ARMS["baseline-claude-search"]
        assert arm.baseline_provider == BaselineProvider.ANTHROPIC
        assert arm.uses_web_search is True
        assert len(arm.allowed_search_domains) > 0

    def test_pipeline_arms_have_no_provider(self) -> None:
        from src.experiment.arms import PIPELINE_ARMS

        for arm in PIPELINE_ARMS.values():
            assert arm.baseline_provider is None

    def test_baseline_provider_enum_values(self) -> None:
        assert BaselineProvider.OPENAI.value == "openai"
        assert BaselineProvider.ANTHROPIC.value == "anthropic"

    def test_resolve_model_id(self) -> None:
        reg = _make_test_registry()
        arm = BASELINE_ARMS["baseline-gpt5-search"]
        model_id = arm.resolve_model_id(reg)
        assert "gpt-5.2" in model_id


# ======================================================================
# 7. Runner dispatch -- run_baseline_arm
# ======================================================================

class TestRunBaselineArm:
    """Tests for run_baseline_arm dispatch logic."""

    def _openai_arm(self) -> ArmConfig:
        return BASELINE_ARMS["baseline-gpt5-search"]

    def _anthropic_arm(self) -> ArmConfig:
        return BASELINE_ARMS["baseline-claude-search"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dispatches_openai_arm(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(
            prediction=pred, input_tokens=100, output_tokens=200, total_tokens=300
        )

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ) as mock_run:
            result = await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                chembl_id="CHEMBL25",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        mock_run.assert_awaited_once()
        assert isinstance(result, ArmResult)
        assert result.prediction is not None
        assert result.prediction.drug_name == "aspirin"
        assert result.error is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dispatches_anthropic_arm(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(
            prediction=pred, input_tokens=50, output_tokens=100, total_tokens=150
        )

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            result = await run_baseline_arm(
                arm=self._anthropic_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert isinstance(result, ArmResult)
        assert result.error is None
        assert result.usage["input_tokens"] == 50

    @pytest.mark.asyncio(loop_scope="session")
    async def test_captures_error_on_failure(self) -> None:
        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "API down" in result.error
        assert result.prediction is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_uses_build_baseline_prompt(self) -> None:
        """Verify the prompt is built by build_baseline_prompt."""
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ) as mock_run:
            await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="metformin",
                chembl_id="CHEMBL1431",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        # First positional arg is the prompt
        prompt_arg = mock_run.call_args[0][0]
        assert "metformin" in prompt_arg
        assert "CHEMBL1431" in prompt_arg

    @pytest.mark.asyncio(loop_scope="session")
    async def test_prompt_override(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ) as mock_run:
            await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
                prompt="custom override prompt",
            )

        prompt_arg = mock_run.call_args[0][0]
        assert prompt_arg == "custom override prompt"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_wall_clock_recorded(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            result = await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert result.wall_clock_seconds >= 0.0

    @pytest.mark.asyncio(loop_scope="session")
    async def test_usage_dict_has_expected_keys(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(
            prediction=pred,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            web_search_requests=2,
        )

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            result = await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 200
        assert result.usage["total_tokens"] == 300
        assert result.usage["web_search_requests"] == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_bad_model_key_returns_error(self) -> None:
        arm = ArmConfig(
            arm_id="bad-key",
            model_key="nonexistent_model",
            arm_type=ArmType.BASELINE,
            description="Bad model key",
            baseline_provider=BaselineProvider.OPENAI,
        )
        result = await run_baseline_arm(
            arm=arm,
            drug_name="aspirin",
            settings=_make_settings(),
            registry=_make_test_registry(),
        )
        assert result.error is not None
        assert "KeyError" in result.error

    @pytest.mark.asyncio(loop_scope="session")
    async def test_none_provider_returns_error(self) -> None:
        """An arm with no baseline_provider should raise ValueError."""
        arm = ArmConfig(
            arm_id="no-provider",
            model_key="gpt_5_2",
            arm_type=ArmType.BASELINE,
            description="No provider set",
            baseline_provider=None,
        )
        result = await run_baseline_arm(
            arm=arm,
            drug_name="aspirin",
            settings=_make_settings(),
            registry=_make_test_registry(),
        )
        assert result.error is not None
        assert "Unknown baseline_provider" in result.error

    @pytest.mark.asyncio(loop_scope="session")
    async def test_pydantic_ai_model_id_passed_correctly(self) -> None:
        """Verify run_baseline receives the resolved PydanticAI model id."""
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ) as mock_run:
            await run_baseline_arm(
                arm=self._openai_arm(),
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        call_kwargs = mock_run.call_args[1]
        assert "openai:" in call_kwargs["pydantic_ai_model_id"]
        assert "gpt-5.2" in call_kwargs["pydantic_ai_model_id"]


# ======================================================================
# 10. Runner dispatch -- run_all_baseline_arms
# ======================================================================

class TestRunAllBaselineArms:
    """Tests for run_all_baseline_arms orchestration."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_runs_all_default_arms(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            results = await run_all_baseline_arms(
                drug_name="aspirin",
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert len(results) == len(BASELINE_ARMS)
        assert all(isinstance(r, ArmResult) for r in results)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_custom_arm_subset(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        arm = BASELINE_ARMS["baseline-gpt5-nosearch"]

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            results = await run_all_baseline_arms(
                drug_name="aspirin",
                arms=[arm],
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert len(results) == 1
        assert results[0].arm_id == "baseline-gpt5-nosearch"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_skips_pipeline_arms(self) -> None:
        from src.experiment.arms import PIPELINE_ARMS

        pipeline_arm = list(PIPELINE_ARMS.values())[0]
        baseline_arm = BASELINE_ARMS["baseline-gpt5-nosearch"]

        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ):
            results = await run_all_baseline_arms(
                drug_name="aspirin",
                arms=[pipeline_arm, baseline_arm],
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        # Only the baseline arm should be run
        assert len(results) == 1
        assert results[0].arm_id == "baseline-gpt5-nosearch"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_errors_captured_not_raised(self) -> None:
        """If one arm fails, others still execute."""
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        # First call raises, second succeeds
        call_count = 0

        async def alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return br

        arms_subset = [
            BASELINE_ARMS["baseline-gpt5-nosearch"],
            BASELINE_ARMS["baseline-gpt5-search"],
        ]

        with patch(
            "src.experiment.runner.run_baseline",
            side_effect=alternating,
        ):
            results = await run_all_baseline_arms(
                drug_name="aspirin",
                arms=arms_subset,
                settings=_make_settings(),
                registry=_make_test_registry(),
            )

        assert len(results) == 2
        assert results[0].error is not None  # first arm failed
        assert results[1].error is None  # second arm succeeded

    @pytest.mark.asyncio(loop_scope="session")
    async def test_passes_settings_through(self) -> None:
        pred = _make_prediction()
        br = BaselineResult(prediction=pred, input_tokens=0, output_tokens=0)

        arm = BASELINE_ARMS["baseline-gpt5-nosearch"]
        custom_settings = _make_settings(openai_api_key="sk-pass-through")

        with patch(
            "src.experiment.runner.run_baseline",
            new_callable=AsyncMock,
            return_value=br,
        ) as mock_run:
            await run_all_baseline_arms(
                drug_name="aspirin",
                arms=[arm],
                settings=custom_settings,
                registry=_make_test_registry(),
            )

        # The settings object is passed to run_baseline_arm, which passes
        # it to the provider runner. Verify the second positional arg
        # (settings) has our custom key.
        call_kwargs = mock_run.call_args
        settings_arg = call_kwargs[0][1]  # second positional arg
        assert settings_arg.openai_api_key == "sk-pass-through"


# ======================================================================
# 9. LIVE integration tests -- real API calls
# ======================================================================
#
# These tests call the real OpenAI and Anthropic APIs via the unified
# ``run_baseline()`` runner (PydanticAI + optional Tavily search).
# They are gated behind ``@pytest.mark.live`` so they only run when
# explicitly requested:
#
#     uv run pytest tests/test_baseline.py -m live -v
#
# Requirements:
#   - Valid API keys in ``.env`` (OPENAI_API_KEY, ANTHROPIC_API_KEY,
#     and optionally TAVILY_API_KEY)
#   - Network access
#
# Each test validates that the provider returns a structurally valid
# ``DrugDiseasePrediction`` for a well-known drug (aspirin).

def _live_settings() -> Settings:
    """Load real API keys from .env."""
    return Settings()


def _skip_if_no_key(settings: Settings, attr: str, label: str) -> None:
    """Skip the test if the required API key is missing or blank."""
    val = getattr(settings, attr, "")
    if not val:
        pytest.skip(f"{label} not set in .env -- skipping live test")


def _validate_prediction(pred: DrugDiseasePrediction, drug: str = "aspirin") -> None:
    """Common assertions for a live prediction."""
    assert pred.drug_name.lower() == drug.lower() or drug.lower() in pred.drug_name.lower()
    assert len(pred.associations) >= 1, "Expected at least one disease association"
    assert pred.reasoning, "Reasoning field should not be empty"
    for assoc in pred.associations:
        assert assoc.disease_name, "Disease name must not be empty"
        assert 0.0 <= assoc.confidence <= 1.0


@pytest.mark.live
class TestLiveOpenAIBaseline:
    """Live integration tests for OpenAI via unified PydanticAI runner."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_openai_no_search(self) -> None:
        settings = _live_settings()
        _skip_if_no_key(settings, "openai_api_key", "OPENAI_API_KEY")

        prompt = build_baseline_prompt("aspirin", chembl_id="CHEMBL25")
        result = await run_baseline(
            prompt,
            settings,
            pydantic_ai_model_id=f"openai:{settings.model_gpt_5_2}",
            uses_web_search=False,
        )

        assert isinstance(result, BaselineResult)
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        _validate_prediction(result.prediction)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_openai_with_tavily_search(self) -> None:
        settings = _live_settings()
        _skip_if_no_key(settings, "openai_api_key", "OPENAI_API_KEY")
        _skip_if_no_key(settings, "tavily_api_key", "TAVILY_API_KEY")

        prompt = build_baseline_prompt("aspirin", chembl_id="CHEMBL25")
        result = await run_baseline(
            prompt,
            settings,
            pydantic_ai_model_id=f"openai:{settings.model_gpt_5_2}",
            uses_web_search=True,
            allowed_domains=[
                "pubmed.ncbi.nlm.nih.gov",
                "opentargets.org",
            ],
        )

        assert isinstance(result, BaselineResult)
        assert result.input_tokens > 0
        _validate_prediction(result.prediction)


@pytest.mark.live
class TestLiveAnthropicBaseline:
    """Live integration tests for Anthropic via unified PydanticAI runner."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_anthropic_no_search(self) -> None:
        settings = _live_settings()
        _skip_if_no_key(settings, "anthropic_api_key", "ANTHROPIC_API_KEY")

        prompt = build_baseline_prompt("aspirin", chembl_id="CHEMBL25")
        result = await run_baseline(
            prompt,
            settings,
            pydantic_ai_model_id=f"anthropic:{settings.model_claude_opus_4_6}",
            uses_web_search=False,
        )

        assert isinstance(result, BaselineResult)
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        _validate_prediction(result.prediction)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_anthropic_with_tavily_search(self) -> None:
        settings = _live_settings()
        _skip_if_no_key(settings, "anthropic_api_key", "ANTHROPIC_API_KEY")
        _skip_if_no_key(settings, "tavily_api_key", "TAVILY_API_KEY")

        prompt = build_baseline_prompt("aspirin", chembl_id="CHEMBL25")
        result = await run_baseline(
            prompt,
            settings,
            pydantic_ai_model_id=f"anthropic:{settings.model_claude_opus_4_6}",
            uses_web_search=True,
            allowed_domains=[
                "pubmed.ncbi.nlm.nih.gov",
                "opentargets.org",
            ],
        )

        assert isinstance(result, BaselineResult)
        assert result.input_tokens > 0
        _validate_prediction(result.prediction)


@pytest.mark.live
class TestLiveRunnerDispatch:
    """Live integration test for the full run_baseline_arm dispatcher."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_run_openai_arm_live(self) -> None:
        """End-to-end: arm config -> dispatch -> OpenAI -> ArmResult."""
        settings = _live_settings()
        _skip_if_no_key(settings, "openai_api_key", "OPENAI_API_KEY")

        arm = BASELINE_ARMS["baseline-gpt5-nosearch"]
        result = await run_baseline_arm(
            arm=arm,
            drug_name="aspirin",
            chembl_id="CHEMBL25",
            settings=settings,
        )

        assert isinstance(result, ArmResult)
        assert result.error is None, f"Arm failed: {result.error}"
        assert result.prediction is not None
        _validate_prediction(result.prediction)
        assert result.wall_clock_seconds > 0.0
        assert result.usage.get("input_tokens", 0) > 0

    @pytest.mark.asyncio(loop_scope="session")
    async def test_run_anthropic_arm_live(self) -> None:
        """End-to-end: arm config -> dispatch -> Anthropic -> ArmResult."""
        settings = _live_settings()
        _skip_if_no_key(settings, "anthropic_api_key", "ANTHROPIC_API_KEY")

        arm = BASELINE_ARMS["baseline-claude-nosearch"]
        result = await run_baseline_arm(
            arm=arm,
            drug_name="aspirin",
            chembl_id="CHEMBL25",
            settings=settings,
        )

        assert isinstance(result, ArmResult)
        assert result.error is None, f"Arm failed: {result.error}"
        assert result.prediction is not None
        _validate_prediction(result.prediction)
        assert result.wall_clock_seconds > 0.0
        assert result.usage.get("input_tokens", 0) > 0
