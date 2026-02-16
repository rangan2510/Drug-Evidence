# Copilot Instructions — Drug-Disease Discovery Pipeline

## Project Context

This repo is being redesigned from v1 → v2 to address peer review concerns. Legacy v1 code is archived in `legacy_v1/` (read-only reference). The v2 structure is shown below.

## Architecture (v2 target)

```
src/
  config/          Pydantic BaseSettings, ModelSpec registry (6 LLMs)
  schemas/         EvidenceDocument, DrugDiseasePrediction, MechanisticEdge
  data/            One async client per source (7 APIs) + aggregator
  vector/          Qdrant hybrid store, MedCPT + SPLADE embeddings
  agents/          PydanticAI evidence agent + baseline agent + tools
  evaluation/      accuracy, evidence_quality, sensitivity, false_negatives
  experiment/      Orchestrator, arm configs, result cache
  viz/             9 plot types, markdown report, CSV/JSON export
  main.py          CLI entry point
scripts/           Staging, parallel runner, subset utilities
tests/             pytest + pytest-asyncio (unit, integration, live)
data/              CTD ground truth, target drug list
docs/latex/        LNCS manuscript, figures, references
```

**Data flow:** CTD ground truth → normalize drug → aggregate evidence from 6 APIs → index into Qdrant → agent extracts DrugDiseasePrediction → evaluate accuracy/evidence quality → report.

## Critical Conventions

- Python 3.14 via `uv` (no pip, no requirements.txt)
- Pydantic everywhere: BaseSettings for config, BaseModel for schemas
- Agent framework: PydanticAI, model selection via ModelSpec
- 6 models across 3 providers (env vars in `.env`): OpenAI, Anthropic, Groq, Tavily
- All data clients return list[EvidenceDocument] (uniform schema)
- Async-first: httpx.AsyncClient, async/await, pytest-asyncio
- Hybrid vector search: Qdrant dense + sparse, RRF fusion
- No selection gates: all drugs passing association-count filter included
- 7 experimental arms, same DrugDiseasePrediction schema

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| N=200 drugs (default), no "too easy" / zero-recall gates | Supports paired Wilcoxon across arms |
| 6 biomedical APIs (not just OpenTargets→PubMed) | Single-source evidence was a reviewer concern |
| Structured mechanistic evidence chains | Replaces regex-parsed confidence scores |
| Frontier baselines get Tavily web search tool | Pipeline must beat baselines on evidence quality |
| Runtime/cost tracking for each arm | Reviewer requested runtime/cost analysis |

## Reviewer Context

See [docs/latex/review/revision-plan.md](../docs/latex/review/revision-plan.md) and [docs/latex/review/reviewer-responses.md](../docs/latex/review/reviewer-responses.md) for full reviewer rationale and experiment changes.
