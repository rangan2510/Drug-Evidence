# Drug-Disease Mechanistic Association Discovery

Deterministic Graph-RAG pipeline for predicting drug-disease associations
with structured mechanistic evidence chains. Compares 4 pipeline arms
against 4 frontier-LLM web-search baselines across 50 drugs (8 arms,
400 total experiments).

Paper: *Deterministic Graph-RAG for Drug-Disease Mechanistic Association
Discovery* (ACIIDS 2026, LNCS format).

## Architecture

```
data sources         vector store           agent             output
  CTD (ground truth)                    +------------+
  OpenTargets    -->  Qdrant hybrid     | PydanticAI |    DrugDiseasePrediction
  DGIdb              (MedCPT + SPLADE   |  evidence  | -->  MechanisticEdge[]
  PubChem             + RRF fusion)     |   agent    |      EvidenceChain[]
  PharmGKB       -->  768d dense +      |  (9 tools) |      Citation[]
  ChEMBL              sparse BM25       +------------+      confidence scores
  PubMed
```

**Pipeline arms** swap the backing LLM (GPT-4.1, GPT-5.2, Claude Sonnet 4.5,
Claude Opus 4.6) while keeping the deterministic retrieval graph fixed.

**Baseline arms** give the same 4 frontier models Tavily web search and free-form
generation, then extract structured predictions via Instructor two-pass parsing.

## Project Structure

```
src/
  config/          Pydantic BaseSettings, ModelSpec registry (6 LLMs)
  schemas/         EvidenceDocument, DrugDiseasePrediction, MechanisticEdge
  data/            One async client per source (7 APIs) + aggregator
  vector/          Qdrant hybrid store, MedCPT + SPLADE embeddings
  agents/          PydanticAI evidence agent + baseline agent + 9 tools
  evaluation/      accuracy, evidence_quality, sensitivity, false_negatives
  experiment/      Orchestrator, arm configs, result cache
  viz/             9 plot types, markdown report, CSV/JSON export
  main.py          CLI entry point
scripts/           Staging, parallel runner, subset utilities
tests/             pytest + pytest-asyncio (unit, integration, live)
data/              CTD ground truth, target drug list
docs/latex/        LNCS manuscript, figures, references
legacy_v1/         Archived v1 code (read-only reference)
```

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups           # install all dependency groups
cp .env.example .env           # fill in API keys
```

### Required API keys (`.env`)

| Key | Provider | Used by |
|-----|----------|---------|
| `OPENAI_API_KEY` | OpenAI | GPT-4.1, GPT-5.2 pipeline + baseline arms |
| `ANTHROPIC_API_KEY` | Anthropic | Claude Sonnet 4.5, Opus 4.6 arms |
| `GROQ_API_KEY` | Groq | OSS models (Llama 4, Qwen3, Kimi K2, GPT-OSS) |
| `ENTREZ_EMAIL` | NCBI | PubMed E-utilities (required for batch queries) |

## Usage

### Run the full experiment

```bash
uv run python -m src.main --drugs 50 \
  --target-file data/target_set.txt \
  --cache-dir .cache/target50_cache \
  --output-dir results/target50
```

### Common flags

| Flag | Description |
|------|-------------|
| `--drugs N` | Number of drugs to evaluate |
| `--arms arm1 arm2 ...` | Subset of arms to run (default: all 8) |
| `--use-staged` | Use pre-staged MongoDB evidence |
| `--run-evidence-quality` | Run evidence quality metrics (adds latency) |
| `--skip-sensitivity` | Skip weight/threshold sweep |
| `--skip-fn-analysis` | Skip false-negative categorisation |
| `--no-resume` | Ignore cached results |
| `--openai-workers N` | Max concurrent OpenAI requests |
| `--anthropic-workers N` | Max concurrent Anthropic requests |

### Stage evidence (one-time prep)

```bash
uv run python scripts/stage_evidence.py
uv run python scripts/stage_missing_targets.py   # backfill any gaps
```

## Development

```bash
uv run pytest                          # unit tests (mocked, fast)
uv run pytest -m integration           # live API integration tests
uv run pytest -m live                  # real LLM tests (costs tokens)
uv run ruff check src/ tests/         # lint
uv run ruff format src/ tests/        # format
```

## Results

The 50-drug experiment output lives in `results/target50/`:

- `report.md` -- full markdown report with statistical tests
- `aggregate.csv` -- per-drug per-arm predictions
- `metrics.csv` -- P@1, P@10, R@1, R@10, ROC-AUC per arm
- `arm_results.csv` -- summary across arms
- `plots/` -- 8 PNG visualisations (ROC, PR, radar, heatmap, etc.)

## Experimental Arms

| Arm ID | Model | Config |
|--------|-------|--------|
| `pipeline-gpt41` | GPT-4.1 | Deterministic Graph-RAG |
| `pipeline-gpt52` | GPT-5.2 | Deterministic Graph-RAG |
| `pipeline-sonnet45` | Claude Sonnet 4.5 | Deterministic Graph-RAG |
| `pipeline-opus46` | Claude Opus 4.6 | Deterministic Graph-RAG |
| `websearch-gpt41` | GPT-4.1 | Tavily web search |
| `websearch-gpt52` | GPT-5.2 | Tavily web search |
| `websearch-sonnet45` | Claude Sonnet 4.5 | Tavily web search |
| `websearch-opus46` | Claude Opus 4.6 | Tavily web search |

## Project Rationale

This project addresses the challenge of predicting drug-disease associations with structured mechanistic evidence chains. The pipeline leverages a deterministic Graph-RAG approach, integrating evidence from six biomedical APIs and frontier LLMs (GPT-4.1, GPT-5.2, Claude Sonnet 4.5, Opus 4.6) to outperform web-search baselines. Mechanistic chains (drug→target→pathway→disease) replace prior regex-based confidence scores, supporting robust, explainable predictions.

## Peer Review Improvements (v2)

Following peer review, the pipeline was redesigned to:
- Expand from N=50 to N=200 drugs (removing cherry-picking and zero-recall gates)
- Integrate evidence from six APIs (OpenTargets, DGIdb, PubChem, PharmGKB, ChEMBL, PubMed)
- Use structured mechanistic chains for evidence, not regex parsing
- Add web-search baselines with Tavily API for fair comparison
- Track runtime and cost for each arm
- Stratify results by drug difficulty (easy/medium/hard)
- Report all drugs, no selection gates

See [docs/latex/review/revision-plan.md](docs/latex/review/revision-plan.md) and [docs/latex/review/reviewer-responses.md](docs/latex/review/reviewer-responses.md) for full reviewer context and rationale.

## Key Results

- Pipeline arms outperform web-search baselines on P@1 (0.760–0.841 vs. 0.540–0.620)
- Pipeline-GPT-5.2 achieves P@10 of 0.525 vs. 0.307 for websearch-GPT-5.2 (+71% relative improvement)
- Ten contrasts are statistically significant at Bonferroni-corrected alpha = 0.0018
- All results stratified by drug difficulty and reported for N=200

For full statistical details, see [docs/latex/main.tex](docs/latex/main.tex) and [results/target50/report.md](results/target50/report.md).
