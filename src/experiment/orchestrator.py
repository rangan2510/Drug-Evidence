"""Experiment orchestrator -- outer loop over drugs x arms.

``ExperimentRunner`` coordinates the full v2 experiment lifecycle:

1. Load CTD ground truth and select candidate drugs.
2. **Pre-build** the Qdrant vector index: normalise every drug, aggregate
   evidence from 6 APIs, chunk, and index -- all before running any arms.
3. Run all 8 arms (4 pipeline + 4 websearch) per drug against the
   pre-built index.
4. Cache every ``ArmResult`` for resumption.
5. After all drugs complete, classify difficulty and compute metrics.

Design constraints (from the project plan):
  - NO selection gates -- all drugs passing the 10-250 association filter
    are included.  Difficulty is a covariate, not a filter.
  - Pre-built index ensures all arms see identical evidence.
  - Hash-based result cache enables resumption of interrupted runs.
  - 4x2 factorial: 4 frontier models x (pipeline vs websearch).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.deps import EvidenceDeps
from src.config.models import MODEL_REGISTRY
from src.config.settings import Settings
from src.data.aggregator import EvidenceAggregator
from src.data.ctd import CTDClient
from src.data.normalizer import DrugNormalizer
from src.evaluation.accuracy import (
    AggregateMetrics,
    DrugMetrics,
    aggregate_metrics,
    evaluate_prediction,
)
from src.evaluation.evidence_quality import (
    EvidenceQualityMetrics,
    evaluate_evidence_quality,
)
from src.evaluation.false_negatives import (
    AggregateFNSummary,
    FalseNegativeSummary,
    aggregate_fn_summaries,
    analyse_false_negatives,
)
from src.evaluation.sensitivity import (
    CachedScore,
    HeatmapCell,
    WeightSweepResult,
    build_heatmap_data,
    extract_cached_scores,
    weight_sweep,
)
from src.experiment.arms import ALL_ARMS, ArmConfig, ArmType
from src.experiment.cache import ResultCache
from src.experiment.difficulty import ClassifiedDrug, classify_batch
from src.experiment.runner import (
    ArmResult,
    run_pipeline_arm,
    run_websearch_arm,
)
from src.schemas.prediction import DrugDifficulty, DrugDiseasePrediction
from src.data.evidence_store import EvidenceStore
from src.vector.chunker import chunk_evidence
from src.vector.embeddings import EmbeddingManager
from src.vector.precomputed_points import load_points_from_dir
from src.vector.store import HybridVectorStore

logger = logging.getLogger(__name__)

# Difficulty is now classified using the websearch-gpt52 arm (no special tools).
_DIFFICULTY_ARM_ID = "websearch-gpt52"


# ------------------------------------------------------------------
# Per-drug result container
# ------------------------------------------------------------------


@dataclass
class DrugResult:
    """All arm results + ground truth for one drug."""

    drug_name: str
    chembl_id: str | None = None
    pubchem_cid: int | None = None
    ground_truth_diseases: set[str] = field(default_factory=set)
    difficulty: DrugDifficulty | None = None
    arm_results: dict[str, ArmResult] = field(default_factory=dict)
    metrics: dict[str, DrugMetrics] = field(default_factory=dict)
    n_evidence_docs: int = 0
    n_chunks_indexed: int = 0


# ------------------------------------------------------------------
# Normalised drug info (carried through index-building phase)
# ------------------------------------------------------------------


@dataclass
class DrugInfo:
    """Lightweight container for a normalised drug (pre-build phase)."""

    drug_name: str
    chembl_id: str | None = None
    pubchem_cid: int | None = None
    n_evidence_docs: int = 0
    n_chunks_indexed: int = 0


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------


class ExperimentRunner:
    """Top-level orchestrator for the drug-disease prediction experiment.

    The v2 design uses a **pre-built Qdrant index**: all evidence is
    aggregated, chunked, and indexed before any arms execute.  This
    guarantees every arm (pipeline and websearch) sees identical evidence.

    Parameters
    ----------
    settings:
        Application settings (API keys, knobs, etc.).
    arms:
        Arm configurations to execute.  Defaults to all 8 arms.
    cache_dir:
        Directory for the result cache.
    resume:
        Whether to skip drugs/arms that already have cached results.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        arms: dict[str, ArmConfig] | None = None,
        cache_dir: str = ".cache/results",
        resume: bool = True,
        *,
        run_fn_analysis: bool = True,
        run_evidence_quality: bool = False,
        run_sensitivity: bool = True,
        openai_workers: int | None = None,
        anthropic_workers: int | None = None,
        skip_index_build: bool = False,
        use_staged: bool = False,
        use_staged_vectors: bool = False,
        target_file: str | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._arms = arms or dict(ALL_ARMS)
        self._cache = ResultCache(cache_dir)
        self._resume = resume
        self._skip_index_build = skip_index_build
        self._use_staged = use_staged
        self._use_staged_vectors = use_staged_vectors
        self._target_file = Path(target_file) if target_file else None

        self._openai_workers = openai_workers or self._settings.openai_workers
        self._anthropic_workers = anthropic_workers or self._settings.anthropic_workers
        self._vendor_semaphores: dict[str, asyncio.Semaphore] = {
            "openai": asyncio.Semaphore(self._openai_workers),
            "anthropic": asyncio.Semaphore(self._anthropic_workers),
        }

        # Evaluation flags
        self._run_fn_analysis = run_fn_analysis
        self._run_evidence_quality = run_evidence_quality
        self._run_sensitivity = run_sensitivity

        # Shared services (initialised in run())
        self._ctd: CTDClient | None = None
        self._normalizer: DrugNormalizer | None = None
        self._aggregator: EvidenceAggregator | None = None
        self._embeddings: EmbeddingManager | None = None
        self._vector_store: HybridVectorStore | None = None
        self._evidence_store: EvidenceStore | None = None

        # Drug info populated during index build phase
        self._drug_info: dict[str, DrugInfo] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        drug_limit: int | None = None,
        arm_ids: list[str] | None = None,
    ) -> ExperimentReport:
        """Execute the full experiment.

        Two-phase design:

        **Phase 1 -- Index Build** (sequential, one drug at a time):
          normalise -> aggregate evidence (6 APIs) -> chunk -> index into Qdrant.
          The entire index is built *before* any arm executes so that all
          arms see identical evidence.

        **Phase 2 -- Arm Execution** (per drug, concurrent across arms):
          For each drug, run all configured arms against the pre-built
          index.  Pipeline arms get 9 evidence tools + Qdrant RAG;
          websearch arms get only Tavily web search.

        Parameters
        ----------
        drug_limit:
            Override ``Settings.target_drugs`` (useful for testing).
        arm_ids:
            Subset of arm IDs to run.  Defaults to all configured arms.

        Returns
        -------
        ExperimentReport
            Summary of all results, metrics, and difficulty labels.
        """
        t0 = time.perf_counter()
        limit = drug_limit or self._settings.target_drugs

        arms_to_run = self._arms
        if arm_ids:
            arms_to_run = {k: v for k, v in self._arms.items() if k in arm_ids}

        logger.info(
            "Starting experiment: target_drugs=%d, arms=%d, resume=%s",
            limit,
            len(arms_to_run),
            self._resume,
        )
        logger.info(
            "Worker limits: openai=%d anthropic=%d",
            self._openai_workers,
            self._anthropic_workers,
        )

        # -- Step 1: Initialise shared services --------------------------
        await self._init_services()

        # -- Step 2: Load CTD and get candidate drugs --------------------
        assert self._ctd is not None
        await self._ctd.load()

        if self._target_file is not None:
            # Use explicit target file instead of CTD association-count filter
            raw_lines = self._target_file.read_text(encoding="utf-8").splitlines()
            all_targets = [
                line.strip() for line in raw_lines if line.strip()
            ]
            candidates = all_targets[:limit] if limit else all_targets
            logger.info(
                "Loaded %d candidate drugs from target file '%s'",
                len(candidates),
                self._target_file,
            )
        else:
            candidates = self._ctd.get_candidate_drugs(
                min_assoc=self._settings.min_associations,
                max_assoc=self._settings.max_associations,
                limit=limit,
            )
            logger.info("Selected %d candidate drugs from CTD", len(candidates))

        # -- Step 3: PHASE 1 -- Pre-build evidence index -------------------
        if self._use_staged:
            logger.info(
                "Using staged evidence from MongoDB -- "
                "skipping live API aggregation and Qdrant indexing."
            )
            if self._use_staged_vectors:
                logger.info(
                    "Staged vector mode enabled: using precomputed point pickles."
                )
            # Drug metadata (chembl_id, pubchem_cid, n_docs) will be read
            # from MongoDB in _run_drug_arms; no need for DrugInfo dicts.
        elif not self._skip_index_build:
            await self._build_index(candidates)
        else:
            logger.info(
                "Skipping index build (skip_index_build=True). "
                "Using existing Qdrant collection."
            )
            # Still need drug info for arm execution
            for drug_name in candidates:
                self._drug_info[drug_name.lower()] = DrugInfo(drug_name=drug_name)

        # -- Step 4: PHASE 2 -- Run arms per drug ------------------------
        drug_results: dict[str, DrugResult] = {}

        for idx, drug_name in enumerate(candidates, 1):
            logger.info(
                "=== Drug %d / %d: %s (arm execution) ===",
                idx,
                len(candidates),
                drug_name,
            )
            dr = await self._run_drug_arms(drug_name, arms_to_run)
            drug_results[drug_name.lower()] = dr

        # -- Step 5: Classify difficulty (post-hoc) ----------------------
        difficulty_results: dict[str, ArmResult] = {}
        ground_truths: dict[str, set[str]] = {}
        for drug_lower, dr in drug_results.items():
            if _DIFFICULTY_ARM_ID in dr.arm_results:
                difficulty_results[drug_lower] = dr.arm_results[_DIFFICULTY_ARM_ID]
            ground_truths[drug_lower] = dr.ground_truth_diseases

        difficulty_map: dict[str, ClassifiedDrug] = {}
        if difficulty_results:
            difficulty_map = classify_batch(difficulty_results, ground_truths)
            for drug_lower, cd in difficulty_map.items():
                if drug_lower in drug_results:
                    drug_results[drug_lower].difficulty = cd.difficulty

        # -- Step 6: Compute per-drug metrics ----------------------------
        for drug_lower, dr in drug_results.items():
            diff = dr.difficulty
            for arm_id, arm_result in dr.arm_results.items():
                if arm_result.prediction is None:
                    continue
                metrics = evaluate_prediction(
                    arm_result.prediction,
                    dr.ground_truth_diseases,
                    arm_id=arm_id,
                    difficulty=diff,
                )
                dr.metrics[arm_id] = metrics

        # -- Step 7: Aggregate per-arm metrics ---------------------------
        arm_aggregates: dict[str, AggregateMetrics] = {}
        for arm_id in arms_to_run:
            arm_drug_metrics = [
                dr.metrics[arm_id]
                for dr in drug_results.values()
                if arm_id in dr.metrics
            ]
            arm_aggregates[arm_id] = aggregate_metrics(arm_drug_metrics, arm_id)

        # -- Step 8: False-negative analysis (optional) ------------------
        fn_summaries: dict[str, AggregateFNSummary] = {}
        if self._run_fn_analysis:
            fn_summaries = self._compute_fn_analysis(
                drug_results,
                arms_to_run,
            )
            logger.info(
                "FN analysis: %d arms evaluated",
                len(fn_summaries),
            )

        # -- Step 9: Evidence quality (optional, async) ------------------
        evidence_quality: dict[str, EvidenceQualityMetrics] = {}
        if self._run_evidence_quality:
            evidence_quality = await self._compute_evidence_quality(
                drug_results,
                arms_to_run,
            )
            logger.info(
                "Evidence quality: %d arms evaluated",
                len(evidence_quality),
            )

        # -- Step 10: Sensitivity analysis (optional) --------------------
        heatmap_data: list[HeatmapCell] = []
        weight_sweeps: dict[str, list[WeightSweepResult]] = {}
        if self._run_sensitivity:
            heatmap_data, weight_sweeps = self._compute_sensitivity(
                drug_results,
                arms_to_run,
            )
            logger.info(
                "Sensitivity: %d heatmap cells, %d arms with weight sweeps",
                len(heatmap_data),
                len(weight_sweeps),
            )

        elapsed = time.perf_counter() - t0

        report = ExperimentReport(
            n_drugs=len(candidates),
            n_arms=len(arms_to_run),
            wall_clock_seconds=elapsed,
            drug_results=drug_results,
            difficulty_map=difficulty_map,
            arm_aggregates=arm_aggregates,
            cached_results=self._cache.count(),
            fn_summaries=fn_summaries,
            evidence_quality=evidence_quality,
            heatmap_data=heatmap_data,
            weight_sweeps=weight_sweeps,
        )

        logger.info(
            "Experiment complete: %d drugs x %d arms in %.1f s (%d cached results)",
            report.n_drugs,
            report.n_arms,
            report.wall_clock_seconds,
            report.cached_results,
        )
        return report

    # ------------------------------------------------------------------
    # PHASE 1: Pre-build Qdrant index
    # ------------------------------------------------------------------

    async def _build_index(self, candidates: list[str]) -> None:
        """Normalise, aggregate, chunk, and index ALL drugs into Qdrant.

        This runs sequentially through every candidate drug so that the
        entire vector index is populated before any arm executes.
        """
        assert self._normalizer is not None
        assert self._aggregator is not None
        assert self._vector_store is not None
        assert self._embeddings is not None

        total_docs = 0
        total_chunks = 0

        for idx, drug_name in enumerate(candidates, 1):
            logger.info(
                "--- Index build %d / %d: %s ---",
                idx,
                len(candidates),
                drug_name,
            )
            info = DrugInfo(drug_name=drug_name)

            # -- Normalise drug -------------------------------------------
            logger.info("[%s] Normalising drug ...", drug_name)
            try:
                norm = await self._normalizer.normalize(drug_name)
                info.chembl_id = norm.chembl_id
                info.pubchem_cid = norm.pubchem_cid
                logger.info(
                    "[%s] Normalised: chembl=%s pubchem=%s",
                    drug_name,
                    info.chembl_id,
                    info.pubchem_cid,
                )
            except Exception:
                logger.exception("Normalisation failed for '%s'", drug_name)

            # -- Aggregate evidence ---------------------------------------
            logger.info("[%s] Aggregating evidence ...", drug_name)
            docs = []
            try:
                docs = await self._aggregator.gather(
                    drug_name,
                    chembl_id=info.chembl_id,
                    pubchem_cid=info.pubchem_cid,
                    skip_normalize=True,
                )
                info.n_evidence_docs = len(docs)
                total_docs += len(docs)
                logger.info(
                    "[%s] Gathered %d evidence documents.", drug_name, len(docs)
                )
            except Exception:
                logger.exception("Evidence aggregation failed for '%s'", drug_name)

            # -- Chunk and index into Qdrant ------------------------------
            if docs:
                try:
                    logger.info("[%s] Chunking %d docs ...", drug_name, len(docs))
                    chunks = chunk_evidence(
                        docs,
                        chunk_size=self._settings.chunk_size,
                        chunk_overlap=self._settings.chunk_overlap,
                    )
                    logger.info(
                        "[%s] Indexing %d chunks into Qdrant ...",
                        drug_name,
                        len(chunks),
                    )
                    n_indexed = self._vector_store.index_chunks(chunks)
                    info.n_chunks_indexed = n_indexed
                    total_chunks += n_indexed
                    logger.info("[%s] Indexed %d chunks.", drug_name, n_indexed)
                except Exception:
                    logger.exception("Chunking/indexing failed for '%s'", drug_name)

            self._drug_info[drug_name.lower()] = info

        logger.info(
            "Index build complete: %d drugs, %d documents, %d chunks indexed.",
            len(candidates),
            total_docs,
            total_chunks,
        )

    # ------------------------------------------------------------------
    # PHASE 2: Run arms per drug
    # ------------------------------------------------------------------

    async def _run_drug_arms(
        self,
        drug_name: str,
        arms: dict[str, ArmConfig],
    ) -> DrugResult:
        """Run all arms for one drug against staged or live evidence."""
        assert self._ctd is not None

        dr = DrugResult(drug_name=drug_name)

        # Copy drug info from index-build phase or MongoDB metadata
        if self._use_staged and self._evidence_store is not None:
            meta = await self._evidence_store.get_drug_meta(drug_name)
            if meta:
                dr.chembl_id = meta.get("chembl_id")
                dr.pubchem_cid = meta.get("pubchem_cid")
                dr.n_evidence_docs = meta.get("n_docs", 0)
            else:
                logger.warning(
                    "No staged evidence for '%s' -- arms may produce poor results",
                    drug_name,
                )
        else:
            info = self._drug_info.get(drug_name.lower())
            if info:
                dr.chembl_id = info.chembl_id
                dr.pubchem_cid = info.pubchem_cid
                dr.n_evidence_docs = info.n_evidence_docs
                dr.n_chunks_indexed = info.n_chunks_indexed

        # Ground truth
        dr.ground_truth_diseases = {
            d.lower() for d in self._ctd.get_therapeutic_diseases(drug_name)
        }

        # Deps for pipeline arms
        deps = EvidenceDeps(
            settings=self._settings,
            vector_store=self._vector_store,
            aggregator=self._aggregator,
            evidence_store=self._evidence_store,
            drug_name=drug_name,
            chembl_id=dr.chembl_id,
            pubchem_cid=dr.pubchem_cid,
        )

        async def _execute_arm(
            arm_id: str,
            arm_cfg: ArmConfig,
        ) -> tuple[str, ArmResult]:
            model_id = self._resolve_model_id_safe(arm_cfg)

            if self._resume and self._cache.has(drug_name, arm_id, model_id):
                cached = self._cache.get(drug_name, arm_id, model_id)
                if cached is not None:
                    logger.info(
                        "Cache hit for %s / %s -- skipping",
                        drug_name,
                        arm_id,
                    )
                    return arm_id, cached

            provider = self._provider_for_arm(arm_cfg)
            sem = self._vendor_semaphores.get(provider)

            if sem is None:
                result = await self._run_single_arm(arm_cfg, deps, drug_name)
            else:
                async with sem:
                    result = await self._run_single_arm(arm_cfg, deps, drug_name)

            return arm_id, result

        tasks = [
            asyncio.create_task(_execute_arm(arm_id, arm_cfg))
            for arm_id, arm_cfg in arms.items()
        ]

        for arm_id, result in await asyncio.gather(*tasks):
            dr.arm_results[arm_id] = result
            self._cache.put(result)

        return dr

    async def _run_single_arm(
        self,
        arm: ArmConfig,
        deps: EvidenceDeps,
        drug_name: str,
    ) -> ArmResult:
        """Dispatch a single arm execution (pipeline or websearch)."""
        if arm.arm_type == ArmType.PIPELINE:
            return await run_pipeline_arm(arm=arm, deps=deps)
        elif arm.arm_type == ArmType.WEBSEARCH:
            return await run_websearch_arm(
                arm=arm,
                drug_name=drug_name,
                chembl_id=deps.chembl_id,
                settings=self._settings,
            )
        else:
            msg = f"Unknown arm type: {arm.arm_type}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Internal: evaluation helpers
    # ------------------------------------------------------------------

    def _compute_fn_analysis(
        self,
        drug_results: dict[str, DrugResult],
        arms: dict[str, ArmConfig],
    ) -> dict[str, AggregateFNSummary]:
        """Run false-negative categorisation for every arm."""
        arm_summaries: dict[str, list[FalseNegativeSummary]] = {}

        for dr in drug_results.values():
            for arm_id in arms:
                ar = dr.arm_results.get(arm_id)
                if ar is None or ar.prediction is None:
                    continue
                summary = analyse_false_negatives(
                    ar.prediction,
                    dr.ground_truth_diseases,
                    arm_id,
                )
                arm_summaries.setdefault(arm_id, []).append(summary)

        return {
            arm_id: aggregate_fn_summaries(summaries, arm_id)
            for arm_id, summaries in arm_summaries.items()
        }

    async def _compute_evidence_quality(
        self,
        drug_results: dict[str, DrugResult],
        arms: dict[str, ArmConfig],
    ) -> dict[str, EvidenceQualityMetrics]:
        """Compute mean evidence quality per arm (async -- hits PubMed).

        Uses heuristic specificity (no LLM calls) to keep costs down.
        Per-arm results are averaged across all drugs.

        Parallelised with a semaphore to respect PubMed rate limits.
        """
        # Ensure embedding manager is available (may be None in staged
        # mode without staged vectors).
        if self._embeddings is None:
            logger.info("Lazy-loading embedding manager for evidence quality")
            self._embeddings = EmbeddingManager(
                query_model_name=self._settings.dense_model_query,
                doc_model_name=self._settings.dense_model_doc,
                sparse_model_name=self._settings.sparse_model,
            )
            self._embeddings.load_query_only()

        arm_metrics: dict[str, list[EvidenceQualityMetrics]] = {}
        sem = asyncio.Semaphore(16)  # limit concurrent eval tasks

        async def _eval_one(
            drug_name: str, arm_id: str, prediction: DrugDiseasePrediction,
        ) -> tuple[str, EvidenceQualityMetrics | None]:
            async with sem:
                try:
                    eq = await evaluate_evidence_quality(
                        prediction,
                        embedding_manager=self._embeddings,
                        settings=self._settings,
                        use_llm_specificity=False,  # heuristic only
                    )
                    return arm_id, eq
                except Exception:
                    logger.exception(
                        "Evidence quality failed for %s / %s",
                        drug_name,
                        arm_id,
                    )
                    return arm_id, None

        # Build tasks for all drug x arm pairs
        tasks: list[asyncio.Task] = []
        for dr in drug_results.values():
            for arm_id in arms:
                ar = dr.arm_results.get(arm_id)
                if ar is None or ar.prediction is None:
                    continue
                tasks.append(
                    asyncio.create_task(
                        _eval_one(dr.drug_name, arm_id, ar.prediction)
                    )
                )

        logger.info(
            "Evidence quality: evaluating %d predictions in parallel (sem=%d)",
            len(tasks),
            sem._value,
        )
        results = await asyncio.gather(*tasks)

        for aid, eq in results:
            if eq is not None:
                arm_metrics.setdefault(aid, []).append(eq)

        # Average per arm
        result: dict[str, EvidenceQualityMetrics] = {}
        for arm_id, metrics_list in arm_metrics.items():
            n = len(metrics_list)
            if n == 0:
                continue
            result[arm_id] = EvidenceQualityMetrics(
                citation_validity_rate=sum(
                    m.citation_validity_rate for m in metrics_list
                )
                / n,
                mean_chain_depth=sum(m.mean_chain_depth for m in metrics_list) / n,
                chain_verifiability_score=sum(
                    m.chain_verifiability_score for m in metrics_list
                )
                / n,
                evidence_relevance=sum(m.evidence_relevance for m in metrics_list) / n,
                mechanistic_specificity=sum(
                    m.mechanistic_specificity for m in metrics_list
                )
                / n,
                n_citations_checked=sum(m.n_citations_checked for m in metrics_list),
                n_citations_valid=sum(m.n_citations_valid for m in metrics_list),
                n_chains=sum(m.n_chains for m in metrics_list),
                n_edges=sum(m.n_edges for m in metrics_list),
            )
        return result

    def _compute_sensitivity(
        self,
        drug_results: dict[str, DrugResult],
        arms: dict[str, ArmConfig],
    ) -> tuple[list[HeatmapCell], dict[str, list[WeightSweepResult]]]:
        """Run weight sweeps and build heatmap data for pipeline arms.

        Only pipeline arms have retrieval scores, so baselines are skipped
        for heatmap generation. Weight sweeps are computed for all arms.
        """
        all_sweeps: dict[str, list[WeightSweepResult]] = {}
        # Collect all cached scores for heatmap (aggregate across drugs)
        all_cached_scores: list[CachedScore] = []
        all_ground_truth: set[str] = set()

        for dr in drug_results.values():
            for arm_id, arm_cfg in arms.items():
                ar = dr.arm_results.get(arm_id)
                if ar is None or ar.prediction is None:
                    continue

                cached = extract_cached_scores(ar.prediction)
                sweep = weight_sweep(
                    cached,
                    dr.ground_truth_diseases,
                    arm_id,
                    dr.drug_name,
                )
                all_sweeps.setdefault(arm_id, []).append(sweep)

                # Collect for heatmap (use first pipeline arm encountered)
                if arm_cfg.arm_type == ArmType.PIPELINE and not all_cached_scores:
                    all_cached_scores = cached
                    all_ground_truth = dr.ground_truth_diseases

        # Build heatmap from the first pipeline drug with data
        heatmap: list[HeatmapCell] = []
        if all_cached_scores:
            heatmap = build_heatmap_data(
                all_cached_scores,
                all_ground_truth,
            )

        return heatmap, all_sweeps

    # ------------------------------------------------------------------
    # Internal: service initialisation
    # ------------------------------------------------------------------

    async def _init_services(self) -> None:
        """Create shared service instances (CTD, normalizer, aggregator, vector store).

        When ``use_staged=True``, connects to MongoDB and skips embedding /
        vector store initialisation (evidence comes from MongoDB, not Qdrant).
        """
        self._ctd = CTDClient(self._settings)
        self._normalizer = DrugNormalizer(self._settings)
        self._aggregator = EvidenceAggregator(self._settings)

        if self._use_staged:
            # MongoDB is the primary evidence store -- skip heavy
            # embedding model loading and Qdrant connection.
            self._evidence_store = EvidenceStore(self._settings)
            await self._evidence_store.connect()
            n_staged = await self._evidence_store.count_staged_drugs()
            logger.info(
                "MongoDB staged mode: %d drugs available in '%s'",
                n_staged,
                self._settings.mongo_db_name,
            )

            if self._use_staged_vectors or self._settings.use_staged_vectors:
                points_dir = Path(self._settings.staged_points_dir)
                if not points_dir.exists():
                    msg = (
                        "Staged vector mode is enabled, but points directory does not exist: "
                        f"{points_dir}"
                    )
                    raise FileNotFoundError(msg)

                self._embeddings = EmbeddingManager(
                    query_model_name=self._settings.dense_model_query,
                    doc_model_name=self._settings.dense_model_doc,
                    sparse_model_name=self._settings.sparse_model,
                )
                self._embeddings.load_query_only()
                self._vector_store = HybridVectorStore.from_settings(
                    self._settings,
                    self._embeddings,
                    in_memory=True,
                )
                self._vector_store.ensure_collection()

                points = load_points_from_dir(points_dir)
                loaded = self._vector_store.upsert_points(points)
                logger.info(
                    "Loaded %d precomputed points from '%s' into in-memory Qdrant.",
                    loaded,
                    points_dir,
                )
        else:
            self._embeddings = EmbeddingManager(
                query_model_name=self._settings.dense_model_query,
                doc_model_name=self._settings.dense_model_doc,
                sparse_model_name=self._settings.sparse_model,
            )
            self._vector_store = HybridVectorStore.from_settings(
                self._settings,
                self._embeddings,
            )

    def _resolve_model_id_safe(self, arm: ArmConfig) -> str:
        """Resolve model ID without raising if the key is missing."""
        try:
            return arm.resolve_model_id()
        except KeyError:
            return f"<unresolved:{arm.model_key}>"

    def _provider_for_arm(self, arm: ArmConfig) -> str:
        """Return provider key used for concurrency semaphore routing."""
        try:
            spec = MODEL_REGISTRY[arm.model_key]
            return spec.provider.value
        except KeyError:
            return "unknown"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release resources (normalizer HTTP client, MongoDB, etc.)."""
        if self._normalizer:
            await self._normalizer.close()
        if self._evidence_store:
            await self._evidence_store.close()


# ------------------------------------------------------------------
# Experiment report container
# ------------------------------------------------------------------


@dataclass
class ExperimentReport:
    """Final output of the experiment run."""

    n_drugs: int
    n_arms: int
    wall_clock_seconds: float
    drug_results: dict[str, DrugResult] = field(default_factory=dict)
    difficulty_map: dict[str, ClassifiedDrug] = field(default_factory=dict)
    arm_aggregates: dict[str, AggregateMetrics] = field(default_factory=dict)
    cached_results: int = 0

    # Evaluation results (populated when corresponding flags are enabled)
    fn_summaries: dict[str, AggregateFNSummary] = field(default_factory=dict)
    evidence_quality: dict[str, EvidenceQualityMetrics] = field(
        default_factory=dict,
    )
    heatmap_data: list[HeatmapCell] = field(default_factory=list)
    weight_sweeps: dict[str, list[WeightSweepResult]] = field(
        default_factory=dict,
    )

    def summary_table(self) -> str:
        """Return a markdown summary table of per-arm aggregate metrics."""
        lines: list[str] = [
            "| Arm | N | P@1 | P@10 | R@1 | R@10 | AUC |",
            "|-----|---|-----|------|-----|------|-----|",
        ]
        for arm_id, agg in sorted(self.arm_aggregates.items()):
            auc_str = (
                f"{agg.mean_roc_auc:.3f}" if agg.mean_roc_auc is not None else "n/a"
            )
            lines.append(
                f"| {arm_id} | {agg.n_drugs} | "
                f"{agg.mean_precision_at_1:.3f} | {agg.mean_precision_at_10:.3f} | "
                f"{agg.mean_recall_at_1:.3f} | {agg.mean_recall_at_10:.3f} | "
                f"{auc_str} |"
            )
        return "\n".join(lines)

    def difficulty_summary(self) -> str:
        """Return difficulty distribution as a string."""
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for cd in self.difficulty_map.values():
            counts[cd.difficulty.value] += 1
        return (
            f"Difficulty: easy={counts['easy']}, "
            f"medium={counts['medium']}, "
            f"hard={counts['hard']}"
        )
