"""Run the frontier-model experiment with vendor-aware worker limits.

Default profile:
- 200 drugs (4 frontier models x 2 configs = 8 arms)
- workers: OpenAI=3, Anthropic=3 (6 total)
- Pre-built Qdrant index (localhost -> on-disk -> in-memory fallback)

Two phases:
1. Index Build -- normalise, aggregate evidence (6 APIs), chunk, index into Qdrant
2. Arm Execution -- run 8 arms per drug (4 pipeline + 4 websearch)

Outputs:
- Standard export artefacts (arm_results.csv, metrics.csv, aggregate.csv, report.json)
- Runtime summary CSV per arm

Usage
-----
uv run python -m scripts.run_250_parallel
uv run python -m scripts.run_250_parallel --drugs 50 --no-resume
uv run python -m scripts.run_250_parallel --skip-index-build
uv run python -m scripts.run_250_parallel --subset-best 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import statistics
import subprocess
import sys
from pathlib import Path

from src.config.settings import Settings
from src.experiment.orchestrator import ExperimentRunner
from src.viz.export import export_results

# Ensure log output is visible on the console during long runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quieten chatty libraries to WARNING so our progress lines stand out.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hishel").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frontier-model experiment driver (4 models x 2 configs)"
    )
    parser.add_argument("--drugs", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/run_250_parallel")
    parser.add_argument("--cache-dir", type=str, default=".cache/results")
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        default=False,
        help="Skip Qdrant index build (use existing collection)",
    )
    parser.add_argument(
        "--use-staged",
        action="store_true",
        default=False,
        help="Use MongoDB staged evidence (no live API calls or Qdrant)",
    )
    parser.add_argument(
        "--use-staged-vectors",
        action="store_true",
        default=False,
        help="Load precomputed point pickles into in-memory Qdrant during staged runs",
    )
    parser.add_argument(
        "--staged-points-dir",
        type=str,
        default="data/phase0_points",
        help="Directory containing points_*.pkl files",
    )
    parser.add_argument("--openai-workers", type=int, default=3)
    parser.add_argument("--anthropic-workers", type=int, default=3)
    parser.add_argument("--qdrant-in-memory", action="store_true", default=False)
    parser.add_argument("--no-qdrant-fallback", action="store_true", default=False)
    parser.add_argument("--run-fn-analysis", action="store_true", default=False)
    parser.add_argument("--run-evidence-quality", action="store_true", default=False)
    parser.add_argument("--run-sensitivity", action="store_true", default=False)
    parser.add_argument(
        "--subset-best",
        type=int,
        default=0,
        help="If >0, run post-hoc top-N subset filtering",
    )
    parser.add_argument(
        "--arm-ids",
        type=str,
        default="",
        help="Comma-separated arm IDs to run (default: all)",
    )
    return parser


def _write_runtime_summary(report, output_dir: Path) -> Path:
    arm_rows: dict[str, list[dict]] = {}
    for dr in report.drug_results.values():
        for arm_id, ar in dr.arm_results.items():
            arm_rows.setdefault(arm_id, []).append(
                {
                    "wall_clock_seconds": float(ar.wall_clock_seconds),
                    "success": ar.prediction is not None and not ar.error,
                    "error": ar.error or "",
                }
            )

    path = output_dir / "runtime_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "arm_id",
                "n_runs",
                "n_success",
                "n_failed",
                "mean_seconds",
                "median_seconds",
                "p95_seconds",
            ],
        )
        writer.writeheader()

        for arm_id in sorted(arm_rows):
            rows = arm_rows[arm_id]
            times = [r["wall_clock_seconds"] for r in rows]
            successes = sum(1 for r in rows if r["success"])
            failures = len(rows) - successes
            if len(times) >= 20:
                p95 = statistics.quantiles(times, n=20, method="inclusive")[-1]
            else:
                p95 = max(times) if times else 0.0

            writer.writerow(
                {
                    "arm_id": arm_id,
                    "n_runs": len(rows),
                    "n_success": successes,
                    "n_failed": failures,
                    "mean_seconds": round(statistics.fmean(times), 3) if times else 0.0,
                    "median_seconds": round(statistics.median(times), 3)
                    if times
                    else 0.0,
                    "p95_seconds": round(p95, 3),
                }
            )

    return path


async def _main() -> None:
    args = _build_parser().parse_args()

    settings = Settings(
        openai_workers=args.openai_workers,
        anthropic_workers=args.anthropic_workers,
        qdrant_in_memory=args.qdrant_in_memory,
        qdrant_fallback_to_in_memory=not args.no_qdrant_fallback,
        staged_points_dir=args.staged_points_dir,
        use_staged_vectors=args.use_staged_vectors,
    )

    # Provider SDKs read API keys from environment variables.
    # Export values loaded via Settings/.env for this process.
    env_exports = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "TAVILY_API_KEY": settings.tavily_api_key,
    }
    for key, value in env_exports.items():
        if value and key not in os.environ:
            os.environ[key] = value

    runner = ExperimentRunner(
        settings=settings,
        cache_dir=args.cache_dir,
        resume=not args.no_resume,
        run_fn_analysis=args.run_fn_analysis,
        run_evidence_quality=args.run_evidence_quality,
        run_sensitivity=args.run_sensitivity,
        openai_workers=args.openai_workers,
        anthropic_workers=args.anthropic_workers,
        skip_index_build=args.skip_index_build,
        use_staged=args.use_staged,
        use_staged_vectors=args.use_staged_vectors,
    )

    try:
        arm_ids = [a.strip() for a in args.arm_ids.split(",") if a.strip()]
        report = await runner.run(drug_limit=args.drugs, arm_ids=arm_ids or None)
    finally:
        await runner.close()

    output_dir = Path(args.output_dir)
    artefacts = export_results(report, output_dir=output_dir)
    runtime_csv = _write_runtime_summary(report, output_dir)

    print("\n=== Frontier Experiment Complete ===")
    print(
        f"drugs={report.n_drugs} arms={report.n_arms} wall_clock={report.wall_clock_seconds:.1f}s"
    )
    print(f"workers: openai={args.openai_workers} anthropic={args.anthropic_workers}")
    print(f"skip_index_build={args.skip_index_build} use_staged={args.use_staged}")
    print(f"use_staged_vectors={args.use_staged_vectors} staged_points_dir={args.staged_points_dir}")
    print(
        f"qdrant_in_memory={settings.qdrant_in_memory} fallback={settings.qdrant_fallback_to_in_memory}"
    )
    print(f"report_json={artefacts['report_json']}")
    print(f"runtime_summary={runtime_csv}")

    if args.subset_best and args.subset_best > 0:
        subset_dir = output_dir / f"subset_top_{args.subset_best}"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/subset_results.py",
            str(artefacts["report_json"]),
            "--min-drugs",
            str(args.subset_best),
            "--max-drugs",
            str(args.subset_best),
            "--output-dir",
            str(subset_dir),
        ]
        print("\nRunning subset filter:")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(
                f"Subset filter failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
        else:
            print(f"Subset output: {subset_dir}")


if __name__ == "__main__":
    asyncio.run(_main())
