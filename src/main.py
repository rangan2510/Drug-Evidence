"""CLI entry point for the drug-disease prediction experiment.

Usage::

    uv run python -m src.main
    uv run python -m src.main --drugs 10 --arms pipeline-gpt-oss pipeline-llama4
    uv run python -m src.main --no-resume --cache-dir .cache/fresh
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config.settings import Settings
from src.experiment.arms import ALL_ARMS
from src.experiment.orchestrator import ExperimentRunner
from src.viz.export import export_results
from src.viz.plots import generate_all_plots
from src.viz.report import write_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.main",
        description="Drug-disease prediction experiment orchestrator.",
    )
    parser.add_argument(
        "--drugs",
        type=int,
        default=None,
        help="Number of drugs to evaluate (default: settings.target_drugs = 200).",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="Arm IDs to run (default: all 8). Example: pipeline-gpt41 websearch-gpt52",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for CSV/JSON/plot outputs (default: results/).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/results",
        help="Directory for the result cache (default: .cache/results).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run everything.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--skip-fn-analysis",
        action="store_true",
        default=False,
        help="Skip false-negative categorisation.",
    )
    parser.add_argument(
        "--run-evidence-quality",
        action="store_true",
        default=False,
        help="Run evidence quality evaluation (hits PubMed, adds latency).",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        default=False,
        help="Skip sensitivity weight/threshold sweeps.",
    )
    parser.add_argument(
        "--skip-index-build",
        action="store_true",
        default=False,
        help="Skip Qdrant index build (use existing persistent collection).",
    )
    parser.add_argument(
        "--openai-workers",
        type=int,
        default=None,
        help="Max concurrent OpenAI workers (default: settings.openai_workers).",
    )
    parser.add_argument(
        "--anthropic-workers",
        type=int,
        default=None,
        help="Max concurrent Anthropic workers (default: settings.anthropic_workers).",
    )
    parser.add_argument(
        "--qdrant-in-memory",
        action="store_true",
        default=False,
        help="Force in-memory Qdrant (no external server).",
    )
    parser.add_argument(
        "--no-qdrant-fallback",
        action="store_true",
        default=False,
        help="Disable automatic fallback to in-memory Qdrant when healthcheck fails.",
    )
    parser.add_argument(
        "--use-staged",
        action="store_true",
        default=False,
        help="Use pre-staged MongoDB evidence instead of live API aggregation.",
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help=(
            "Path to a text file with one drug name per line. "
            "Overrides CTD association-count filter for drug selection. "
            "Combined with --drugs N to limit to the first N entries."
        ),
    )
    return parser


def _validate_arms(arm_ids: list[str] | None) -> list[str] | None:
    """Validate that requested arm IDs exist."""
    if arm_ids is None:
        return None
    unknown = [a for a in arm_ids if a not in ALL_ARMS]
    if unknown:
        valid = ", ".join(sorted(ALL_ARMS.keys()))
        print(f"ERROR: Unknown arm(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Valid arms: {valid}", file=sys.stderr)
        sys.exit(1)
    return arm_ids


async def _async_main(args: argparse.Namespace) -> None:
    """Run the experiment asynchronously."""
    import os

    settings_overrides: dict[str, object] = {}
    if args.openai_workers is not None:
        settings_overrides["openai_workers"] = args.openai_workers
    if args.anthropic_workers is not None:
        settings_overrides["anthropic_workers"] = args.anthropic_workers
    if args.qdrant_in_memory:
        settings_overrides["qdrant_in_memory"] = True
    if args.no_qdrant_fallback:
        settings_overrides["qdrant_fallback_to_in_memory"] = False

    settings = Settings(**settings_overrides)

    # PydanticAI providers look up API keys via os.environ, not through
    # our Settings object.  Ensure .env values are visible to them.
    _ENV_EXPORTS = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "TAVILY_API_KEY": settings.tavily_api_key,
    }
    for key, value in _ENV_EXPORTS.items():
        if value and key not in os.environ:
            os.environ[key] = value

    arm_ids = _validate_arms(args.arms)

    runner = ExperimentRunner(
        settings=settings,
        cache_dir=args.cache_dir,
        resume=not args.no_resume,
        run_fn_analysis=not args.skip_fn_analysis,
        run_evidence_quality=args.run_evidence_quality,
        run_sensitivity=not args.skip_sensitivity,
        openai_workers=settings.openai_workers,
        anthropic_workers=settings.anthropic_workers,
        skip_index_build=args.skip_index_build,
        use_staged=args.use_staged,
        target_file=args.target_file,
    )

    try:
        report = await runner.run(
            drug_limit=args.drugs,
            arm_ids=arm_ids,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        print(
            f"Drugs: {report.n_drugs}  |  Arms: {report.n_arms}  |  "
            f"Time: {report.wall_clock_seconds:.1f}s  |  "
            f"Cached: {report.cached_results}"
        )
        print()
        print(report.difficulty_summary())
        print()
        print(report.summary_table())
        print("=" * 70)

        # Export CSV/JSON artefacts
        artefacts = export_results(report, output_dir=args.output_dir)
        print(f"\nExported {len(artefacts)} artefacts to {args.output_dir}/:")
        for name, path in sorted(artefacts.items()):
            print(f"  {name}: {path}")

        # Generate plots (pass through eval data from report)
        plot_dir = Path(args.output_dir) / "plots"
        plot_paths = generate_all_plots(
            report,
            output_dir=plot_dir,
            heatmap_data=report.heatmap_data or None,
            fn_summaries=report.fn_summaries or None,
            evidence_quality=report.evidence_quality or None,
        )
        if plot_paths:
            print(f"\nGenerated {len(plot_paths)} plots in {plot_dir}/:")
            for name, path in sorted(plot_paths.items()):
                print(f"  {name}: {path}")

        # Generate markdown report (pass through eval data from report)
        report_path = write_report(
            report,
            output_dir=args.output_dir,
            plot_paths=plot_paths,
            evidence_quality=report.evidence_quality or None,
            fn_summaries=report.fn_summaries or None,
        )
        print(f"\nReport: {report_path}")

    finally:
        await runner.close()


def main() -> None:
    """Parse args and run the experiment."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
