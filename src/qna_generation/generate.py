#!/usr/bin/env python3
"""QA Dataset Generation CLI.

This script generates Question & Answer pairs from graph chunks for RAG evaluation.
Questions are organized into 5 categories (pillars) for comprehensive coverage.

Categories:
    1. characters - Who people really are (names, origins, roles)
    2. events     - What happened (battles, deaths, plot points)
    3. magic      - How the system works (weaves, techniques, abilities)
    4. artifacts  - Special objects/locations (ter'angreal, places)
    5. lore       - Deep history & metaphysics (prophecies, cosmology)

Usage:
    # Generate "characters" category for all chunks
    uv run python -m src.qna_generation.generate --category characters

    # Generate all 5 categories (run separately to manage quotas)
    uv run python -m src.qna_generation.generate --category characters
    uv run python -m src.qna_generation.generate --category events
    uv run python -m src.qna_generation.generate --category magic
    uv run python -m src.qna_generation.generate --category artifacts
    uv run python -m src.qna_generation.generate --category lore

    # Process specific chunks only
    uv run python -m src.qna_generation.generate --category characters \\
        --chunk-ids wheel_of_time_section_01 wheel_of_time_section_02

    # Show statistics only
    uv run python -m src.qna_generation.generate --stats-only

Rate Limits (Gemini 3 Flash Paid Tier 1):
    - Requests per minute: 1,000 (default)
    - Requests per day: 10,000 (default)
    - Input tokens per minute: 1,000,000 (default)

Expected Output:
    - 47 chunks × 5 categories × ~15-25 questions = ~3,500-5,800 questions
"""

import argparse
import asyncio
import sys
from datetime import datetime

import structlog

# Configure logging before other imports
from src.utils.logger import get_logger

from .prompts import CATEGORY_INFO, QuestionCategory
from .service import QAGenerationService

logger = get_logger(__name__)


def print_categories() -> None:
    """Print available question categories with descriptions."""
    print("\nAvailable Question Categories:")
    print("-" * 60)
    for cat in QuestionCategory:
        info = CATEGORY_INFO[cat]
        print(f"\n  {cat.value:<12} - {info.name}")
        print(f"                 {info.description}")
    print()


async def main(args: argparse.Namespace) -> int:
    """Main entry point for QA generation.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Show categories if requested
    if args.list_categories:
        print_categories()
        return 0

    log = logger.bind(series=args.series, category=args.category)
    log.info("starting_qa_generation", args=vars(args))

    service = QAGenerationService(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        requests_per_minute=args.rpm,
        requests_per_day=args.rpd,
        tokens_per_minute=args.tpm,
    )

    try:
        await service.initialize()

        # Stats only mode
        if args.stats_only:
            stats = await service.get_qa_stats(args.series)
            log.info("qa_statistics", **stats)
            print("\n" + "=" * 60)
            print(f"QA Statistics for '{args.series}'")
            print("=" * 60)
            print(f"Total QA Pairs: {stats['total_qa_pairs']}")
            print(f"Unique Source Chunks: {stats['unique_source_chunks']}")
            print("\nBy Category:")
            for cat, count in stats.get("by_category", {}).items():
                print(f"  {cat}: {count}")
            print("\nBy Complexity:")
            for comp, count in stats.get("by_complexity", {}).items():
                print(f"  {comp}: {count}")
            print()
            return 0

        # Validate category
        try:
            category = QuestionCategory(args.category)
        except ValueError:
            log.error("invalid_category", category=args.category)
            print(f"\nError: Invalid category '{args.category}'")
            print_categories()
            return 1

        # Process chunks
        chunk_ids = args.chunk_ids if args.chunk_ids else None
        progress = await service.process_series(
            series=args.series,
            category=category,
            skip_processed=not args.no_skip_processed,
            chunk_ids=chunk_ids,
        )

        # Print final summary
        duration = datetime.now() - progress.started_at.replace(tzinfo=None)
        log.info(
            "generation_complete",
            series=progress.series,
            category=progress.category,
            total_chunks=progress.total_chunks,
            processed_chunks=progress.processed_chunks,
            total_qa_pairs=progress.total_qa_pairs,
            failed_chunks=progress.failed_chunks,
            duration=str(duration),
        )

        print("\n" + "=" * 60)
        print(f"Generation Complete: {progress.category}")
        print("=" * 60)
        print(f"Series: {progress.series}")
        print(f"Category: {progress.category}")
        print(f"Chunks Processed: {progress.processed_chunks}/{progress.total_chunks}")
        print(f"QA Pairs Generated: {progress.total_qa_pairs}")
        print(f"Failed Chunks: {len(progress.failed_chunks)}")
        print(f"Duration: {duration}")

        if progress.failed_chunks:
            print(f"\nFailed Chunk IDs: {', '.join(progress.failed_chunks)}")

        # Get final stats
        stats = await service.get_qa_stats(args.series)
        print("\n" + "-" * 60)
        print("Overall Statistics:")
        print(f"  Total QA Pairs (all categories): {stats['total_qa_pairs']}")
        print()

        return 0 if not progress.failed_chunks else 1

    except KeyboardInterrupt:
        log.warning("interrupted_by_user")
        return 130

    except Exception as e:
        log.exception("generation_failed", error=str(e))
        return 1

    finally:
        await service.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from graph chunks for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate questions for the "characters" category
  %(prog)s --category characters

  # Generate questions for the "magic" category
  %(prog)s --category magic

  # Process only specific chunks
  %(prog)s --category events --chunk-ids wheel_of_time_section_01

  # Show current statistics
  %(prog)s --stats-only

  # List all available categories
  %(prog)s --list-categories
""",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="characters",
        choices=["characters", "events", "magic", "artifacts", "lore"],
        help="Question category to generate (default: characters)",
    )

    parser.add_argument(
        "--series",
        type=str,
        default="wheel_of_time",
        help="Series identifier (default: wheel_of_time)",
    )

    parser.add_argument(
        "--chunk-ids",
        type=str,
        nargs="+",
        help="Specific chunk IDs to process (default: all chunks)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=65536,
        help="Maximum output tokens per request (default: 65536)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature (default: 0.3)",
    )

    # Rate limiting arguments
    parser.add_argument(
        "--rpm",
        type=int,
        default=1000,
        help="Requests per minute limit (default: 1000)",
    )

    parser.add_argument(
        "--rpd",
        type=int,
        default=10000,
        help="Requests per day limit (default: 10000)",
    )

    parser.add_argument(
        "--tpm",
        type=int,
        default=1000000,
        help="Input tokens per minute limit (default: 1000000)",
    )

    parser.add_argument(
        "--no-skip-processed",
        action="store_true",
        help="Reprocess chunks that already have QA pairs for this category",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't generate new QA pairs",
    )

    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available question categories and exit",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
