#!/usr/bin/env python3
"""RAG Evaluation Q&A Dataset Generation CLI.

This script generates Question & Answer pairs specifically designed for
evaluating RAG (Retrieval-Augmented Generation) systems.

Key differences from regular QA generation (generate.py):
- Uses RAG chunks (~1000 tokens each) instead of graph chunks (~1M tokens)
- Processes chunks in batches of ~100 at a time
- Tracks source_chunk_ids for each Q&A pair
- Category is always "rag_evaluation"
- No 5-pillar category system

Usage:
    # Generate RAG evaluation Q&A for wheel_of_time
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time

    # Start from a specific batch (for resuming)
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time --start-batch 5

    # Process only a few batches (for testing)
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time --max-batches 2

    # Show statistics only
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time --stats-only

    # Custom batch size
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time --batch-size 50
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from src.utils.logger import get_logger

from .rag_eval_service import RAGEvalQAService

logger = get_logger(__name__)


async def main(args: argparse.Namespace) -> int:
    """Main entry point for RAG evaluation Q&A generation.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    log = logger.bind(series=args.series, mode="rag_evaluation")
    log.info("starting_rag_eval_generation", args=vars(args))

    service = RAGEvalQAService(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        requests_per_minute=args.rpm,
        requests_per_day=args.rpd,
        tokens_per_minute=args.tpm,
    )

    try:
        await service.initialize()

        # Stats only mode
        if args.stats_only:
            stats = await service.get_rag_eval_stats(args.series)
            log.info("rag_eval_statistics", **stats)
            print("\n" + "=" * 60)
            print(f"RAG Evaluation Statistics for '{args.series}'")
            print("=" * 60)
            print(f"Total Q&A Pairs: {stats['total_qa_pairs']}")
            print(f"Unique Batches Processed: {stats['unique_batches']}")
            print(f"Unique Source Chunks Referenced: {stats['unique_source_chunks']}")
            print()
            return 0

        # Process series
        progress = await service.process_series(
            series=args.series,
            skip_processed=not args.no_skip_processed,
            start_batch=args.start_batch,
            max_batches=args.max_batches,
        )

        # Print summary
        duration = datetime.now(timezone.utc) - progress.started_at
        log.info(
            "rag_eval_generation_complete",
            series=progress.series,
            total_chunks=progress.total_chunks,
            processed_batches=progress.processed_batches,
            total_batches=progress.total_batches,
            total_qa_pairs=progress.total_qa_pairs,
            failed_batches=progress.failed_batches,
            duration=str(duration),
        )

        print("\n" + "=" * 60)
        print("RAG Evaluation Q&A Generation Complete")
        print("=" * 60)
        print(f"Series: {progress.series}")
        print(f"Total RAG Chunks: {progress.total_chunks}")
        print(f"Batches Processed: {progress.processed_batches}/{progress.total_batches}")
        print(f"Q&A Pairs Generated: {progress.total_qa_pairs}")
        print(f"Failed Batches: {len(progress.failed_batches)}")
        print(f"Duration: {duration}")

        if progress.failed_batches:
            print(f"\nFailed Batch Indices: {progress.failed_batches}")

        # Get final stats
        stats = await service.get_rag_eval_stats(args.series)
        print("\n" + "-" * 60)
        print("Overall RAG Evaluation Statistics:")
        print(f"  Total Q&A Pairs: {stats['total_qa_pairs']}")
        print(f"  Unique Source Chunks: {stats['unique_source_chunks']}")
        print()

        return 0 if not progress.failed_batches else 1

    except KeyboardInterrupt:
        log.warning("interrupted_by_user")
        return 130

    except Exception as e:
        log.exception("rag_eval_generation_failed", error=str(e))
        return 1

    finally:
        await service.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate RAG evaluation Q&A pairs from RAG chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate RAG evaluation Q&A for wheel_of_time
  %(prog)s --series wheel_of_time

  # Start from batch 5 (resume interrupted processing)
  %(prog)s --series wheel_of_time --start-batch 5

  # Process only 2 batches (testing)
  %(prog)s --series wheel_of_time --max-batches 2

  # Show current statistics
  %(prog)s --series wheel_of_time --stats-only

  # Custom batch size (50 chunks instead of 100)
  %(prog)s --series wheel_of_time --batch-size 50
""",
    )

    parser.add_argument(
        "--series",
        type=str,
        default="wheel_of_time",
        help="Series identifier (default: wheel_of_time)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per batch (default: 100, ~100k tokens)",
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Batch index to start from (for resuming, default: 0)",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process (for testing, default: all)",
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

    # Rate limiting
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
        default=1_000_000,
        help="Input tokens per minute limit (default: 1000000)",
    )

    parser.add_argument(
        "--no-skip-processed",
        action="store_true",
        help="Reprocess batches that already have Q&A pairs",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't generate new Q&A pairs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
