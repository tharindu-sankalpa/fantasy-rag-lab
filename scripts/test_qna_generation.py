#!/usr/bin/env python3
"""Test script for QA generation service.

This script validates the QA generation pipeline by:
1. Testing MongoDB connection and wot_qna collection operations
2. Testing rate limiter functionality
3. Testing QA generation for a single chunk with a category (optional)

Usage:
    # Basic connectivity test
    uv run python scripts/test_qna_generation.py

    # Full test with actual generation (requires API key and chunks)
    uv run python scripts/test_qna_generation.py --generate --category characters

    # Show existing QA stats
    uv run python scripts/test_qna_generation.py --stats
"""

import argparse
import asyncio
import sys

import structlog

# Initialize logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_mongodb_operations() -> bool:
    """Test MongoDB wot_qna collection operations.

    Returns:
        True if all tests pass, False otherwise.
    """
    log = logger.bind(test="mongodb_operations")
    log.info("starting_mongodb_test")

    from src.services.mongodb_service import MongoDBService

    try:
        service = MongoDBService()
        await service.connect()
        log.info("mongodb_connected")

        # Test create indexes
        await service.create_indexes()
        log.info("indexes_created")

        # Test upsert
        test_doc = {
            "qa_id": "test_chunk_characters_0000",
            "question": "What is the test question?",
            "answer": "This is a detailed test answer for validation purposes.",
            "category": "characters",
            "complexity": "simple",
            "evidence_quote": "This is the evidence quote from the source.",
            "metadata": {
                "source_chunk_id": "test_chunk",
                "included_books": ["test_book.epub"],
                "series": "test_series",
            },
        }

        qa_id = await service.upsert_qa_pair(test_doc)
        log.info("qa_pair_upserted", qa_id=qa_id)

        # Test get
        retrieved = await service.get_qa_pair(qa_id)
        assert retrieved is not None, "Failed to retrieve QA pair"
        assert retrieved["question"] == test_doc["question"]
        log.info("qa_pair_retrieved", qa_id=qa_id)

        # Test get by chunk
        by_chunk = await service.get_qa_pairs_by_chunk("test_chunk")
        assert len(by_chunk) >= 1, "Failed to get QA pairs by chunk"
        log.info("qa_pairs_by_chunk", count=len(by_chunk))

        # Test stats
        stats = await service.get_qa_stats("test_series")
        log.info("qa_stats", **stats)

        # Cleanup test data
        deleted = await service.delete_qa_pairs_by_series("test_series")
        log.info("test_data_cleaned", deleted=deleted)

        # Get collection stats
        all_stats = await service.get_collection_stats()
        log.info("collection_stats", **all_stats)

        await service.disconnect()
        log.info("mongodb_test_passed")
        return True

    except Exception as e:
        log.exception("mongodb_test_failed", error=str(e))
        return False


async def test_rate_limiter() -> bool:
    """Test rate limiter functionality.

    Returns:
        True if tests pass, False otherwise.
    """
    log = logger.bind(test="rate_limiter")
    log.info("starting_rate_limiter_test")

    from src.qna_generation.service import RateLimiter

    try:
        limiter = RateLimiter(
            requests_per_minute=5,  # Low limit for testing
            requests_per_day=100,
            tokens_per_minute=10000,
        )

        # Test acquiring permits
        for i in range(3):
            await limiter.acquire(estimated_tokens=1000)
            log.info("permit_acquired", request=i + 1)

        # Check status
        status = limiter.get_status()
        log.info("rate_limiter_status", **status)

        assert status["requests_per_minute"]["current"] == 3
        assert status["tokens_per_minute"]["current"] == 3000

        log.info("rate_limiter_test_passed")
        return True

    except Exception as e:
        log.exception("rate_limiter_test_failed", error=str(e))
        return False


async def test_qa_generation(category: str, chunk_id: str = None) -> bool:
    """Test actual QA generation with a chunk.

    Args:
        category: Question category to test
        chunk_id: Specific chunk ID to test with (uses first available if None)

    Returns:
        True if generation succeeds, False otherwise.
    """
    log = logger.bind(test="qa_generation", category=category)
    log.info("starting_qa_generation_test")

    from src.qna_generation.prompts import QuestionCategory
    from src.qna_generation.service import QAGenerationService

    try:
        # Validate category
        try:
            cat = QuestionCategory(category)
        except ValueError:
            log.error("invalid_category", category=category)
            return False

        service = QAGenerationService()
        await service.initialize()
        log.info("service_initialized")

        # Get a chunk to test with
        chunks = await service.mongodb.get_graph_chunks_by_series("wheel_of_time")

        if not chunks:
            log.warning("no_chunks_available_for_testing")
            await service.close()
            return True  # Not a failure, just no data

        # Use specified chunk or first available
        if chunk_id:
            chunk = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
            if not chunk:
                log.error("specified_chunk_not_found", chunk_id=chunk_id)
                await service.close()
                return False
        else:
            chunk = chunks[0]

        log.info(
            "testing_with_chunk",
            chunk_id=chunk["chunk_id"],
            token_count=chunk.get("token_count", "unknown"),
            category=category,
        )

        # Generate QA pairs
        qa_count = await service.process_chunk(chunk, cat)
        log.info("qa_pairs_generated", count=qa_count, category=category)

        # Verify storage
        stored = await service.mongodb.get_qa_pairs_by_chunk(chunk["chunk_id"])
        log.info("qa_pairs_stored", count=len(stored))

        # Get stats
        stats = await service.get_qa_stats("wheel_of_time")
        log.info("final_stats", **stats)

        await service.close()
        log.info("qa_generation_test_passed")
        return True

    except Exception as e:
        log.exception("qa_generation_test_failed", error=str(e))
        return False


async def show_stats() -> None:
    """Display current QA statistics."""
    log = logger.bind(task="show_stats")

    from src.services.mongodb_service import MongoDBService

    try:
        service = MongoDBService()
        await service.connect()

        # Get overall stats
        collection_stats = await service.get_collection_stats()
        log.info("collection_counts", **collection_stats)

        # Get QA-specific stats
        qa_stats = await service.get_qa_stats("wheel_of_time")
        log.info("wheel_of_time_qa_stats", **qa_stats)

        # Get breakdown by category
        pipeline = [
            {"$match": {"metadata.series": "wheel_of_time"}},
            {
                "$group": {
                    "_id": "$category",
                    "count": {"$sum": 1},
                }
            },
        ]
        cursor = service.db.wot_qna.aggregate(pipeline)
        category_breakdown = await cursor.to_list(length=None)
        log.info("qa_by_category", breakdown=category_breakdown)

        # Get breakdown by complexity
        pipeline[1]["$group"]["_id"] = "$complexity"
        cursor = service.db.wot_qna.aggregate(pipeline)
        complexity_breakdown = await cursor.to_list(length=None)
        log.info("qa_by_complexity", breakdown=complexity_breakdown)

        # Print summary
        print("\n" + "=" * 60)
        print("QA Generation Statistics")
        print("=" * 60)
        print(f"\nTotal QA Pairs: {qa_stats.get('total_qa_pairs', 0)}")
        print(f"Unique Source Chunks: {qa_stats.get('unique_source_chunks', 0)}")
        print("\nBy Category:")
        for item in category_breakdown:
            print(f"  {item['_id']}: {item['count']}")
        print("\nBy Complexity:")
        for item in complexity_breakdown:
            print(f"  {item['_id']}: {item['count']}")
        print()

        await service.disconnect()

    except Exception as e:
        log.exception("stats_failed", error=str(e))


async def main(args: argparse.Namespace) -> int:
    """Run tests based on arguments.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    log = logger.bind(mode="test")
    log.info("starting_tests", args=vars(args))

    if args.stats:
        await show_stats()
        return 0

    all_passed = True

    # Always run MongoDB tests
    if not await test_mongodb_operations():
        all_passed = False

    # Always run rate limiter tests
    if not await test_rate_limiter():
        all_passed = False

    # Optionally run generation test
    if args.generate:
        if not await test_qa_generation(args.category, args.chunk_id):
            all_passed = False

    if all_passed:
        log.info("all_tests_passed")
        return 0
    else:
        log.error("some_tests_failed")
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test QA generation service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run actual QA generation test (requires API key and chunks)",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="characters",
        choices=["characters", "events", "magic", "artifacts", "lore"],
        help="Question category to test (default: characters)",
    )

    parser.add_argument(
        "--chunk-id",
        type=str,
        help="Specific chunk ID to test with",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current QA statistics and exit",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
